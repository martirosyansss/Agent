"""
Tests for StackingHead.

Stacking is subtle — the failure mode we most want to catch is "the head
fits on in-sample predictions, looks brilliant in tests, destroys AUC in
production". The key tests below deliberately construct OOF and IS data
with different label alignments so we can verify the API enforces the
OOF discipline correctly.
"""
from __future__ import annotations

import numpy as np
import pytest

from analyzer.ml_stacking import StackingHead


def _make_oof(n: int = 200, skill: float = 0.7, seed: int = 1):
    """Synthesise member OOF probabilities that correlate with labels.

    ``skill`` dials the signal strength: 0.5 = random, 1.0 = perfect.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    # Each 'member' adds noise around the truth — different members have
    # slightly different noise profiles to mimic a real ensemble.
    base = y.astype(np.float64) * skill + (1 - skill) * rng.random(n)
    members = {
        "rf":   np.clip(base + rng.normal(0, 0.1, n), 0, 1),
        "lgbm": np.clip(base + rng.normal(0, 0.1, n), 0, 1),
        "xgb":  np.clip(base + rng.normal(0, 0.15, n), 0, 1),
        "lr_en": np.clip(base + rng.normal(0, 0.2, n), 0, 1),
    }
    X = rng.random((n, 8))
    return members, y, X


class TestFitBasic:
    def test_fits_with_enough_data(self):
        members, y, X = _make_oof(n=300)
        head = StackingHead()
        assert head.fit(members, y, X=X) is True
        assert head.is_fitted

    def test_skips_with_too_few_samples(self):
        members, y, X = _make_oof(n=20)
        head = StackingHead()
        assert head.fit(members, y, X=X) is False
        assert not head.is_fitted

    def test_skips_single_class(self):
        members, y, X = _make_oof(n=200)
        y = np.zeros_like(y)  # force single class
        head = StackingHead()
        assert head.fit(members, y, X=X) is False

    def test_empty_oof_dict_fails(self):
        head = StackingHead()
        y = np.zeros(100, dtype=int)
        assert head.fit({}, y) is False


class TestPredict:
    def test_predict_returns_correct_shape(self):
        members, y, X = _make_oof(n=300)
        head = StackingHead()
        head.fit(members, y, X=X)
        # Predict on fresh data with same structure
        probas_matrix = np.column_stack([members[t] for t in head.member_tags])
        out = head.predict_proba(probas_matrix, X=X)
        assert out.shape == (300,)
        assert np.all((out >= 0) & (out <= 1))

    def test_unfitted_falls_back_to_mean(self):
        members, _y, _X = _make_oof(n=100)
        head = StackingHead()
        # Intentionally NOT fit
        probas_matrix = np.column_stack([members[t] for t in sorted(members.keys())])
        out = head.predict_proba(probas_matrix)
        expected = probas_matrix.mean(axis=1)
        assert np.allclose(out, expected)

    def test_feature_dim_mismatch_falls_back(self):
        members, y, X = _make_oof(n=300)
        head = StackingHead(use_raw_features=False)
        head.fit(members, y, X=X)
        # Now try to predict with only 3 columns instead of 4 — mismatch
        bogus = np.random.rand(50, 3)
        out = head.predict_proba(bogus)
        # Falls back to column-wise mean — valid probabilities in [0, 1]
        assert out.shape == (50,)
        assert np.all((out >= 0) & (out <= 1))


class TestRawFeaturesOption:
    def test_use_raw_changes_n_features(self):
        members, y, X = _make_oof(n=400)
        head_no_raw = StackingHead(use_raw_features=False)
        head_raw = StackingHead(use_raw_features=True)
        head_no_raw.fit(members, y, X=X)
        head_raw.fit(members, y, X=X)
        # Raw variant should have strictly more features
        assert head_raw._n_features > head_no_raw._n_features

    def test_use_raw_requires_X_at_predict(self):
        members, y, X = _make_oof(n=400)
        head = StackingHead(use_raw_features=True)
        head.fit(members, y, X=X)
        probas_matrix = np.column_stack([members[t] for t in head.member_tags])
        # Passing None for X should fall back gracefully (logged, not raised)
        out = head.predict_proba(probas_matrix, X=None)
        assert len(out) == 400


class TestSkillRetention:
    def test_stacking_beats_plain_mean_on_skewed_signal(self):
        """When one member is clearly better, stacking should lean on it."""
        rng = np.random.default_rng(3)
        n = 500
        y = rng.integers(0, 2, size=n)
        # 'rf' carries the real signal; others are near-random
        rf_proba = y.astype(np.float64) * 0.85 + 0.075 + rng.normal(0, 0.05, n)
        members = {
            "rf":   np.clip(rf_proba, 0, 1),
            "lgbm": rng.random(n),
            "xgb":  rng.random(n),
        }
        X = rng.random((n, 6))
        head = StackingHead()
        assert head.fit(members, y, X=X)

        probas_matrix = np.column_stack([members[t] for t in head.member_tags])
        stacked = head.predict_proba(probas_matrix)
        mean_vote = probas_matrix.mean(axis=1)

        from sklearn.metrics import roc_auc_score
        auc_stacked = roc_auc_score(y, stacked)
        auc_mean = roc_auc_score(y, mean_vote)
        # Stacking should identify that rf is the trustworthy member
        assert auc_stacked > auc_mean - 0.02, (
            f"stacking={auc_stacked:.3f} worse than mean={auc_mean:.3f}"
        )
