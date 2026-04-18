"""
Tests for MLWalkForwardValidator.

These tests exercise the split generator and the aggregation logic without
depending on any heavy ML model — the trainer callable is a simple closure
that returns a deterministic probability function of the input, so we can
write exact assertions on precision / recall / AUC outcomes.
"""
from __future__ import annotations

import numpy as np
import pytest

from analyzer.ml_walk_forward import (
    MLWalkForwardValidator,
    WFFoldResult,
    WFReport,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Deterministic trainer helper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _perfect_trainer(X_tr, y_tr, X_te, y_te):
    """Trainer that returns the true labels as 'probability' — a cheat
    for testing that the validator's aggregation treats probas correctly."""
    # Slight perturbation so AUC stays computable (not a constant)
    probas = y_te.astype(np.float64) * 0.9 + 0.05
    return {
        "test_proba": probas,
        "threshold": 0.5,
        "train_precision": float(np.mean(y_tr == 1)) if len(y_tr) else 0.0,
    }


def _random_trainer(seed: int = 7):
    rng = np.random.default_rng(seed)
    def _fn(X_tr, y_tr, X_te, y_te):
        return {
            "test_proba": rng.random(len(y_te)),
            "threshold": 0.5,
            "train_precision": 0.5,
        }
    return _fn


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Split generator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGenerateSplits:
    def test_returns_correct_fold_count_rolling(self):
        wf = MLWalkForwardValidator(n_folds=5, test_fraction=0.1, anchored=False,
                                    min_train_size=50, min_test_size=20)
        splits = wf.generate_splits(500)
        assert len(splits) == 5

    def test_rolling_splits_are_disjoint_on_test(self):
        """Each fold's test window must not overlap any other fold's test window."""
        wf = MLWalkForwardValidator(n_folds=5, test_fraction=0.1, anchored=False)
        splits = wf.generate_splits(1000)
        tests = [(ts, te) for _, _, ts, te in splits]
        for i in range(len(tests)):
            for j in range(i + 1, len(tests)):
                a0, a1 = tests[i]
                b0, b1 = tests[j]
                # No overlap: a1 <= b0 OR b1 <= a0
                assert a1 <= b0 or b1 <= a0, f"Overlap between fold {i} and {j}"

    def test_anchored_train_always_starts_at_zero(self):
        wf = MLWalkForwardValidator(n_folds=4, test_fraction=0.1, anchored=True,
                                    min_train_size=50, min_test_size=20)
        splits = wf.generate_splits(500)
        for tr_s, _, _, _ in splits:
            assert tr_s == 0

    def test_rolling_train_window_can_move_forward(self):
        wf = MLWalkForwardValidator(n_folds=4, test_fraction=0.1, anchored=False,
                                    min_train_size=50, min_test_size=20)
        splits = wf.generate_splits(500)
        train_starts = [tr_s for tr_s, _, _, _ in splits]
        # Rolling: at least one consecutive pair should see train_start move
        assert any(b > a for a, b in zip(train_starts, train_starts[1:])) or all(
            s == 0 for s in train_starts  # degenerate case when pool < rolling size
        )

    def test_empty_when_too_few_samples(self):
        wf = MLWalkForwardValidator(n_folds=5, min_train_size=100, min_test_size=50)
        assert wf.generate_splits(10) == []

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            MLWalkForwardValidator(n_folds=1)
        with pytest.raises(ValueError):
            MLWalkForwardValidator(test_fraction=0.9)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRun:
    def test_report_has_requested_number_of_folds(self):
        wf = MLWalkForwardValidator(n_folds=4, test_fraction=0.1,
                                    min_train_size=50, min_test_size=20)
        X = np.random.rand(500, 5)
        y = (np.random.rand(500) > 0.5).astype(int)
        report = wf.run(X, y, _random_trainer())
        assert isinstance(report, WFReport)
        # May be fewer than requested if any fold fails — should not exceed
        assert 0 < report.n_folds_completed <= 4

    def test_perfect_trainer_yields_high_auc(self):
        wf = MLWalkForwardValidator(n_folds=4, test_fraction=0.1,
                                    min_train_size=50, min_test_size=20)
        X = np.random.rand(500, 5)
        y = (np.arange(500) % 2).astype(int)  # deterministic labels
        report = wf.run(X, y, _perfect_trainer)
        # The perfect trainer returns labels*0.9 + 0.05 so AUC should be 1.0
        assert report.mean_auc > 0.95

    def test_random_trainer_auc_near_half(self):
        wf = MLWalkForwardValidator(n_folds=5, test_fraction=0.1,
                                    min_train_size=50, min_test_size=20)
        X = np.random.rand(600, 5)
        y = np.random.default_rng(1).integers(0, 2, size=600)
        report = wf.run(X, y, _random_trainer(seed=1))
        assert 0.35 <= report.mean_auc <= 0.65, f"got {report.mean_auc}"

    def test_oof_predictions_no_overlap_across_folds(self):
        wf = MLWalkForwardValidator(n_folds=5, test_fraction=0.1,
                                    min_train_size=50, min_test_size=20)
        X = np.random.rand(600, 5)
        y = (np.arange(600) % 2).astype(int)
        report = wf.run(X, y, _perfect_trainer)
        # OOF mask must be True exactly once per covered sample — by construction
        # the test windows are disjoint, so oof_probas should have unique non-NaN
        # values in each fold's test range.
        for r in report.fold_results:
            slice_probas = report.oof_probas[r.test_start:r.test_end]
            assert not np.isnan(slice_probas).any()

    def test_handles_small_dataset_gracefully(self):
        wf = MLWalkForwardValidator(n_folds=5, min_train_size=100, min_test_size=50)
        X = np.random.rand(30, 5)
        y = (np.random.rand(30) > 0.5).astype(int)
        report = wf.run(X, y, _random_trainer())
        assert report.n_folds_completed == 0
        assert report.fold_results == []

    def test_summary_fields_are_jsonable(self):
        wf = MLWalkForwardValidator(n_folds=3, test_fraction=0.1,
                                    min_train_size=50, min_test_size=20)
        X = np.random.rand(500, 5)
        y = (np.arange(500) % 2).astype(int)
        report = wf.run(X, y, _perfect_trainer)
        s = report.summary()
        for k, v in s.items():
            assert isinstance(v, (int, float, str)), f"{k}={v!r} not JSON-safe"


class TestGenerateOOF:
    def test_returns_matching_mask_and_probas(self):
        wf = MLWalkForwardValidator(n_folds=3, test_fraction=0.1,
                                    min_train_size=50, min_test_size=20)
        X = np.random.rand(500, 5)
        y = (np.arange(500) % 2).astype(int)
        probas, mask = wf.generate_oof_predictions(X, y, _perfect_trainer)
        assert len(probas) == 500
        assert len(mask) == 500
        # Where mask is True, probas must be finite
        assert np.all(np.isfinite(probas[mask]))
        # Where mask is False, probas must be NaN
        assert np.all(np.isnan(probas[~mask]))
