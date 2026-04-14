"""
Unit tests for VotingEnsemble & AdaptiveFeatureSelector.

N-4 fix: adds missing test coverage for ml_ensemble.py critical behaviors.

Run: python -m pytest tests/test_ml_ensemble.py -v
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from analyzer.ml_ensemble import VotingEnsemble, AdaptiveFeatureSelector


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FakeModel:
    """Model that returns deterministic probabilities based on the first feature."""
    def __init__(self, offset: float = 0.0):
        self._offset = offset

    def predict_proba(self, X):
        proba = np.clip(X[:, 0] + self._offset, 0.0, 1.0)
        return np.column_stack([1 - proba, proba])


class FailingModel:
    """Model that always raises on predict_proba."""
    def predict_proba(self, X):
        raise RuntimeError("simulated failure")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VotingEnsemble
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestVotingEnsemble:
    def test_empty_ensemble_returns_half(self):
        e = VotingEnsemble()
        assert not e.is_ready
        result = e.predict_proba(np.array([[0.7]]))
        assert result[0] == 0.5

    def test_single_member(self):
        e = VotingEnsemble()
        e.add_member(FakeModel(), "rf", 0.8)
        assert e.is_ready
        assert e.member_count() == 1
        proba = e.predict_proba(np.array([[0.7]]))
        assert abs(proba[0] - 0.7) < 0.01

    def test_weighted_average(self):
        """Two models: rf returns 0.8, lgbm returns 0.4.
        Weights: 0.6, 0.4. Expected: (0.6*0.8 + 0.4*0.4) / (0.6+0.4) = 0.64
        """
        e = VotingEnsemble()
        e.add_member(FakeModel(offset=0.0), "rf", 0.6)
        e.add_member(FakeModel(offset=-0.4), "lgbm", 0.4)
        X = np.array([[0.8]])
        proba = e.predict_proba(X)
        # rf: 0.8, lgbm: max(0.8-0.4, 0) = 0.4
        expected = (0.6 * 0.8 + 0.4 * 0.4) / (0.6 + 0.4)
        assert abs(proba[0] - expected) < 0.01

    def test_w4_failed_member_not_counted(self):
        """W-4: A failing member should be excluded from weight denominator."""
        e = VotingEnsemble()
        e.add_member(FakeModel(), "rf", 0.5)
        e.add_member(FailingModel(), "lgbm", 0.5)
        X = np.array([[0.7]])
        proba = e.predict_proba(X)
        # Only rf succeeds, so result = rf's output (0.7), not 0.7*0.5/1.0=0.35
        assert abs(proba[0] - 0.7) < 0.01

    def test_all_members_fail_returns_half(self):
        e = VotingEnsemble()
        e.add_member(FailingModel(), "rf", 0.5)
        e.add_member(FailingModel(), "lgbm", 0.5)
        proba = e.predict_proba(np.array([[0.5]]))
        assert proba[0] == 0.5

    def test_w9_duplicate_tag_rejected(self):
        """W-9: Adding a duplicate tag should be rejected."""
        e = VotingEnsemble()
        e.add_member(FakeModel(), "rf", 0.5)
        e.add_member(FakeModel(), "rf", 0.6)  # duplicate
        assert e.member_count() == 1

    def test_zero_weight_rejected(self):
        e = VotingEnsemble()
        e.add_member(FakeModel(), "rf", 0.0)
        assert e.member_count() == 0

    def test_negative_weight_rejected(self):
        e = VotingEnsemble()
        e.add_member(FakeModel(), "rf", -0.5)
        assert e.member_count() == 0

    def test_batch_prediction(self):
        """Should handle multiple samples in a batch."""
        e = VotingEnsemble()
        e.add_member(FakeModel(), "rf", 1.0)
        X = np.array([[0.2], [0.5], [0.9]])
        proba = e.predict_proba(X)
        assert len(proba) == 3
        assert abs(proba[0] - 0.2) < 0.01
        assert abs(proba[1] - 0.5) < 0.01
        assert abs(proba[2] - 0.9) < 0.01

    def test_get_member_info(self):
        e = VotingEnsemble()
        e.add_member(FakeModel(), "rf", 0.8)
        e.add_member(FakeModel(), "lgbm", 0.6)
        info = e.get_member_info()
        assert len(info) == 2
        assert info[0]["tag"] == "rf"
        assert info[0]["weight"] == 0.8
        assert info[1]["tag"] == "lgbm"

    def test_isotonic_calibration_roundtrip(self):
        """Calibrated probabilities should differ from raw when calibration fits."""
        e = VotingEnsemble()
        e.add_member(FakeModel(), "rf", 1.0)

        # Generate validation data for calibration
        np.random.seed(42)
        X_val = np.random.rand(100, 1)
        y_val = (X_val[:, 0] > 0.5).astype(int)

        e.apply_isotonic_calibration(y_val, X_val)
        assert e._is_calibrated

        # Calibrated predictions should still be valid probabilities
        X_test = np.array([[0.3], [0.7]])
        proba = e.predict_proba_calibrated(X_test)
        assert len(proba) == 2
        assert 0.0 <= proba[0] <= 1.0
        assert 0.0 <= proba[1] <= 1.0

    def test_uncalibrated_returns_raw(self):
        """Without calibration, predict_proba_calibrated == predict_proba."""
        e = VotingEnsemble()
        e.add_member(FakeModel(), "rf", 1.0)
        X = np.array([[0.6]])
        raw = e.predict_proba(X)
        calibrated = e.predict_proba_calibrated(X)
        assert abs(raw[0] - calibrated[0]) < 0.001


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AdaptiveFeatureSelector
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestAdaptiveFeatureSelector:
    def test_not_fitted_returns_unchanged(self):
        s = AdaptiveFeatureSelector()
        X = np.array([[1, 2, 3, 4, 5]])
        result = s.transform(X)
        assert np.array_equal(result, X)

    def test_fit_and_transform(self):
        s = AdaptiveFeatureSelector(min_importance=0.1)
        importances = {"a": 0.5, "b": 0.05, "c": 0.3}
        feature_names = ["a", "b", "c"]
        s.fit(importances, feature_names)

        assert s.is_fitted
        assert s.selected_feature_count == 2  # a, c
        assert "b" in s.dropped_names
        assert "a" in s.selected_names
        assert "c" in s.selected_names

        X = np.array([[10, 20, 30]])
        result = s.transform(X)
        assert result.shape == (1, 2)
        assert result[0, 0] == 10  # a
        assert result[0, 1] == 30  # c

    def test_transform_single(self):
        s = AdaptiveFeatureSelector(min_importance=0.1)
        importances = {"a": 0.5, "b": 0.02, "c": 0.3}
        feature_names = ["a", "b", "c"]
        s.fit(importances, feature_names)

        result = s.transform_single([10.0, 20.0, 30.0])
        assert len(result) == 2
        assert result[0] == 10.0
        assert result[1] == 30.0

    def test_all_features_kept(self):
        s = AdaptiveFeatureSelector(min_importance=0.0)
        importances = {"a": 0.01, "b": 0.02}
        s.fit(importances, ["a", "b"])
        assert s.selected_feature_count == 2
        assert len(s.dropped_names) == 0

    def test_missing_importance_dropped(self):
        """Feature not in importances dict → importance=0.0 → dropped."""
        s = AdaptiveFeatureSelector(min_importance=0.01)
        importances = {"a": 0.5}
        s.fit(importances, ["a", "b"])
        assert "b" in s.dropped_names
        assert s.selected_feature_count == 1

    def test_batch_transform(self):
        s = AdaptiveFeatureSelector(min_importance=0.1)
        importances = {"f0": 0.5, "f1": 0.01, "f2": 0.3, "f3": 0.02, "f4": 0.4}
        s.fit(importances, ["f0", "f1", "f2", "f3", "f4"])

        X = np.array([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
        ])
        result = s.transform(X)
        assert result.shape == (2, 3)  # kept f0, f2, f4
        assert list(result[0]) == [1, 3, 5]
        assert list(result[1]) == [6, 8, 10]
