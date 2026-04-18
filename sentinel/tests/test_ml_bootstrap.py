"""
Tests for MLBootstrap — Monte Carlo confidence intervals.
"""
from __future__ import annotations

import numpy as np
import pytest

from analyzer.ml_bootstrap import MLBootstrap, BootstrapCI


class TestBootstrapMetrics:
    def test_returns_all_three_metrics(self):
        rng = np.random.default_rng(42)
        y = rng.integers(0, 2, size=200)
        p = rng.random(200)
        bs = MLBootstrap(n_simulations=200, seed=7)
        cis = bs.bootstrap_metrics(y, p)
        assert set(cis.keys()) == {"precision", "recall", "roc_auc"}

    def test_ci_contains_point_estimate_on_strong_signal(self):
        """When y and p are perfectly aligned, the AUC CI should hug 1.0."""
        n = 300
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, size=n)
        # Strong-signal probas
        p = y.astype(np.float64) * 0.9 + 0.05 + rng.normal(0, 0.02, n)
        p = np.clip(p, 0, 1)
        bs = MLBootstrap(n_simulations=300, seed=7)
        cis = bs.bootstrap_metrics(y, p, threshold=0.5)
        # Point estimate (p50) should be very high
        assert cis["roc_auc"].p50 > 0.95
        # CI must bracket the point estimate
        assert cis["roc_auc"].p5 <= cis["roc_auc"].p50 <= cis["roc_auc"].p95

    def test_random_data_auc_ci_near_half(self):
        rng = np.random.default_rng(2)
        y = rng.integers(0, 2, size=500)
        p = rng.random(500)
        bs = MLBootstrap(n_simulations=500, seed=7)
        cis = bs.bootstrap_metrics(y, p)
        # Median AUC on genuinely random data clusters around 0.5 ± 0.05
        assert 0.40 <= cis["roc_auc"].p50 <= 0.60
        # CI width should be notable (this is the whole point)
        assert cis["roc_auc"].p95 - cis["roc_auc"].p5 > 0.03

    def test_too_small_sample_returns_empty(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([0.1, 0.9, 0.2, 0.8])
        bs = MLBootstrap(n_simulations=100)
        assert bs.bootstrap_metrics(y, p) == {}

    def test_rejects_mismatched_shapes(self):
        bs = MLBootstrap(n_simulations=100)
        with pytest.raises(ValueError):
            bs.bootstrap_metrics(np.zeros(100), np.zeros(50))


class TestProbabilityAboveBaseline:
    def test_strong_signal_probability_approaches_one(self):
        rng = np.random.default_rng(3)
        n = 300
        y = rng.integers(0, 2, size=n)
        p = y.astype(np.float64) * 0.85 + 0.075 + rng.normal(0, 0.03, n)
        p = np.clip(p, 0, 1)
        bs = MLBootstrap(n_simulations=400, seed=7)
        prob = bs.probability_above_baseline(y, p, baseline_auc=0.5)
        assert prob > 0.95

    def test_random_signal_probability_near_half(self):
        rng = np.random.default_rng(4)
        n = 500
        y = rng.integers(0, 2, size=n)
        p = rng.random(n)
        bs = MLBootstrap(n_simulations=400, seed=7)
        prob = bs.probability_above_baseline(y, p, baseline_auc=0.5)
        assert 0.30 <= prob <= 0.70

    def test_monotonic_in_baseline(self):
        """P(AUC > baseline) must be monotonically non-increasing in baseline."""
        rng = np.random.default_rng(5)
        n = 300
        y = rng.integers(0, 2, size=n)
        p = y.astype(np.float64) * 0.8 + 0.1 + rng.normal(0, 0.05, n)
        p = np.clip(p, 0, 1)
        bs = MLBootstrap(n_simulations=400, seed=7)
        probs = [bs.probability_above_baseline(y, p, baseline_auc=b)
                 for b in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
        for a, b in zip(probs, probs[1:]):
            assert a >= b - 0.03  # monotone modulo sampling noise


class TestCIStructure:
    def test_summary_is_json_safe(self):
        rng = np.random.default_rng(7)
        y = rng.integers(0, 2, size=200)
        p = rng.random(200)
        bs = MLBootstrap(n_simulations=200)
        cis = bs.bootstrap_metrics(y, p)
        for ci in cis.values():
            assert isinstance(ci, BootstrapCI)
            s = ci.summary()
            for k, v in s.items():
                assert isinstance(v, (int, float, str)), f"{k}={v!r}"

    def test_rejects_tiny_n_simulations(self):
        with pytest.raises(ValueError):
            MLBootstrap(n_simulations=10)
