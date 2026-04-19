"""Tests for the Probabilistic Sharpe Ratio / Deflated Sharpe module.

Contracts locked in:

* **Noise returns → PSR ≈ 0.5** — a pure coin-flip series must not pass
  the gate. This is the main thing multiple-testing wants to catch.
* **Strong positive returns → PSR → 1** — persistent edge dominates
  the kurtosis/skew penalties at realistic N.
* **DSR ≤ PSR** — deflation can never *loosen* the test.
* **Negative skew penalises PSR** — a fat left tail matters even when
  the mean looks fine.
* **Gate is conjunctive** — PSR ≥ 0.95 alone is not enough when DSR
  is computed; both must pass.
* **Degenerate inputs** — n < 3, zero-variance, constant returns all
  return PSR = 0, gate_passed = False. No NaN, no crash.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyzer.ml.domain.psr import (
    PSRResult,
    probabilistic_sharpe_ratio,
    _normal_cdf,
    _normal_ppf,
    _sample_skew,
    _sample_excess_kurtosis,
)


class TestNormalUtilities:
    def test_cdf_known_values(self):
        assert _normal_cdf(0.0) == pytest.approx(0.5, abs=1e-12)
        assert _normal_cdf(1.96) == pytest.approx(0.975, abs=1e-3)
        assert _normal_cdf(-1.96) == pytest.approx(0.025, abs=1e-3)

    def test_ppf_inverts_cdf(self):
        for p in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
            assert _normal_cdf(_normal_ppf(p)) == pytest.approx(p, abs=1e-6)

    def test_skew_of_gaussian_near_zero(self):
        rng = np.random.default_rng(42)
        g = rng.standard_normal(10_000)
        assert abs(_sample_skew(g)) < 0.1

    def test_excess_kurtosis_of_gaussian_near_zero(self):
        rng = np.random.default_rng(42)
        g = rng.standard_normal(10_000)
        assert abs(_sample_excess_kurtosis(g)) < 0.2


class TestPSRBasics:
    def test_pure_noise_does_not_pass_gate(self):
        rng = np.random.default_rng(0)
        noise = rng.standard_normal(200) * 0.01
        res = probabilistic_sharpe_ratio(noise)
        # Mean is essentially zero; PSR against SR*=0 should be around 0.5.
        assert 0.2 < res.psr < 0.8
        assert res.gate_passed is False

    def test_strong_positive_edge_passes(self):
        # Mean 0.002 per-period, std 0.01 → per-period Sharpe ≈ 0.2.
        # Over n=500 that's a comfortably significant edge.
        rng = np.random.default_rng(1)
        returns = 0.002 + rng.standard_normal(500) * 0.01
        res = probabilistic_sharpe_ratio(returns)
        assert res.sharpe > 0.15
        assert res.psr > 0.95
        assert res.gate_passed is True

    def test_strong_negative_edge_fails(self):
        rng = np.random.default_rng(2)
        returns = -0.002 + rng.standard_normal(500) * 0.01
        res = probabilistic_sharpe_ratio(returns)
        assert res.sharpe < 0
        assert res.psr < 0.05
        assert res.gate_passed is False

    def test_benchmark_sr_shifts_threshold(self):
        # With mean 0.001/std 0.01 → SR ≈ 0.1 per-period.
        # Against benchmark 0 → should pass; against 0.2 → should fail.
        rng = np.random.default_rng(3)
        returns = 0.001 + rng.standard_normal(500) * 0.01

        res_zero = probabilistic_sharpe_ratio(returns, benchmark_sr=0.0)
        res_high = probabilistic_sharpe_ratio(returns, benchmark_sr=0.2)
        assert res_zero.psr > res_high.psr


class TestDSR:
    def test_dsr_is_none_for_single_trial(self):
        rng = np.random.default_rng(4)
        returns = 0.001 + rng.standard_normal(300) * 0.01
        res = probabilistic_sharpe_ratio(returns, n_trials=1)
        assert res.dsr is None
        assert res.deflated_benchmark is None

    def test_dsr_never_exceeds_psr(self):
        rng = np.random.default_rng(5)
        returns = 0.001 + rng.standard_normal(300) * 0.01
        res = probabilistic_sharpe_ratio(returns, n_trials=20)
        assert res.dsr is not None
        # Deflation shifts the benchmark right → P(sr > shifted) ≤ P(sr > 0).
        assert res.dsr <= res.psr + 1e-9

    def test_more_trials_lower_dsr(self):
        rng = np.random.default_rng(6)
        returns = 0.0015 + rng.standard_normal(300) * 0.01
        res10 = probabilistic_sharpe_ratio(returns, n_trials=10)
        res100 = probabilistic_sharpe_ratio(returns, n_trials=100)
        # More trials ⇒ higher deflated benchmark ⇒ lower DSR.
        assert res100.dsr is not None and res10.dsr is not None
        assert res100.dsr <= res10.dsr + 1e-9

    def test_gate_requires_both(self):
        # Construct a case where PSR passes but DSR (with many trials) fails.
        rng = np.random.default_rng(7)
        # Marginal edge: mean 0.001, std 0.01, n=120 → PSR ~ 0.87, not 0.95.
        # Raise mean slightly so PSR passes but DSR fails after heavy deflation.
        returns = 0.0015 + rng.standard_normal(150) * 0.01
        res = probabilistic_sharpe_ratio(returns, n_trials=1000)
        # Gate is conjunctive; if DSR fails, gate must fail regardless of PSR.
        if res.dsr is not None and res.dsr < 0.95:
            assert res.gate_passed is False


class TestSkewSensitivity:
    def test_negative_skew_reduces_psr_vs_symmetric(self):
        rng = np.random.default_rng(8)
        n = 500
        # Symmetric series with the target mean/std.
        symmetric = 0.0015 + rng.standard_normal(n) * 0.01

        # Skewed version: most returns small and positive, rare large
        # negative losses — same mean/std, but a fat left tail.
        base = rng.standard_normal(n)
        skewed = 0.0015 + 0.01 * (base - 0.5 * (base ** 2 - 1))  # negatively skewed
        # Normalise to roughly the same std so the comparison is about shape.
        skewed = skewed * (symmetric.std() / skewed.std())

        res_sym = probabilistic_sharpe_ratio(symmetric)
        res_skew = probabilistic_sharpe_ratio(skewed)
        # The skewed series has negative skew ⇒ PSR penalty.
        assert res_skew.skewness < 0
        assert res_skew.psr <= res_sym.psr + 1e-6


class TestDegenerate:
    def test_empty_returns(self):
        res = probabilistic_sharpe_ratio(np.array([]))
        assert res.psr == 0.0
        assert res.gate_passed is False
        assert res.n == 0

    def test_too_few_returns(self):
        res = probabilistic_sharpe_ratio(np.array([1.0, 2.0]))
        assert res.psr == 0.0
        assert res.gate_passed is False

    def test_constant_returns(self):
        res = probabilistic_sharpe_ratio(np.full(50, 0.005))
        # Variance = 0 → Sharpe undefined → PSR = 0, gate fails.
        assert res.psr == 0.0
        assert res.sharpe == 0.0
        assert res.gate_passed is False
