"""Tests for the minimal Gaussian HMM regime detector.

Contracts:

* **Two-regime synthetic data recovers structure** — given returns
  generated from a mixture of two Gaussians with clearly different
  variance, the fit's ``means`` or ``variances`` must span similar
  magnitudes to the ground truth.
* **Viterbi output is valid** — every state index ∈ [0, K), length
  matches the input.
* **Posterior rows sum to 1** — ``predict_proba`` rows are valid
  probability distributions.
* **Numerical stability** — tiny-variance inputs don't blow up; the
  ``_MIN_VAR`` floor does its job.
* **Multi-start improves likelihood** — ``n_starts > 1`` should never
  produce a worse log-likelihood than ``n_starts = 1``.
* **Degenerate input** — too few samples raise clearly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyzer.ml.domain.hmm_regime import (
    GaussianHMMFit,
    _forward_backward,
    _gaussian_log_pdf,
    _log_sum_exp,
    fit_gaussian_hmm,
)


class TestUtilities:
    def test_log_sum_exp_matches_numpy(self):
        a = np.array([[1.0, 2.0, 3.0], [-10.0, 0.0, 5.0]])
        # Direct computation
        expected = np.log(np.exp(a).sum(axis=1))
        assert np.allclose(_log_sum_exp(a, axis=1), expected)

    def test_gaussian_log_pdf_at_mean(self):
        # log N(μ | μ, σ²) = -0.5 * log(2πσ²)
        val = _gaussian_log_pdf(np.array([5.0]), 5.0, 1.0)
        expected = -0.5 * np.log(2 * np.pi * 1.0)
        assert np.allclose(val[0], expected)


class TestFit:
    def test_two_regime_synthetic_recovery(self):
        rng = np.random.default_rng(0)
        # 400 samples of low-vol (σ=1), then 400 of high-vol (σ=5).
        low = rng.normal(0, 1.0, size=400)
        high = rng.normal(0, 5.0, size=400)
        series = np.concatenate([low, high])

        fit = fit_gaussian_hmm(series, n_states=2, max_iter=50, n_starts=3)
        # One state must have markedly higher variance than the other.
        v_sorted = np.sort(fit.variances)
        assert v_sorted[1] / v_sorted[0] > 3.0

    def test_viterbi_returns_valid_state_sequence(self):
        rng = np.random.default_rng(1)
        series = rng.normal(0, 1, size=300)
        fit = fit_gaussian_hmm(series, n_states=2, n_starts=2)
        states = fit.predict(series)
        assert states.shape == (300,)
        assert states.min() >= 0
        assert states.max() < 2

    def test_predict_proba_normalised(self):
        rng = np.random.default_rng(2)
        series = rng.normal(0, 1, size=200)
        fit = fit_gaussian_hmm(series, n_states=2, n_starts=1, max_iter=20)
        proba = fit.predict_proba(series)
        assert proba.shape == (200, 2)
        row_sums = proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-8)


class TestStability:
    def test_tiny_variance_series_does_not_blow_up(self):
        # Constant-ish series — variance floor should kick in.
        series = np.full(200, 5.0) + np.random.default_rng(3).normal(0, 1e-10, size=200)
        fit = fit_gaussian_hmm(series, n_states=2, n_starts=1, max_iter=10)
        assert np.all(np.isfinite(fit.means))
        assert np.all(fit.variances > 0)

    def test_multi_start_never_worse_than_single(self):
        rng = np.random.default_rng(4)
        series = np.concatenate([
            rng.normal(0, 1, size=200),
            rng.normal(0, 3, size=200),
        ])
        fit_1 = fit_gaussian_hmm(series, n_states=2, n_starts=1, seed=0)
        fit_5 = fit_gaussian_hmm(series, n_states=2, n_starts=5, seed=0)
        assert fit_5.log_likelihood >= fit_1.log_likelihood - 1e-6


class TestValidation:
    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError):
            fit_gaussian_hmm(np.array([1.0, 2.0, 3.0]), n_states=2)

    def test_empty_predict_returns_empty(self):
        rng = np.random.default_rng(5)
        series = rng.normal(0, 1, size=100)
        fit = fit_gaussian_hmm(series, n_states=2, n_starts=1, max_iter=10)
        assert fit.predict(np.array([])).size == 0
        assert fit.predict_proba(np.array([])).shape == (0, 2)


class TestForwardBackward:
    def test_alpha_beta_lengths(self):
        # Simple 2-state model, 5 observations.
        T, K = 5, 2
        log_emission = np.log(np.array([
            [0.6, 0.4], [0.5, 0.5], [0.3, 0.7], [0.2, 0.8], [0.4, 0.6],
        ]))
        log_pi = np.log(np.array([0.6, 0.4]))
        log_A = np.log(np.array([[0.7, 0.3], [0.4, 0.6]]))
        log_a, log_b, ll = _forward_backward(log_emission, log_pi, log_A)
        assert log_a.shape == (T, K)
        assert log_b.shape == (T, K)
        assert np.isfinite(ll)
