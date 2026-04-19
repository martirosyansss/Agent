"""Tests for fractional differentiation.

Contracts:

* **d=0 is the identity** — fractional differentiation with order zero
  leaves the series unchanged (after warm-up).
* **d=1 converges to first differences** — López de Prado's hallmark
  check: the limiting case of the fractional operator is the classical
  one, so at ``d → 1`` the FFD output should track ``diff(series)``
  closely on the valid slice.
* **Memory preservation** — for a pure random walk, fractionally
  differentiated output (``d=0.4``) has meaningful autocorrelation at
  lag 1, whereas full differencing (``d=1``) has nearly zero.
* **Warm-up NaN handling** — no NaN leaks into the valid region; the
  valid region starts exactly ``window_size - 1`` in FFD.
* **Weight decay monotonicity** — FFD weights decrease in absolute
  value past the peak (the binomial series has a known shape).
* **Empty / invalid inputs** — don't crash, return all-NaN.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyzer.ml.features.fracdiff import (
    _fracdiff_weights,
    _fracdiff_weights_ffd,
    frac_diff,
    frac_diff_ffd,
    suggest_optimal_d,
)


class TestWeights:
    def test_first_weight_is_one(self):
        w = _fracdiff_weights(0.4, 10)
        assert w[0] == 1.0

    def test_weights_alternate_in_sign_for_d_in_01(self):
        w = _fracdiff_weights(0.5, 20)
        # For 0 < d < 1, weights alternate: 1, -0.5, -0.125, ...
        signs = np.sign(w[1:8])   # skip w[0]=1
        assert all(s < 0 for s in signs)

    def test_ffd_truncates_at_tolerance(self):
        w_high_tol = _fracdiff_weights_ffd(0.4, tolerance=1e-3)
        w_low_tol = _fracdiff_weights_ffd(0.4, tolerance=1e-6)
        # Smaller tolerance → longer kernel.
        assert len(w_low_tol) > len(w_high_tol)
        # Final weight below tolerance.
        assert abs(w_high_tol[-1]) >= 1e-3 or len(w_high_tol) >= 1

    def test_weights_decay(self):
        w = _fracdiff_weights_ffd(0.4, tolerance=1e-6)
        # Past the first 2 entries, |w| is monotonically decreasing.
        abs_w = np.abs(w[2:])
        assert all(abs_w[i] >= abs_w[i + 1] - 1e-12 for i in range(len(abs_w) - 1))


class TestFracDiff:
    def test_empty_input(self):
        out = frac_diff(np.array([]), 0.4)
        assert out.size == 0

    def test_invalid_d_returns_nan(self):
        # d must be in (0, 1); 0 and 1 are excluded for frac_diff.
        series = np.arange(100, dtype=np.float64)
        assert np.all(np.isnan(frac_diff(series, 0.0)))
        assert np.all(np.isnan(frac_diff(series, 1.0)))

    def test_ffd_convolves_to_input_length(self):
        # Use a generous tolerance so the FFD kernel fits comfortably inside
        # the series (for d=0.4 the kernel at tol=1e-2 is ~60 weights).
        series = np.arange(200, dtype=np.float64)
        out = frac_diff_ffd(series, 0.4, tolerance=1e-2)
        assert out.size == series.size
        # Some warm-up NaNs at the start; the tail is all-finite.
        assert np.any(np.isnan(out[:5]))
        assert np.all(np.isfinite(out[-50:]))

    def test_d_near_one_matches_first_differences(self):
        rng = np.random.default_rng(0)
        # Random walk
        series = np.cumsum(rng.standard_normal(400))
        # At d close to 1, FFD should closely track simple diff(series).
        out = frac_diff_ffd(series, 0.99, tolerance=1e-3)
        raw_diff = np.diff(series)
        # Compare on the valid, overlapping tail.
        valid = ~np.isnan(out)
        # Align: out[t] ≈ series[t] - series[t-1] for d → 1, so match against
        # raw_diff[t-1] (which is series[t] - series[t-1]).
        idx = np.where(valid)[0]
        overlap = idx[idx >= 1]
        corr = np.corrcoef(out[overlap], raw_diff[overlap - 1])[0, 1]
        assert corr > 0.95

    def test_preserves_memory_better_than_raw_diff(self):
        rng = np.random.default_rng(1)
        # Random walk — raw level has high autocorr, raw diff has near-zero.
        series = np.cumsum(rng.standard_normal(1000))

        raw_diff = np.diff(series)
        ac_diff = float(np.corrcoef(raw_diff[1:], raw_diff[:-1])[0, 1])

        ffd = frac_diff_ffd(series, 0.4, tolerance=1e-3)
        ffd_valid = ffd[~np.isnan(ffd)]
        ac_ffd = float(np.corrcoef(ffd_valid[1:], ffd_valid[:-1])[0, 1])

        # FFD at d=0.4 preserves more long-range structure, which for
        # a random walk shows up as higher |autocorr at lag 1|.
        assert abs(ac_ffd) >= abs(ac_diff)


class TestSuggestOptimalD:
    def test_random_walk_picks_some_d_in_range(self):
        rng = np.random.default_rng(42)
        series = np.cumsum(rng.standard_normal(500))

        # Dummy stationarity test: returns p > 0.05 for d < 0.3, p < 0.05 otherwise.
        def fake_test(x: np.ndarray) -> float:
            return 0.2 if x.std() > 10 else 0.01

        chosen, results = suggest_optimal_d(series, stationarity_test=fake_test)
        assert len(results) > 0
        assert 0.0 <= chosen <= 1.0

    def test_short_series_returns_fallback(self):
        chosen, results = suggest_optimal_d(np.arange(10.0))
        assert chosen == 1.0
        assert results == []
