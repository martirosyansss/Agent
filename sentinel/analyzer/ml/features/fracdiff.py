"""
Fractional Differentiation (López de Prado, Ch. 5 of *Advances in
Financial Machine Learning*).

Price-based features in a trading ML pipeline carry two conflicting
requirements:

1. **Stationarity** — the model needs inputs whose distribution doesn't
   drift over time. Raw prices fail this badly (unit-root behaviour);
   the classical fix is first-differencing (returns).
2. **Memory** — the model also needs inputs that still carry predictive
   information. First-differencing wipes the memory: returns have
   almost zero autocorrelation after 1–2 lags, so the model sees only
   the present moment with no context.

Fractional differentiation resolves the dilemma: instead of `d = 1`
(returns), use a real-valued `d ∈ (0, 1)` that is large enough to make
the series stationary but small enough to preserve autocorrelation. The
optimal `d` is typically in `[0.3, 0.5]` for daily crypto closes; it
gives an ADF-stationary series whose correlation with the original
price sits around 0.9, compared to ~0.05 for simple returns.

Two kernels are supported:

* **Expanding window** — convolves each output with *every* prior
  observation, weighted by the binomial-series coefficients. Loses no
  information but grows unbounded; used for research.
* **Fixed window (``frac_diff_ffd``)** — truncates the weight series
  once `|w| < tolerance`. Produces a stationary series with bounded
  look-back (suitable for live inference) and matches López de Prado's
  "FFD" implementation.

Both return a float NumPy array of the same length as the input; the
first few values (until enough history accumulates) are NaN-filled.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _fracdiff_weights(d: float, size: int) -> np.ndarray:
    """Binomial-series weights for fractional differentiation.

    ``w[0] = 1``; ``w[k] = -w[k-1] * (d - k + 1) / k``. The series is
    alternating in sign and decays as ``k^{-d-1}`` for large ``k`` —
    fast enough that truncation at ``|w| < tol`` converges quickly for
    ``d ∈ (0, 1)``.
    """
    if size <= 0:
        return np.empty(0, dtype=np.float64)
    w = np.empty(size, dtype=np.float64)
    w[0] = 1.0
    for k in range(1, size):
        w[k] = -w[k - 1] * (d - k + 1) / k
    return w


def _fracdiff_weights_ffd(d: float, tolerance: float = 1e-5, max_size: int = 10_000) -> np.ndarray:
    """Fixed-window weights: grow the binomial series until the next
    coefficient is below ``tolerance`` in absolute value.

    ``max_size`` is a safety cap — a very small ``d`` and tolerance
    combination can produce a huge window otherwise.
    """
    weights: list[float] = [1.0]
    for k in range(1, max_size):
        next_w = -weights[-1] * (d - k + 1) / k
        if abs(next_w) < tolerance:
            break
        weights.append(next_w)
    return np.asarray(weights, dtype=np.float64)


def frac_diff(
    series: npt.ArrayLike,
    d: float,
    *,
    tolerance: float = 1e-5,
) -> np.ndarray:
    """Standard (expanding-window) fractional differentiation.

    For each index ``t`` returns ``sum_k w[k] * series[t - k]`` over all
    valid ``k``, where ``w`` is the binomial series for ``d``. Leading
    positions (where the cumulative weight mass below ``tolerance`` is
    still significant) are filled with ``NaN``.

    Args:
        series: 1D array-like of observations. Price-level series —
            feed this directly.
        d: Fractional order in ``(0, 1)``. Larger → more stationary,
            less memory. Typical crypto choice 0.3–0.5.
        tolerance: Determines how many leading values get NaN-filled:
            an index is valid once the cumulative weight mass beyond
            that index is below ``tolerance``.

    Returns:
        ``np.ndarray`` of the same length as ``series``, ``dtype=float64``.
    """
    arr = np.asarray(series, dtype=np.float64).ravel()
    n = arr.size
    if n == 0 or not (0.0 < d < 1.0):
        return np.full(n, np.nan)

    w = _fracdiff_weights(d, n)
    # Skip-size: leading indices where the sum of |w[k:]| exceeds tolerance.
    # Those output positions would mix too little of the tail to be stable.
    w_abs_cum = np.cumsum(np.abs(w[::-1]))[::-1]
    # Fraction of tail weight still outside the truncation; first index
    # where it's below tolerance is the first "valid" output position.
    skip = int(np.searchsorted(w_abs_cum[::-1], tolerance))
    skip = n - skip if skip < n else n

    out = np.full(n, np.nan)
    # Loop — the vectorised version is a Toeplitz matmul but memory-
    # quadratic. For n ≤ 10_000 the Python loop is fine; if you hit
    # higher n in practice, switch to scipy.signal.convolve.
    for t in range(skip, n):
        window = arr[: t + 1]
        # Use w[: t+1] reversed so newest price is w[0], oldest is w[t].
        out[t] = float(np.dot(w[: t + 1], window[::-1]))
    return out


def frac_diff_ffd(
    series: npt.ArrayLike,
    d: float,
    *,
    tolerance: float = 1e-5,
    max_window: int = 10_000,
) -> np.ndarray:
    """Fixed-width fractional differentiation (López de Prado's FFD).

    Unlike ``frac_diff`` the kernel has a fixed length — once computed
    it never grows. Suitable for live inference because the compute
    cost per new observation is constant.

    The kernel window is determined by ``tolerance``: weights stop being
    appended once the next coefficient's absolute value is below it. The
    first ``window_size − 1`` output positions are NaN.

    Args:
        series: 1D array-like of observations.
        d: Fractional order in ``(0, 1)``.
        tolerance: Weight-decay cutoff. Smaller ⇒ longer kernel.
        max_window: Hard upper bound on kernel length (safety cap).

    Returns:
        Array of same length as ``series``. NaN in the warm-up positions.
    """
    arr = np.asarray(series, dtype=np.float64).ravel()
    n = arr.size
    if n == 0 or not (0.0 < d < 1.0):
        return np.full(n, np.nan)

    w = _fracdiff_weights_ffd(d, tolerance=tolerance, max_size=max_window)
    k = w.size
    if k > n:
        # Not enough history for even one valid output.
        return np.full(n, np.nan)

    out = np.full(n, np.nan)
    for t in range(k - 1, n):
        # Newest observation is arr[t], oldest in the window is arr[t - k + 1].
        out[t] = float(np.dot(w, arr[t - k + 1 : t + 1][::-1]))
    return out


def suggest_optimal_d(
    series: npt.ArrayLike,
    d_values: np.ndarray | None = None,
    *,
    tolerance: float = 1e-5,
    stationarity_test=None,
) -> tuple[float, list[tuple[float, float]]]:
    """Search for the smallest ``d`` that produces a stationary series.

    Evaluates ADF stationarity (via ``statsmodels.tsa.stattools.adfuller``)
    across a grid of ``d`` values and returns the smallest that rejects
    the unit-root null at p < 0.05. Smaller ``d`` means more memory
    preserved, so picking the *boundary* is the López de Prado recipe.

    Args:
        series: The raw (un-differenced) price-level series.
        d_values: Candidate grid; defaults to ``np.arange(0, 1.01, 0.1)``.
        tolerance: Passed to FFD weights.
        stationarity_test: Callable accepting a 1D array and returning
            a p-value float. Defaults to ``statsmodels.tsa.stattools.adfuller``
            when available; when neither is available, returns
            ``(1.0, [])`` so the caller falls back to a hand-picked ``d``.

    Returns:
        ``(optimal_d, [(d, p_value), ...])``. ``optimal_d == 1.0`` means
        no fractional ``d`` in the grid produced stationarity; the caller
        should keep the raw series (and accept the memory loss) or widen
        the grid.
    """
    arr = np.asarray(series, dtype=np.float64).ravel()
    if arr.size < 30:
        return 1.0, []

    grid = d_values if d_values is not None else np.round(np.arange(0.0, 1.01, 0.1), 2)

    if stationarity_test is None:
        try:
            from statsmodels.tsa.stattools import adfuller

            def stationarity_test(x: np.ndarray) -> float:
                return float(adfuller(x, maxlag=1, regression="c", autolag=None)[1])
        except Exception:
            return 1.0, []

    results: list[tuple[float, float]] = []
    chosen = 1.0
    for d in grid:
        try:
            if d == 0.0:
                s = arr
            else:
                s = frac_diff_ffd(arr, float(d), tolerance=tolerance)
            s = s[~np.isnan(s)]
            if s.size < 30:
                continue
            p = stationarity_test(s)
            results.append((float(d), float(p)))
            if p < 0.05 and chosen == 1.0:
                chosen = float(d)
        except Exception:
            continue
    return chosen, results
