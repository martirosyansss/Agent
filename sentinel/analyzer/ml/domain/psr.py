"""
Probabilistic Sharpe Ratio (PSR) & Deflated Sharpe Ratio (DSR).

The classical Sharpe ratio is a *point estimate*: it answers "what did I
observe?" but not "is what I observed real skill or sampling noise?". On
a short sample, or on returns with heavy tails, a large observed Sharpe
can be an artefact — and hyper-parameter search over many strategies
guarantees one will win the lottery even when none has true edge.

López de Prado (2012, 2014) addresses both failure modes:

* **PSR(SR*)** is the probability that the *true* Sharpe exceeds the
  benchmark ``SR*``, given the observed mean, standard deviation,
  skewness, and kurtosis of the returns. It replaces "is this number
  big?" with "would I still believe this number with another sample?".

* **DSR** (Deflated Sharpe) adjusts the benchmark upward to compensate
  for multiple-testing inflation — if you tried N strategy variants,
  the best observed Sharpe is biased high by the max of N Gaussians.
  DSR passing ≥ 0.95 means "even accounting for the N trials we ran,
  this result is probably not luck".

Both return a probability ∈ [0, 1]. The usual deployment gate is
``PSR ≥ 0.95`` and ``DSR ≥ 0.95``. For a backtest with 150 trades this
typically requires a raw Sharpe of roughly 1.2–1.6 depending on tails;
for 500 trades, roughly 0.6–0.9.

All functions operate on an array of *per-period* returns (one entry per
bar or per trade). Annualisation is a display concern only — PSR/DSR
are invariant to the period choice because the numerator and denominator
scale together.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


# Euler-Mascheroni constant. Appears in the asymptotic of the expected
# max of N iid standard-normal draws.
_EULER_MASCHERONI: float = 0.5772156649015329


@dataclass(slots=True)
class PSRResult:
    """Structured output of the PSR/DSR test."""
    sharpe: float            # observed per-period Sharpe ratio
    skewness: float          # sample skew γ₃ — negative ⇒ fat left tail
    kurtosis: float          # sample excess kurtosis γ₄ − 3
    n: int                   # number of observations used
    psr: float               # P(true SR > benchmark_sr), ∈ [0, 1]
    dsr: Optional[float]     # deflated version (adjusted for N trials); None when trials=1
    benchmark_sr: float      # the SR* threshold used in the PSR test
    deflated_benchmark: Optional[float]   # SR0 used by DSR, or None
    gate_passed: bool        # convenience: PSR ≥ psr_threshold AND (DSR ≥ dsr_threshold OR DSR is None)


def _safe_std(r: np.ndarray) -> float:
    """Sample standard deviation (ddof=1). Returns 0 for size ≤ 1."""
    if r.size <= 1:
        return 0.0
    return float(r.std(ddof=1))


def _sample_skew(r: np.ndarray) -> float:
    """Sample skewness γ₃, Fisher definition. Returns 0 on degenerate input."""
    std = _safe_std(r)
    if std <= 0.0 or r.size < 3:
        return 0.0
    z = (r - r.mean()) / std
    return float((z ** 3).mean())


def _sample_excess_kurtosis(r: np.ndarray) -> float:
    """Sample excess kurtosis γ₄ − 3 (so a Gaussian scores 0)."""
    std = _safe_std(r)
    if std <= 0.0 or r.size < 4:
        return 0.0
    z = (r - r.mean()) / std
    return float((z ** 4).mean() - 3.0)


def _normal_cdf(x: float) -> float:
    """Standard-normal CDF via erf so we don't pull scipy for one call."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation).

    Used by DSR to pick the adjusted benchmark ``SR0``. Accurate to ~1e-9
    across ``p ∈ (0, 1)`` — more than enough; we only evaluate it at
    ``1 − 1/N`` and ``1 − 1/(N·e)`` where ``N ≥ 2``.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"_normal_ppf: p must be in (0, 1), got {p}")

    a = [-39.69683028665376, 220.9460984245205, -275.9285104469687,
         138.3577518672690, -30.66479806614716, 2.506628277459239]
    b = [-54.47609879822406, 161.5858368580409, -155.6989798598866,
         66.80131188771972, -13.28068155288572]
    c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996,
         3.754408661907416]

    plow = 0.02425
    phigh = 1.0 - plow

    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
    )


def probabilistic_sharpe_ratio(
    returns: np.ndarray | list[float],
    *,
    benchmark_sr: float = 0.0,
    n_trials: int = 1,
    psr_threshold: float = 0.95,
    dsr_threshold: float = 0.95,
) -> PSRResult:
    """Compute PSR and (optionally) DSR for a per-period return series.

    Args:
        returns: Per-period returns (one entry per bar or per trade).
            Must be a flat 1D sequence of floats.
        benchmark_sr: Per-period Sharpe threshold to compare against. The
            conservative default 0 asks "is there any skill at all?".
        n_trials: Number of independent strategies / hyper-parameter
            configurations tested. When > 1, DSR inflates the benchmark
            upward by the expected max of N Gaussians — the central trick
            of deflation. ``n_trials=1`` skips DSR entirely (returns None).
        psr_threshold: PSR cutoff for ``gate_passed``. Textbook 0.95.
        dsr_threshold: DSR cutoff for ``gate_passed``. Textbook 0.95.

    Returns:
        ``PSRResult`` with all intermediate statistics so the caller can
        log/display each step of the test, plus a ``gate_passed`` flag
        for a one-line production check.
    """
    arr = np.asarray(returns, dtype=np.float64).ravel()
    n = int(arr.size)
    if n < 3:
        return PSRResult(
            sharpe=0.0, skewness=0.0, kurtosis=0.0, n=n,
            psr=0.0, dsr=None,
            benchmark_sr=benchmark_sr, deflated_benchmark=None,
            gate_passed=False,
        )

    std = _safe_std(arr)
    if std <= 0.0:
        # All-zero or constant returns — Sharpe is undefined. Treat as
        # no-skill: PSR = 0, gate fails.
        return PSRResult(
            sharpe=0.0, skewness=0.0, kurtosis=0.0, n=n,
            psr=0.0, dsr=None if n_trials <= 1 else 0.0,
            benchmark_sr=benchmark_sr, deflated_benchmark=None,
            gate_passed=False,
        )

    sr = float(arr.mean() / std)
    skew = _sample_skew(arr)
    kurt = _sample_excess_kurtosis(arr)

    # PSR formula (López de Prado, 2012):
    #   PSR(SR*) = Φ( (SR - SR*) * sqrt(n - 1) / denom )
    # with denom = sqrt(1 - γ3·SR + ((γ4 - 1)/4)·SR²).
    # Note: kurtosis here is *excess* (γ4 − 3), so the formula uses
    # (γ4 + 3 − 1) / 4 = (γ4 + 2) / 4 when γ4 is excess. We stick to the
    # standard form and supply the full (non-excess) kurtosis where
    # needed. Converting excess back: γ4_full = excess + 3.
    gamma4_full = kurt + 3.0
    denom_sq = 1.0 - skew * sr + ((gamma4_full - 1.0) / 4.0) * sr * sr
    if denom_sq <= 0.0:
        # Degenerate tails make the PSR denominator imaginary. Conservatively
        # treat as no skill rather than propagating NaN into downstream gates.
        psr_val = 0.0
    else:
        z = (sr - benchmark_sr) * math.sqrt(max(n - 1, 1)) / math.sqrt(denom_sq)
        psr_val = _normal_cdf(z)

    dsr_val: Optional[float] = None
    deflated_bench: Optional[float] = None
    if n_trials > 1:
        # Expected max of N iid standard-normal draws (Bai, 2003 approximation)
        #   E[max] ≈ (1 - γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N·e))
        # γ is Euler-Mascheroni. Used to set the *deflated* benchmark
        # SR0 = sigma(SR) · E[max], where sigma(SR) under the null is
        # approx sqrt((1 - γ3·0 + (γ4-1)/4·0) / (n - 1)) = 1/sqrt(n-1).
        try:
            term1 = _normal_ppf(1.0 - 1.0 / n_trials)
            term2 = _normal_ppf(1.0 - 1.0 / (n_trials * math.e))
        except ValueError:
            term1 = term2 = 0.0
        exp_max = (1.0 - _EULER_MASCHERONI) * term1 + _EULER_MASCHERONI * term2
        sigma_sr = 1.0 / math.sqrt(max(n - 1, 1))
        sr0 = sigma_sr * exp_max
        deflated_bench = float(sr0)
        # DSR = PSR evaluated at the deflated benchmark
        if denom_sq <= 0.0:
            dsr_val = 0.0
        else:
            z_d = (sr - sr0) * math.sqrt(max(n - 1, 1)) / math.sqrt(denom_sq)
            dsr_val = _normal_cdf(z_d)

    gate = (psr_val >= psr_threshold) and (
        dsr_val is None or dsr_val >= dsr_threshold
    )

    return PSRResult(
        sharpe=float(sr),
        skewness=float(skew),
        kurtosis=float(kurt),
        n=n,
        psr=float(psr_val),
        dsr=float(dsr_val) if dsr_val is not None else None,
        benchmark_sr=float(benchmark_sr),
        deflated_benchmark=deflated_bench,
        gate_passed=bool(gate),
    )
