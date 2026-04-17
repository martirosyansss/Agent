"""
Correlation Guard — generalised concentration check across open positions.

Replaces the hard-coded BTC/ETH cluster in ``risk/sentinel.py`` with a rolling
Pearson correlation matrix computed from log-returns. Two refinements over a
naive pairwise threshold:

1. **Cluster transitivity.** If A↔B and A↔C are both above the threshold,
   we treat {A,B,C} as one cluster even when ρ(B,C) is below the line. This
   is what stops a portfolio of "diversified" tech names that all move on
   the same NVDA news.
2. **Effective independent positions (ENP).** A scalar measure of how many
   *independent* bets you hold. Built from the eigenvalues of the correlation
   sub-matrix of currently-open symbols. Used for soft caps where a hard
   "no new entries" is too aggressive.

The guard is data-source agnostic: callers pass a dict of ``{symbol → list of
returns}``. The caller decides on the lookback (60 daily bars or 240 4h bars
are typical) and how to refresh history. The guard never fetches prices.

This module is deliberately dependency-free (pure Python, no NumPy required)
so it can run inside the Risk Sentinel hot path without import cost.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CorrelationConfig:
    """Tunables for the guard."""
    threshold: float = 0.70           # |ρ| above this counts as "correlated"
    min_observations: int = 30        # need this many returns or skip the pair
    max_cluster_size: int = 2         # max correlated open positions in one cluster
    # Soft cap: if Effective N (ENP) of the *post-trade* portfolio falls
    # below this, reject the trade even when the cluster check passed.
    min_effective_positions: float = 1.5


@dataclass
class CorrelationDecision:
    """Result of a pre-trade check."""
    approved: bool
    reason: str = ""
    cluster: tuple[str, ...] = field(default_factory=tuple)
    pair_correlations: dict[str, float] = field(default_factory=dict)
    effective_positions: float = 0.0


# ──────────────────────────────────────────────
# Pure math helpers (no numpy dependency)
# ──────────────────────────────────────────────

def _pearson(xs: list[float], ys: list[float]) -> Optional[float]:
    """Pearson correlation. Returns None if undefined (zero variance / too short)."""
    n = min(len(xs), len(ys))
    if n < 2:
        return None
    xs = xs[-n:]
    ys = ys[-n:]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = 0.0
    var_x = 0.0
    var_y = 0.0
    for x, y in zip(xs, ys):
        dx = x - mean_x
        dy = y - mean_y
        cov += dx * dy
        var_x += dx * dx
        var_y += dy * dy
    if var_x <= 0 or var_y <= 0:
        return None
    return cov / math.sqrt(var_x * var_y)


def _log_returns(prices: list[float]) -> list[float]:
    """Convert price series to log returns. Drops invalid (≤0) prices in pairs."""
    out: list[float] = []
    for i in range(1, len(prices)):
        prev, cur = prices[i - 1], prices[i]
        if prev > 0 and cur > 0:
            out.append(math.log(cur / prev))
    return out


def effective_n_positions(corr_matrix: list[list[float]]) -> float:
    """ENP = (Tr C)² / ‖C‖²_F = (Σᵢ Cᵢᵢ)² / Σᵢⱼ Cᵢⱼ².

    Equivalent to (Σλᵢ)² / Σλᵢ² for symmetric C, but computed directly from
    matrix entries — avoids the numerical instability of power iteration on
    near-identity matrices.

    For an n×n correlation matrix Tr(C) = n, so:
      - identity matrix      → ENP = n²/n = n           (all independent)
      - all-ones matrix      → ENP = n²/n² = 1          (all perfectly correlated)
      - intermediate         → ENP ∈ (1, n)
    """
    n = len(corr_matrix)
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    trace = sum(corr_matrix[i][i] for i in range(n))
    frob_sq = sum(corr_matrix[i][j] ** 2 for i in range(n) for j in range(n))
    if frob_sq <= 0:
        return float(n)
    return (trace ** 2) / frob_sq


# ──────────────────────────────────────────────
# Guard
# ──────────────────────────────────────────────

class CorrelationGuard:
    """Stateless guard: caller injects price history and open-position symbols.

    The guard caches nothing — the caller (typically PriceHistoryProvider in
    the live executor or a backtest harness) owns the prices and decides
    refresh cadence. This keeps the guard usable from both hot path and
    tests, and avoids subtle bugs around cache invalidation.
    """

    def __init__(self, config: Optional[CorrelationConfig] = None) -> None:
        self._cfg = config or CorrelationConfig()

    def check(
        self,
        candidate_symbol: str,
        open_symbols: Iterable[str],
        price_history: dict[str, list[float]],
    ) -> CorrelationDecision:
        """Decide whether opening ``candidate_symbol`` would breach concentration limits.

        Args:
            candidate_symbol: symbol the caller wants to open.
            open_symbols: symbols currently held.
            price_history: ``{symbol → list of close prices}``. Lookback is
                determined by what the caller passes; we only require enough
                samples to satisfy ``min_observations`` per pair.

        Returns:
            CorrelationDecision. If ``approved`` is False, ``reason`` and
            ``cluster`` describe which positions blocked the entry.
        """
        open_list = [s for s in open_symbols if s != candidate_symbol]

        # No open positions → nothing to be correlated with.
        if not open_list:
            return CorrelationDecision(approved=True, reason="No open positions")

        # Candidate must have a price series.
        cand_returns = _log_returns(price_history.get(candidate_symbol, []))
        if len(cand_returns) < self._cfg.min_observations:
            # Fail-open: insufficient data is the caller's problem to log,
            # but we shouldn't block a trade just because we lack history.
            logger.warning(
                "CorrelationGuard: insufficient history for %s (have %d, need %d) — skipping check",
                candidate_symbol, len(cand_returns), self._cfg.min_observations,
            )
            return CorrelationDecision(approved=True, reason="Insufficient history (skipped)")

        # Pairwise correlations: candidate vs each open symbol.
        pair_corrs: dict[str, float] = {}
        correlated_with: list[str] = []
        for s in open_list:
            other_returns = _log_returns(price_history.get(s, []))
            n = min(len(cand_returns), len(other_returns))
            if n < self._cfg.min_observations:
                continue
            rho = _pearson(cand_returns[-n:], other_returns[-n:])
            if rho is None:
                continue
            pair_corrs[s] = rho
            if abs(rho) >= self._cfg.threshold:
                correlated_with.append(s)

        # Cluster transitivity: candidate forms a cluster with all symbols
        # it's directly correlated with, plus any symbol that's correlated
        # with one of those (one hop). This catches indirect concentration.
        cluster: set[str] = set(correlated_with)
        if cluster:
            for s in open_list:
                if s in cluster:
                    continue
                s_returns = _log_returns(price_history.get(s, []))
                if len(s_returns) < self._cfg.min_observations:
                    continue
                for c in list(cluster):
                    c_returns = _log_returns(price_history.get(c, []))
                    n = min(len(s_returns), len(c_returns))
                    if n < self._cfg.min_observations:
                        continue
                    rho = _pearson(s_returns[-n:], c_returns[-n:])
                    if rho is not None and abs(rho) >= self._cfg.threshold:
                        cluster.add(s)
                        break

        # Hard cap: cluster size including the candidate.
        if len(cluster) + 1 > self._cfg.max_cluster_size:
            cluster_sorted = tuple(sorted(cluster))
            return CorrelationDecision(
                approved=False,
                reason=(
                    f"Correlation cluster too large: {candidate_symbol} would join "
                    f"{cluster_sorted} (size {len(cluster) + 1} > "
                    f"{self._cfg.max_cluster_size})"
                ),
                cluster=cluster_sorted,
                pair_correlations=pair_corrs,
            )

        # Soft cap: post-trade ENP across all open + candidate.
        post_trade_symbols = open_list + [candidate_symbol]
        corr_matrix = self._build_corr_matrix(post_trade_symbols, price_history)
        enp = effective_n_positions(corr_matrix)
        if enp > 0 and enp < self._cfg.min_effective_positions:
            return CorrelationDecision(
                approved=False,
                reason=(
                    f"Effective independent positions {enp:.2f} < "
                    f"{self._cfg.min_effective_positions} — portfolio too concentrated"
                ),
                cluster=tuple(sorted(cluster)),
                pair_correlations=pair_corrs,
                effective_positions=enp,
            )

        return CorrelationDecision(
            approved=True,
            reason="Correlation within limits",
            cluster=tuple(sorted(cluster)),
            pair_correlations=pair_corrs,
            effective_positions=enp,
        )

    def _build_corr_matrix(
        self,
        symbols: list[str],
        price_history: dict[str, list[float]],
    ) -> list[list[float]]:
        """Build a symmetric Pearson correlation matrix for ``symbols``.

        Missing pairs (insufficient data) get 0.0 — treating as independent
        is safer than treating as correlated when computing ENP, because
        it's an upper bound on diversification.
        """
        n = len(symbols)
        returns_cache: dict[str, list[float]] = {
            s: _log_returns(price_history.get(s, [])) for s in symbols
        }
        matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            matrix[i][i] = 1.0
            for j in range(i + 1, n):
                xs = returns_cache[symbols[i]]
                ys = returns_cache[symbols[j]]
                m = min(len(xs), len(ys))
                if m < self._cfg.min_observations:
                    rho = 0.0
                else:
                    rho = _pearson(xs[-m:], ys[-m:]) or 0.0
                matrix[i][j] = matrix[j][i] = rho
        return matrix
