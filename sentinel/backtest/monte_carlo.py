"""
Monte Carlo analysis for strategy robustness.

A single backtest produces ONE equity curve from ONE realisation of history.
Monte Carlo resamples the per-trade returns thousands of times to estimate
the *distribution* of outcomes — answering questions a point estimate can't:

- "What's the worst max-drawdown I can plausibly experience?"
  → Bootstrap the trade sequence, compute max-DD per simulation, take the
    5th percentile (worst-95%) as a realistic stress scenario.

- "What's my probability of losing money over the next 30 / 90 / 180 trades?"
  → Bootstrap forward, compound, count loss outcomes.

- "What's the probability of ruin (account drawdown ≥ X%)?"
  → Same idea: count simulations where peak-to-trough loss breaches X%.

- "What's the 95% confidence interval for next-year return?"
  → Annualised return from N-trade simulations, take 2.5/97.5 percentiles.

The module is dependency-free Python — uses ``random`` for sampling so
results are reproducible via seed and don't pull in NumPy.

Inputs are per-trade returns expressed as decimal fractions (e.g. 0.012
for +1.2%) — the same units BacktestResult.trades carry as ``pnl_pct/100``.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MonteCarloConfig:
    n_simulations: int = 1000
    confidence: float = 0.95          # for two-sided intervals
    ruin_threshold_pct: float = 25.0  # drawdown that constitutes "ruin"
    seed: Optional[int] = None        # deterministic when set


@dataclass
class MonteCarloReport:
    n_simulations: int
    n_trades_per_sim: int
    # Drawdown distribution (positive %, as fraction of peak)
    expected_max_dd_pct: float = 0.0
    median_max_dd_pct: float = 0.0
    worst_p05_max_dd_pct: float = 0.0     # 5th percentile = stress case
    worst_max_dd_pct: float = 0.0          # absolute worst across sims
    # Probability of ruin
    probability_of_ruin: float = 0.0
    ruin_threshold_pct: float = 0.0
    # Probability of loss after N compounded trades
    prob_loss_by_horizon: dict[int, float] = field(default_factory=dict)
    # Cumulative-return distribution at full horizon
    expected_return_pct: float = 0.0
    return_lower_bound_pct: float = 0.0   # confidence-interval lower
    return_upper_bound_pct: float = 0.0
    return_std_pct: float = 0.0


def _equity_path(returns: Sequence[float], starting_equity: float = 1.0) -> list[float]:
    """Geometric compounding of per-trade returns to an equity series."""
    equity = starting_equity
    out = [equity]
    for r in returns:
        equity *= (1.0 + r)
        out.append(equity)
    return out


def _max_drawdown_pct(equity: Sequence[float]) -> float:
    """Largest peak-to-trough drawdown of an equity curve, in % of peak."""
    if not equity:
        return 0.0
    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd * 100.0


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolated percentile (0 ≤ pct ≤ 100)."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (pct / 100.0) * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


class MonteCarloAnalyser:
    """Bootstrap-based Monte Carlo over a per-trade return series.

    Two assumptions worth knowing:

    1. **i.i.d. trades.** Bootstrapping with replacement assumes trades are
       independent and identically distributed. Real trade sequences often
       have positive serial correlation (winners cluster in regime windows).
       The bootstrap therefore tends to UNDERESTIMATE worst-case drawdown.
       For a more realistic stress test, use ``block_bootstrap`` with a
       block size around the typical regime length.

    2. **Stationarity.** The future return distribution is assumed to match
       the input sample. If the sample is from a single bull market, the
       MC will not predict bear-market drawdowns — those need a separately
       sourced stress sample (e.g. 2022 bear).
    """

    def __init__(self, config: Optional[MonteCarloConfig] = None) -> None:
        self._cfg = config or MonteCarloConfig()
        self._rng = random.Random(self._cfg.seed)

    # ──────────────────────────────────────────────
    # Sampling
    # ──────────────────────────────────────────────

    def bootstrap(
        self,
        returns: Sequence[float],
        n_periods: Optional[int] = None,
    ) -> list[list[float]]:
        """Resample with replacement. Returns a list of N simulations."""
        if not returns:
            return []
        n = n_periods or len(returns)
        return [
            [self._rng.choice(returns) for _ in range(n)]
            for _ in range(self._cfg.n_simulations)
        ]

    def block_bootstrap(
        self,
        returns: Sequence[float],
        block_size: int,
        n_periods: Optional[int] = None,
    ) -> list[list[float]]:
        """Stationary block bootstrap — preserves short-run serial correlation.

        Samples random STARTING indices and concatenates contiguous blocks of
        ``block_size`` until the simulation horizon is filled. Periods that
        run off the end wrap modulo len(returns).
        """
        if not returns or block_size < 1:
            return self.bootstrap(returns, n_periods)
        L = len(returns)
        n = n_periods or L
        sims: list[list[float]] = []
        for _ in range(self._cfg.n_simulations):
            sim: list[float] = []
            while len(sim) < n:
                start = self._rng.randint(0, L - 1)
                for i in range(block_size):
                    if len(sim) >= n:
                        break
                    sim.append(returns[(start + i) % L])
            sims.append(sim)
        return sims

    # ──────────────────────────────────────────────
    # Analyses
    # ──────────────────────────────────────────────

    def analyse(
        self,
        returns: Sequence[float],
        horizon: Optional[int] = None,
        block_size: int = 1,
        loss_horizons: Sequence[int] = (21, 63, 126, 252),
    ) -> MonteCarloReport:
        """Run all standard analyses and return a single report.

        Args:
            returns: per-trade returns as decimal fractions
            horizon: number of trades per simulation (default = len(returns))
            block_size: 1 = i.i.d. bootstrap; ≥2 = block bootstrap
                       (recommended block_size ≈ √n for moderate serial corr)
            loss_horizons: trade counts to evaluate probability-of-loss at
        """
        if not returns:
            return MonteCarloReport(n_simulations=0, n_trades_per_sim=0)
        if len(returns) < 5:
            logger.warning("MC needs at least 5 trades to be meaningful, got %d", len(returns))

        horizon = horizon or len(returns)
        sims = (
            self.block_bootstrap(returns, block_size, horizon)
            if block_size > 1
            else self.bootstrap(returns, horizon)
        )

        # Equity paths and max drawdowns.
        max_dds_pct: list[float] = []
        final_equities: list[float] = []
        for sim in sims:
            eq = _equity_path(sim)
            max_dds_pct.append(_max_drawdown_pct(eq))
            final_equities.append(eq[-1])

        max_dds_pct.sort()
        final_returns_pct = sorted([(e - 1.0) * 100.0 for e in final_equities])

        # Probability of ruin: % of sims where max-DD breaches threshold.
        ruin_count = sum(1 for dd in max_dds_pct if dd >= self._cfg.ruin_threshold_pct)
        prob_ruin = ruin_count / len(sims) if sims else 0.0

        # Probability-of-loss at each horizon (via independent shorter sims —
        # cheap because we already have the trade sample).
        prob_loss_by_horizon: dict[int, float] = {}
        for h in loss_horizons:
            if h <= 0 or h > 5_000:
                continue
            sub_sims = (
                self.block_bootstrap(returns, block_size, h)
                if block_size > 1
                else self.bootstrap(returns, h)
            )
            losses = sum(1 for sim in sub_sims if (math.prod(1 + r for r in sim) - 1.0) < 0)
            prob_loss_by_horizon[h] = losses / len(sub_sims) if sub_sims else 0.0

        # Confidence interval on final return.
        alpha = (1 - self._cfg.confidence) / 2 * 100
        ret_lower = _percentile(final_returns_pct, alpha)
        ret_upper = _percentile(final_returns_pct, 100 - alpha)
        mean_ret = sum(final_returns_pct) / len(final_returns_pct) if final_returns_pct else 0.0
        if len(final_returns_pct) > 1:
            var_ret = sum((r - mean_ret) ** 2 for r in final_returns_pct) / (len(final_returns_pct) - 1)
            std_ret = math.sqrt(var_ret)
        else:
            std_ret = 0.0

        return MonteCarloReport(
            n_simulations=len(sims),
            n_trades_per_sim=horizon,
            expected_max_dd_pct=round(sum(max_dds_pct) / len(max_dds_pct), 2),
            median_max_dd_pct=round(_percentile(max_dds_pct, 50), 2),
            worst_p05_max_dd_pct=round(_percentile(max_dds_pct, 95), 2),  # 95th of DDs = "worst 5%"
            worst_max_dd_pct=round(max_dds_pct[-1], 2) if max_dds_pct else 0.0,
            probability_of_ruin=round(prob_ruin, 4),
            ruin_threshold_pct=self._cfg.ruin_threshold_pct,
            prob_loss_by_horizon={h: round(p, 4) for h, p in prob_loss_by_horizon.items()},
            expected_return_pct=round(mean_ret, 2),
            return_lower_bound_pct=round(ret_lower, 2),
            return_upper_bound_pct=round(ret_upper, 2),
            return_std_pct=round(std_ret, 2),
        )

    @staticmethod
    def format_report(report: MonteCarloReport) -> str:
        if report.n_simulations == 0:
            return "Monte Carlo: no trades to analyse"
        lines = [
            f"Monte Carlo ({report.n_simulations} sims × {report.n_trades_per_sim} trades):",
            f"  Drawdown:",
            f"    expected max-DD:   {report.expected_max_dd_pct:.1f}%",
            f"    median max-DD:     {report.median_max_dd_pct:.1f}%",
            f"    worst 5% max-DD:   {report.worst_p05_max_dd_pct:.1f}%   ← stress case",
            f"    absolute worst:    {report.worst_max_dd_pct:.1f}%",
            f"  Probability of ruin (DD ≥ {report.ruin_threshold_pct}%): {report.probability_of_ruin * 100:.1f}%",
            f"  Probability of loss by horizon:",
        ]
        for h, p in sorted(report.prob_loss_by_horizon.items()):
            lines.append(f"    after {h:>3} trades: {p * 100:.1f}%")
        lines.extend([
            f"  Final return:",
            f"    expected:          {report.expected_return_pct:+.1f}%",
            f"    95% CI:            [{report.return_lower_bound_pct:+.1f}%, {report.return_upper_bound_pct:+.1f}%]",
            f"    std:               {report.return_std_pct:.1f}%",
        ])
        return "\n".join(lines)


def from_backtest_result(result, config: Optional[MonteCarloConfig] = None) -> MonteCarloReport:
    """Convenience wrapper: extract per-trade returns from a BacktestResult
    and run the standard MC analysis. ``result`` is duck-typed to avoid
    circular imports — must expose ``.trades`` with ``.pnl_pct`` attributes."""
    returns = [t.pnl_pct / 100.0 for t in getattr(result, "trades", [])]
    analyser = MonteCarloAnalyser(config)
    return analyser.analyse(returns)
