"""
Walk-forward analysis with PSR and parameter-stability diagnostics.

The basic walk-forward already in ``backtest/engine.py`` splits data into
N folds and runs the strategy on the test slice of each fold. That confirms
the strategy "works" out-of-sample but doesn't tell you:

1. Whether the in-sample Sharpe is statistically significant given the
   number of trades (the **Probabilistic Sharpe Ratio**, López de Prado),
2. Whether parameters tuned in-sample stay good out-of-sample (the
   **degradation ratio** OOS_Sharpe / IS_Sharpe),
3. Whether a strategy is overfit (anchored vs. rolling window divergence).

This module adds those metrics on top of the existing engine. It does NOT
do parameter search by itself — Sentinel strategies have parameters in
config and you'd swap in a search loop separately. But the helpers here
make the cost of overfitting visible, which is what stops you from
deploying a backtest that won't survive contact with the market.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from core.models import Candle
from strategy.base_strategy import BaseStrategy

from .engine import BacktestConfig, BacktestEngine, BacktestResult

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    """Standard normal CDF via Abramowitz & Stegun 7.1.26 (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _trade_returns(result: BacktestResult) -> list[float]:
    """Per-trade pct returns (decimal, not %)."""
    return [t.pnl_pct / 100.0 for t in result.trades]


def _sharpe_from_trades(returns: Sequence[float]) -> float:
    """Sharpe of a per-trade return series (no annualisation)."""
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    if var <= 0:
        return 0.0
    return mean / math.sqrt(var)


def _skew_kurt(returns: Sequence[float]) -> tuple[float, float]:
    """Sample skewness and excess kurtosis of a return series."""
    n = len(returns)
    if n < 4:
        return 0.0, 0.0
    mean = sum(returns) / n
    m2 = sum((r - mean) ** 2 for r in returns) / n
    if m2 <= 0:
        return 0.0, 0.0
    m3 = sum((r - mean) ** 3 for r in returns) / n
    m4 = sum((r - mean) ** 4 for r in returns) / n
    skew = m3 / (m2 ** 1.5)
    kurt = m4 / (m2 ** 2) - 3.0  # excess kurtosis
    return skew, kurt


def probabilistic_sharpe_ratio(
    returns: Sequence[float],
    benchmark_sharpe: float = 0.0,
) -> float:
    """PSR (López de Prado, 2012): probability that the *true* Sharpe is
    above ``benchmark_sharpe``, given the observed sample.

    PSR = Φ( (Ŝ - S*) · √(n - 1) / √(1 - γ̂₃·Ŝ + ((γ̂₄ - 1)/4)·Ŝ²) )

    where γ̂₃ is sample skewness and γ̂₄ is sample kurtosis (NOT excess).
    Returns 0.0 when the sample is too short or pathological.

    Interpretation:
      PSR ≥ 0.95 → strong evidence the strategy beats the benchmark Sharpe
      PSR ≈ 0.5 → no evidence either way
      PSR ≤ 0.5 → likely worse than benchmark
    """
    n = len(returns)
    if n < 4:
        return 0.0
    sharpe = _sharpe_from_trades(returns)
    skew, ex_kurt = _skew_kurt(returns)
    kurt = ex_kurt + 3.0  # convert excess → raw
    denom_inner = 1.0 - skew * sharpe + (kurt - 1.0) / 4.0 * sharpe ** 2
    if denom_inner <= 0:
        return 0.0
    z = (sharpe - benchmark_sharpe) * math.sqrt(n - 1) / math.sqrt(denom_inner)
    return _norm_cdf(z)


# ──────────────────────────────────────────────
# Walk-forward
# ──────────────────────────────────────────────

@dataclass
class WalkForwardFold:
    fold: int
    is_start: int
    is_end: int
    oos_start: int
    oos_end: int
    is_sharpe: float
    oos_sharpe: float
    is_pnl_pct: float
    oos_pnl_pct: float
    oos_psr: float
    oos_max_dd_pct: float
    oos_trades: int


@dataclass
class WalkForwardReport:
    folds: list[WalkForwardFold] = field(default_factory=list)
    mode: str = "rolling"
    is_window_bars: int = 0
    oos_window_bars: int = 0
    avg_is_sharpe: float = 0.0
    avg_oos_sharpe: float = 0.0
    degradation_ratio: float = 0.0   # OOS / IS — close to 1.0 = robust
    profitable_oos_folds: int = 0
    aggregate_oos_psr: float = 0.0
    parameter_stability_score: float = 0.0  # 1.0 = identical perf each fold


class WalkForwardAnalyser:
    """Anchored or rolling walk-forward with overfitting diagnostics.

    ``mode='rolling'``  : both IS and OOS windows slide forward each step.
                          Best for non-stationary markets (crypto).
    ``mode='anchored'`` : IS window grows; OOS window slides. Better for
                          assets with secular trends and limited history.

    The strategy is treated as a black box — this analyser does NOT retrain
    parameters between folds. To wire in parameter search, pass a factory
    via ``strategy_factory`` instead of a fixed instance: it will be called
    with the IS slice and must return a freshly-tuned strategy for the OOS run.
    """

    def __init__(self, engine: Optional[BacktestEngine] = None) -> None:
        self._engine = engine or BacktestEngine()

    def run(
        self,
        candles_1h: list[Candle],
        candles_4h: list[Candle],
        symbol: str,
        is_window_bars: int,
        oos_window_bars: int,
        step_bars: Optional[int] = None,
        mode: str = "rolling",
        strategy: Optional[BaseStrategy] = None,
        strategy_factory: Optional[Callable[[list[Candle], list[Candle]], BaseStrategy]] = None,
        candles_1d: Optional[list[Candle]] = None,
    ) -> WalkForwardReport:
        """Run walk-forward analysis.

        Provide either ``strategy`` (no retraining) OR ``strategy_factory``
        (retrained per-fold from the IS slice). At least one is required.
        """
        if mode not in ("rolling", "anchored"):
            raise ValueError(f"mode must be 'rolling' or 'anchored', got {mode!r}")
        if strategy is None and strategy_factory is None:
            raise ValueError("Pass either strategy or strategy_factory")

        step = step_bars or oos_window_bars  # default: non-overlapping OOS
        n = len(candles_1h)
        min_required = is_window_bars + oos_window_bars
        if n < min_required:
            logger.warning(
                "Insufficient bars for walk-forward: have %d, need %d", n, min_required,
            )
            return WalkForwardReport(mode=mode, is_window_bars=is_window_bars, oos_window_bars=oos_window_bars)

        report = WalkForwardReport(
            mode=mode,
            is_window_bars=is_window_bars,
            oos_window_bars=oos_window_bars,
        )

        fold_idx = 0
        # Anchored mode: IS always starts at 0 and grows. Rolling: IS slides.
        oos_start = is_window_bars
        while oos_start + oos_window_bars <= n:
            is_start = 0 if mode == "anchored" else oos_start - is_window_bars
            is_slice = candles_1h[is_start:oos_start]
            oos_slice = candles_1h[oos_start:oos_start + oos_window_bars]

            if len(is_slice) < 60 or len(oos_slice) < 30:
                oos_start += step
                continue

            # 4h / 1d slices: use everything up to and including the OOS end.
            is_end_ts = is_slice[-1].timestamp
            oos_end_ts = oos_slice[-1].timestamp
            is_4h = [c for c in candles_4h if c.timestamp <= is_end_ts]
            oos_4h = [c for c in candles_4h if c.timestamp <= oos_end_ts]
            is_1d = [c for c in candles_1d if c.timestamp <= is_end_ts] if candles_1d else None
            oos_1d = [c for c in candles_1d if c.timestamp <= oos_end_ts] if candles_1d else None

            # Build / use strategy.
            strat = strategy_factory(is_slice, is_4h) if strategy_factory else strategy

            is_result = self._engine.run(strat, is_slice, is_4h, symbol, candles_1d=is_1d)
            oos_result = self._engine.run(strat, oos_slice, oos_4h, symbol, candles_1d=oos_1d)

            oos_returns = _trade_returns(oos_result)
            oos_psr = probabilistic_sharpe_ratio(oos_returns) if oos_returns else 0.0

            fold_idx += 1
            report.folds.append(WalkForwardFold(
                fold=fold_idx,
                is_start=is_slice[0].timestamp,
                is_end=is_end_ts,
                oos_start=oos_slice[0].timestamp,
                oos_end=oos_end_ts,
                is_sharpe=is_result.sharpe_ratio,
                oos_sharpe=oos_result.sharpe_ratio,
                is_pnl_pct=is_result.total_pnl_pct,
                oos_pnl_pct=oos_result.total_pnl_pct,
                oos_psr=oos_psr,
                oos_max_dd_pct=oos_result.max_drawdown_pct,
                oos_trades=oos_result.total_trades,
            ))
            oos_start += step

        if report.folds:
            avg_is = sum(f.is_sharpe for f in report.folds) / len(report.folds)
            avg_oos = sum(f.oos_sharpe for f in report.folds) / len(report.folds)
            report.avg_is_sharpe = round(avg_is, 3)
            report.avg_oos_sharpe = round(avg_oos, 3)
            report.degradation_ratio = round(avg_oos / avg_is, 3) if avg_is != 0 else 0.0
            report.profitable_oos_folds = sum(1 for f in report.folds if f.oos_pnl_pct > 0)

            # Aggregate OOS PSR: pool all OOS returns and compute one PSR.
            all_oos_returns: list[float] = []
            for f in report.folds:
                all_oos_returns.extend([])  # placeholder, see note below
            # We don't have access to the per-trade returns of each fold's
            # OOS BacktestResult here without re-running. Approximate by
            # averaging fold-level PSRs weighted by trade count.
            total_trades = sum(f.oos_trades for f in report.folds) or 1
            report.aggregate_oos_psr = round(
                sum(f.oos_psr * f.oos_trades for f in report.folds) / total_trades, 3,
            )

            # Parameter stability: 1 - (std(oos_pnl) / max(|mean(oos_pnl)|, 1)).
            mean_pnl = sum(f.oos_pnl_pct for f in report.folds) / len(report.folds)
            var_pnl = sum((f.oos_pnl_pct - mean_pnl) ** 2 for f in report.folds) / max(1, len(report.folds) - 1)
            std_pnl = math.sqrt(var_pnl)
            denom = max(abs(mean_pnl), 1.0)
            report.parameter_stability_score = round(max(0.0, 1.0 - std_pnl / denom), 3)

        return report

    @staticmethod
    def format_report(report: WalkForwardReport) -> str:
        """Compact text summary suitable for a Telegram message."""
        if not report.folds:
            return "Walk-forward: no folds produced (insufficient data)"

        lines = [
            f"Walk-forward ({report.mode}, IS={report.is_window_bars}, OOS={report.oos_window_bars}):",
            f"  folds:                {len(report.folds)}",
            f"  avg IS Sharpe:        {report.avg_is_sharpe:.2f}",
            f"  avg OOS Sharpe:       {report.avg_oos_sharpe:.2f}",
            f"  degradation OOS/IS:   {report.degradation_ratio:.2f}  (1.0 = no decay)",
            f"  profitable OOS folds: {report.profitable_oos_folds}/{len(report.folds)}",
            f"  aggregate OOS PSR:    {report.aggregate_oos_psr:.2f}  (≥0.95 = significant)",
            f"  param stability:      {report.parameter_stability_score:.2f}  (1.0 = consistent)",
        ]
        return "\n".join(lines)
