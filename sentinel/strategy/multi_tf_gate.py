"""
Multi-timeframe AND-gate for trend-following strategies.

Trend strategies (EMA crossover, Bollinger breakout, MACD trend) lose money
when the higher timeframe trend disagrees with the entry signal. The
existing ``features.indicators.trend_alignment`` produces a continuous
0..1 confluence score that strategies use as a soft modifier — but soft
modifiers can be overpowered by strong signals on a single timeframe.

This module enforces a HARD gate: a trend BUY is rejected unless EVERY
configured timeframe agrees on the trend direction. Mean-reversion
strategies (which intentionally fade the higher TF) opt out by setting
``StrategyType.MEAN_REVERSION``.

The gate reads only the FeatureVector produced by ``FeatureBuilder`` —
no I/O, no async. It can be invoked from inside a strategy's
``generate_signal`` method or from a meta-layer just before the
risk sentinel.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from core.models import Direction, FeatureVector

logger = logging.getLogger(__name__)


class StrategyType(str, Enum):
    """How the strategy relates to the prevailing trend."""
    TREND_FOLLOWING = "trend_following"   # must agree across TFs
    MEAN_REVERSION = "mean_reversion"     # intentionally fades trend; gate skipped
    BREAKOUT = "breakout"                 # treated like trend-following
    NEUTRAL = "neutral"                   # gate skipped


# Registry mapping strategy NAME → StrategyType. Used by RiskSentinel to
# decide whether to apply the multi-TF gate. Update when adding strategies.
# Mean-reversion / DCA / grid intentionally fade or ignore the prevailing
# trend, so the AND-gate would reject every signal — they opt out.
STRATEGY_TYPE_REGISTRY: dict[str, StrategyType] = {
    "ema_crossover_rsi": StrategyType.TREND_FOLLOWING,
    "macd_divergence":   StrategyType.TREND_FOLLOWING,
    "bollinger_breakout": StrategyType.BREAKOUT,
    "mean_reversion":    StrategyType.MEAN_REVERSION,
    "dca_bot":           StrategyType.NEUTRAL,
    "grid_trading":      StrategyType.NEUTRAL,
}


def classify_strategy(name: str) -> StrategyType:
    """Look up a strategy by name; defaults to TREND_FOLLOWING (strict gate)
    rather than NEUTRAL because false-positive blocks are safer than false
    negatives for an unknown strategy."""
    return STRATEGY_TYPE_REGISTRY.get(name, StrategyType.TREND_FOLLOWING)


@dataclass(frozen=True)
class MultiTFGateConfig:
    require_4h_alignment: bool = True
    require_1d_alignment: bool = True
    # If trend_alignment score is set by FeatureBuilder, also require it
    # above this threshold for trend strategies.
    min_trend_alignment_score: float = 0.6
    # When 1d EMA50 is missing (early in data lifecycle), fail-open vs
    # fail-closed. fail-closed is safer for live; fail-open keeps backtests
    # from trivially rejecting all signals on short series.
    fail_closed_on_missing_data: bool = True


@dataclass
class MultiTFDecision:
    approved: bool
    reason: str = ""
    checks: dict[str, bool] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.checks is None:
            self.checks = {}


class MultiTFGate:
    """Strict multi-timeframe confluence gate.

    Approval rule for a long entry from a trend strategy:
        - close > ema_50 (4h)            ← higher-TF uptrend
        - close > ema_50_daily            ← daily-TF uptrend
        - trend_alignment ≥ threshold     ← composite confluence score

    Short entries (Direction.SELL meaning enter-short) mirror the rules.
    For Sentinel, SELL means "close existing long" — the gate does NOT
    block exits, since holding into adverse conditions is worse than
    closing late. Always allow SELL.
    """

    def __init__(self, config: Optional[MultiTFGateConfig] = None) -> None:
        self._cfg = config or MultiTFGateConfig()

    def check(
        self,
        direction: Direction,
        features: FeatureVector,
        strategy_type: StrategyType,
    ) -> MultiTFDecision:
        """Return approval decision. Always approves SELL/HOLD and non-trend types."""
        # SELL is closing — never block.
        if direction == Direction.SELL or direction == Direction.HOLD:
            return MultiTFDecision(approved=True, reason="Exit/hold not gated")

        # Mean reversion / neutral don't gate; they intentionally fade the trend.
        if strategy_type in (StrategyType.MEAN_REVERSION, StrategyType.NEUTRAL):
            return MultiTFDecision(approved=True, reason=f"{strategy_type.value} bypasses gate")

        checks: dict[str, bool] = {}

        # 4h trend.
        if self._cfg.require_4h_alignment:
            if features.ema_50 <= 0 or features.close <= 0:
                if self._cfg.fail_closed_on_missing_data:
                    return MultiTFDecision(
                        approved=False,
                        reason="4h EMA50 unavailable (fail-closed)",
                        checks={"4h_alignment": False},
                    )
                checks["4h_alignment"] = True  # fail-open
            else:
                checks["4h_alignment"] = features.close > features.ema_50

        # Daily trend.
        if self._cfg.require_1d_alignment:
            if features.ema_50_daily <= 0 or features.close <= 0:
                if self._cfg.fail_closed_on_missing_data:
                    return MultiTFDecision(
                        approved=False,
                        reason="Daily EMA50 unavailable (fail-closed)",
                        checks=checks | {"1d_alignment": False},
                    )
                checks["1d_alignment"] = True
            else:
                checks["1d_alignment"] = features.close > features.ema_50_daily

        # Composite trend alignment score.
        checks["trend_alignment_score"] = (
            features.trend_alignment >= self._cfg.min_trend_alignment_score
        )

        # AND-gate: all configured checks must pass.
        if not all(checks.values()):
            failed = [k for k, v in checks.items() if not v]
            return MultiTFDecision(
                approved=False,
                reason=f"Multi-TF mismatch: {', '.join(failed)} (TA={features.trend_alignment:.2f}, "
                       f"4h_EMA50={features.ema_50:.2f}, 1d_EMA50={features.ema_50_daily:.2f}, "
                       f"close={features.close:.2f})",
                checks=checks,
            )

        return MultiTFDecision(
            approved=True,
            reason="All TFs aligned",
            checks=checks,
        )
