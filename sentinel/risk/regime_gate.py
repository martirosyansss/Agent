"""
Regime gate — hard block for adverse market-regime / strategy combinations.

Sentinel's existing ``base_strategy.adaptive_min_confidence`` raises the
required confidence in adverse regimes (e.g. +0.10 for trending_down). That
is a SOFT adjustment: a strategy producing confidence 0.95 still passes.

Empirically the worst trades come from trend-following entries against a
clear higher-TF downtrend, or from any directional entry during regime
``transitioning`` (chop right before a breakout). This guard blocks those
combinations OUTRIGHT regardless of confidence — a hard floor below the
soft confidence ladder.

Asymmetric by design:
- Trend strategies: blocked in trending_down / volatile / transitioning
- Mean-reversion: ALLOWED in trending_down (contrarian opportunity) but
  blocked in trending_up (no edge fading a clear trend) and transitioning
- Breakout: blocked only in transitioning (false-breakout factory)
- Neutral (DCA, grid): never blocked here (their own logic decides)

Tunables live in ``RegimeGateConfig`` so a user can soften / harden per-class.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from core.models import FeatureVector
from strategy.multi_tf_gate import StrategyType, classify_strategy

logger = logging.getLogger(__name__)


# Hard-block matrix: {strategy_type → {regime → block?}}.
# True = reject the entry signal regardless of confidence.
DEFAULT_BLOCK_MATRIX: dict[StrategyType, dict[str, bool]] = {
    StrategyType.TREND_FOLLOWING: {
        "trending_down": True,
        "volatile":      True,
        "transitioning": True,
    },
    StrategyType.BREAKOUT: {
        "transitioning": True,   # false-breakout regime
    },
    StrategyType.MEAN_REVERSION: {
        "trending_up":   True,   # don't fade a clear uptrend
        "transitioning": True,
    },
    StrategyType.NEUTRAL: {},    # DCA/grid handle their own regime logic
}


@dataclass(frozen=True)
class RegimeGateConfig:
    block_matrix: dict[StrategyType, dict[str, bool]] = field(
        default_factory=lambda: {k: dict(v) for k, v in DEFAULT_BLOCK_MATRIX.items()}
    )


@dataclass
class RegimeDecision:
    approved: bool
    reason: str = ""
    strategy_type: str = ""
    regime: str = ""


class RegimeGate:
    """Hard regime/strategy block. Stateless."""

    def __init__(self, config: Optional[RegimeGateConfig] = None) -> None:
        self._cfg = config or RegimeGateConfig()

    def check(self, strategy_name: str, features: FeatureVector) -> RegimeDecision:
        regime = getattr(features, "market_regime", "unknown") or "unknown"
        strat_type = classify_strategy(strategy_name)

        # Unknown regime is a soft state — features.market_regime is set lazily,
        # so blocking on it would prevent trading at startup. Skip with a debug log.
        if regime == "unknown":
            return RegimeDecision(
                approved=True,
                reason="Regime unknown (allowed)",
                strategy_type=strat_type.value,
                regime=regime,
            )

        block_map = self._cfg.block_matrix.get(strat_type, {})
        if block_map.get(regime, False):
            return RegimeDecision(
                approved=False,
                reason=f"Regime gate: {strat_type.value} hard-blocked in {regime} regime",
                strategy_type=strat_type.value,
                regime=regime,
            )

        return RegimeDecision(
            approved=True,
            reason=f"Regime {regime} compatible with {strat_type.value}",
            strategy_type=strat_type.value,
            regime=regime,
        )
