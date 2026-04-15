"""
Strategy Selector — автоматический выбор стратегий по режиму рынка.

ALLOCATION_TABLE определяет базовые % капитала для каждой стратегии в каждом режиме.
Adaptive weighting корректирует по скилу каждой стратегии (win rate за 30 дней).
Фактическая экспозиция ≤ 60%, max 2 направленных + 1 grid + 1 DCA.
Auto-selection отключён по умолчанию (auto_strategy_selection=False).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from core.models import MarketRegime, MarketRegimeType, StrategyTrade

logger = logging.getLogger(__name__)


# Allocation table: regime → {strategy: allocation_pct}
# "reserve" = что остаётся в кэше
ALLOCATION_TABLE: dict[str, dict[str, float]] = {
    "trending_up": {
        "ema_crossover_rsi": 25, "grid_trading": 5, "mean_reversion": 0,
        "bollinger_breakout": 15, "dca_bot": 5, "macd_divergence": 0,
    },
    "trending_down": {
        "ema_crossover_rsi": 0, "grid_trading": 0, "mean_reversion": 5,
        "bollinger_breakout": 0, "dca_bot": 10, "macd_divergence": 0,
        # MACD divergence removed from trending_down — catching knives is too risky
    },
    "sideways": {
        "ema_crossover_rsi": 5, "grid_trading": 25, "mean_reversion": 10,
        "bollinger_breakout": 5, "dca_bot": 5, "macd_divergence": 0,
    },
    "volatile": {
        "ema_crossover_rsi": 5, "grid_trading": 0, "mean_reversion": 5,
        "bollinger_breakout": 10, "dca_bot": 5, "macd_divergence": 5,
    },
    "transitioning": {
        # TRANSITIONING = dangerous zone. Only high-conviction strategies.
        # Reduced exposure, DCA for dollar-cost averaging, small mean reversion.
        "ema_crossover_rsi": 5, "grid_trading": 0, "mean_reversion": 5,
        "bollinger_breakout": 5, "dca_bot": 10, "macd_divergence": 0,
    },
    "unknown": {
        "ema_crossover_rsi": 5, "grid_trading": 0, "mean_reversion": 0,
        "bollinger_breakout": 0, "dca_bot": 5, "macd_divergence": 0,
    },
}

ALL_STRATEGY_NAMES = [
    "ema_crossover_rsi", "grid_trading", "mean_reversion",
    "bollinger_breakout", "dca_bot", "macd_divergence",
]


@dataclass
class StrategyAllocation:
    """Аллокация для одной стратегии."""
    strategy_name: str
    allocation_pct: float
    is_active: bool


def get_allocations(regime: MarketRegime) -> list[StrategyAllocation]:
    """Получить аллокации для текущего режима.

    Returns:
        Список аллокаций с is_active=True для стратегий с allocation > 0.
    """
    regime_key = regime.regime.value
    table = ALLOCATION_TABLE.get(regime_key, ALLOCATION_TABLE["unknown"])

    result = []
    for name in ALL_STRATEGY_NAMES:
        pct = table.get(name, 0)
        result.append(StrategyAllocation(
            strategy_name=name,
            allocation_pct=pct,
            is_active=pct > 0,
        ))
    return result


def get_active_strategies(regime: MarketRegime) -> list[str]:
    """Получить имена активных стратегий для режима."""
    return [a.strategy_name for a in get_allocations(regime) if a.is_active]


def get_strategy_budget_pct(regime: MarketRegime, strategy_name: str) -> float:
    """Получить бюджет стратегии в % от капитала."""
    regime_key = regime.regime.value
    table = ALLOCATION_TABLE.get(regime_key, ALLOCATION_TABLE["unknown"])
    return table.get(strategy_name, 0.0)


# ──────────────────────────────────────────────
# Adaptive Strategy Weighting (Phase 1)
# ──────────────────────────────────────────────

class AdaptiveAllocator:
    """Корректирует аллокации по скилу каждой стратегии за последние N дней."""

    def __init__(self, lookback_trades: int = 50) -> None:
        self._lookback = lookback_trades
        self._skill_scores: dict[str, float] = {}

    def update_skills(self, trades: list[StrategyTrade]) -> None:
        """Пересчитать skill score каждой стратегии на основе последних сделок."""
        for strategy in ALL_STRATEGY_NAMES:
            strat_trades = [t for t in trades if t.strategy_name == strategy]
            recent = strat_trades[-self._lookback:] if strat_trades else []

            if len(recent) < 5:
                self._skill_scores[strategy] = 0.5
                continue

            wins = sum(1 for t in recent if t.is_win)
            win_rate = wins / len(recent)

            # Profit factor
            gross_profit = sum(t.pnl_usd for t in recent if t.pnl_usd > 0) or 0.001
            gross_loss = abs(sum(t.pnl_usd for t in recent if t.pnl_usd < 0)) or 0.001
            profit_factor = min(gross_profit / gross_loss, 3.0)

            # Skill = weighted win_rate + profit_factor
            skill = 0.60 * win_rate + 0.40 * min(profit_factor / 2.0, 1.0)
            self._skill_scores[strategy] = max(0.05, min(skill, 1.0))

        if self._skill_scores:
            top = sorted(self._skill_scores.items(), key=lambda x: x[1], reverse=True)
            logger.info("Strategy skills: %s",
                         ", ".join(f"{n}={v:.2f}" for n, v in top))

    def get_adaptive_allocations(self, regime: MarketRegime) -> list[StrategyAllocation]:
        """Получить аллокации, скорректированные по скилу."""
        base_allocs = get_allocations(regime)

        if not self._skill_scores:
            return base_allocs

        adjusted = []
        total = 0.0
        for alloc in base_allocs:
            skill = self._skill_scores.get(alloc.strategy_name, 0.5)
            # Quadratic scaling with cap: penalty for skill < 0.5, max 1.5x boost
            multiplier = min((skill / 0.5) ** 2, 1.5)
            new_pct = alloc.allocation_pct * multiplier
            adjusted.append((alloc.strategy_name, new_pct))
            total += new_pct

        # Re-normalize to original total
        orig_total = sum(a.allocation_pct for a in base_allocs)
        if total > 0 and orig_total > 0:
            scale = orig_total / total
        else:
            scale = 1.0

        result = []
        for name, pct in adjusted:
            final_pct = pct * scale
            result.append(StrategyAllocation(
                strategy_name=name,
                allocation_pct=final_pct,
                is_active=final_pct > 0,
            ))
        return result
