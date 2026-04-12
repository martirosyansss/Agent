"""
Strategy Selector — автоматический выбор стратегий по режиму рынка.

ALLOCATION_TABLE определяет % капитала для каждой стратегии в каждом режиме.
Фактическая экспозиция ≤ 60%, max 2 направленных + 1 grid + 1 DCA.
Auto-selection отключён по умолчанию (auto_strategy_selection=False).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.models import MarketRegime, MarketRegimeType


# Allocation table: regime → {strategy: allocation_pct}
# "reserve" = что остаётся в кэше
ALLOCATION_TABLE: dict[str, dict[str, float]] = {
    "trending_up": {
        "ema_crossover_rsi": 25, "grid_trading": 5, "mean_reversion": 0,
        "bollinger_breakout": 15, "dca_bot": 5, "macd_divergence": 0,
    },
    "trending_down": {
        "ema_crossover_rsi": 0, "grid_trading": 0, "mean_reversion": 5,
        "bollinger_breakout": 0, "dca_bot": 10, "macd_divergence": 5,
    },
    "sideways": {
        "ema_crossover_rsi": 5, "grid_trading": 25, "mean_reversion": 10,
        "bollinger_breakout": 5, "dca_bot": 5, "macd_divergence": 0,
    },
    "volatile": {
        "ema_crossover_rsi": 5, "grid_trading": 0, "mean_reversion": 5,
        "bollinger_breakout": 10, "dca_bot": 5, "macd_divergence": 5,
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
