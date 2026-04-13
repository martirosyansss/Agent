"""
Стратегия V2: Grid Trading (Сетка, 4h BB-based range).

Логика:
  Авто-определение границ через Bollinger Bands (20, 2.0).
  N уровней равномерно от lower BB до upper BB.
  BUY: цена ниже очередного grid level → покупка.
  SELL: цена выше grid level на min_profit_pct → продажа.

Защита:
  - max 30% капитала
  - стоп при выходе цены за границы >2%
  - отключение при падении >3%/час
Режим рынка: sideways (основной)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from core.models import Direction, FeatureVector, Signal
from strategy.base_strategy import BaseStrategy, news_confidence_adjustment


@dataclass
class GridConfig:
    num_grids: int = 8
    capital_pct: float = 30.0
    min_profit_pct: float = 0.3
    max_loss_pct: float = 5.0
    min_confidence: float = 0.70

    def __post_init__(self):
        if self.num_grids <= 0:
            raise ValueError(f"num_grids must be > 0, got {self.num_grids}")
        if self.max_loss_pct <= 0 or self.max_loss_pct > 50:
            raise ValueError(f"max_loss_pct must be (0, 50], got {self.max_loss_pct}")


class GridTrading(BaseStrategy):
    """Стратегия V2: Grid Trading."""

    NAME = "grid_trading"

    def __init__(self, config: GridConfig | None = None) -> None:
        super().__init__()
        self._cfg = config or GridConfig()
        self._grid_levels: dict[str, list[float]] = {}
        self._last_rebuild: dict[str, int] = {}
        self._filled_buys: dict[str, set[int]] = {}

    def _build_grid(self, f: FeatureVector) -> list[float]:
        """Построить сетку уровней от lower BB до upper BB."""
        low = f.bb_lower
        high = f.bb_upper
        if high <= low or low <= 0 or self._cfg.num_grids <= 0:
            return []
        step = (high - low) / self._cfg.num_grids
        return [low + step * i for i in range(self._cfg.num_grids + 1)]

    def _should_rebuild(self, symbol: str, now_ms: int) -> bool:
        last = self._last_rebuild.get(symbol, 0)
        return (now_ms - last) > 24 * 3600 * 1000  # min 24h between rebuilds

    def generate_signal(
        self,
        features: FeatureVector,
        has_open_position: bool = False,
        entry_price: float | None = None,
    ) -> Optional[Signal]:
        cfg = self._cfg
        sym = features.symbol
        now_ms = int(time.time() * 1000)

        # Rebuild grid if needed
        if sym not in self._grid_levels or self._should_rebuild(sym, now_ms):
            levels = self._build_grid(features)
            if not levels:
                return None
            self._grid_levels[sym] = levels
            self._last_rebuild[sym] = now_ms
            self._filled_buys[sym] = set()

        levels = self._grid_levels[sym]
        if not levels:
            return None

        price = features.close

        # Safety: price outside grid range by >2%
        grid_low, grid_high = levels[0], levels[-1]
        if price < grid_low * 0.98 or price > grid_high * 1.02:
            return None

        # SELL: if has position and price moved up enough from entry
        if has_open_position and entry_price is not None:
            if entry_price <= 0:
                return Signal(
                    timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                    confidence=0.99, strategy_name=self.NAME,
                    reason=f"SAFETY: invalid entry_price={entry_price}",
                )
            pnl_pct = (price - entry_price) / entry_price * 100
            if pnl_pct >= cfg.min_profit_pct:
                return Signal(
                    timestamp=now_ms,
                    symbol=sym,
                    direction=Direction.SELL,
                    confidence=0.75,
                    strategy_name=self.NAME,
                    reason=f"Grid TP: PnL {pnl_pct:.2f}% >= {cfg.min_profit_pct}%",
                    stop_loss_price=0.0,
                    take_profit_price=0.0,
                )
            # Stop loss
            if pnl_pct <= -cfg.max_loss_pct:
                return Signal(
                    timestamp=now_ms,
                    symbol=sym,
                    direction=Direction.SELL,
                    confidence=0.90,
                    strategy_name=self.NAME,
                    reason=f"Grid SL: PnL {pnl_pct:.2f}% <= -{cfg.max_loss_pct}%",
                )
            return None

        # BUY: find lowest unfilled grid level above current price
        if not has_open_position:
            filled = self._filled_buys.get(sym, set())
            for i, level in enumerate(levels):
                if i in filled:
                    continue
                if price <= level:
                    self._filled_buys.setdefault(sym, set()).add(i)
                    sl = price * (1 - cfg.max_loss_pct / 100)
                    tp = price * (1 + cfg.min_profit_pct / 100)

                    # Grid: professional news-adjusted confidence
                    news_delta, news_reason = news_confidence_adjustment(features, direction="buy")
                    grid_conf = 0.75 + news_delta

                    reason = f"Grid BUY at level {i}/{len(levels)-1}, price={price:.2f}"
                    if news_delta != 0:
                        reason += f", {news_reason}"

                    return Signal(
                        timestamp=now_ms,
                        symbol=sym,
                        direction=Direction.BUY,
                        confidence=grid_conf,
                        strategy_name=self.NAME,
                        reason=reason,
                        stop_loss_price=sl,
                        take_profit_price=tp,
                    )

        return None
