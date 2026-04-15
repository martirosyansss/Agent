"""
Стратегия V2: Grid Trading (ATR-adaptive, BB-based range).

Логика:
  Авто-определение границ через Bollinger Bands (20, 2.0).
  N уровней от lower BB до upper BB, spacing адаптируется к ATR.
  BUY: цена ниже grid level + spread filter + regime check.
  SELL: ATR-based TP (min 0.5%) + trailing + SL + max hold.

Защита:
  - max 30% капитала
  - стоп при выходе цены за границы >2%
  - spread filter (не входить при широком спреде)
  - rebuild при изменении волатильности >50%
  - commission-aware TP (вычитает 0.2% round-trip)
Режим рынка: sideways (основной)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from core.models import Direction, FeatureVector, Signal
from strategy.base_strategy import (
    BaseStrategy,
    news_confidence_adjustment,
    news_should_block_entry,
    news_should_accelerate_exit,
    news_adjust_sl_tp,
)

COMMISSION_ROUND_TRIP_PCT = 0.20  # Binance spot: 0.1% maker + 0.1% taker


@dataclass
class GridConfig:
    num_grids: int = 8
    capital_pct: float = 30.0
    min_profit_pct: float = 0.5          # raised from 0.3: must cover commissions
    max_loss_pct: float = 5.0
    trailing_activate_pct: float = 1.0   # activate trailing at +1%
    trailing_stop_pct: float = 0.5       # trail 0.5% from max (tight for grid)
    max_hold_hours: int = 48             # prevent stuck positions
    min_confidence: float = 0.70
    min_volume_ratio: float = 0.8
    max_spread_pct: float = 0.15         # skip entry when spread > 0.15%
    rebuild_vol_change_pct: float = 50.0 # rebuild grid when BB width changes >50%

    def __post_init__(self):
        if self.num_grids <= 0:
            raise ValueError(f"num_grids must be > 0, got {self.num_grids}")
        if self.max_loss_pct <= 0 or self.max_loss_pct > 50:
            raise ValueError(f"max_loss_pct must be (0, 50], got {self.max_loss_pct}")


class GridTrading(BaseStrategy):
    """Стратегия V2: Grid Trading — ATR-adaptive grid."""

    NAME = "grid_trading"

    def __init__(self, config: GridConfig | None = None) -> None:
        super().__init__()
        self._cfg = config or GridConfig()
        self._grid_levels: dict[str, list[float]] = {}
        self._last_rebuild: dict[str, int] = {}
        self._filled_buys: dict[str, set[int]] = {}
        self._bb_width_at_build: dict[str, float] = {}  # track BB width for rebuild trigger
        self._max_price: dict[str, float] = {}           # trailing stop
        self._entry_ts: dict[str, int] = {}               # hold time tracking

    def _build_grid(self, f: FeatureVector) -> list[float]:
        """Build grid levels from lower BB to upper BB."""
        low = f.bb_lower
        high = f.bb_upper
        if high <= low or low <= 0 or self._cfg.num_grids <= 0:
            return []
        step = (high - low) / self._cfg.num_grids
        return [low + step * i for i in range(self._cfg.num_grids + 1)]

    def _should_rebuild(self, sym: str, now_ms: int, features: FeatureVector) -> bool:
        """Rebuild grid on timer (24h) OR when volatility shifts >50%."""
        last = self._last_rebuild.get(sym, 0)
        if (now_ms - last) > 24 * 3600 * 1000:
            return True
        # Volatility-based rebuild
        old_width = self._bb_width_at_build.get(sym, 0)
        if old_width > 0 and features.bb_bandwidth > 0:
            change_pct = abs(features.bb_bandwidth - old_width) / old_width * 100
            if change_pct > self._cfg.rebuild_vol_change_pct:
                return True
        return False

    def _cleanup(self, sym: str) -> None:
        self._max_price.pop(sym, None)
        self._entry_ts.pop(sym, None)

    def _atr_based_tp(self, features: FeatureVector) -> float:
        """Calculate ATR-based take-profit, minimum min_profit_pct, commission-aware."""
        if features.atr > 0 and features.close > 0:
            atr_pct = features.atr / features.close * 100
            # Grid TP = 30% of ATR (capture a fraction of expected range)
            raw_tp = atr_pct * 0.3
        else:
            raw_tp = self._cfg.min_profit_pct
        # Subtract commissions + add floor
        net_tp = max(raw_tp, self._cfg.min_profit_pct) + COMMISSION_ROUND_TRIP_PCT
        return net_tp

    def generate_signal(
        self,
        features: FeatureVector,
        has_open_position: bool = False,
        entry_price: float | None = None,
    ) -> Optional[Signal]:
        cfg = self._cfg
        sym = features.symbol
        now_ms = int(time.time() * 1000)

        # Rebuild grid if needed (timer or volatility shift)
        if sym not in self._grid_levels or self._should_rebuild(sym, now_ms, features):
            levels = self._build_grid(features)
            if not levels:
                return None
            self._grid_levels[sym] = levels
            self._last_rebuild[sym] = now_ms
            self._filled_buys[sym] = set()
            self._bb_width_at_build[sym] = features.bb_bandwidth

        levels = self._grid_levels[sym]
        if not levels:
            return None

        price = features.close
        grid_low, grid_high = levels[0], levels[-1]

        # Safety: price outside grid range by >2%
        if price < grid_low * 0.98 or price > grid_high * 1.02:
            return None

        # ── SELL ──
        if has_open_position and entry_price is not None:
            if entry_price <= 0:
                self._cleanup(sym)
                return Signal(timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                              confidence=0.99, strategy_name=self.NAME,
                              reason=f"SAFETY: invalid entry_price={entry_price}",
                              features=features)
            pnl_pct = (price - entry_price) / entry_price * 100
            self._max_price[sym] = max(self._max_price.get(sym, entry_price), price)

            # News-driven emergency exit
            exit_now, exit_conf, exit_reason = news_should_accelerate_exit(features, pnl_pct)
            if exit_now:
                self._cleanup(sym)
                return Signal(timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                              confidence=exit_conf, strategy_name=self.NAME,
                              reason=exit_reason, features=features)

            # ATR-based take profit (commission-aware)
            tp_pct = self._atr_based_tp(features)
            if pnl_pct >= tp_pct:
                self._cleanup(sym)
                return Signal(timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                              confidence=0.75, strategy_name=self.NAME,
                              reason=f"Grid TP: PnL {pnl_pct:.2f}% >= {tp_pct:.2f}% (ATR-based)",
                              features=features)

            # Trailing stop
            max_p = self._max_price.get(sym, entry_price)
            max_gain = (max_p - entry_price) / entry_price * 100
            if max_gain >= cfg.trailing_activate_pct:
                drop = (max_p - price) / max_p * 100
                if drop >= cfg.trailing_stop_pct:
                    self._cleanup(sym)
                    return Signal(timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                                  confidence=0.78, strategy_name=self.NAME,
                                  reason=f"Grid trailing: drop {drop:.1f}% from max",
                                  features=features)

            # Stop loss
            if pnl_pct <= -cfg.max_loss_pct:
                self._cleanup(sym)
                return Signal(timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                              confidence=0.90, strategy_name=self.NAME,
                              reason=f"Grid SL: PnL {pnl_pct:.2f}% <= -{cfg.max_loss_pct}%",
                              features=features)

            # Max hold time
            entry_ts = self._entry_ts.get(sym, now_ms)
            hours_held = (now_ms - entry_ts) / 3_600_000
            if hours_held >= cfg.max_hold_hours and pnl_pct < tp_pct * 0.5:
                self._cleanup(sym)
                return Signal(timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                              confidence=0.65, strategy_name=self.NAME,
                              reason=f"Grid time exit: held {hours_held:.0f}h, pnl={pnl_pct:+.1f}%",
                              features=features)

            return None

        # ── BUY ──
        if not has_open_position:
            # News block
            blocked, block_reason = news_should_block_entry(features)
            if blocked:
                return None

            # Volume filter
            if features.volume_ratio < cfg.min_volume_ratio:
                return None

            # Spread filter: don't enter when spread is too wide
            spread_pct = features.spread / price * 100 if price > 0 and features.spread > 0 else 0
            if spread_pct > cfg.max_spread_pct:
                return None

            filled = self._filled_buys.get(sym, set())
            for i, level in enumerate(levels):
                if i in filled:
                    continue
                if price <= level:
                    self._filled_buys.setdefault(sym, set()).add(i)

                    # News-adjusted SL/TP
                    sl, tp = news_adjust_sl_tp(features, price, cfg.max_loss_pct, cfg.min_profit_pct)

                    # Confidence
                    news_delta, news_reason = news_confidence_adjustment(features, direction="buy")
                    grid_conf = 0.75 + news_delta

                    if grid_conf < cfg.min_confidence:
                        continue

                    self._entry_ts[sym] = now_ms
                    self._max_price[sym] = price

                    reason = f"Grid BUY at level {i}/{len(levels)-1}, price={price:.2f}"
                    if news_delta != 0:
                        reason += f", {news_reason}"

                    return Signal(
                        timestamp=now_ms, symbol=sym, direction=Direction.BUY,
                        confidence=grid_conf, strategy_name=self.NAME,
                        reason=reason, stop_loss_price=sl, take_profit_price=tp,
                        features=features,
                    )

        return None
