"""
Стратегия V5: DCA Bot (Smart Dollar-Cost Averaging).

Логика:
  BUY:  каждые 24h, с множителем при dip (-3%->1.5x, -5%->2x, -10%->3x)
  SELL: trailing stop at +5%, full TP at +8%, drawdown stop at -15%

Confidence: base 0.80 (математический подход).
Защита: max 40% invested, min $100 reserve, drawdown < 15%.
Режим рынка: ALL (лучше в trending_down / sideways)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from core.models import Direction, FeatureVector, Signal
from strategy.base_strategy import (
    BaseStrategy,
    news_confidence_adjustment,
    news_should_accelerate_exit,
    news_should_block_entry,
    adaptive_min_confidence,
)


@dataclass
class DCAConfig:
    base_amount_usd: float = 10.0
    interval_hours: int = 24
    max_daily_buys: int = 3
    max_invested_pct: float = 40.0
    stop_drawdown_pct: float = 15.0
    take_profit_pct: float = 8.0
    trailing_activate_pct: float = 5.0
    trailing_stop_pct: float = 2.5
    min_confidence: float = 0.70

    # Dip multipliers: (price_drop_pct, amount_multiplier)
    dip_thresholds: list[tuple[float, float]] = None

    def __post_init__(self):
        if self.dip_thresholds is None:
            self.dip_thresholds = [(-3.0, 1.5), (-5.0, 2.0), (-10.0, 3.0)]
        if self.base_amount_usd <= 0:
            raise ValueError(f"base_amount_usd must be > 0, got {self.base_amount_usd}")
        if self.stop_drawdown_pct <= 0 or self.stop_drawdown_pct > 50:
            raise ValueError(f"stop_drawdown_pct must be (0, 50], got {self.stop_drawdown_pct}")


class DCABot(BaseStrategy):
    """Стратегия V5: DCA Bot — умное усреднение."""

    NAME = "dca_bot"

    def __init__(self, config: DCAConfig | None = None) -> None:
        super().__init__()
        self._cfg = config or DCAConfig()
        self._last_buy_time: dict[str, int] = {}
        self._daily_buys: dict[str, int] = {}
        self._daily_reset: dict[str, int] = {}
        self._max_price: dict[str, float] = {}

    def restore_state(
        self,
        last_buy_times: dict[str, int],
        daily_buys: dict[str, int],
    ) -> None:
        """Восстановить персистентное состояние из БД после перезапуска."""
        now_ms = int(time.time() * 1000)
        for sym, ts in last_buy_times.items():
            if ts > 0:
                self._last_buy_time[sym] = ts
        for sym, count in daily_buys.items():
            if count > 0:
                self._daily_buys[sym] = count
                self._daily_reset[sym] = now_ms

    def _get_dip_multiplier(self, features: FeatureVector) -> float:
        pct = features.price_change_15m
        multiplier = 1.0
        for threshold, mult in sorted(self._cfg.dip_thresholds, key=lambda x: x[0], reverse=True):
            if pct <= threshold:
                multiplier = mult
        return multiplier

    def _reset_daily(self, sym: str, now_ms: int) -> None:
        day_ms = 86400 * 1000
        last_reset = self._daily_reset.get(sym, 0)
        if now_ms - last_reset > day_ms:
            self._daily_buys[sym] = 0
            self._daily_reset[sym] = now_ms

    def _cleanup(self, sym: str) -> None:
        self._max_price.pop(sym, None)

    def generate_signal(
        self,
        features: FeatureVector,
        has_open_position: bool = False,
        entry_price: float | None = None,
    ) -> Optional[Signal]:
        cfg = self._cfg
        sym = features.symbol
        now_ms = features.timestamp or int(time.time() * 1000)
        self._reset_daily(sym, now_ms)

        # ── SELL ──
        if has_open_position and entry_price is not None:
            if entry_price <= 0:
                self._cleanup(sym)
                return Signal(timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                              confidence=0.99, strategy_name=self.NAME,
                              reason=f"SAFETY: invalid entry_price={entry_price}", features=features)
            pnl_pct = (features.close - entry_price) / entry_price * 100
            self._max_price[sym] = max(self._max_price.get(sym, entry_price), features.close)

            # News-driven emergency exit
            exit_now, exit_conf, exit_reason = news_should_accelerate_exit(features, pnl_pct)
            if exit_now:
                self._cleanup(sym)
                return Signal(timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                              confidence=exit_conf, strategy_name=self.NAME,
                              reason=exit_reason, features=features)

            # Full take profit
            if pnl_pct >= cfg.take_profit_pct:
                self._cleanup(sym)
                return Signal(timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                              confidence=0.85, strategy_name=self.NAME,
                              reason=f"DCA full TP: +{pnl_pct:.1f}% >= {cfg.take_profit_pct}%",
                              features=features)

            # Trailing stop (replaces broken partial TP)
            max_p = self._max_price.get(sym, entry_price)
            max_gain_pct = (max_p - entry_price) / entry_price * 100
            if max_gain_pct >= cfg.trailing_activate_pct:
                drop_from_max = (max_p - features.close) / max_p * 100
                if drop_from_max >= cfg.trailing_stop_pct:
                    self._cleanup(sym)
                    return Signal(timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                                  confidence=0.82, strategy_name=self.NAME,
                                  reason=f"DCA trailing: drop {drop_from_max:.1f}% from max (activated at +{max_gain_pct:.1f}%)",
                                  features=features)

            # Drawdown stop
            if pnl_pct <= -cfg.stop_drawdown_pct:
                self._cleanup(sym)
                return Signal(timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                              confidence=0.95, strategy_name=self.NAME,
                              reason=f"DCA drawdown stop: {pnl_pct:.1f}% <= -{cfg.stop_drawdown_pct}%",
                              features=features)
            return None

        # ── BUY ──
        last_buy = self._last_buy_time.get(sym, 0)
        interval_ms = cfg.interval_hours * 3600 * 1000
        if now_ms - last_buy < interval_ms:
            return None
        daily = self._daily_buys.get(sym, 0)
        if daily >= cfg.max_daily_buys:
            return None

        # News block
        blocked, block_reason = news_should_block_entry(features)
        if blocked:
            return None

        multiplier = self._get_dip_multiplier(features)
        news_delta, news_reason = news_confidence_adjustment(features, direction="buy")
        dca_confidence = 0.80 + news_delta

        regime = getattr(features, 'market_regime', 'unknown')
        eff_threshold = adaptive_min_confidence(cfg.min_confidence, regime, "dca")
        if dca_confidence < eff_threshold:
            return None

        # News-aware sizing
        news_mult = 1.0
        if features.news_critical_alert:
            news_mult = 0.5
        elif features.news_composite_score < -0.2 and features.news_actionable:
            news_mult = 1.3
        elif features.news_composite_score > 0.3 and features.news_actionable:
            news_mult = 0.7
        multiplier *= news_mult

        amount = cfg.base_amount_usd * multiplier
        qty = amount / features.close if features.close > 0 else 0

        # Update cooldown AFTER all validation passed
        self._last_buy_time[sym] = now_ms
        self._daily_buys[sym] = daily + 1
        self._max_price[sym] = features.close

        reason = f"DCA buy: ${amount:.2f} (mult={multiplier:.1f}x)"
        if news_delta != 0:
            reason += f", {news_reason}"

        # SL for Risk Sentinel compliance — use ATR-aware SL if available.
        # Internal drawdown stop at stop_drawdown_pct is handled in SELL logic.
        _sl_pct_for_risk = min(features.atr / features.close * 100 * 2.5, 8.0) if features.atr > 0 and features.close > 0 else 3.0
        return Signal(
            timestamp=now_ms, symbol=sym, direction=Direction.BUY,
            confidence=dca_confidence, strategy_name=self.NAME,
            reason=reason, suggested_quantity=qty,
            stop_loss_price=features.close * (1 - _sl_pct_for_risk / 100),
            take_profit_price=features.close * (1 + cfg.take_profit_pct / 100),
            features=features,
        )
