"""
Стратегия V4: Bollinger Band Breakout.

Логика:
  BUY:  close > upper BB + volume > 1.5× avg + squeeze предшествовал + RSI < 80 + ADX > 20
  SELL: close < upper BB (ослабление) ИЛИ trailing stop ИЛИ SL/TP ИЛИ RSI > 85

Confidence: base=0.60 + vol>2x(+0.10) + squeeze(+0.10) + ADX>30(+0.05) + EMA9>EMA21(+0.05)
Режим рынка: trending_up, volatile
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
    news_adjust_sl_tp,
)


@dataclass
class BBBreakoutConfig:
    bb_period: int = 20
    bb_std_dev: float = 2.0
    volume_confirm_mult: float = 1.5
    squeeze_threshold: float = 0.05
    stop_loss_pct: float = 3.0
    take_profit_pct: float = 6.0
    trailing_stop_pct: float = 2.0
    trailing_activate_pct: float = 3.0
    min_confidence: float = 0.70
    max_position_pct: float = 15.0

    def __post_init__(self):
        if self.stop_loss_pct <= 0 or self.stop_loss_pct > 50:
            raise ValueError(f"stop_loss_pct must be (0, 50], got {self.stop_loss_pct}")
        if self.take_profit_pct <= 0:
            raise ValueError(f"take_profit_pct must be > 0, got {self.take_profit_pct}")


class BollingerBreakout(BaseStrategy):
    """Стратегия V4: Bollinger Band Breakout."""

    NAME = "bollinger_breakout"

    def __init__(self, config: BBBreakoutConfig | None = None) -> None:
        super().__init__()
        self._cfg = config or BBBreakoutConfig()
        self._max_price: dict[str, float] = {}

    def generate_signal(
        self,
        features: FeatureVector,
        has_open_position: bool = False,
        entry_price: float | None = None,
    ) -> Optional[Signal]:
        cfg = self._cfg
        sym = features.symbol
        now_ms = int(time.time() * 1000)

        # ── SELL ──
        if has_open_position and entry_price is not None:
            if entry_price <= 0:
                self._max_price.pop(sym, None)
                return Signal(
                    timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                    confidence=0.99, strategy_name=self.NAME,
                    reason=f"SAFETY: invalid entry_price={entry_price}",
                )
            pnl_pct = (features.close - entry_price) / entry_price * 100

            # News-driven emergency exit (critical bearish / security event / profit lock)
            exit_now, exit_conf, exit_reason = news_should_accelerate_exit(features, pnl_pct)
            if exit_now:
                self._max_price.pop(sym, None)
                return Signal(
                    timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                    confidence=exit_conf, strategy_name=self.NAME,
                    reason=exit_reason,
                )

            # Update max price for trailing stop
            self._max_price[sym] = max(self._max_price.get(sym, entry_price), features.close)

            # Take profit
            if pnl_pct >= cfg.take_profit_pct:
                self._max_price.pop(sym, None)
                return Signal(
                    timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                    confidence=0.90, strategy_name=self.NAME,
                    reason=f"BB TP: +{pnl_pct:.1f}% >= {cfg.take_profit_pct}%",
                )
            # Stop loss
            if pnl_pct <= -cfg.stop_loss_pct:
                self._max_price.pop(sym, None)
                return Signal(
                    timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                    confidence=0.95, strategy_name=self.NAME,
                    reason=f"BB SL: {pnl_pct:.1f}% <= -{cfg.stop_loss_pct}%",
                )
            # Trailing stop (activate after +3%, trail at 2%)
            if pnl_pct >= cfg.trailing_activate_pct:
                max_p = self._max_price.get(sym, features.close)
                drawdown_from_max = (max_p - features.close) / max_p * 100
                if drawdown_from_max >= cfg.trailing_stop_pct:
                    self._max_price.pop(sym, None)
                    return Signal(
                        timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                        confidence=0.85, strategy_name=self.NAME,
                        reason=f"BB trailing stop: {drawdown_from_max:.1f}% from max",
                    )
            # Weakness: price back below upper BB
            if features.close < features.bb_upper and pnl_pct > 0:
                if features.rsi_14 > 85:
                    self._max_price.pop(sym, None)
                    return Signal(
                        timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                        confidence=0.75, strategy_name=self.NAME,
                        reason=f"BB weakness: close<upperBB + RSI={features.rsi_14:.0f}>85",
                    )
            return None

        # ── BUY ──
        if has_open_position:
            return None

        # Breakout above upper BB
        if features.bb_upper <= 0 or features.close <= features.bb_upper:
            return None
        # Volume confirmation
        if features.volume_ratio < cfg.volume_confirm_mult:
            return None
        # RSI not too hot
        if features.rsi_14 >= 70:
            return None
        # ADX confirms trend
        if features.adx < 20:
            return None
        # Squeeze preceded — use bb_pct_b (relative) instead of fixed threshold
        # bb_pct_b near 1.0 means at upper BB; low bb_bandwidth = squeeze
        # Use hist_volatility as adaptive reference if available
        if hasattr(features, 'hist_volatility') and features.hist_volatility > 0:
            is_squeeze = features.bb_bandwidth < features.hist_volatility * 0.5
        else:
            is_squeeze = features.bb_bandwidth < cfg.squeeze_threshold

        # Confidence
        confidence = 0.60
        if features.volume_ratio > 2.0:
            confidence += 0.10
        if is_squeeze:
            confidence += 0.10
        if features.adx > 30:
            confidence += 0.05
        if features.ema_9 > features.ema_21:
            confidence += 0.05

        # News: block entry on critical events (black swan / security)
        blocked, block_reason = news_should_block_entry(features)
        if blocked:
            return None

        # News confidence adjustment (composite_score, strength, category, F&G)
        news_delta, news_reason = news_confidence_adjustment(features, "buy", "breakout")
        confidence += news_delta

        confidence = min(confidence, 0.95)

        if confidence < cfg.min_confidence:
            return None

        # SL / TP (adjusted for news-driven volatility)
        sl, tp = news_adjust_sl_tp(features, features.close, cfg.stop_loss_pct, cfg.take_profit_pct)

        reason = f"BB Breakout: close={features.close:.2f}>upperBB={features.bb_upper:.2f}, ADX={features.adx:.0f}, squeeze={is_squeeze}"
        if news_delta != 0:
            reason += f", {news_reason}"

        return Signal(
            timestamp=now_ms, symbol=sym, direction=Direction.BUY,
            confidence=confidence, strategy_name=self.NAME,
            reason=reason,
            stop_loss_price=sl, take_profit_price=tp,
        )
