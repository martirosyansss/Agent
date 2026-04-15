"""
Стратегия V4: Bollinger Band Breakout.

Логика:
  BUY:  close > upper BB + breakout magnitude + volume > 1.5x avg + RSI < 78 + ADX > 20
  SELL: back inside bands + trailing stop + SL/TP + RSI > 85 + max hold time

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
    trailing_stop_pct: float = 1.5       # tighter trail to lock in breakout profits
    trailing_activate_pct: float = 3.0   # activate after meaningful move
    min_confidence: float = 0.70
    max_position_pct: float = 15.0
    max_hold_hours: int = 72

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
        self._entry_ts: dict[str, int] = {}
        self._bars_inside_bands: dict[str, int] = {}
        self._squeeze_bars: dict[str, int] = {}      # track squeeze duration

    def _cleanup(self, sym: str) -> None:
        self._max_price.pop(sym, None)
        self._entry_ts.pop(sym, None)
        self._bars_inside_bands.pop(sym, None)
        # Don't clear squeeze_bars — it's pre-entry tracking

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
                self._cleanup(sym)
                return Signal(
                    timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                    confidence=0.99, strategy_name=self.NAME,
                    reason=f"SAFETY: invalid entry_price={entry_price}",
                    features=features,
                )
            pnl_pct = (features.close - entry_price) / entry_price * 100

            # News-driven emergency exit
            exit_now, exit_conf, exit_reason = news_should_accelerate_exit(features, pnl_pct)
            if exit_now:
                self._cleanup(sym)
                return Signal(
                    timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                    confidence=exit_conf, strategy_name=self.NAME,
                    reason=exit_reason, features=features,
                )

            # Update max price for trailing stop
            self._max_price[sym] = max(self._max_price.get(sym, entry_price), features.close)

            # Take profit (net of 0.2% round-trip commission)
            net_tp = cfg.take_profit_pct - 0.20
            if pnl_pct >= net_tp:
                self._cleanup(sym)
                return Signal(
                    timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                    confidence=0.90, strategy_name=self.NAME,
                    reason=f"BB TP: +{pnl_pct:.1f}% >= {net_tp:.1f}% (net)",
                    features=features,
                )
            # Stop loss
            if pnl_pct <= -cfg.stop_loss_pct:
                self._cleanup(sym)
                return Signal(
                    timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                    confidence=0.95, strategy_name=self.NAME,
                    reason=f"BB SL: {pnl_pct:.1f}% <= -{cfg.stop_loss_pct}%",
                    features=features,
                )
            # Trailing stop
            if pnl_pct >= cfg.trailing_activate_pct:
                max_p = self._max_price.get(sym, features.close)
                if max_p <= 0:
                    max_p = features.close
                drawdown_from_max = (max_p - features.close) / max_p * 100 if max_p > 0 else 0.0
                if drawdown_from_max >= cfg.trailing_stop_pct:
                    self._cleanup(sym)
                    return Signal(
                        timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                        confidence=0.85, strategy_name=self.NAME,
                        reason=f"BB trailing stop: {drawdown_from_max:.1f}% from max",
                        features=features,
                    )

            # Back inside bands: failed breakout detection
            if features.bb_upper > 0 and features.close < features.bb_upper:
                self._bars_inside_bands[sym] = self._bars_inside_bands.get(sym, 0) + 1
            else:
                self._bars_inside_bands[sym] = 0

            if self._bars_inside_bands.get(sym, 0) >= 2 and pnl_pct < cfg.take_profit_pct * 0.5:
                self._cleanup(sym)
                return Signal(
                    timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                    confidence=0.72, strategy_name=self.NAME,
                    reason=f"BB failed breakout: {self._bars_inside_bands.get(sym, 0)} bars inside bands, pnl={pnl_pct:+.1f}%",
                    features=features,
                )

            # Weakness: price back below upper BB + extreme RSI
            if features.close < features.bb_upper and pnl_pct > 0:
                if features.rsi_14 > 85:
                    self._cleanup(sym)
                    return Signal(
                        timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                        confidence=0.75, strategy_name=self.NAME,
                        reason=f"BB weakness: close<upperBB + RSI={features.rsi_14:.0f}>85",
                        features=features,
                    )

            # Max hold time exit
            entry_ts = self._entry_ts.get(sym, now_ms)
            hours_held = (now_ms - entry_ts) / 3_600_000
            if hours_held >= cfg.max_hold_hours and pnl_pct < cfg.take_profit_pct * 0.5:
                self._cleanup(sym)
                return Signal(
                    timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                    confidence=0.70, strategy_name=self.NAME,
                    reason=f"BB time exit: held {hours_held:.0f}h (max {cfg.max_hold_hours}h), pnl={pnl_pct:+.1f}%",
                    features=features,
                )

            return None

        # ── BUY ──
        if has_open_position:
            return None

        # Breakout above upper BB
        if features.bb_upper <= 0 or features.close <= features.bb_upper:
            return None
        # Breakout magnitude filter
        if (features.close - features.bb_upper) / features.bb_upper <= 0.001:
            return None
        # Volume confirmation
        if features.volume_ratio < cfg.volume_confirm_mult:
            return None
        # RSI not too hot (78 instead of 70 — breakouts often have RSI 70-77)
        if features.rsi_14 >= 78:
            return None
        # ADX confirms trend
        if features.adx < 20:
            return None

        # Squeeze detection with duration tracking
        if hasattr(features, 'hist_volatility') and features.hist_volatility > 0:
            is_squeeze = features.bb_bandwidth < features.hist_volatility * 0.5
        else:
            is_squeeze = features.bb_bandwidth < cfg.squeeze_threshold

        # Track squeeze duration (min 5 bars for reliable breakout)
        if is_squeeze:
            self._squeeze_bars[sym] = self._squeeze_bars.get(sym, 0) + 1
        else:
            self._squeeze_bars[sym] = 0

        squeeze_duration = self._squeeze_bars.get(sym, 0)
        long_squeeze = squeeze_duration >= 5  # 5+ bars of squeeze = high energy

        # Confidence
        confidence = 0.60
        if features.volume_ratio > 2.0:
            confidence += 0.10
        if long_squeeze:
            confidence += 0.12  # long squeeze = higher energy breakout
        elif is_squeeze:
            confidence += 0.06  # short squeeze = weaker signal
        if features.adx > 30:
            confidence += 0.05
        if features.ema_9 > features.ema_21:
            confidence += 0.05

        # News block
        blocked, block_reason = news_should_block_entry(features)
        if blocked:
            return None

        news_delta, news_reason = news_confidence_adjustment(features, "buy", "breakout")
        confidence += news_delta
        confidence = min(confidence, 0.95)

        if confidence < cfg.min_confidence:
            return None

        # ATR-adaptive SL/TP: use 1.5x ATR as SL floor, 3x ATR as TP floor
        if features.atr > 0 and features.close > 0:
            atr_pct = features.atr / features.close * 100
            sl_pct = max(cfg.stop_loss_pct, atr_pct * 1.5)
            tp_pct = max(cfg.take_profit_pct, atr_pct * 3.0)
        else:
            sl_pct = cfg.stop_loss_pct
            tp_pct = cfg.take_profit_pct
        sl, tp = news_adjust_sl_tp(features, features.close, sl_pct, tp_pct)

        reason = f"BB Breakout: close={features.close:.2f}>upperBB={features.bb_upper:.2f}, ADX={features.adx:.0f}, squeeze={is_squeeze}"
        if news_delta != 0:
            reason += f", {news_reason}"

        self._entry_ts[sym] = now_ms

        return Signal(
            timestamp=now_ms, symbol=sym, direction=Direction.BUY,
            confidence=confidence, strategy_name=self.NAME,
            reason=reason,
            stop_loss_price=sl, take_profit_price=tp,
            features=features,
        )
