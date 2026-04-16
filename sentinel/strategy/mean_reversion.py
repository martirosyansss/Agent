"""
Стратегия V3: Mean Reversion (RSI Extreme + Bollinger Bands).

Логика:
  BUY:  RSI(4h) < 25 + цена < lower BB(4h) + Volume > 1.8× avg + not falling knife
  SELL: RSI(4h) > 75 + trailing stop + BB middle revert + SL/TP + max hold

Confidence: base=0.65 + RSI extreme(+0.10) + volume(+0.08) + trend(+0.07)
Режим рынка: any (ловит экстремумы)
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
class MeanRevConfig:
    rsi_oversold: float = 25.0
    rsi_overbought: float = 75.0
    stop_loss_pct: float = 3.0
    take_profit_pct: float = 6.5          # R:R = 2.17 (was 5% = R:R 1.67)
    trailing_activate_pct: float = 3.0
    trailing_stop_pct: float = 1.5
    min_volume_ratio: float = 1.8
    min_confidence: float = 0.65
    max_position_pct: float = 15.0
    max_hold_hours: int = 48              # if no revert in 48h, thesis is invalid

    def __post_init__(self):
        if self.stop_loss_pct <= 0 or self.stop_loss_pct > 50:
            raise ValueError(f"stop_loss_pct must be (0, 50], got {self.stop_loss_pct}")
        if self.take_profit_pct <= 0:
            raise ValueError(f"take_profit_pct must be > 0, got {self.take_profit_pct}")


class MeanReversion(BaseStrategy):
    NAME = "mean_reversion"

    def __init__(self, config: MeanRevConfig | None = None) -> None:
        super().__init__()
        self._cfg = config or MeanRevConfig()
        self._max_price: dict[str, float] = {}
        self._entry_ts: dict[str, int] = {}
        self._rsi_history: dict[str, list[float]] = {}  # RSI divergence detection

    def _cleanup(self, sym: str) -> None:
        self._max_price.pop(sym, None)
        self._entry_ts.pop(sym, None)

    def generate_signal(self, features: FeatureVector, has_open_position: bool = False, entry_price: float | None = None) -> Optional[Signal]:
        cfg = self._cfg
        sym = features.symbol
        now_ms = features.timestamp or int(time.time() * 1000)

        if has_open_position and entry_price is not None:
            if entry_price <= 0:
                self._cleanup(sym)
                return Signal(timestamp=now_ms, symbol=sym, direction=Direction.SELL, confidence=0.99, strategy_name=self.NAME, reason=f"SAFETY: invalid entry_price={entry_price}", features=features)
            pnl_pct = (features.close - entry_price) / entry_price * 100
            self._max_price[sym] = max(self._max_price.get(sym, entry_price), features.close)

            exit_now, exit_conf, exit_reason = news_should_accelerate_exit(features, pnl_pct)
            if exit_now:
                self._cleanup(sym)
                return Signal(timestamp=now_ms, symbol=sym, direction=Direction.SELL, confidence=exit_conf, strategy_name=self.NAME, reason=exit_reason, features=features)
            if pnl_pct >= cfg.take_profit_pct:
                self._cleanup(sym)
                return Signal(timestamp=now_ms, symbol=sym, direction=Direction.SELL, confidence=0.90, strategy_name=self.NAME, reason=f"MeanRev TP: +{pnl_pct:.1f}% >= {cfg.take_profit_pct}%", features=features)
            if pnl_pct <= -cfg.stop_loss_pct:
                self._cleanup(sym)
                return Signal(timestamp=now_ms, symbol=sym, direction=Direction.SELL, confidence=0.95, strategy_name=self.NAME, reason=f"MeanRev SL: {pnl_pct:.1f}% <= -{cfg.stop_loss_pct}%", features=features)

            max_p = self._max_price.get(sym, entry_price)
            max_gain = (max_p - entry_price) / entry_price * 100
            if max_gain >= cfg.trailing_activate_pct:
                drop = (max_p - features.close) / max_p * 100
                if drop >= cfg.trailing_stop_pct:
                    self._cleanup(sym)
                    return Signal(timestamp=now_ms, symbol=sym, direction=Direction.SELL, confidence=0.85, strategy_name=self.NAME, reason=f"MeanRev trailing: drop {drop:.1f}% from max", features=features)

            if features.rsi_14 > cfg.rsi_overbought:
                self._cleanup(sym)
                return Signal(timestamp=now_ms, symbol=sym, direction=Direction.SELL, confidence=0.80, strategy_name=self.NAME, reason=f"MeanRev exit: RSI {features.rsi_14:.1f} > {cfg.rsi_overbought}", features=features)

            entry_ts = self._entry_ts.get(sym, now_ms)
            hours_held = (now_ms - entry_ts) / 3_600_000
            bb_mid = (features.bb_upper + features.bb_lower) / 2 if features.bb_upper > 0 and features.bb_lower > 0 else 0
            if hours_held >= 4.0 and bb_mid > 0 and features.close >= bb_mid and features.rsi_14 > 72:
                self._cleanup(sym)
                return Signal(timestamp=now_ms, symbol=sym, direction=Direction.SELL, confidence=0.75, strategy_name=self.NAME, reason=f"MeanRev revert: price>BB_mid, RSI={features.rsi_14:.0f} (held {hours_held:.1f}h)", features=features)

            if hours_held >= cfg.max_hold_hours and pnl_pct < cfg.take_profit_pct * 0.5:
                self._cleanup(sym)
                return Signal(timestamp=now_ms, symbol=sym, direction=Direction.SELL, confidence=0.70, strategy_name=self.NAME, reason=f"MeanRev time exit: held {hours_held:.0f}h", features=features)
            return None

        if has_open_position:
            return None

        # Track RSI for divergence detection
        rsi_hist = self._rsi_history.setdefault(sym, [])
        rsi_hist.append(features.rsi_14)
        if len(rsi_hist) > 30:
            self._rsi_history[sym] = rsi_hist[-30:]

        if features.rsi_14 >= cfg.rsi_oversold:
            return None
        # Allow 1% tolerance above lower BB — RSI extreme + near BB is valid entry
        if features.bb_lower > 0 and features.close >= features.bb_lower * 1.01:
            return None
        if features.volume_ratio < cfg.min_volume_ratio:
            return None
        if features.ema_9 < features.ema_21 < features.ema_50 and features.ema_50 > 0 and features.adx > 25:
            return None
        if features.ema_50 > 0 and features.close < features.ema_50 and features.adx > 30:
            return None
        dmi_spread = getattr(features, 'dmi_spread', None)
        if dmi_spread is not None and dmi_spread < -20:
            return None

        # RSI divergence check: if RSI keeps making lower lows without any bounce,
        # this is a continuation, not mean reversion. Require at least a subtle
        # RSI uptick vs previous low (bullish divergence hint).
        rsi_hist = self._rsi_history.get(sym, [])
        if len(rsi_hist) >= 5:
            recent_rsi_min = min(rsi_hist[-5:-1])  # previous 4 bars
            if features.rsi_14 < recent_rsi_min and recent_rsi_min < cfg.rsi_oversold:
                # RSI still making lower lows — continuation, not reversal
                return None

        confidence = 0.65
        if features.rsi_14 < 20:
            confidence += 0.10
        if features.volume_ratio > 2.0:
            confidence += 0.08
        if features.ema_50 > 0 and features.close > features.ema_50 * 0.95:
            confidence += 0.07

        # Williams %R extreme oversold confirmation
        if hasattr(features, 'williams_r') and features.williams_r < -90:
            confidence += 0.05

        # Ichimoku: if price is above cloud even when RSI oversold, stronger reversion
        if features.ichimoku_senkou_a > 0 and features.ichimoku_senkou_b > 0:
            cloud_bottom = min(features.ichimoku_senkou_a, features.ichimoku_senkou_b)
            if features.close > cloud_bottom:
                confidence += 0.05  # still in uptrend structure despite oversold

        blocked, block_reason = news_should_block_entry(features)
        if blocked:
            return None
        news_delta, news_reason = news_confidence_adjustment(features, "buy", "mean_reversion")
        confidence += news_delta
        confidence = min(confidence, 0.95)
        if confidence < cfg.min_confidence:
            return None

        # ATR-adaptive SL/TP
        if features.atr > 0 and features.close > 0:
            atr_pct = features.atr / features.close * 100
            sl_pct = max(cfg.stop_loss_pct, atr_pct * 1.5)
            tp_pct = max(cfg.take_profit_pct, atr_pct * 3.0)
        else:
            sl_pct, tp_pct = cfg.stop_loss_pct, cfg.take_profit_pct
        sl, tp = news_adjust_sl_tp(features, features.close, sl_pct, tp_pct)
        reason = f"MeanRev BUY: RSI={features.rsi_14:.1f}, price<lowerBB, vol_ratio={features.volume_ratio:.1f}"
        if news_delta != 0:
            reason += f", {news_reason}"
        self._entry_ts[sym] = now_ms
        return Signal(timestamp=now_ms, symbol=sym, direction=Direction.BUY, confidence=confidence, strategy_name=self.NAME, reason=reason, stop_loss_price=sl, take_profit_price=tp, features=features)
