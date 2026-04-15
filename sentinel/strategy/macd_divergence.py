"""
Стратегия V6: MACD Divergence (proper swing-point detection).

Логика:
  Bullish divergence BUY: 2 price swing lows (lower-lower) + 2 MACD swing lows (higher-lower)
                          + RSI < 30 + vol confirm
  Bearish divergence SELL: 2 price swing highs (higher-higher) + 2 MACD swing highs (lower-higher)

Confidence: base=0.55 + RSI confirm(+0.10) + vol>1.3x(+0.08) + MACD zero cross(+0.05)
Режим рынка: trending_down → trending_up (ловит развороты)
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
class MACDDivConfig:
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal_period: int = 9
    lookback_candles: int = 50
    min_divergence_bars: int = 10        # min distance between swing points
    swing_window: int = 3                # window for swing point detection
    require_rsi_confirm: bool = True
    rsi_oversold: float = 30.0           # tightened from 40
    rsi_overbought: float = 65.0
    require_vol_confirm: bool = True
    min_volume_ratio: float = 1.2
    stop_loss_pct: float = 3.0
    take_profit_pct: float = 7.0
    trailing_activate_pct: float = 3.5
    trailing_stop_pct: float = 1.5
    max_position_pct: float = 15.0
    min_confidence: float = 0.72

    def __post_init__(self):
        if self.stop_loss_pct <= 0 or self.stop_loss_pct > 50:
            raise ValueError(f"stop_loss_pct must be (0, 50], got {self.stop_loss_pct}")
        if self.min_divergence_bars < 2:
            raise ValueError(f"min_divergence_bars must be >= 2, got {self.min_divergence_bars}")


class MACDDivergence(BaseStrategy):
    """Стратегия V6: MACD Divergence — торговля расходимостью с swing-point detection."""

    NAME = "macd_divergence"

    def __init__(self, config: MACDDivConfig | None = None) -> None:
        super().__init__()
        self._cfg = config or MACDDivConfig()
        self._price_history: dict[str, list[float]] = {}
        self._macd_history: dict[str, list[float]] = {}
        self._last_ts: dict[str, int] = {}
        self._max_price: dict[str, float] = {}
        self._entry_ts: dict[str, int] = {}

    def _update_history(self, sym: str, price: float, macd_val: float, timestamp: int = 0) -> None:
        last_ts = self._last_ts.get(sym, 0)
        if timestamp == last_ts and last_ts != 0:
            ph = self._price_history.get(sym, [])
            mh = self._macd_history.get(sym, [])
            if ph:
                ph[-1] = price
            if mh:
                mh[-1] = macd_val
            return
        self._last_ts[sym] = timestamp
        maxlen = self._cfg.lookback_candles
        ph = self._price_history.setdefault(sym, [])
        mh = self._macd_history.setdefault(sym, [])
        ph.append(price)
        mh.append(macd_val)
        if len(ph) > maxlen:
            self._price_history[sym] = ph[-maxlen:]
        if len(mh) > maxlen:
            self._macd_history[sym] = mh[-maxlen:]

    def _clear_history(self, sym: str) -> None:
        self._price_history.pop(sym, None)
        self._macd_history.pop(sym, None)
        self._last_ts.pop(sym, None)
        self._max_price.pop(sym, None)
        self._entry_ts.pop(sym, None)

    @staticmethod
    def _find_swing_lows(data: list[float], window: int = 3) -> list[tuple[int, float]]:
        swings = []
        for i in range(window, len(data) - window):
            segment = data[i - window: i + window + 1]
            if data[i] == min(segment):
                swings.append((i, data[i]))
        return swings

    @staticmethod
    def _find_swing_highs(data: list[float], window: int = 3) -> list[tuple[int, float]]:
        swings = []
        for i in range(window, len(data) - window):
            segment = data[i - window: i + window + 1]
            if data[i] == max(segment):
                swings.append((i, data[i]))
        return swings

    @staticmethod
    def _nearest_swing(swings: list[tuple[int, float]], target_idx: int, tolerance: int) -> Optional[float]:
        best_val = None
        best_dist = tolerance + 1
        for idx, val in swings:
            dist = abs(idx - target_idx)
            if dist < best_dist:
                best_dist = dist
                best_val = val
        return best_val if best_dist <= tolerance else None

    def _detect_bullish_divergence(self, sym: str) -> bool:
        ph = self._price_history.get(sym, [])
        mh = self._macd_history.get(sym, [])
        cfg = self._cfg
        if len(ph) < cfg.min_divergence_bars + cfg.swing_window * 2:
            return False
        price_lows = self._find_swing_lows(ph, cfg.swing_window)
        macd_lows = self._find_swing_lows(mh, cfg.swing_window)
        if len(price_lows) < 2 or len(macd_lows) < 2:
            return False
        p1_idx, p1_val = price_lows[-2]
        p2_idx, p2_val = price_lows[-1]
        if p2_idx - p1_idx < cfg.min_divergence_bars:
            return False
        if p2_val >= p1_val:
            return False
        macd_at_p1 = self._nearest_swing(macd_lows, p1_idx, cfg.swing_window + 2)
        macd_at_p2 = self._nearest_swing(macd_lows, p2_idx, cfg.swing_window + 2)
        if macd_at_p1 is None or macd_at_p2 is None:
            return False
        return macd_at_p2 > macd_at_p1

    def _detect_bearish_divergence(self, sym: str) -> bool:
        ph = self._price_history.get(sym, [])
        mh = self._macd_history.get(sym, [])
        cfg = self._cfg
        if len(ph) < cfg.min_divergence_bars + cfg.swing_window * 2:
            return False
        price_highs = self._find_swing_highs(ph, cfg.swing_window)
        macd_highs = self._find_swing_highs(mh, cfg.swing_window)
        if len(price_highs) < 2 or len(macd_highs) < 2:
            return False
        p1_idx, p1_val = price_highs[-2]
        p2_idx, p2_val = price_highs[-1]
        if p2_idx - p1_idx < cfg.min_divergence_bars:
            return False
        if p2_val <= p1_val:
            return False
        macd_at_p1 = self._nearest_swing(macd_highs, p1_idx, cfg.swing_window + 2)
        macd_at_p2 = self._nearest_swing(macd_highs, p2_idx, cfg.swing_window + 2)
        if macd_at_p1 is None or macd_at_p2 is None:
            return False
        return macd_at_p2 < macd_at_p1

    def generate_signal(
        self,
        features: FeatureVector,
        has_open_position: bool = False,
        entry_price: float | None = None,
    ) -> Optional[Signal]:
        cfg = self._cfg
        sym = features.symbol
        now_ms = int(time.time() * 1000)
        macd_val = getattr(features, 'macd', None) or features.macd_histogram
        ts = getattr(features, 'timestamp', now_ms)
        self._update_history(sym, features.close, macd_val, ts)

        if has_open_position and entry_price is not None:
            if entry_price <= 0:
                self._clear_history(sym)
                return Signal(
                    timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                    confidence=0.99, strategy_name=self.NAME,
                    reason=f"SAFETY: invalid entry_price={entry_price}",
                    features=features,
                )
            pnl_pct = (features.close - entry_price) / entry_price * 100
            self._max_price[sym] = max(self._max_price.get(sym, entry_price), features.close)

            exit_now, exit_conf, exit_reason = news_should_accelerate_exit(features, pnl_pct)
            if exit_now:
                self._clear_history(sym)
                return Signal(
                    timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                    confidence=exit_conf, strategy_name=self.NAME,
                    reason=exit_reason, features=features,
                )
            if pnl_pct >= cfg.take_profit_pct:
                self._clear_history(sym)
                return Signal(
                    timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                    confidence=0.90, strategy_name=self.NAME,
                    reason=f"MACD Div TP: +{pnl_pct:.1f}%", features=features,
                )
            if pnl_pct <= -cfg.stop_loss_pct:
                self._clear_history(sym)
                return Signal(
                    timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                    confidence=0.95, strategy_name=self.NAME,
                    reason=f"MACD Div SL: {pnl_pct:.1f}%", features=features,
                )
            max_p = self._max_price.get(sym, entry_price)
            max_gain_pct = (max_p - entry_price) / entry_price * 100
            if max_gain_pct >= cfg.trailing_activate_pct:
                drop = (max_p - features.close) / max_p * 100
                if drop >= cfg.trailing_stop_pct:
                    self._clear_history(sym)
                    return Signal(
                        timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                        confidence=0.85, strategy_name=self.NAME,
                        reason=f"MACD Div trailing: drop {drop:.1f}% from max",
                        features=features,
                    )
            if self._detect_bearish_divergence(sym) and features.rsi_14 > cfg.rsi_overbought:
                self._clear_history(sym)
                return Signal(
                    timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                    confidence=0.80, strategy_name=self.NAME,
                    reason=f"MACD bearish divergence + RSI={features.rsi_14:.0f}>{cfg.rsi_overbought}",
                    features=features,
                )
            return None

        if has_open_position:
            return None

        if not self._detect_bullish_divergence(sym):
            return None
        if cfg.require_rsi_confirm and features.rsi_14 >= cfg.rsi_oversold:
            return None
        if cfg.require_vol_confirm and features.volume_ratio < cfg.min_volume_ratio:
            return None

        confidence = 0.55
        if features.rsi_14 < cfg.rsi_oversold:
            confidence += 0.10
        if features.volume_ratio > cfg.min_volume_ratio:
            confidence += 0.08
        mh = self._macd_history.get(sym, [])
        if len(mh) >= 2 and mh[-2] < 0 <= mh[-1]:
            confidence += 0.05

        blocked, block_reason = news_should_block_entry(features)
        if blocked:
            return None
        news_delta, news_reason = news_confidence_adjustment(features, "buy", "divergence")
        confidence += news_delta
        confidence = min(confidence, 0.95)
        if confidence < cfg.min_confidence:
            return None

        sl, tp = news_adjust_sl_tp(features, features.close, cfg.stop_loss_pct, cfg.take_profit_pct)
        reason = f"MACD bullish divergence: RSI={features.rsi_14:.0f}, vol_r={features.volume_ratio:.1f}"
        if news_delta != 0:
            reason += f", {news_reason}"
        self._entry_ts[sym] = now_ms
        self._max_price[sym] = features.close
        return Signal(
            timestamp=now_ms, symbol=sym, direction=Direction.BUY,
            confidence=confidence, strategy_name=self.NAME,
            reason=reason, stop_loss_price=sl, take_profit_price=tp,
            features=features,
        )
