"""
Стратегия V6: MACD Divergence.

Логика:
  Bullish divergence BUY: цена lower low + MACD higher low + RSI < 35 + vol confirm
  Bearish divergence SELL: цена higher high + MACD lower high + RSI > 65

Confidence: base=0.55 + RSI confirm(+0.10) + vol>1.3x(+0.08) + divergence>15 свечей(+0.07) + MACD crossed 0(+0.05)
Режим рынка: trending_down → trending_up (ловит развороты)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from core.models import Direction, FeatureVector, Signal
from strategy.base_strategy import BaseStrategy


@dataclass
class MACDDivConfig:
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal_period: int = 9
    lookback_candles: int = 30
    min_divergence_bars: int = 5
    require_rsi_confirm: bool = True
    rsi_oversold: float = 35.0
    rsi_overbought: float = 65.0
    require_vol_confirm: bool = True
    min_volume_ratio: float = 1.3
    stop_loss_pct: float = 3.5
    take_profit_pct: float = 7.0
    max_position_pct: float = 15.0
    min_confidence: float = 0.72

    def __post_init__(self):
        if self.stop_loss_pct <= 0 or self.stop_loss_pct > 50:
            raise ValueError(f"stop_loss_pct must be (0, 50], got {self.stop_loss_pct}")
        if self.min_divergence_bars < 2:
            raise ValueError(f"min_divergence_bars must be >= 2, got {self.min_divergence_bars}")


class MACDDivergence(BaseStrategy):
    """Стратегия V6: MACD Divergence — торговля расходимостью."""

    NAME = "macd_divergence"

    def __init__(self, config: MACDDivConfig | None = None) -> None:
        self._cfg = config or MACDDivConfig()
        # History buffers for divergence detection
        self._price_history: dict[str, list[float]] = {}
        self._macd_history: dict[str, list[float]] = {}

    def _update_history(self, sym: str, price: float, macd_val: float) -> None:
        """Обновить историю для детекции дивергенции."""
        maxlen = self._cfg.lookback_candles
        ph = self._price_history.setdefault(sym, [])
        mh = self._macd_history.setdefault(sym, [])
        ph.append(price)
        mh.append(macd_val)
        if len(ph) > maxlen:
            self._price_history[sym] = ph[-maxlen:]
        if len(mh) > maxlen:
            self._macd_history[sym] = mh[-maxlen:]

    def _detect_bullish_divergence(self, sym: str) -> bool:
        """Бычья дивергенция: цена lower low, MACD higher low."""
        ph = self._price_history.get(sym, [])
        mh = self._macd_history.get(sym, [])
        min_bars = self._cfg.min_divergence_bars
        if len(ph) < min_bars or len(mh) < min_bars:
            return False

        recent = min_bars
        # Price made lower low
        price_new_low = ph[-1] < min(ph[-recent:-1])
        # MACD made higher low
        macd_higher_low = mh[-1] > min(mh[-recent:-1])

        return price_new_low and macd_higher_low

    def _detect_bearish_divergence(self, sym: str) -> bool:
        """Медвежья дивергенция: цена higher high, MACD lower high."""
        ph = self._price_history.get(sym, [])
        mh = self._macd_history.get(sym, [])
        min_bars = self._cfg.min_divergence_bars
        if len(ph) < min_bars or len(mh) < min_bars:
            return False

        recent = min_bars
        price_new_high = ph[-1] > max(ph[-recent:-1])
        macd_lower_high = mh[-1] < max(mh[-recent:-1])

        return price_new_high and macd_lower_high

    def generate_signal(
        self,
        features: FeatureVector,
        has_open_position: bool = False,
        entry_price: float | None = None,
    ) -> Optional[Signal]:
        cfg = self._cfg
        sym = features.symbol
        now_ms = int(time.time() * 1000)

        # Update history
        self._update_history(sym, features.close, features.macd_histogram)

        # ── SELL (если есть позиция) ──
        if has_open_position and entry_price is not None:
            if entry_price <= 0:
                return Signal(
                    timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                    confidence=0.99, strategy_name=self.NAME,
                    reason=f"SAFETY: invalid entry_price={entry_price}",
                )
            pnl_pct = (features.close - entry_price) / entry_price * 100

            if pnl_pct >= cfg.take_profit_pct:
                return Signal(
                    timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                    confidence=0.90, strategy_name=self.NAME,
                    reason=f"MACD Div TP: +{pnl_pct:.1f}%",
                )
            if pnl_pct <= -cfg.stop_loss_pct:
                return Signal(
                    timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                    confidence=0.95, strategy_name=self.NAME,
                    reason=f"MACD Div SL: {pnl_pct:.1f}%",
                )
            # Bearish divergence exit
            if self._detect_bearish_divergence(sym) and features.rsi_14 > cfg.rsi_overbought:
                return Signal(
                    timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                    confidence=0.80, strategy_name=self.NAME,
                    reason=f"MACD bearish divergence + RSI={features.rsi_14:.0f}>{cfg.rsi_overbought}",
                )
            return None

        # ── BUY ──
        if has_open_position:
            return None

        # Bullish divergence
        if not self._detect_bullish_divergence(sym):
            return None

        # RSI confirmation
        if cfg.require_rsi_confirm and features.rsi_14 >= cfg.rsi_oversold:
            return None

        # Volume confirmation
        if cfg.require_vol_confirm and features.volume_ratio < cfg.min_volume_ratio:
            return None

        # Confidence
        confidence = 0.55
        if features.rsi_14 < cfg.rsi_oversold:
            confidence += 0.10
        if features.volume_ratio > cfg.min_volume_ratio:
            confidence += 0.08
        # MACD crossed zero line
        mh = self._macd_history.get(sym, [])
        if len(mh) >= 2 and mh[-2] < 0 <= mh[-1]:
            confidence += 0.05

        # News sentiment boost/penalty (±0.08)
        if features.news_sentiment > 0.3:
            confidence += 0.08
        elif features.news_sentiment > 0.15:
            confidence += 0.04
        elif features.news_sentiment < -0.3:
            confidence -= 0.08
        elif features.news_sentiment < -0.15:
            confidence -= 0.04

        confidence = min(confidence, 0.95)

        if confidence < cfg.min_confidence:
            return None

        sl = features.close * (1 - cfg.stop_loss_pct / 100)
        tp = features.close * (1 + cfg.take_profit_pct / 100)

        reason = f"MACD bullish divergence: RSI={features.rsi_14:.0f}, vol_r={features.volume_ratio:.1f}"
        if features.news_sentiment != 0.0:
            reason += f", sentiment={features.news_sentiment:+.2f}"

        return Signal(
            timestamp=now_ms, symbol=sym, direction=Direction.BUY,
            confidence=confidence, strategy_name=self.NAME,
            reason=reason,
            stop_loss_price=sl, take_profit_price=tp,
        )
