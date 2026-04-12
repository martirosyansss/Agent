"""
Стратегия V3: Mean Reversion (RSI Extreme + Bollinger Bands).

Логика:
  BUY:  RSI(4h) < 25 + цена < lower BB(4h) + Volume > 1.5× avg + EMA50(1d) растёт
  SELL: RSI(4h) > 75 + цена > upper BB ИЛИ цена вернулась к EMA21 ИЛИ SL/TP

Confidence: base=0.65 + RSI extreme(+0.10) + volume(+0.08) + trend(+0.07)
Режим рынка: any (ловит экстремумы)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from core.models import Direction, FeatureVector, Signal
from strategy.base_strategy import BaseStrategy


@dataclass
class MeanRevConfig:
    rsi_oversold: float = 25.0
    rsi_overbought: float = 75.0
    stop_loss_pct: float = 4.0
    take_profit_pct: float = 6.0
    min_volume_ratio: float = 1.5
    min_confidence: float = 0.80
    max_position_pct: float = 15.0


class MeanReversion(BaseStrategy):
    """Стратегия V3: Mean Reversion — торговля от экстремумов RSI + BB."""

    NAME = "mean_reversion"

    def __init__(self, config: MeanRevConfig | None = None) -> None:
        self._cfg = config or MeanRevConfig()

    def generate_signal(
        self,
        features: FeatureVector,
        has_open_position: bool = False,
        entry_price: float | None = None,
    ) -> Optional[Signal]:
        cfg = self._cfg
        sym = features.symbol
        now_ms = int(time.time() * 1000)

        # ── SELL (если есть позиция) ──
        if has_open_position and entry_price is not None:
            pnl_pct = (features.close - entry_price) / entry_price * 100

            # Take profit
            if pnl_pct >= cfg.take_profit_pct:
                return Signal(
                    timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                    confidence=0.90, strategy_name=self.NAME,
                    reason=f"MeanRev TP: +{pnl_pct:.1f}% >= {cfg.take_profit_pct}%",
                )
            # Stop loss
            if pnl_pct <= -cfg.stop_loss_pct:
                return Signal(
                    timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                    confidence=0.95, strategy_name=self.NAME,
                    reason=f"MeanRev SL: {pnl_pct:.1f}% <= -{cfg.stop_loss_pct}%",
                )
            # RSI overbought exit
            if features.rsi_14 > cfg.rsi_overbought:
                return Signal(
                    timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                    confidence=0.80, strategy_name=self.NAME,
                    reason=f"MeanRev exit: RSI {features.rsi_14:.1f} > {cfg.rsi_overbought}",
                )
            # Price reverted to EMA21
            if features.close >= features.ema_21 > 0:
                return Signal(
                    timestamp=now_ms, symbol=sym, direction=Direction.SELL,
                    confidence=0.75, strategy_name=self.NAME,
                    reason=f"MeanRev revert to EMA21: {features.close:.2f} >= {features.ema_21:.2f}",
                )
            return None

        # ── BUY (нет позиции) ──
        if has_open_position:
            return None

        # RSI oversold
        if features.rsi_14 >= cfg.rsi_oversold:
            return None
        # Price below lower BB
        if features.bb_lower > 0 and features.close >= features.bb_lower:
            return None
        # Volume confirmation
        if features.volume_ratio < cfg.min_volume_ratio:
            return None

        # Confidence scoring
        confidence = 0.65
        if features.rsi_14 < 20:
            confidence += 0.10
        if features.volume_ratio > 2.0:
            confidence += 0.08
        if features.ema_50 > 0 and features.close > features.ema_50 * 0.95:
            confidence += 0.07
        confidence = min(confidence, 0.95)

        if confidence < cfg.min_confidence:
            return None

        sl = features.close * (1 - cfg.stop_loss_pct / 100)
        tp = features.close * (1 + cfg.take_profit_pct / 100)

        return Signal(
            timestamp=now_ms, symbol=sym, direction=Direction.BUY,
            confidence=confidence, strategy_name=self.NAME,
            reason=f"MeanRev BUY: RSI={features.rsi_14:.1f}, price<lowerBB, vol_ratio={features.volume_ratio:.1f}",
            stop_loss_price=sl, take_profit_price=tp,
        )
