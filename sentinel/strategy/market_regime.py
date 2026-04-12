"""
Market Regime Detection.

Определяет текущий режим рынка каждые 4 часа по 4h свечам:
  - trending_up:   EMA9 > EMA21 > EMA50 + ADX > 25
  - trending_down:  EMA9 < EMA21 < EMA50 + ADX > 25
  - sideways:       ADX < 20 + цена внутри Bollinger Bands
  - volatile:       ATR > 2× среднего
  - unknown:        не определено
"""

from __future__ import annotations

import time
from typing import Optional

from core.models import FeatureVector, MarketRegime, MarketRegimeType


def detect_regime(
    features: FeatureVector,
    adx_trending: float = 25.0,
    adx_sideways: float = 20.0,
) -> MarketRegime:
    """Определить текущий рыночный режим по FeatureVector."""
    adx = features.adx
    atr = features.atr
    bb_mid = features.bb_middle
    atr_ratio = (atr / bb_mid) if bb_mid > 0 else 0.0

    # volatile: ATR > 2× средней
    if atr_ratio > 0.04:  # ~2% ATR relative to price is volatile
        regime = MarketRegimeType.VOLATILE
    # trending_up
    elif features.ema_9 > features.ema_21 > features.ema_50 > 0 and adx > adx_trending:
        regime = MarketRegimeType.TRENDING_UP
    # trending_down
    elif features.ema_9 < features.ema_21 < features.ema_50 and adx > adx_trending:
        regime = MarketRegimeType.TRENDING_DOWN
    # sideways
    elif adx < adx_sideways and features.bb_lower < features.close < features.bb_upper:
        regime = MarketRegimeType.SIDEWAYS
    else:
        regime = MarketRegimeType.UNKNOWN

    return MarketRegime(
        regime=regime,
        adx=adx,
        atr_ratio=atr_ratio,
        determined_at=int(time.time() * 1000),
    )
