"""
Market Regime Detection with Hysteresis.

Определяет текущий режим рынка каждые 4 часа по 4h свечам:
  - trending_up:   EMA9 > EMA21 > EMA50 + ADX > 25
  - trending_down:  EMA9 < EMA21 < EMA50 + ADX > 25
  - sideways:       ADX < 20 + цена внутри Bollinger Bands
  - volatile:       ATR > 2× среднего
  - unknown:        не определено

Гистерезис: для смены режима нужно 3 подряд подтверждения нового режима.
"""

from __future__ import annotations

import time
from typing import Optional

from core.models import FeatureVector, MarketRegime, MarketRegimeType


# Module-level state for hysteresis
_prev_regime: MarketRegimeType = MarketRegimeType.UNKNOWN
_pending_regime: MarketRegimeType | None = None
_pending_count: int = 0
_HYSTERESIS_CONFIRMS: int = 2  # require 2 consecutive signals to switch


def reset_hysteresis() -> None:
    """Reset hysteresis state (for testing or regime recalibration)."""
    global _prev_regime, _pending_regime, _pending_count
    _prev_regime = MarketRegimeType.UNKNOWN
    _pending_regime = None
    _pending_count = 0


def detect_regime(
    features: FeatureVector,
    adx_trending: float = 25.0,
    adx_sideways: float = 20.0,
) -> MarketRegime:
    """Определить текущий рыночный режим по FeatureVector с гистерезисом."""
    global _prev_regime, _pending_regime, _pending_count

    adx = features.adx
    atr = features.atr
    bb_mid = features.bb_middle
    atr_ratio = (atr / bb_mid) if bb_mid > 0 else 0.0

    # Raw regime detection — check trend BEFORE volatile to avoid masking
    if features.ema_9 > features.ema_21 > features.ema_50 > 0 and adx > adx_trending:
        raw_regime = MarketRegimeType.TRENDING_UP
    elif features.ema_9 < features.ema_21 < features.ema_50 and adx > adx_trending:
        raw_regime = MarketRegimeType.TRENDING_DOWN
    elif atr_ratio > 0.04:
        raw_regime = MarketRegimeType.VOLATILE
    elif adx < adx_sideways and features.bb_lower < features.close < features.bb_upper:
        raw_regime = MarketRegimeType.SIDEWAYS
    else:
        raw_regime = MarketRegimeType.UNKNOWN

    # Hysteresis logic: require N consecutive confirmations to change regime
    if raw_regime == _prev_regime:
        # Still in current regime — reset pending
        _pending_regime = None
        _pending_count = 0
        final_regime = _prev_regime
    elif raw_regime == _pending_regime:
        # Same new regime again — increment counter
        _pending_count += 1
        if _pending_count >= _HYSTERESIS_CONFIRMS:
            # Confirmed regime change
            _prev_regime = raw_regime
            _pending_regime = None
            _pending_count = 0
            final_regime = raw_regime
        else:
            final_regime = _prev_regime  # stay in old regime
    else:
        # Different new regime — start fresh count
        _pending_regime = raw_regime
        _pending_count = 1
        final_regime = _prev_regime  # stay in old regime

    return MarketRegime(
        regime=final_regime,
        adx=adx,
        atr_ratio=atr_ratio,
        determined_at=int(time.time() * 1000),
    )
