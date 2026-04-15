"""
Market Regime Detection with Hysteresis.

Определяет текущий режим рынка каждые 4 часа по 4h свечам:
  - trending_up:   EMA9 > EMA21 AND close > EMA50 + ADX > 25
  - trending_down:  EMA9 < EMA21 AND close < EMA50 + ADX > 25
  - sideways:       ADX < 20 + цена внутри Bollinger Bands
  - volatile:       ATR/close > 4%
  - transitioning:  между режимами (EMA partially aligned, 20 < ADX < 25)
  - unknown:        не определено

Гистерезис: для смены режима нужно 2 подряд подтверждения нового режима.
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
    close = features.close
    # Use ATR/close instead of ATR/bb_middle — more stable metric
    atr_ratio = (atr / close) if close > 0 else 0.0

    # Raw regime detection — relaxed EMA alignment for trending
    # Trending: EMA9 vs EMA21 determines direction, close vs EMA50 confirms structure
    ema_bullish = features.ema_9 > features.ema_21 and close > features.ema_50 > 0
    ema_bearish = features.ema_9 < features.ema_21 and close < features.ema_50 and features.ema_50 > 0

    if ema_bullish and adx > adx_trending:
        raw_regime = MarketRegimeType.TRENDING_UP
    elif ema_bearish and adx > adx_trending:
        raw_regime = MarketRegimeType.TRENDING_DOWN
    elif atr_ratio > 0.04:
        raw_regime = MarketRegimeType.VOLATILE
    elif adx < adx_sideways and features.bb_lower < close < features.bb_upper:
        raw_regime = MarketRegimeType.SIDEWAYS
    elif adx_sideways <= adx <= adx_trending:
        # TRANSITIONING: ADX between sideways and trending thresholds
        # This is the most dangerous zone — trend is forming or fading
        raw_regime = MarketRegimeType.TRANSITIONING
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
