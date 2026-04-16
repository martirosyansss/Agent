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


# Per-symbol hysteresis state (prevents cross-symbol regime confusion)
_HYSTERESIS_CONFIRMS: int = 2  # require 2 consecutive signals to switch

_prev_regime: dict[str, MarketRegimeType] = {}
_pending_regime: dict[str, MarketRegimeType | None] = {}
_pending_count: dict[str, int] = {}


def reset_hysteresis(symbol: str | None = None) -> None:
    """Reset hysteresis state (for testing or regime recalibration).

    Args:
        symbol: Reset for specific symbol. None resets all symbols.
    """
    if symbol is None:
        _prev_regime.clear()
        _pending_regime.clear()
        _pending_count.clear()
    else:
        _prev_regime.pop(symbol, None)
        _pending_regime.pop(symbol, None)
        _pending_count.pop(symbol, None)


def detect_regime(
    features: FeatureVector,
    adx_trending: float = 25.0,
    adx_sideways: float = 20.0,
) -> MarketRegime:
    """Определить текущий рыночный режим по FeatureVector с per-symbol гистерезисом."""
    sym = features.symbol

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

    prev = _prev_regime.get(sym, MarketRegimeType.UNKNOWN)
    pending = _pending_regime.get(sym)
    count = _pending_count.get(sym, 0)

    # Hysteresis logic: require N consecutive confirmations to change regime
    if raw_regime == prev:
        # Still in current regime — reset pending
        _pending_regime[sym] = None
        _pending_count[sym] = 0
        final_regime = prev
    elif raw_regime == pending:
        # Same new regime again — increment counter
        count += 1
        _pending_count[sym] = count
        if count >= _HYSTERESIS_CONFIRMS:
            # Confirmed regime change
            _prev_regime[sym] = raw_regime
            _pending_regime[sym] = None
            _pending_count[sym] = 0
            final_regime = raw_regime
        else:
            final_regime = prev  # stay in old regime
    else:
        # Different new regime — start fresh count
        _pending_regime[sym] = raw_regime
        _pending_count[sym] = 1
        final_regime = prev  # stay in old regime

    return MarketRegime(
        regime=final_regime,
        adx=adx,
        atr_ratio=atr_ratio,
        determined_at=int(time.time() * 1000),
    )
