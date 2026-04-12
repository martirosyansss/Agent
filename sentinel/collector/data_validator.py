"""
Валидация входящих рыночных данных.

Проверки: price > 0, volume > 0, timestamp не из будущего,
цена в допустимом диапазоне (PRICE_RANGES из absolute_limits).
"""

from __future__ import annotations

import time

from loguru import logger

from core.absolute_limits import PRICE_RANGES
from core.models import Candle, MarketTrade

log = logger.bind(module="validator")

# Допуск: timestamp может быть на 5 секунд из будущего (clock skew)
_MAX_FUTURE_MS = 5_000


def validate_trade(t: MarketTrade) -> bool:
    """Проверить сырую рыночную сделку. True = валидна."""
    now_ms = int(time.time() * 1000)

    if t.price <= 0:
        log.warning("Trade rejected: price <= 0 ({} {})", t.symbol, t.price)
        return False

    if t.quantity <= 0:
        log.warning("Trade rejected: quantity <= 0 ({} {})", t.symbol, t.quantity)
        return False

    if t.timestamp > now_ms + _MAX_FUTURE_MS:
        log.warning("Trade rejected: timestamp from future ({} {}ms ahead)", t.symbol, t.timestamp - now_ms)
        return False

    price_range = PRICE_RANGES.get(t.symbol)
    if price_range:
        lo, hi = price_range
        if not (lo <= t.price <= hi):
            log.warning(
                "Trade rejected: price {} out of range [{}, {}] for {}",
                t.price, lo, hi, t.symbol,
            )
            return False

    return True


def validate_candle(c: Candle) -> bool:
    """Проверить свечу. True = валидна."""
    now_ms = int(time.time() * 1000)

    if c.close <= 0 or c.open <= 0 or c.high <= 0 or c.low <= 0:
        log.warning("Candle rejected: OHLC <= 0 ({} {})", c.symbol, c.interval)
        return False

    if c.high < c.low:
        log.warning("Candle rejected: high < low ({} {} h={} l={})", c.symbol, c.interval, c.high, c.low)
        return False

    if c.volume < 0:
        log.warning("Candle rejected: negative volume ({} {})", c.symbol, c.interval)
        return False

    if c.timestamp > now_ms + _MAX_FUTURE_MS:
        log.warning("Candle rejected: timestamp from future ({} {})", c.symbol, c.interval)
        return False

    price_range = PRICE_RANGES.get(c.symbol)
    if price_range:
        lo, hi = price_range
        if not (lo <= c.close <= hi):
            log.warning(
                "Candle rejected: close {} out of range [{}, {}] for {}",
                c.close, lo, hi, c.symbol,
            )
            return False

    return True
