"""
Валидация входящих рыночных данных.

Проверки: price > 0, volume > 0, timestamp не из будущего,
цена в допустимом диапазоне (PRICE_RANGES из absolute_limits).
Sequence validation: gaps, duplicates, outliers.
"""

from __future__ import annotations

import time
from typing import Optional

from loguru import logger

from core.absolute_limits import PRICE_RANGES
from core.models import Candle, MarketTrade

log = logger.bind(module="validator")

# Допуск: timestamp может быть на 5 секунд из будущего (clock skew)
_MAX_FUTURE_MS = 5_000

# Интервалы свечей в миллисекундах
_INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


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

    if not (c.low <= c.close <= c.high):
        log.warning("Candle rejected: close outside high/low ({} {} c={} h={} l={})",
                    c.symbol, c.interval, c.close, c.high, c.low)
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


# ──────────────────────────────────────────────
# Sequence validation (gaps, duplicates, outliers)
# ──────────────────────────────────────────────

def validate_candle_sequence(candles: list[Candle]) -> dict:
    """Проверить последовательность свечей на gaps / duplicates / out-of-order.

    Returns dict: {"valid": bool, "gaps": int, "duplicates": int, "out_of_order": int}
    """
    if len(candles) < 2:
        return {"valid": True, "gaps": 0, "duplicates": 0, "out_of_order": 0}

    gaps = 0
    duplicates = 0
    out_of_order = 0

    for i in range(1, len(candles)):
        prev = candles[i - 1]
        curr = candles[i]

        if curr.timestamp == prev.timestamp:
            duplicates += 1
        elif curr.timestamp < prev.timestamp:
            out_of_order += 1
        else:
            expected_gap = _INTERVAL_MS.get(curr.interval, 3_600_000)
            actual_gap = curr.timestamp - prev.timestamp
            if actual_gap > expected_gap * 1.5:
                gaps += 1

    total_issues = gaps + duplicates + out_of_order
    if total_issues > 0:
        log.warning(
            "Candle sequence issues: {} gaps, {} duplicates, {} out_of_order ({} candles)",
            gaps, duplicates, out_of_order, len(candles),
        )

    return {
        "valid": total_issues == 0,
        "gaps": gaps,
        "duplicates": duplicates,
        "out_of_order": out_of_order,
    }


def detect_price_outliers(
    candles: list[Candle],
    z_threshold: float = 4.0,
) -> list[int]:
    """Обнаружить ценовые выбросы по z-score returns.

    Returns: list индексов подозрительных свечей.
    """
    if len(candles) < 20:
        return []

    closes = [c.close for c in candles]
    returns = [(closes[i] - closes[i - 1]) / closes[i - 1]
               for i in range(1, len(closes)) if closes[i - 1] != 0]

    if len(returns) < 10:
        return []

    mean_r = sum(returns) / len(returns)
    variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
    std_r = variance ** 0.5

    if std_r == 0:
        return []

    outlier_indices = []
    for i, r in enumerate(returns):
        z_score = abs(r - mean_r) / std_r
        if z_score > z_threshold:
            outlier_indices.append(i + 1)
            log.warning(
                "Price outlier at index {}: return={:.4f}% z={:.1f} ({})",
                i + 1, r * 100, z_score, candles[i + 1].symbol,
            )

    return outlier_indices
