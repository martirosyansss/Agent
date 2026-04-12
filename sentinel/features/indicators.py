"""
Расчёт технических индикаторов из массивов свечей.

Все функции принимают список close/high/low/volume (float) и возвращают
последнее значение индикатора. При недостатке данных — None.
"""

from __future__ import annotations

import math
from typing import Optional

from guards.safe_math import safe_value


# ──────────────────────────────────────────────
# EMA — Exponential Moving Average
# ──────────────────────────────────────────────

def ema(closes: list[float], period: int) -> Optional[float]:
    """Вычислить EMA по массиву close-цен. Возвращает последнее значение."""
    if len(closes) < period:
        return None
    k = 2.0 / (period + 1)
    result = sum(closes[:period]) / period  # SMA seed
    for price in closes[period:]:
        result = price * k + result * (1 - k)
    return safe_value(result)


def ema_series(closes: list[float], period: int) -> list[float]:
    """Полная серия EMA (для MACD / crossover detection)."""
    if len(closes) < period:
        return []
    k = 2.0 / (period + 1)
    result = [sum(closes[:period]) / period]
    for price in closes[period:]:
        result.append(price * k + result[-1] * (1 - k))
    return result


# ──────────────────────────────────────────────
# RSI — Relative Strength Index
# ──────────────────────────────────────────────

def rsi(closes: list[float], period: int = 14) -> Optional[float]:
    """RSI (Wilder's smoothing). Возвращает последнее значение 0-100."""
    if len(closes) < period + 1:
        return None

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0.0 for d in deltas]
    losses = [-d if d < 0 else 0.0 for d in deltas]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return safe_value(100.0 - 100.0 / (1.0 + rs))


# ──────────────────────────────────────────────
# MACD
# ──────────────────────────────────────────────

def macd(
    closes: list[float],
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> Optional[tuple[float, float, float]]:
    """MACD → (macd_line, signal_line, histogram) или None."""
    if len(closes) < slow + signal_period:
        return None

    fast_ema = ema_series(closes, fast)
    slow_ema = ema_series(closes, slow)

    # Выровнять длины (fast_ema длиннее slow_ema)
    offset = len(fast_ema) - len(slow_ema)
    macd_line = [f - s for f, s in zip(fast_ema[offset:], slow_ema)]

    if len(macd_line) < signal_period:
        return None

    signal_line = ema_series(macd_line, signal_period)
    if not signal_line:
        return None

    offset2 = len(macd_line) - len(signal_line)
    histogram = macd_line[-1] - signal_line[-1]

    return (
        safe_value(macd_line[-1]),
        safe_value(signal_line[-1]),
        safe_value(histogram),
    )


# ──────────────────────────────────────────────
# ADX — Average Directional Index
# ──────────────────────────────────────────────

def adx(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
) -> Optional[float]:
    """ADX (0-100). Нужно min period*2 + 1 свечей."""
    n = len(closes)
    if n < period * 2 + 1 or len(highs) != n or len(lows) != n:
        return None

    plus_dm = []
    minus_dm = []
    tr_list = []

    for i in range(1, n):
        h_diff = highs[i] - highs[i - 1]
        l_diff = lows[i - 1] - lows[i]
        plus_dm.append(h_diff if h_diff > l_diff and h_diff > 0 else 0.0)
        minus_dm.append(l_diff if l_diff > h_diff and l_diff > 0 else 0.0)
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        tr_list.append(tr)

    # Wilder smoothing
    def wilder_smooth(data: list[float], p: int) -> list[float]:
        s = [sum(data[:p])]
        for v in data[p:]:
            s.append(s[-1] - s[-1] / p + v)
        return s

    smoothed_tr = wilder_smooth(tr_list, period)
    smoothed_plus = wilder_smooth(plus_dm, period)
    smoothed_minus = wilder_smooth(minus_dm, period)

    dx_list = []
    for i in range(len(smoothed_tr)):
        if smoothed_tr[i] == 0:
            dx_list.append(0.0)
            continue
        plus_di = 100.0 * smoothed_plus[i] / smoothed_tr[i]
        minus_di = 100.0 * smoothed_minus[i] / smoothed_tr[i]
        di_sum = plus_di + minus_di
        if di_sum == 0:
            dx_list.append(0.0)
        else:
            dx_list.append(100.0 * abs(plus_di - minus_di) / di_sum)

    if len(dx_list) < period:
        return None

    adx_val = sum(dx_list[:period]) / period
    for dx in dx_list[period:]:
        adx_val = (adx_val * (period - 1) + dx) / period

    return safe_value(adx_val)


# ──────────────────────────────────────────────
# Bollinger Bands
# ──────────────────────────────────────────────

def bollinger_bands(
    closes: list[float],
    period: int = 20,
    std_dev: float = 2.0,
) -> Optional[tuple[float, float, float, float]]:
    """(upper, middle, lower, bandwidth) или None."""
    if len(closes) < period:
        return None

    window = closes[-period:]
    middle = sum(window) / period
    variance = sum((x - middle) ** 2 for x in window) / period
    sd = math.sqrt(variance)
    upper = middle + std_dev * sd
    lower = middle - std_dev * sd
    bandwidth = (upper - lower) / middle if middle != 0 else 0.0

    return (
        safe_value(upper),
        safe_value(middle),
        safe_value(lower),
        safe_value(bandwidth),
    )


# ──────────────────────────────────────────────
# ATR — Average True Range
# ──────────────────────────────────────────────

def atr(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
) -> Optional[float]:
    """ATR (Wilder smoothing)."""
    n = len(closes)
    if n < period + 1 or len(highs) != n or len(lows) != n:
        return None

    tr_list = []
    for i in range(1, n):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        tr_list.append(tr)

    atr_val = sum(tr_list[:period]) / period
    for tr in tr_list[period:]:
        atr_val = (atr_val * (period - 1) + tr) / period

    return safe_value(atr_val)


# ──────────────────────────────────────────────
# Volume
# ──────────────────────────────────────────────

def volume_sma(volumes: list[float], period: int = 20) -> Optional[float]:
    if len(volumes) < period:
        return None
    return sum(volumes[-period:]) / period


def volume_ratio(volumes: list[float], period: int = 20) -> Optional[float]:
    """Текущий объём / средний за period."""
    avg = volume_sma(volumes, period)
    if avg is None or avg == 0:
        return None
    return safe_value(volumes[-1] / avg)


def obv(closes: list[float], volumes: list[float]) -> Optional[float]:
    """On-Balance Volume — последнее значение."""
    if len(closes) < 2 or len(volumes) != len(closes):
        return None
    result = 0.0
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            result += volumes[i]
        elif closes[i] < closes[i - 1]:
            result -= volumes[i]
    return result


# ──────────────────────────────────────────────
# Stochastic RSI
# ──────────────────────────────────────────────

def stochastic_rsi(closes: list[float], rsi_period: int = 14, stoch_period: int = 14) -> Optional[float]:
    """Stoch RSI (0-100)."""
    if len(closes) < rsi_period + stoch_period + 1:
        return None

    # Вычислить серию RSI
    rsi_values = []
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0.0 for d in deltas]
    losses = [-d if d < 0 else 0.0 for d in deltas]

    avg_gain = sum(gains[:rsi_period]) / rsi_period
    avg_loss = sum(losses[:rsi_period]) / rsi_period

    for i in range(rsi_period, len(gains)):
        avg_gain = (avg_gain * (rsi_period - 1) + gains[i]) / rsi_period
        avg_loss = (avg_loss * (rsi_period - 1) + losses[i]) / rsi_period
        if avg_loss == 0:
            rsi_values.append(100.0)
        else:
            rsi_values.append(100.0 - 100.0 / (1.0 + avg_gain / avg_loss))

    if len(rsi_values) < stoch_period:
        return None

    window = rsi_values[-stoch_period:]
    min_rsi = min(window)
    max_rsi = max(window)
    if max_rsi == min_rsi:
        return 50.0
    return safe_value(100.0 * (rsi_values[-1] - min_rsi) / (max_rsi - min_rsi))


# ──────────────────────────────────────────────
# Momentum / Price Change
# ──────────────────────────────────────────────

def price_change_pct(closes: list[float], lookback: int = 1) -> Optional[float]:
    """Процентное изменение цены за lookback свечей."""
    if len(closes) < lookback + 1:
        return None
    old = closes[-(lookback + 1)]
    if old == 0:
        return None
    return safe_value((closes[-1] - old) / old * 100.0)


def momentum(closes: list[float], period: int = 10) -> Optional[float]:
    """Rate of Change за period свечей (в %)."""
    return price_change_pct(closes, period)
