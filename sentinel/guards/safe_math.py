"""
Безопасные математические операции — защита от NaN, Infinity.

Каждый финансовый расчёт должен проходить через safe_value() или safe_div().
"""

from __future__ import annotations

import math


def safe_value(v: float, default: float = 0.0) -> float:
    """Вернуть default если v is NaN/Inf."""
    if v is None or math.isnan(v) or math.isinf(v):
        return default
    return v


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    """Безопасное деление: a / b. При b==0 или NaN — default."""
    a = safe_value(a)
    b = safe_value(b)
    if b == 0.0:
        return default
    result = a / b
    return safe_value(result, default)


def safe_pct(part: float, total: float, default: float = 0.0) -> float:
    """Безопасный процент: (part / total) * 100."""
    return safe_div(part, total, default) * 100.0


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Ограничить значение в диапазоне [min_val, max_val]."""
    return max(min_val, min(safe_value(value, min_val), max_val))
