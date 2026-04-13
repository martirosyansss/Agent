"""
Dynamic Position Sizing — Kelly Criterion + ATR-based volatility scaling.

Replaces fixed max_position_pct with adaptive sizing based on:
1. Kelly fraction from strategy win-rate / avg-win / avg-loss
2. ATR-based volatility scaling (smaller in volatile markets)
3. Regime-aware dampening
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class SizingInput:
    """Inputs for the position sizer."""
    balance: float
    price: float
    atr: float                          # current ATR value
    win_rate: float = 0.5               # strategy win rate (0-1)
    avg_win_pct: float = 3.0            # average winning trade %
    avg_loss_pct: float = 2.0           # average losing trade %
    regime_adx: float = 25.0            # current ADX
    max_position_pct: float = 20.0      # hard cap from config
    max_order_usd: float = 100.0        # hard cap from config


@dataclass(slots=True)
class SizingResult:
    """Output of position sizer."""
    quantity: float
    budget_usd: float
    budget_pct: float
    method: str                          # 'kelly_atr', 'fixed', 'minimum'
    kelly_fraction: float = 0.0
    volatility_factor: float = 1.0


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Half-Kelly fraction (conservative) clamped to [0, 0.25].

    Formula: f = (p * b - q) / b  where p=win_rate, q=1-p, b=avg_win/avg_loss
    Then halved for safety.
    """
    if avg_loss <= 0 or avg_win <= 0 or win_rate <= 0:
        return 0.0
    b = avg_win / avg_loss
    q = 1.0 - win_rate
    f = (win_rate * b - q) / b
    # Half-Kelly for safety
    f *= 0.5
    return max(0.0, min(f, 0.25))


def volatility_factor(atr: float, price: float, base_atr_pct: float = 1.5) -> float:
    """Scale position inversely with volatility.

    If ATR% > base_atr_pct, shrink position. If lower, allow slightly larger.
    Returns factor in [0.3, 1.5].
    """
    if price <= 0 or atr <= 0:
        return 1.0
    atr_pct = (atr / price) * 100
    if atr_pct <= 0:
        return 1.0
    factor = base_atr_pct / atr_pct
    return max(0.3, min(factor, 1.5))


def regime_dampener(adx: float) -> float:
    """Reduce size in low-conviction (low ADX) environments.

    ADX >= 30: full size (1.0)
    ADX 20-30: linear scale 0.6-1.0
    ADX < 20: 0.5
    """
    if adx >= 30:
        return 1.0
    if adx >= 20:
        return 0.6 + (adx - 20) / 10 * 0.4
    return 0.5


def calculate_position_size(inp: SizingInput) -> SizingResult:
    """Calculate position size using Kelly + ATR + regime.

    Falls back to fixed sizing if insufficient data (win_rate near 50%).
    """
    if inp.balance <= 0 or inp.price <= 0:
        return SizingResult(
            quantity=0.0, budget_usd=0.0, budget_pct=0.0, method="minimum"
        )

    # 1. Kelly fraction
    kf = kelly_fraction(inp.win_rate, inp.avg_win_pct, inp.avg_loss_pct)

    # 2. Volatility factor
    vf = volatility_factor(inp.atr, inp.price)

    # 3. Regime dampener
    rd = regime_dampener(inp.regime_adx)

    # Combine: kelly-based pct, scaled by vol and regime
    if kf > 0.01:
        base_pct = kf * 100  # kelly as percentage
        adjusted_pct = base_pct * vf * rd
        method = "kelly_atr"
    else:
        # Insufficient edge, use minimum fixed size
        adjusted_pct = 5.0 * vf * rd
        method = "fixed"

    # Clamp to hard limits
    budget_pct = max(2.0, min(adjusted_pct, inp.max_position_pct))
    budget_usd = inp.balance * budget_pct / 100
    budget_usd = min(budget_usd, inp.max_order_usd)
    budget_pct = budget_usd / inp.balance * 100

    quantity = budget_usd / inp.price if inp.price > 0 else 0.0

    return SizingResult(
        quantity=quantity,
        budget_usd=budget_usd,
        budget_pct=budget_pct,
        method=method,
        kelly_fraction=kf,
        volatility_factor=vf,
    )
