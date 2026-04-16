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
    win_rate: float = 0.5               # strategy win rate (0-1), gross
    avg_win_pct: float = 3.0            # average winning trade % (gross)
    avg_loss_pct: float = 2.0           # average losing trade % (gross)
    sample_size: int = 0                # number of trades backing the above
                                        # stats — drives shrinkage strength
    round_trip_cost_pct: float = 0.20   # entry+exit commission+slippage budget
    regime_adx: float = 25.0            # current ADX
    max_position_pct: float = 20.0      # hard cap from config
    max_order_usd: float = 100.0        # hard cap from config
    symbol: str = ""                    # trading symbol
    open_symbols: list[str] | None = None  # already open positions' symbols
    consecutive_losses: int = 0         # current loss streak
    stop_loss_pct: float = 0.0          # actual SL% for risk-based sizing
    max_risk_per_trade_pct: float = 3.0 # max % of portfolio to risk per trade


@dataclass(slots=True)
class SizingResult:
    """Output of position sizer."""
    quantity: float
    budget_usd: float
    budget_pct: float
    method: str                          # 'kelly_atr', 'fixed', 'minimum'
    kelly_fraction: float = 0.0
    volatility_factor: float = 1.0


def shrinkage_win_rate(
    observed_wr: float,
    sample_size: int,
    prior_wr: float = 0.5,
    prior_weight: int = 30,
) -> float:
    """Beta-Binomial posterior-mean shrinkage of observed WR toward a prior.

    Formula: ŵ = (n·p̂ + k·w₀) / (n + k)

    Motivation: MLE of a Bernoulli parameter from small samples has variance
    p(1-p)/n — 10 trades at 70% WR have ~14% stderr. Raw plug-in of that
    estimate into Kelly systematically over-sizes. Shrinkage toward 0.5
    with k=30 pseudo-trades is equivalent to a James-Stein-type estimator
    and dominates the MLE under squared-error loss when the true rate is
    near neutral.

    Convergence: ŵ → p̂ as n → ∞; ŵ → w₀ as n → 0.
    """
    if sample_size <= 0:
        return prior_wr
    return (sample_size * observed_wr + prior_weight * prior_wr) / (
        sample_size + prior_weight
    )


def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    sample_size: int = 0,
    round_trip_cost_pct: float = 0.0,
    prior_weight: int = 30,
) -> float:
    """Half-Kelly fraction, net of transaction costs, with WR shrinkage.

    Core formula: f* = (p·b - q) / b  where p=win_rate, q=1-p, b=avg_win/avg_loss
    Half-Kelly: f = 0.5·f* (Thorp-style margin for estimation error)
    Clamped to [0, 0.25] — a hard defence against runaway sizing.

    Two corrections relative to textbook Kelly:

    1. Transaction-cost adjustment (round_trip_cost_pct):
       Textbook Kelly assumes frictionless execution. A Binance spot
       round-trip is ≈0.2% (0.1% each side) plus slippage. Subtracting
       the cost from avg_win and adding it to avg_loss gives the *net*
       payoff-ratio b_net, which is what the geometric-growth objective
       actually optimises.

    2. Shrinkage of observed WR (sample_size):
       With n ≪ 100 trades, raw WR is noisy; a 60% WR from 10 trades
       has a 95% CI of roughly [30%, 85%]. Shrinking toward 0.5 via a
       Beta-Binomial prior with prior_weight pseudo-trades protects
       against over-sizing on fluke early wins.

    Args:
        win_rate: observed win-rate ∈ [0, 1]
        avg_win: average gross winning-trade size (e.g. %)
        avg_loss: average gross losing-trade size (e.g. %)
        sample_size: number of trades backing win_rate (drives shrinkage).
            Pass 0 to disable shrinkage (use observed WR directly).
        round_trip_cost_pct: total commission + slippage per round-trip
            (default 0 for backwards compat; use ≈0.20 for Binance spot).
        prior_weight: shrinkage prior strength in pseudo-trades.

    Returns:
        Half-Kelly fraction clamped to [0, 0.25].
    """
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0

    # Shrink raw WR toward 0.5 to correct small-sample over-estimation
    effective_wr = (
        shrinkage_win_rate(win_rate, sample_size, 0.5, prior_weight)
        if sample_size > 0
        else win_rate
    )
    if effective_wr <= 0:
        return 0.0

    # Net-of-cost payoff (textbook Kelly is gross; real growth compounds net)
    net_win = max(0.01, avg_win - round_trip_cost_pct)
    net_loss = avg_loss + round_trip_cost_pct

    b = net_win / net_loss
    q = 1.0 - effective_wr
    f = (effective_wr * b - q) / b
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


# BTC/ETH correlation ~0.85 — reduce size when both are held
_CORRELATED_PAIRS: dict[str, set[str]] = {
    "BTCUSDT": {"ETHUSDT"},
    "ETHUSDT": {"BTCUSDT"},
}
_CORRELATION_PENALTY: float = 0.70  # reduce to 70% of calculated size


def correlation_factor(symbol: str, open_symbols: list[str] | None) -> float:
    """Reduce position size when correlated assets are already held.

    BTC and ETH have ~0.85 correlation — holding both is not true diversification.
    """
    if not open_symbols:
        return 1.0
    correlated = _CORRELATED_PAIRS.get(symbol, set())
    for s in open_symbols:
        if s in correlated:
            return _CORRELATION_PENALTY
    return 1.0


def loss_streak_dampener(consecutive_losses: int) -> float:
    """Reduce position size after consecutive losses.

    0 losses: 1.0 (full size)
    1 loss: 0.85
    2 losses: 0.65
    3+ losses: 0.50 (minimum — preserve capital)
    """
    if consecutive_losses <= 0:
        return 1.0
    if consecutive_losses == 1:
        return 0.85
    if consecutive_losses == 2:
        return 0.65
    return 0.50


def calculate_position_size(inp: SizingInput) -> SizingResult:
    """Calculate position size using Kelly + ATR + regime + correlation.

    Falls back to fixed sizing if insufficient data (win_rate near 50%).
    """
    if inp.balance <= 0 or inp.price <= 0:
        return SizingResult(
            quantity=0.0, budget_usd=0.0, budget_pct=0.0, method="minimum"
        )

    # 1. Kelly fraction (shrunk WR + net-of-cost payoff)
    kf = kelly_fraction(
        inp.win_rate,
        inp.avg_win_pct,
        inp.avg_loss_pct,
        sample_size=inp.sample_size,
        round_trip_cost_pct=inp.round_trip_cost_pct,
    )

    # 2. Volatility factor
    vf = volatility_factor(inp.atr, inp.price)

    # 3. Regime dampener
    rd = regime_dampener(inp.regime_adx)

    # 4. Correlation factor (reduce when correlated assets already held)
    cf = correlation_factor(inp.symbol, inp.open_symbols)

    # 5. Loss streak dampener (reduce after consecutive losses)
    ld = loss_streak_dampener(inp.consecutive_losses)

    # Combine: kelly-based pct, scaled by vol, regime, correlation, and loss streak
    if kf > 0.01:
        base_pct = kf * 100  # kelly as percentage
        adjusted_pct = base_pct * vf * rd * cf * ld
        method = "kelly_atr"
    else:
        # Insufficient edge, use minimum fixed size
        adjusted_pct = 5.0 * vf * rd * cf * ld
        method = "fixed"

    # Risk-based cap: if SL is known, ensure dollar_risk ≤ max_risk_per_trade_pct% of balance
    # This is the key fix: wider SL → smaller position → same dollar risk
    # Include 0.1% exit commission in effective SL (Binance spot = 0.1% per trade)
    if inp.stop_loss_pct > 0 and inp.balance > 0:
        max_risk_usd = inp.balance * inp.max_risk_per_trade_pct / 100
        effective_sl_pct = inp.stop_loss_pct + 0.10  # SL% + 0.1% exit commission
        max_position_usd = max_risk_usd / (effective_sl_pct / 100)
        risk_cap_pct = max_position_usd / inp.balance * 100
        adjusted_pct = min(adjusted_pct, risk_cap_pct)

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
