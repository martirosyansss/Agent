"""
Chandelier Exit — ATR-based ratchet trailing stop.

Classical definition (Chuck LeBeau): stop = highest_high_since_entry - ATR × N.
For a risk-managed long-only system the standalone formula is refined:

* **Activation gate** — the stop is armed only after price has advanced by
  ``activate_pct`` above entry. Before that, the fixed entry stop-loss
  remains authoritative (the trade hasn't earned a trail yet).
* **Ratchet** — once armed, the stop can only move up. If ATR expands and
  the raw formula would lower the stop, we keep the previous level —
  that's the whole point of a trailing mechanism.
* **Breakeven floor** — after activation the stop is never allowed to sit
  below ``entry_price × (1 + buffer_pct/100)``. Locks in the commission
  and prevents a trade that was already in profit from closing at a loss
  because ATR blew out.

Why this over fixed-% trailing:

A 1.5% fixed trailing stop is equally tight in a dead-calm session (where
you'd want a 3% buffer to avoid noise) and in a 10%-range day (where 1.5%
is a single wick). Chandelier scales with realised volatility so the trader
isn't forced to pick a compromise.

Config per strategy lives in ``STRATEGY_CHANDELIER_DEFAULTS`` — momentum
strategies get a wider multiplier (let winners run), mean-reversion gets
tighter (take profit close to the mean).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ChandelierConfig:
    """Per-strategy Chandelier Exit tuning."""
    strategy_name: str
    atr_mult: float = 3.0                 # stop = max_price - atr * atr_mult
    activate_pct: float = 1.0             # arm only after +N% profit
    floor_at_breakeven: bool = True       # after activation, never below entry+buffer
    breakeven_buffer_pct: float = 0.1     # safety margin above entry


STRATEGY_CHANDELIER_DEFAULTS: dict[str, ChandelierConfig] = {
    # Momentum — wider mult, let winners run through noise
    "ema_crossover_rsi":  ChandelierConfig("ema_crossover_rsi",  atr_mult=3.0, activate_pct=1.5),
    "macd_divergence":    ChandelierConfig("macd_divergence",    atr_mult=3.0, activate_pct=1.5),
    # Breakout — activate later (breakouts retrace), tighter trail once running
    "bollinger_breakout": ChandelierConfig("bollinger_breakout", atr_mult=2.5, activate_pct=2.0),
    # Mean-reversion — cashflow near the mean, early & tight
    "mean_reversion":     ChandelierConfig("mean_reversion",     atr_mult=2.5, activate_pct=1.0),
    # DCA — long horizon, widest trail
    "dca_bot":            ChandelierConfig("dca_bot",            atr_mult=3.5, activate_pct=2.0),
    # Grid — tiny moves, tightest trail
    "grid_trading":       ChandelierConfig("grid_trading",       atr_mult=2.0, activate_pct=0.5),
}


def get_chandelier_config(strategy_name: str) -> ChandelierConfig:
    """Return the tuned config for a strategy, or a neutral default."""
    cfg = STRATEGY_CHANDELIER_DEFAULTS.get(strategy_name)
    if cfg is not None:
        return cfg
    return ChandelierConfig(strategy_name=strategy_name or "default")


def compute_chandelier_stop(
    max_price: float,
    atr: float,
    atr_mult: float,
    *,
    entry_price: float = 0.0,
    floor_at_breakeven: bool = True,
    breakeven_buffer_pct: float = 0.1,
) -> float:
    """Raw Chandelier Exit stop price for a long position.

    Args:
        max_price: Highest high (or highest close) since entry.
        atr: Current ATR value on the signal timeframe.
        atr_mult: Chandelier multiplier (typical 2.5–3.5).
        entry_price: Position entry price — only used when flooring.
        floor_at_breakeven: If True, never let the stop sit below
            ``entry_price × (1 + buffer/100)``. Recommended ON for risk-
            managed trading; OFF matches LeBeau's textbook formula.
        breakeven_buffer_pct: Buffer over entry (default 0.1% — covers
            round-trip commission on Binance spot).

    Returns:
        Stop price, or 0.0 when inputs are invalid (caller must treat as
        "no stop computed yet").
    """
    if max_price <= 0 or atr <= 0 or atr_mult <= 0:
        return 0.0

    raw = max_price - atr * atr_mult

    if floor_at_breakeven and entry_price > 0:
        breakeven = entry_price * (1.0 + breakeven_buffer_pct / 100.0)
        if raw < breakeven:
            return breakeven

    return raw
