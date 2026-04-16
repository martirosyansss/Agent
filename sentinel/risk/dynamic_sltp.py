"""
Dynamic SL/TP Calculator — ATR-based, regime-aware stop-loss and take-profit.

Replaces fixed stop_loss_pct / take_profit_pct with adaptive levels based on:
1. ATR (volatility) — wider stops in volatile markets
2. Strategy type — mean reversion gets wider SL, momentum tighter
3. Risk-Reward ratio config per strategy
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SLTPConfig:
    """Per-strategy SL/TP tuning."""
    strategy_name: str
    rr_ratio: float = 2.0           # risk:reward ratio (TP = SL * rr)
    atr_sl_mult: float = 2.0        # SL = atr * mult
    min_sl_pct: float = 1.0         # floor SL %
    max_sl_pct: float = 8.0         # ceiling SL %
    min_tp_pct: float = 1.5         # floor TP %
    max_tp_pct: float = 15.0        # ceiling TP %


# Default tuning per strategy type
STRATEGY_SLTP_DEFAULTS: dict[str, SLTPConfig] = {
    # max_sl_pct is capped at 2.9% to stay within the RiskSentinel.max_loss_per_trade_pct (3.0%).
    # Dynamic ATR-based SL for volatile assets like ETH/BTC can easily exceed 3% —
    # without this cap every signal would be rejected at the risk check stage.
    "ema_crossover_rsi": SLTPConfig("ema_crossover_rsi", rr_ratio=2.0, atr_sl_mult=2.0, max_sl_pct=2.9),
    "bollinger_breakout": SLTPConfig("bollinger_breakout", rr_ratio=2.5, atr_sl_mult=1.8, max_sl_pct=2.9),
    "mean_reversion":     SLTPConfig("mean_reversion", rr_ratio=2.2, atr_sl_mult=2.5, min_sl_pct=1.5, max_sl_pct=2.9),
    "macd_divergence":    SLTPConfig("macd_divergence", rr_ratio=2.0, atr_sl_mult=2.2, max_sl_pct=2.9),
    "dca_bot":            SLTPConfig("dca_bot", rr_ratio=2.0, atr_sl_mult=3.0, max_sl_pct=2.9),
    "grid_trading":       SLTPConfig("grid_trading", rr_ratio=1.8, atr_sl_mult=1.5, min_sl_pct=0.5, max_sl_pct=2.9),
}


@dataclass(slots=True)
class SLTPResult:
    """Calculated SL/TP prices."""
    stop_loss_price: float
    take_profit_price: float
    stop_loss_pct: float
    take_profit_pct: float
    method: str  # 'atr_dynamic' or 'fixed_fallback'


def calculate_dynamic_sltp(
    entry_price: float,
    atr: float,
    strategy_name: str,
    fallback_sl_pct: float = 3.0,
    fallback_tp_pct: float = 5.0,
) -> SLTPResult:
    """Calculate dynamic SL/TP based on ATR and strategy config.

    Args:
        entry_price: Signal entry price.
        atr: Current ATR value.
        strategy_name: Name of strategy generating the signal.
        fallback_sl_pct: Fixed SL% if ATR unavailable.
        fallback_tp_pct: Fixed TP% if ATR unavailable.

    Returns:
        SLTPResult with prices and percentages.
    """
    if entry_price <= 0:
        return SLTPResult(0, 0, 0, 0, "fixed_fallback")

    cfg = STRATEGY_SLTP_DEFAULTS.get(strategy_name)
    if cfg is None:
        cfg = SLTPConfig(strategy_name)

    # If ATR is valid, use dynamic calculation
    if atr > 0 and entry_price > 0:
        atr_pct = (atr / entry_price) * 100

        sl_pct = atr_pct * cfg.atr_sl_mult
        sl_pct = max(cfg.min_sl_pct, min(sl_pct, cfg.max_sl_pct))

        tp_pct = sl_pct * cfg.rr_ratio
        tp_pct = max(cfg.min_tp_pct, min(tp_pct, cfg.max_tp_pct))

        sl_price = entry_price * (1 - sl_pct / 100)
        tp_price = entry_price * (1 + tp_pct / 100)

        return SLTPResult(
            stop_loss_price=round(sl_price, 8),
            take_profit_price=round(tp_price, 8),
            stop_loss_pct=round(sl_pct, 2),
            take_profit_pct=round(tp_pct, 2),
            method="atr_dynamic",
        )

    # Fallback to fixed percentages
    sl_price = entry_price * (1 - fallback_sl_pct / 100)
    tp_price = entry_price * (1 + fallback_tp_pct / 100)

    return SLTPResult(
        stop_loss_price=round(sl_price, 8),
        take_profit_price=round(tp_price, 8),
        stop_loss_pct=fallback_sl_pct,
        take_profit_pct=fallback_tp_pct,
        method="fixed_fallback",
    )
