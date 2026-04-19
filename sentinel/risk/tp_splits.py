"""
Take-Profit Split Configs — per-strategy scale-out ladders.

Replaces the previous hard-coded "TP1 @ 1R close 50%, TP2 @ 2R close 60%
of remaining, 20% rides trailing" that every strategy used regardless of
its statistical edge shape. A momentum breakout and a mean-reversion
snap-back want very different scale-out ladders:

* **Momentum / breakout** — let winners run. First partial deep, second
  even deeper; the surviving size rides a loose trailing stop so a real
  trend captures the tail of the move.
* **Mean reversion** — cash out near the mean. Earlier partials, more of
  them, tighter trailing. The expected value of holding past 2R is low
  because price is supposed to revert.
* **DCA / grid** — very different shapes; explicit per-strategy entries
  below rather than "close enough" defaults.

Each ladder is a list of ``TpStage`` — at position ``i``, when price
reaches ``entry + risk × r_multiple``, close ``close_pct_of_remaining``
percent of what's currently open (NOT of the original size), then arm
``(trail_activate_pct, trail_pct)`` if provided.

After the last stage fires, whatever's left rides the trailing stop
configured on that last stage until it either trails out or hits the
strategy's signal-based exit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True, frozen=True)
class TpStage:
    """One rung of the scale-out ladder."""
    r_multiple: float                       # TP = entry + risk × r_multiple
    close_pct_of_remaining: float           # 50 = close half of what's left
    # Trailing stop to arm after this stage fires. Keeps behaviour matched
    # to the scale-out shape — deeper partial + looser trail, shallower
    # partial + tighter trail.
    trail_activate_pct: Optional[float] = None
    trail_pct: Optional[float] = None


# Default ladder — matches pre-Phase-5 behaviour so strategies that don't
# override see no regression.
DEFAULT_TP_STAGES: list[TpStage] = [
    TpStage(r_multiple=1.0, close_pct_of_remaining=50.0,
            trail_activate_pct=1.5, trail_pct=1.0),
    TpStage(r_multiple=2.0, close_pct_of_remaining=60.0,
            trail_activate_pct=0.5, trail_pct=0.8),
]


STRATEGY_TP_STAGES: dict[str, list[TpStage]] = {
    # Momentum — default ladder (let winners run)
    "ema_crossover_rsi":  DEFAULT_TP_STAGES,
    "bollinger_breakout": DEFAULT_TP_STAGES,
    "macd_divergence":    DEFAULT_TP_STAGES,

    # Mean reversion — cash out faster; price is supposed to revert so
    # holding past 1.5R is low-EV. Tighter trailing because the runner
    # doesn't need room to breathe.
    "mean_reversion": [
        TpStage(r_multiple=0.75, close_pct_of_remaining=50.0,
                trail_activate_pct=0.5, trail_pct=0.3),
        TpStage(r_multiple=1.5,  close_pct_of_remaining=60.0,
                trail_activate_pct=0.3, trail_pct=0.3),
    ],

    # DCA — one wide target, most of the position rides a very loose
    # trail. The strategy expects multi-day holds and will exit on
    # signal conditions in most cases anyway.
    "dca_bot": [
        TpStage(r_multiple=1.5, close_pct_of_remaining=30.0,
                trail_activate_pct=2.0, trail_pct=1.5),
    ],

    # Grid — tiny targets, aggressive scale-out. Most of the edge is in
    # repeatedly harvesting small moves, not riding tails.
    "grid_trading": [
        TpStage(r_multiple=0.5, close_pct_of_remaining=50.0,
                trail_activate_pct=0.3, trail_pct=0.2),
        TpStage(r_multiple=1.0, close_pct_of_remaining=60.0,
                trail_activate_pct=0.2, trail_pct=0.2),
    ],
}


def get_tp_stages(strategy_name: str) -> list[TpStage]:
    """Return the ladder for ``strategy_name`` or the default."""
    stages = STRATEGY_TP_STAGES.get(strategy_name)
    return stages if stages is not None else DEFAULT_TP_STAGES


@dataclass(slots=True)
class TpStagePriced:
    """A ``TpStage`` with the TP price baked in for this specific position."""
    price: float
    close_pct_of_remaining: float
    trail_activate_pct: Optional[float]
    trail_pct: Optional[float]

    @property
    def trailing_after(self) -> Optional[tuple[float, float]]:
        """Convenience for ``apply_tp_stage_transition`` — returns the
        ``(activate_pct, trail_pct)`` tuple or ``None`` when not set."""
        if self.trail_activate_pct is None or self.trail_pct is None:
            return None
        return (float(self.trail_activate_pct), float(self.trail_pct))


def build_priced_ladder(
    entry_price: float,
    risk_per_unit: float,
    stages: list[TpStage],
) -> list[TpStagePriced]:
    """Convert R-multiples into absolute prices for one position.

    ``risk_per_unit`` is ``entry − stop_loss``. When either input is
    non-positive the ladder is empty — callers must treat that as
    "no TP levels configured" and fall back to the strategy's full TP.
    """
    if entry_price <= 0 or risk_per_unit <= 0 or not stages:
        return []
    return [
        TpStagePriced(
            price=entry_price + risk_per_unit * stage.r_multiple,
            close_pct_of_remaining=stage.close_pct_of_remaining,
            trail_activate_pct=stage.trail_activate_pct,
            trail_pct=stage.trail_pct,
        )
        for stage in stages
    ]
