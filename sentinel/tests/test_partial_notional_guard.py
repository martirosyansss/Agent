"""Regression test for the SL/TP partial-notional guard.

Production incident 2026-04-20 05:32 UTC: a tp1_partial slice on a small
ETHUSDT position priced out to $5.72 (< $10 min-notional). The paper
executor returned None, the caller retried once more, both attempts hit
the same gate, and the kill-switch halted trading while the position
stayed open. See the event in ``events.jsonl`` — ``type=component_error``,
``reason='SL/TP close failed for ETHUSDT after retry'``.

The fix is in :func:`risk.tp_splits.evaluate_partial_notional`: it inspects
the slice *before* the executor call and reports one of three decisions.
These tests lock the three branches so a future refactor can't silently
reintroduce the HALT path.
"""
from __future__ import annotations

import pytest

from risk.tp_splits import (
    PartialNotionalDecision,
    evaluate_partial_notional,
)


class TestEvaluatePartialNotional:
    def test_slice_above_min_executes(self):
        # Standard case: full-sized position, 50% partial comfortably clears min.
        decision = evaluate_partial_notional(
            remaining_qty=0.05, close_pct=50.0, price=2300.0, min_notional_usd=10.0,
        )
        assert decision is PartialNotionalDecision.EXECUTE

    def test_slice_below_min_but_full_clears_escalates(self):
        # The exact prod scenario: qty=0.004989 @ 2263.39 → slice=$5.65, full=$11.29.
        # Partial dust-sized, but closing 100% still clears $10 min → escalate.
        decision = evaluate_partial_notional(
            remaining_qty=0.004989,
            close_pct=50.0,
            price=2263.39,
            min_notional_usd=10.0,
        )
        assert decision is PartialNotionalDecision.ESCALATE

    def test_both_below_min_skips(self):
        # Shrunk-down position where even a full close would be dust.
        decision = evaluate_partial_notional(
            remaining_qty=0.002, close_pct=50.0, price=2000.0, min_notional_usd=10.0,
        )
        assert decision is PartialNotionalDecision.SKIP

    def test_boundary_exactly_at_min_executes(self):
        # A slice sitting exactly on the min-notional line must not trip the
        # guard — Binance's check is ``>=``, and we mirror that.
        decision = evaluate_partial_notional(
            remaining_qty=1.0, close_pct=50.0, price=20.0, min_notional_usd=10.0,
        )
        assert decision is PartialNotionalDecision.EXECUTE

    @pytest.mark.parametrize(
        "qty, close_pct, price",
        [
            (0.0, 50.0, 2000.0),   # zero qty
            (-0.1, 50.0, 2000.0),  # negative qty
            (0.1, 0.0, 2000.0),    # zero close_pct
            (0.1, 50.0, 0.0),      # zero price
        ],
    )
    def test_degenerate_inputs_skip(self, qty, close_pct, price):
        # Protect the executor from being called with garbage inputs — skip
        # rather than EXECUTE/ESCALATE when the math doesn't make sense.
        decision = evaluate_partial_notional(
            remaining_qty=qty, close_pct=close_pct, price=price,
            min_notional_usd=10.0,
        )
        assert decision is PartialNotionalDecision.SKIP
