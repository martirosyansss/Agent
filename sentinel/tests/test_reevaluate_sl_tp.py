"""Tests for PositionManager.reevaluate_sl_tp — the Phase-2 dynamic SL.

Locks in the invariants of the re-evaluation layer:

* **Monotonicity** — the stop only moves UP, never down. A losing ATR
  spike, a regime flip against an unprofitable position, or repeated
  calls with stale data must never loosen the stop.
* **Breakeven raise** — once a position has earned ≥ ``threshold_pct``
  of profit, the SL is ratcheted to ``entry × 1.001`` so commission is
  covered and the trade can't round-trip into a loss.
* **Adverse-regime tighten** — if the regime turns against a long and
  the position is in profit, the SL pulls in to
  ``current − atr × mult``, floored at breakeven.
* **No double-apply** — a second call with identical inputs is a no-op
  (actions empty) so the event stream doesn't flood with duplicates.
* **TP untouched** — re-evaluation only moves SL; TP stays at entry-time
  level. Raising TP is a strategy-level decision, not a risk one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.events import EventBus
from core.models import Direction, Order, OrderStatus, OrderType
from position.manager import PositionManager


def _make_buy_order(symbol: str = "BTCUSDT", qty: float = 0.001, price: float = 100.0) -> Order:
    return Order(
        timestamp=1700000000000,
        symbol=symbol, side=Direction.BUY, order_type=OrderType.MARKET,
        quantity=qty, fill_price=price, fill_quantity=qty,
        commission=qty * price * 0.001,
        status=OrderStatus.FILLED, is_paper=True,
        strategy_name="ema_crossover_rsi",
    )


@pytest.fixture
def pm():
    return PositionManager(event_bus=EventBus(), initial_balance=500.0, max_open_positions=4)


class TestReevaluateNoops:
    @pytest.mark.asyncio
    async def test_noop_when_no_position(self, pm):
        res = await pm.reevaluate_sl_tp("BTCUSDT", current_atr=1.0, regime="trending_up")
        assert res["actions"] == []
        assert res["reason"] == "no_position"

    @pytest.mark.asyncio
    async def test_noop_when_no_sl(self, pm):
        # Position opened with sl=0 — re-eval can't anchor anything.
        await pm.open_position(_make_buy_order(), stop_loss_price=0.0, take_profit_price=0.0)
        res = await pm.reevaluate_sl_tp("BTCUSDT", current_atr=1.0, regime="trending_up")
        assert res["actions"] == []
        assert res["reason"] == "no_sl"

    @pytest.mark.asyncio
    async def test_noop_when_pnl_below_threshold(self, pm):
        # +1% is below the default 1.5% breakeven threshold AND regime is
        # bullish, so no rule fires.
        await pm.open_position(_make_buy_order(price=100.0), stop_loss_price=95.0, take_profit_price=110.0)
        await pm.update_price("BTCUSDT", 101.0)
        res = await pm.reevaluate_sl_tp("BTCUSDT", current_atr=1.0, regime="trending_up")
        assert res["actions"] == []
        assert res["sl_before"] == pytest.approx(95.0)
        assert res["sl_after"] == pytest.approx(95.0)


class TestBreakevenRaise:
    @pytest.mark.asyncio
    async def test_fires_when_pnl_above_threshold(self, pm):
        await pm.open_position(_make_buy_order(price=100.0), stop_loss_price=95.0, take_profit_price=110.0)
        await pm.update_price("BTCUSDT", 102.0)   # +2% — past 1.5% gate
        res = await pm.reevaluate_sl_tp("BTCUSDT", current_atr=1.0, regime="trending_up")

        assert len(res["actions"]) == 1
        assert res["actions"][0]["type"] == "breakeven_raise"
        assert res["sl_after"] == pytest.approx(100.1)
        # Position state reflects the change
        assert pm.get_position("BTCUSDT").stop_loss_price == pytest.approx(100.1)

    @pytest.mark.asyncio
    async def test_noop_when_sl_already_at_breakeven(self, pm):
        # First pass raises to breakeven; second pass must be a no-op.
        await pm.open_position(_make_buy_order(price=100.0), stop_loss_price=95.0, take_profit_price=110.0)
        await pm.update_price("BTCUSDT", 103.0)
        await pm.reevaluate_sl_tp("BTCUSDT", current_atr=1.0, regime="trending_up")

        res = await pm.reevaluate_sl_tp("BTCUSDT", current_atr=1.0, regime="trending_up")
        assert res["actions"] == []

    @pytest.mark.asyncio
    async def test_configurable_threshold(self, pm):
        # With threshold=0.5%, +0.6% already fires breakeven raise.
        await pm.open_position(_make_buy_order(price=100.0), stop_loss_price=95.0, take_profit_price=110.0)
        await pm.update_price("BTCUSDT", 100.6)
        res = await pm.reevaluate_sl_tp(
            "BTCUSDT", current_atr=1.0, regime="trending_up",
            breakeven_threshold_pct=0.5,
        )
        assert len(res["actions"]) == 1
        assert res["actions"][0]["type"] == "breakeven_raise"


class TestAdverseRegimeTighten:
    @pytest.mark.asyncio
    async def test_tightens_sl_in_profit_on_bearish_regime(self, pm):
        # Entry 100, price 110 (+10%), ATR 1, mult 1.5 → tight SL at 108.5.
        # Breakeven (100.1) will fire first; then regime_tighten pulls SL
        # from 100.1 to 108.5.
        await pm.open_position(_make_buy_order(price=100.0), stop_loss_price=95.0, take_profit_price=130.0)
        await pm.update_price("BTCUSDT", 110.0)
        res = await pm.reevaluate_sl_tp(
            "BTCUSDT", current_atr=1.0, regime="trending_down",
        )
        kinds = [a["type"] for a in res["actions"]]
        assert "breakeven_raise" in kinds
        assert "regime_tighten" in kinds
        assert res["sl_after"] == pytest.approx(108.5)

    @pytest.mark.asyncio
    async def test_does_not_tighten_when_losing(self, pm):
        # Position in loss — regime rule must NOT fire (would stop us out
        # immediately if current_price - atr*mult > current_sl).
        await pm.open_position(_make_buy_order(price=100.0), stop_loss_price=95.0, take_profit_price=110.0)
        await pm.update_price("BTCUSDT", 98.0)   # -2%
        res = await pm.reevaluate_sl_tp(
            "BTCUSDT", current_atr=1.0, regime="trending_down",
        )
        assert res["actions"] == []
        assert res["sl_after"] == pytest.approx(95.0)

    @pytest.mark.asyncio
    async def test_sideways_regime_also_triggers_tighten(self, pm):
        # "sideways" is in the adverse set for longs — there's no trend to
        # ride, so profits should be defended aggressively.
        await pm.open_position(_make_buy_order(price=100.0), stop_loss_price=95.0, take_profit_price=130.0)
        await pm.update_price("BTCUSDT", 110.0)
        res = await pm.reevaluate_sl_tp(
            "BTCUSDT", current_atr=1.0, regime="sideways",
        )
        kinds = [a["type"] for a in res["actions"]]
        assert "regime_tighten" in kinds

    @pytest.mark.asyncio
    async def test_tightened_stop_floored_at_breakeven(self, pm):
        # Entry 100, price 100.5 (+0.5%) — but breakeven threshold is
        # 1.5% by default so rule #1 does NOT fire. Regime is adverse;
        # raw tightened = 100.5 - 1.0*1.5 = 99.0, BELOW breakeven 100.1.
        # Floor must bring it up to 100.1.
        await pm.open_position(_make_buy_order(price=100.0), stop_loss_price=95.0, take_profit_price=110.0)
        await pm.update_price("BTCUSDT", 100.5)
        res = await pm.reevaluate_sl_tp(
            "BTCUSDT", current_atr=1.0, regime="trending_down",
        )
        # Only regime_tighten fires (breakeven rule didn't arm — PnL < 1.5%)
        kinds = [a["type"] for a in res["actions"]]
        assert kinds == ["regime_tighten"]
        assert res["sl_after"] == pytest.approx(100.1)   # floored


class TestMonotonicity:
    @pytest.mark.asyncio
    async def test_sl_never_moves_down(self, pm):
        # Arrange: raise SL to 108.5 via regime tighten.
        await pm.open_position(_make_buy_order(price=100.0), stop_loss_price=95.0, take_profit_price=130.0)
        await pm.update_price("BTCUSDT", 110.0)
        await pm.reevaluate_sl_tp("BTCUSDT", current_atr=1.0, regime="trending_down")
        assert pm.get_position("BTCUSDT").stop_loss_price == pytest.approx(108.5)

        # Now price pulls back to 105 and regime stays adverse — a naive
        # recompute would give 105 - 1.5 = 103.5, which is LOWER than the
        # current SL. The method must keep the higher SL.
        await pm.update_price("BTCUSDT", 105.0)
        res = await pm.reevaluate_sl_tp("BTCUSDT", current_atr=1.0, regime="trending_down")
        assert res["actions"] == []
        assert pm.get_position("BTCUSDT").stop_loss_price == pytest.approx(108.5)


class TestTpUntouched:
    @pytest.mark.asyncio
    async def test_take_profit_unchanged(self, pm):
        await pm.open_position(_make_buy_order(price=100.0), stop_loss_price=95.0, take_profit_price=110.0)
        await pm.update_price("BTCUSDT", 110.0)
        await pm.reevaluate_sl_tp("BTCUSDT", current_atr=1.0, regime="trending_down")
        # Internal TP state is preserved
        sl, tp = pm._sl_tp["BTCUSDT"]
        assert tp == pytest.approx(110.0)
