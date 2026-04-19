"""Tests for Phase 5 — flexible multi-stage TP ladder.

Contracts:

* **Default ladder preserves legacy behaviour** — a position opened with
  no explicit ladder configuration (the old call pattern) sees the same
  TP prices, close percentages, and trailing-after config as before
  Phase 5. Existing backtests are bit-for-bit compatible.
* **Per-strategy defaults differentiate** — ``mean_reversion`` cashes
  out earlier than ``ema_crossover_rsi``; ``dca_bot`` has a single
  wider rung. Running the wrong ladder on a strategy materially
  changes realised P&L, so the dispatch must be exact.
* **Triggers are ``tp{N}_partial`` for arbitrary N** — a 4-rung ladder
  must emit ``tp1_partial`` … ``tp4_partial`` in order and stop firing
  TP triggers once ``pos.tp_stage`` advances past the last rung.
* **get_current_tp_stage returns the stage being triggered** — so
  ``main.py`` can read ``close_pct_of_remaining`` and ``trailing_after``
  without re-deriving the ladder shape. Returns ``None`` when the
  ladder is exhausted.
* **Empty risk → empty ladder** — a position with ``risk_per_unit <= 0``
  gets no priced ladder (falls through to the strategy's full TP).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.events import EventBus
from core.models import Direction, Order, OrderStatus, OrderType
from position.manager import PositionManager
from risk.tp_splits import (
    DEFAULT_TP_STAGES,
    TpStage,
    build_priced_ladder,
    get_tp_stages,
)


def _make_buy(symbol: str = "BTCUSDT", price: float = 100.0, qty: float = 0.001, strategy: str = "ema_crossover_rsi") -> Order:
    return Order(
        timestamp=1700000000000,
        symbol=symbol, side=Direction.BUY, order_type=OrderType.MARKET,
        quantity=qty, fill_price=price, fill_quantity=qty,
        commission=qty * price * 0.001,
        status=OrderStatus.FILLED, is_paper=True,
        strategy_name=strategy,
    )


@pytest.fixture
def pm():
    return PositionManager(event_bus=EventBus(), initial_balance=5000.0, max_open_positions=4)


# ──────────────────────────────────────────────
# Pure config layer
# ──────────────────────────────────────────────


class TestTpSplitsConfig:
    def test_default_stages_unchanged(self):
        # The default ladder must match pre-Phase-5 behaviour so existing
        # strategies don't see silent P&L drift.
        assert len(DEFAULT_TP_STAGES) == 2
        assert DEFAULT_TP_STAGES[0].r_multiple == 1.0
        assert DEFAULT_TP_STAGES[0].close_pct_of_remaining == 50.0
        assert DEFAULT_TP_STAGES[0].trail_activate_pct == 1.5
        assert DEFAULT_TP_STAGES[0].trail_pct == 1.0

    def test_known_strategy_uses_default(self):
        assert get_tp_stages("ema_crossover_rsi") == DEFAULT_TP_STAGES
        assert get_tp_stages("bollinger_breakout") == DEFAULT_TP_STAGES

    def test_mean_reversion_has_tighter_ladder(self):
        stages = get_tp_stages("mean_reversion")
        assert stages[0].r_multiple < DEFAULT_TP_STAGES[0].r_multiple

    def test_unknown_strategy_falls_back(self):
        assert get_tp_stages("future_strategy_xyz") == DEFAULT_TP_STAGES

    def test_build_priced_ladder_maths(self):
        stages = [
            TpStage(r_multiple=1.0, close_pct_of_remaining=50.0,
                    trail_activate_pct=1.5, trail_pct=1.0),
            TpStage(r_multiple=2.5, close_pct_of_remaining=40.0),
        ]
        priced = build_priced_ladder(
            entry_price=100.0, risk_per_unit=5.0, stages=stages,
        )
        assert len(priced) == 2
        assert priced[0].price == pytest.approx(105.0)
        assert priced[1].price == pytest.approx(112.5)
        assert priced[0].trailing_after == (1.5, 1.0)
        assert priced[1].trailing_after is None   # neither trail arg set

    def test_build_priced_ladder_empty_on_zero_risk(self):
        assert build_priced_ladder(100.0, 0.0, DEFAULT_TP_STAGES) == []
        assert build_priced_ladder(0.0, 5.0, DEFAULT_TP_STAGES) == []


# ──────────────────────────────────────────────
# PositionManager integration
# ──────────────────────────────────────────────


class TestPositionManagerLadder:
    @pytest.mark.asyncio
    async def test_setup_uses_strategy_default(self, pm):
        await pm.open_position(_make_buy(price=100.0, strategy="ema_crossover_rsi"),
                               stop_loss_price=95.0, take_profit_price=115.0)
        await pm.setup_tp_levels("BTCUSDT")
        ladder = pm._tp_levels["BTCUSDT"]
        assert len(ladder) == 2
        # Risk = 5, TP1 at +1R = 105, TP2 at +2R = 110
        assert ladder[0].price == pytest.approx(105.0)
        assert ladder[1].price == pytest.approx(110.0)
        assert ladder[0].close_pct_of_remaining == 50.0
        assert ladder[1].close_pct_of_remaining == 60.0

    @pytest.mark.asyncio
    async def test_setup_noop_when_risk_nonpositive(self, pm):
        # SL above entry — "negative risk" — no ladder is built.
        await pm.open_position(_make_buy(price=100.0),
                               stop_loss_price=101.0, take_profit_price=110.0)
        await pm.setup_tp_levels("BTCUSDT")
        assert "BTCUSDT" not in pm._tp_levels

    @pytest.mark.asyncio
    async def test_explicit_ladder_overrides_strategy_default(self, pm):
        await pm.open_position(_make_buy(price=100.0, strategy="ema_crossover_rsi"),
                               stop_loss_price=95.0, take_profit_price=200.0)
        custom = [
            TpStage(r_multiple=0.5, close_pct_of_remaining=33.0,
                    trail_activate_pct=0.3, trail_pct=0.2),
            TpStage(r_multiple=1.0, close_pct_of_remaining=33.0),
            TpStage(r_multiple=2.0, close_pct_of_remaining=50.0),
        ]
        await pm.setup_tp_levels("BTCUSDT", tp_stages=custom)

        ladder = pm._tp_levels["BTCUSDT"]
        assert len(ladder) == 3
        assert ladder[0].price == pytest.approx(102.5)   # entry + 0.5R
        assert ladder[2].close_pct_of_remaining == 50.0

    @pytest.mark.asyncio
    async def test_triggers_in_order(self, pm):
        await pm.open_position(_make_buy(price=100.0), stop_loss_price=95.0, take_profit_price=130.0)
        await pm.setup_tp_levels("BTCUSDT")

        # Reach TP1 at 105 → tp1_partial
        await pm.update_price("BTCUSDT", 106.0)
        assert pm.check_stop_loss_take_profit("BTCUSDT") == "tp1_partial"

        # Simulate the stage-transition that main.py applies:
        await pm.apply_tp_stage_transition("BTCUSDT", stage=1, move_to_breakeven=True,
                                            trailing=(1.5, 1.0))

        # Now TP2 at 110 → tp2_partial
        await pm.update_price("BTCUSDT", 111.0)
        assert pm.check_stop_loss_take_profit("BTCUSDT") == "tp2_partial"

        # Advance past the last rung — TP checks stop firing.
        await pm.apply_tp_stage_transition("BTCUSDT", stage=2, trailing=(0.5, 0.8))
        await pm.update_price("BTCUSDT", 115.0)
        trig = pm.check_stop_loss_take_profit("BTCUSDT")
        assert trig != "tp3_partial"   # no such stage exists
        # Trailing may or may not fire depending on price path; either
        # None or "trailing_stop" is acceptable here.
        assert trig in (None, "trailing_stop")

    @pytest.mark.asyncio
    async def test_triggers_for_ladder_depth_4(self, pm):
        await pm.open_position(_make_buy(price=100.0), stop_loss_price=95.0, take_profit_price=300.0)
        await pm.setup_tp_levels("BTCUSDT", tp_stages=[
            TpStage(r_multiple=1.0, close_pct_of_remaining=25.0),
            TpStage(r_multiple=2.0, close_pct_of_remaining=33.0),
            TpStage(r_multiple=3.0, close_pct_of_remaining=50.0),
            TpStage(r_multiple=4.0, close_pct_of_remaining=100.0),
        ])

        for expected_stage in range(1, 5):
            price = 100.0 + 5.0 * expected_stage + 0.5
            await pm.update_price("BTCUSDT", price)
            trig = pm.check_stop_loss_take_profit("BTCUSDT")
            assert trig == f"tp{expected_stage}_partial"
            await pm.apply_tp_stage_transition("BTCUSDT", stage=expected_stage)


class TestGetCurrentTpStage:
    @pytest.mark.asyncio
    async def test_returns_stage_at_current_index(self, pm):
        await pm.open_position(_make_buy(price=100.0), stop_loss_price=95.0, take_profit_price=130.0)
        await pm.setup_tp_levels("BTCUSDT")

        stage0 = pm.get_current_tp_stage("BTCUSDT")
        assert stage0 is not None
        assert stage0.price == pytest.approx(105.0)
        assert stage0.close_pct_of_remaining == 50.0
        assert stage0.trailing_after == (1.5, 1.0)

        await pm.apply_tp_stage_transition("BTCUSDT", stage=1)
        stage1 = pm.get_current_tp_stage("BTCUSDT")
        assert stage1 is not None
        assert stage1.price == pytest.approx(110.0)

    @pytest.mark.asyncio
    async def test_returns_none_past_last_rung(self, pm):
        await pm.open_position(_make_buy(price=100.0), stop_loss_price=95.0, take_profit_price=130.0)
        await pm.setup_tp_levels("BTCUSDT")
        await pm.apply_tp_stage_transition("BTCUSDT", stage=2)  # past last rung (0-indexed 2 for 2-stage ladder)
        assert pm.get_current_tp_stage("BTCUSDT") is None

    @pytest.mark.asyncio
    async def test_returns_none_when_ladder_not_set(self, pm):
        await pm.open_position(_make_buy(price=100.0), stop_loss_price=95.0, take_profit_price=110.0)
        assert pm.get_current_tp_stage("BTCUSDT") is None
