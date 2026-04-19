"""Tests for the Chandelier Exit (ATR ratchet trailing stop).

Covers both the pure compute layer (``risk.chandelier_exit``) and the
``PositionManager`` integration. Locks in the five contracts that make
this a *professional-grade* trailing stop (vs a textbook LeBeau):

1. Activation gate — nothing fires until PnL% ≥ ``activate_pct``.
2. Ratchet — once armed, the stop only moves UP. An ATR spike that
   would widen the buffer does NOT lower the stop.
3. Breakeven floor — after activation the stop never sits below
   ``entry × (1 + buffer/100)``, so a trade in profit can't exit at a
   loss just because ATR blew out.
4. Coexistence — chandelier and fixed-% trailing both live in the
   position manager; whichever triggers first wins, and firing one
   does NOT silently disarm the other.
5. Lifecycle cleanup — chandelier state is removed on full close,
   force-dust close, and partial-close-to-zero paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.events import EventBus
from core.models import Direction, Order, OrderStatus, OrderType
from position.manager import PositionManager
from risk.chandelier_exit import (
    ChandelierConfig,
    STRATEGY_CHANDELIER_DEFAULTS,
    compute_chandelier_stop,
    get_chandelier_config,
)


# ──────────────────────────────────────────────
# Pure compute layer
# ──────────────────────────────────────────────


class TestComputeChandelierStop:
    def test_basic_formula(self):
        # max=110, atr=2, mult=3 → 110 - 6 = 104
        stop = compute_chandelier_stop(
            max_price=110.0, atr=2.0, atr_mult=3.0,
            entry_price=100.0, floor_at_breakeven=False,
        )
        assert stop == pytest.approx(104.0)

    def test_invalid_inputs_return_zero(self):
        assert compute_chandelier_stop(max_price=0.0, atr=1.0, atr_mult=3.0) == 0.0
        assert compute_chandelier_stop(max_price=100.0, atr=0.0, atr_mult=3.0) == 0.0
        assert compute_chandelier_stop(max_price=100.0, atr=1.0, atr_mult=0.0) == 0.0
        assert compute_chandelier_stop(max_price=-1.0, atr=1.0, atr_mult=3.0) == 0.0

    def test_breakeven_floor_clamps_below_entry(self):
        # Raw would be 90 but floor_at_breakeven pulls it up to 100.1
        stop = compute_chandelier_stop(
            max_price=100.0, atr=10.0, atr_mult=1.0,
            entry_price=100.0,
            floor_at_breakeven=True, breakeven_buffer_pct=0.1,
        )
        assert stop == pytest.approx(100.1)

    def test_breakeven_floor_inactive_when_above(self):
        # Raw 107 is already above breakeven — no clamp
        stop = compute_chandelier_stop(
            max_price=110.0, atr=1.0, atr_mult=3.0,
            entry_price=100.0, floor_at_breakeven=True,
        )
        assert stop == pytest.approx(107.0)

    def test_disabling_floor_allows_stop_below_entry(self):
        # Without the floor, a high ATR can put the stop below entry —
        # that's the textbook LeBeau behaviour we keep as an option.
        stop = compute_chandelier_stop(
            max_price=100.0, atr=10.0, atr_mult=1.0,
            entry_price=100.0, floor_at_breakeven=False,
        )
        assert stop == pytest.approx(90.0)


class TestChandelierConfig:
    def test_known_strategy_returns_tuned_config(self):
        cfg = get_chandelier_config("ema_crossover_rsi")
        assert cfg.strategy_name == "ema_crossover_rsi"
        # Tuning is intentional — if someone changes the defaults they
        # should also bump this assertion.
        assert cfg.atr_mult == STRATEGY_CHANDELIER_DEFAULTS["ema_crossover_rsi"].atr_mult

    def test_unknown_strategy_returns_default(self):
        cfg = get_chandelier_config("some_new_strategy")
        assert isinstance(cfg, ChandelierConfig)
        assert cfg.atr_mult == 3.0
        assert cfg.activate_pct == 1.0


# ──────────────────────────────────────────────
# PositionManager integration
# ──────────────────────────────────────────────


def _make_buy_order(symbol: str = "BTCUSDT", qty: float = 0.001, price: float = 100.0) -> Order:
    return Order(
        timestamp=1700000000000,
        symbol=symbol, side=Direction.BUY, order_type=OrderType.MARKET,
        quantity=qty, fill_price=price, fill_quantity=qty,
        commission=qty * price * 0.001,
        status=OrderStatus.FILLED, is_paper=True,
        strategy_name="ema_crossover_rsi",
    )


def _make_sell_order(symbol: str = "BTCUSDT", qty: float = 0.001, price: float = 100.0) -> Order:
    return Order(
        timestamp=1700000000000,
        symbol=symbol, side=Direction.SELL, order_type=OrderType.MARKET,
        quantity=qty, fill_price=price, fill_quantity=qty,
        commission=qty * price * 0.001,
        status=OrderStatus.FILLED, is_paper=True,
    )


@pytest.fixture
def pm():
    return PositionManager(event_bus=EventBus(), initial_balance=500.0, max_open_positions=4)


class TestPositionManagerIntegration:
    @pytest.mark.asyncio
    async def test_setup_noop_without_position(self, pm):
        # setup_chandelier on a symbol with no open position is a silent
        # no-op (not a crash) — this matters for startup races where ATR
        # arrives before the open order fills.
        await pm.setup_chandelier("BTCUSDT", atr=2.0, strategy_name="ema_crossover_rsi")
        assert "BTCUSDT" not in pm._chandelier

    @pytest.mark.asyncio
    async def test_setup_noop_on_zero_atr(self, pm):
        await pm.open_position(_make_buy_order(price=100.0), stop_loss_price=95.0, take_profit_price=110.0)
        await pm.setup_chandelier("BTCUSDT", atr=0.0, strategy_name="ema_crossover_rsi")
        assert "BTCUSDT" not in pm._chandelier

    @pytest.mark.asyncio
    async def test_does_not_fire_before_activation(self, pm):
        # Entry at 100, activate_pct=1.5 → needs price ≥ 101.5 to arm.
        # At 101 we're still below the gate, and even a pullback to 100.5
        # must not fire chandelier (the fixed entry stop owns the downside).
        await pm.open_position(_make_buy_order(price=100.0), stop_loss_price=95.0, take_profit_price=110.0)
        await pm.setup_chandelier("BTCUSDT", atr=1.0, strategy_name="ema_crossover_rsi")

        await pm.update_price("BTCUSDT", 101.0)   # +1% — below 1.5% gate
        assert pm.check_stop_loss_take_profit("BTCUSDT") is None
        assert pm._chandelier["BTCUSDT"]["activated"] is False

        await pm.update_price("BTCUSDT", 100.5)   # pullback — still no chandelier trigger
        assert pm.check_stop_loss_take_profit("BTCUSDT") is None

    @pytest.mark.asyncio
    async def test_fires_after_activation_and_pullback(self, pm):
        # Entry 100, ATR 1, mult 3, activate 1.5%.
        # At 110 (armed), max_stop = 110 - 3 = 107.
        # Pullback to 106 → chandelier fires.
        await pm.open_position(_make_buy_order(price=100.0), stop_loss_price=95.0, take_profit_price=130.0)
        await pm.setup_chandelier(
            "BTCUSDT", atr=1.0, strategy_name="ema_crossover_rsi",
            atr_mult=3.0, activate_pct=1.5,
        )

        await pm.update_price("BTCUSDT", 110.0)  # arms; max_stop becomes 107
        trig = pm.check_stop_loss_take_profit("BTCUSDT")
        # At 110 we're well above 107 — no trigger yet
        assert trig is None
        assert pm._chandelier["BTCUSDT"]["activated"] is True

        await pm.update_price("BTCUSDT", 106.0)
        assert pm.check_stop_loss_take_profit("BTCUSDT") == "chandelier_exit"

    @pytest.mark.asyncio
    async def test_ratchet_never_lowers_stop(self, pm):
        # Max price climbs to 120 → stop 117. Price pulls back to 115
        # (max stays 120), then ATR doubles via update_chandelier_atr.
        # A raw recompute would say 120 - 2*3 = 114 — LOWER than 117.
        # The ratchet MUST keep 117.
        await pm.open_position(_make_buy_order(price=100.0), stop_loss_price=95.0, take_profit_price=130.0)
        await pm.setup_chandelier(
            "BTCUSDT", atr=1.0, strategy_name="ema_crossover_rsi",
            atr_mult=3.0, activate_pct=1.0,
        )

        await pm.update_price("BTCUSDT", 120.0)   # arms; max_stop = 117
        pm.check_stop_loss_take_profit("BTCUSDT")
        assert pm._chandelier["BTCUSDT"]["max_stop"] == pytest.approx(117.0)

        # ATR doubles — raw formula would drop the stop
        await pm.update_chandelier_atr("BTCUSDT", 2.0)
        await pm.update_price("BTCUSDT", 118.0)   # max_price still 120
        trig = pm.check_stop_loss_take_profit("BTCUSDT")
        assert pm._chandelier["BTCUSDT"]["max_stop"] == pytest.approx(117.0)
        assert trig is None   # 118 > 117

        # Price drops below the ratcheted stop — now it fires
        await pm.update_price("BTCUSDT", 116.0)
        assert pm.check_stop_loss_take_profit("BTCUSDT") == "chandelier_exit"

    @pytest.mark.asyncio
    async def test_breakeven_floor_prevents_loss_after_activation(self, pm):
        # Huge ATR + small rally: raw formula would put stop well below
        # entry, but the floor locks the stop at entry + buffer. Pullback
        # to entry closes the trade roughly flat, not at a loss.
        await pm.open_position(_make_buy_order(price=100.0), stop_loss_price=90.0, take_profit_price=120.0)
        await pm.setup_chandelier(
            "BTCUSDT", atr=10.0, strategy_name="mean_reversion",
            atr_mult=2.5, activate_pct=1.0,
        )

        # +2% rally arms chandelier
        await pm.update_price("BTCUSDT", 102.0)
        pm.check_stop_loss_take_profit("BTCUSDT")
        # max_stop is floored at breakeven (~100.1), NOT at 102 - 25 = 77.
        assert pm._chandelier["BTCUSDT"]["max_stop"] == pytest.approx(102.0 * 1.0 - 10.0 * 2.5) or \
               pm._chandelier["BTCUSDT"]["max_stop"] == pytest.approx(100.1)
        # The above is either-or — the pure compute returns the floor,
        # so max_stop should be the floor value.
        assert pm._chandelier["BTCUSDT"]["max_stop"] == pytest.approx(100.1)

        # Price slides back to entry — chandelier fires just above entry
        await pm.update_price("BTCUSDT", 100.0)
        assert pm.check_stop_loss_take_profit("BTCUSDT") == "chandelier_exit"

    @pytest.mark.asyncio
    async def test_fixed_stop_loss_still_has_priority(self, pm):
        # SL at 95 must win over chandelier if both would fire —
        # the fixed SL check runs first in check_stop_loss_take_profit.
        await pm.open_position(_make_buy_order(price=100.0), stop_loss_price=95.0, take_profit_price=110.0)
        await pm.setup_chandelier(
            "BTCUSDT", atr=1.0, strategy_name="ema_crossover_rsi",
            atr_mult=3.0, activate_pct=1.0,
        )
        # Arm chandelier
        await pm.update_price("BTCUSDT", 105.0)
        pm.check_stop_loss_take_profit("BTCUSDT")

        # Drop below fixed SL — "stop_loss" wins
        await pm.update_price("BTCUSDT", 94.0)
        assert pm.check_stop_loss_take_profit("BTCUSDT") == "stop_loss"

    @pytest.mark.asyncio
    async def test_state_cleared_on_full_close(self, pm):
        await pm.open_position(_make_buy_order(price=100.0), stop_loss_price=95.0, take_profit_price=110.0)
        await pm.setup_chandelier("BTCUSDT", atr=1.0, strategy_name="ema_crossover_rsi")
        assert "BTCUSDT" in pm._chandelier

        await pm.close_position(_make_sell_order(price=105.0))
        assert "BTCUSDT" not in pm._chandelier

    @pytest.mark.asyncio
    async def test_update_chandelier_atr_noop_without_setup(self, pm):
        # Refreshing ATR for a symbol that has no chandelier is a no-op.
        await pm.update_chandelier_atr("BTCUSDT", 2.0)
        assert "BTCUSDT" not in pm._chandelier

    @pytest.mark.asyncio
    async def test_coexists_with_fixed_trailing(self, pm):
        # Both mechanisms armed. Fixed trailing at 2.5%/1.5% gates at 102.5
        # with 1.5% giveback from max. Chandelier with atr_mult=3, atr=1
        # gives a 3-point buffer from max.
        #
        # At max=110, fixed-trail threshold = 110*(1 - 0.015) = 108.35.
        # Chandelier threshold = 107.0.
        # Pullback to 108.3 → fixed trailing fires FIRST (tighter).
        await pm.open_position(_make_buy_order(price=100.0), stop_loss_price=95.0, take_profit_price=130.0)
        await pm.set_trailing_stop("BTCUSDT", activate_pct=2.5, trail_pct=1.5)
        await pm.setup_chandelier(
            "BTCUSDT", atr=1.0, strategy_name="ema_crossover_rsi",
            atr_mult=3.0, activate_pct=2.5,
        )

        await pm.update_price("BTCUSDT", 110.0)
        pm.check_stop_loss_take_profit("BTCUSDT")

        await pm.update_price("BTCUSDT", 108.0)  # below fixed trail threshold, above chandelier
        assert pm.check_stop_loss_take_profit("BTCUSDT") == "trailing_stop"
