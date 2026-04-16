"""Тесты Phase 8 — Risk Sentinel, State Machine, Circuit Breakers, Kill Switch."""

import asyncio
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.events import EventBus
from core.models import Direction, RiskCheckResult, RiskState, Signal
from risk.circuit_breakers import CircuitBreakers, CircuitBreakerState
from risk.kill_switch import KillSwitch
from risk.sentinel import RiskLimits, RiskSentinel
from risk.state_machine import RiskStateMachine


# ──────────────────────────────────────────────
# State Machine
# ──────────────────────────────────────────────

class TestRiskStateMachine:
    @pytest.fixture
    def bus(self):
        return EventBus()

    @pytest.fixture
    def sm(self, bus):
        return RiskStateMachine(event_bus=bus, max_daily_loss=50.0)

    def test_initial_state(self, sm):
        assert sm.state == RiskState.NORMAL

    def test_evaluate_normal(self, sm):
        assert sm.evaluate(0.0) == RiskState.NORMAL
        assert sm.evaluate(-10.0) == RiskState.NORMAL

    def test_evaluate_reduced(self, sm):
        # 30% of $50 = $15
        assert sm.evaluate(-16.0) == RiskState.REDUCED

    def test_evaluate_safe(self, sm):
        # 70% of $50 = $35
        assert sm.evaluate(-36.0) == RiskState.SAFE

    def test_evaluate_stop(self, sm):
        assert sm.evaluate(-50.0) == RiskState.STOP
        assert sm.evaluate(-60.0) == RiskState.STOP

    @pytest.mark.asyncio
    async def test_update_changes_state(self, sm):
        result = await sm.update(-20.0)
        assert result == RiskState.REDUCED
        assert sm.state == RiskState.REDUCED

    @pytest.mark.asyncio
    async def test_update_no_change(self, sm):
        result = await sm.update(-5.0)
        assert result is None  # Still NORMAL
        assert sm.state == RiskState.NORMAL

    @pytest.mark.asyncio
    async def test_event_emitted_on_change(self, bus, sm):
        received = []
        async def handler(old, new, reason):
            received.append((old, new, reason))
        bus.subscribe("risk_state_changed", handler)

        await sm.update(-20.0)
        assert len(received) == 1
        assert received[0][0] == RiskState.NORMAL
        assert received[0][1] == RiskState.REDUCED

    def test_reset(self, sm):
        sm._state = RiskState.STOP
        sm.reset()
        assert sm.state == RiskState.NORMAL


# ──────────────────────────────────────────────
# Risk Sentinel
# ──────────────────────────────────────────────

class TestRiskSentinel:
    @pytest.fixture
    def bus(self):
        return EventBus()

    @pytest.fixture
    def sentinel(self, bus):
        limits = RiskLimits(
            max_daily_loss_usd=50.0,
            max_daily_trades=6,
            max_open_positions=2,
            max_total_exposure_pct=60.0,
            max_trades_per_hour=2,
            min_trade_interval_sec=0,  # Отключаем для тестов
            min_order_usd=10.0,
            max_order_usd=100.0,
            max_loss_per_trade_pct=3.0,
        )
        sm = RiskStateMachine(event_bus=bus, max_daily_loss=50.0)
        return RiskSentinel(limits=limits, state_machine=sm)

    def _make_buy_signal(self, **kwargs) -> Signal:
        defaults = dict(
            timestamp=1700000000000,
            symbol="BTCUSDT",
            direction=Direction.BUY,
            confidence=0.85,
            strategy_name="ema_crossover_rsi",
            reason="Test",
            suggested_quantity=0.001,
            stop_loss_price=65000.0,  # ~3% от 67000
            take_profit_price=70350.0,
        )
        defaults.update(kwargs)
        return Signal(**defaults)

    def _make_sell_signal(self) -> Signal:
        return Signal(
            timestamp=1700000000000,
            symbol="BTCUSDT",
            direction=Direction.SELL,
            confidence=0.90,
            strategy_name="ema_crossover_rsi",
            reason="Stop-loss",
            suggested_quantity=0.001,
        )

    def test_approved_signal(self, sentinel):
        signal = self._make_buy_signal()
        result = sentinel.check_signal(
            signal, daily_pnl=0.0, open_positions_count=0,
            total_exposure_pct=0.0, balance=500.0, current_market_price=67000.0,
        )
        assert result.approved is True

    def test_reject_daily_loss(self, sentinel):
        signal = self._make_buy_signal()
        result = sentinel.check_signal(
            signal, daily_pnl=-50.0, open_positions_count=0,
            total_exposure_pct=0.0, balance=450.0, current_market_price=67000.0,
        )
        assert result.approved is False
        assert "Daily loss" in result.reason

    def test_reject_position_limit(self, sentinel):
        signal = self._make_buy_signal()
        result = sentinel.check_signal(
            signal, daily_pnl=0.0, open_positions_count=2,
            total_exposure_pct=40.0, balance=500.0, current_market_price=67000.0,
        )
        assert result.approved is False
        assert "positions" in result.reason.lower()

    def test_reject_exposure(self, sentinel):
        signal = self._make_buy_signal(suggested_quantity=0.005)  # ~$335
        result = sentinel.check_signal(
            signal, daily_pnl=0.0, open_positions_count=0,
            total_exposure_pct=50.0, balance=500.0, current_market_price=67000.0,
        )
        assert result.approved is False
        assert "Exposure" in result.reason

    def test_reject_daily_trade_limit(self, sentinel):
        for _ in range(6):
            sentinel.record_trade()
        signal = self._make_buy_signal()
        result = sentinel.check_signal(
            signal, daily_pnl=0.0, open_positions_count=0,
            total_exposure_pct=0.0, balance=500.0, current_market_price=67000.0,
        )
        assert result.approved is False
        assert "Daily trade limit" in result.reason

    def test_reject_order_too_small(self, sentinel):
        signal = self._make_buy_signal(suggested_quantity=0.0001)  # ~$6.70
        result = sentinel.check_signal(
            signal, daily_pnl=0.0, open_positions_count=0,
            total_exposure_pct=0.0, balance=500.0, current_market_price=67000.0,
        )
        assert result.approved is False
        assert "too small" in result.reason

    def test_reject_no_stop_loss(self, sentinel):
        signal = self._make_buy_signal(stop_loss_price=0.0)
        result = sentinel.check_signal(
            signal, daily_pnl=0.0, open_positions_count=0,
            total_exposure_pct=0.0, balance=500.0, current_market_price=67000.0,
        )
        assert result.approved is False
        assert "mandatory" in result.reason.lower()

    def test_reject_stop_loss_too_wide(self, sentinel):
        # SL at $60000 from current $67000 = ~10.4%
        signal = self._make_buy_signal(stop_loss_price=60000.0)
        result = sentinel.check_signal(
            signal, daily_pnl=0.0, open_positions_count=0,
            total_exposure_pct=0.0, balance=500.0, current_market_price=67000.0,
        )
        assert result.approved is False
        assert "too wide" in result.reason.lower()

    def test_sell_always_allowed(self, sentinel):
        """SELL не блокируется лимитами позиций/частоты."""
        signal = self._make_sell_signal()
        result = sentinel.check_signal(
            signal, daily_pnl=-40.0, open_positions_count=2,
            total_exposure_pct=50.0, balance=460.0, current_market_price=67000.0,
        )
        assert result.approved is True

    def test_stop_state_blocks_buy(self, sentinel):
        """STOP state blocks BUY. SELL is never blocked (must close positions)."""
        sentinel._sm._state = RiskState.STOP
        signal = self._make_buy_signal()
        result = sentinel.check_signal(
            signal, daily_pnl=-25.0, open_positions_count=0,
            total_exposure_pct=0.0, balance=475.0, current_market_price=67000.0,
        )
        assert result.approved is False
        assert "STOP" in result.reason

    def test_safe_state_blocks_buy(self, sentinel):
        sentinel._sm._state = RiskState.SAFE
        signal = self._make_buy_signal()
        result = sentinel.check_signal(
            signal, daily_pnl=-36.0, open_positions_count=0,
            total_exposure_pct=0.0, balance=464.0, current_market_price=67000.0,
        )
        assert result.approved is False
        assert "SAFE" in result.reason

    def test_reduced_state_requires_high_confidence(self, sentinel):
        sentinel._sm._state = RiskState.REDUCED
        signal = self._make_buy_signal(confidence=0.70)
        result = sentinel.check_signal(
            signal, daily_pnl=-16.0, open_positions_count=0,
            total_exposure_pct=0.0, balance=484.0, current_market_price=67000.0,
        )
        assert result.approved is False
        assert "REDUCED" in result.reason

    def test_record_trade(self, sentinel):
        sentinel.record_trade(commission=0.1)
        assert sentinel._daily_trades == 1

    def test_runtime_metrics_snapshot(self, sentinel):
        sentinel.record_trade(commission=0.1)
        sentinel.record_trade(commission=0.2, increment_trade=False)

        metrics = sentinel.get_runtime_metrics(balance=500.0)
        assert metrics["daily_trades"] == 1
        assert metrics["trades_last_hour"] == 1
        assert metrics["daily_commission"] == pytest.approx(0.3)
        assert metrics["commission_pct"] > 0

    def test_reset_daily(self, sentinel):
        sentinel.record_trade()
        sentinel.reset_daily()
        assert sentinel._daily_trades == 0
        assert sentinel.state == RiskState.NORMAL


# ──────────────────────────────────────────────
# Circuit Breakers
# ──────────────────────────────────────────────

class TestCircuitBreakers:
    @pytest.fixture
    def cbs(self):
        return CircuitBreakers()

    def test_initial_state(self, cbs):
        assert cbs.is_trading_allowed() is True
        assert cbs.any_tripped is False

    def test_cb1_price_anomaly(self, cbs):
        result = cbs.check_price_anomaly(6.0)
        assert result is not None
        assert "Price anomaly" in result
        assert cbs.is_trading_allowed() is False

    def test_cb1_normal_price(self, cbs):
        result = cbs.check_price_anomaly(2.0)
        assert result is None
        assert cbs.is_trading_allowed() is True

    def test_cb2_consecutive_losses(self, cbs):
        cbs.record_trade_result(False)
        cbs.record_trade_result(False)
        cbs.record_trade_result(False)
        cbs.record_trade_result(False)
        assert cbs.is_trading_allowed() is True
        result = cbs.record_trade_result(False)
        assert result is not None
        assert "consecutive" in result

    def test_cb2_win_resets(self, cbs):
        cbs.record_trade_result(False)
        cbs.record_trade_result(False)
        cbs.record_trade_result(True)
        result = cbs.record_trade_result(False)
        assert result is None  # Reset by win

    def test_cb3_spread(self, cbs):
        result = cbs.check_spread(0.7)
        assert result is not None
        assert cbs.is_trading_allowed() is False

    def test_cb4_volume(self, cbs):
        result = cbs.check_volume_anomaly(15.0)
        assert result is not None
        assert "Volume" in result

    def test_cb5_api_errors(self, cbs):
        for _ in range(5):
            cbs.record_api_error()
        assert cbs.is_trading_allowed() is True
        result = cbs.record_api_error()  # 6th
        assert result is not None
        assert "API error" in result

    def test_cb6_latency(self, cbs):
        cbs.check_latency(6.0)
        cbs.check_latency(7.0)
        result = cbs.check_latency(8.0)
        assert result is not None
        assert "latency" in result.lower()

    def test_cb6_latency_reset_on_good(self, cbs):
        cbs.check_latency(6.0)
        cbs.check_latency(6.0)
        cbs.check_latency(1.0)  # Reset
        result = cbs.check_latency(6.0)
        assert result is None  # Only 1 violation after reset

    def test_cb7_balance_mismatch(self, cbs):
        result = cbs.check_balance_mismatch(500.0, 480.0)
        assert result is not None
        assert "mismatch" in result.lower()

    def test_cb7_balance_ok(self, cbs):
        result = cbs.check_balance_mismatch(500.0, 498.0)
        assert result is None

    def test_cb8_commission_spike(self, cbs):
        result = cbs.check_commission_spike(6.0, 500.0)  # 1.2%
        assert result is not None
        assert "Commission" in result

    def test_permanent_stop_after_3_trips(self, cbs):
        cbs.check_price_anomaly(6.0)
        cbs._breakers["CB-1"].is_tripped = False  # Manual reset
        cbs.check_price_anomaly(7.0)
        cbs._breakers["CB-1"].is_tripped = False
        cbs.check_price_anomaly(8.0)
        assert cbs._breakers["CB-1"].permanent_stop is True
        assert cbs.any_permanent is True

    def test_reset_daily(self, cbs):
        cbs.check_price_anomaly(10.0)
        cbs.reset_daily()
        assert cbs.is_trading_allowed() is True
        assert not cbs.any_permanent

    def test_get_active_breakers(self, cbs):
        assert cbs.get_active_breakers() == []
        cbs.check_price_anomaly(6.0)
        active = cbs.get_active_breakers()
        assert "CB-1" in active


# ──────────────────────────────────────────────
# Kill Switch
# ──────────────────────────────────────────────

class TestKillSwitch:
    @pytest.fixture
    def bus(self):
        return EventBus()

    @pytest.fixture
    def ks(self, bus):
        return KillSwitch(event_bus=bus)

    def test_initial_state(self, ks):
        assert ks.is_activated is False

    @pytest.mark.asyncio
    async def test_activate(self, ks):
        await ks.activate("Test kill")
        assert ks.is_activated is True

    @pytest.mark.asyncio
    async def test_double_activate(self, ks):
        await ks.activate("First")
        await ks.activate("Second")  # Should not error
        assert ks.is_activated is True

    @pytest.mark.asyncio
    async def test_callbacks_called(self, ks):
        closed = []
        cancelled = []
        stopped = []

        async def close_all(): closed.append(True)
        async def cancel_all(): cancelled.append(True)
        async def stop_trading(): stopped.append(True)

        ks.on_close_all_positions = close_all
        ks.on_cancel_all_orders = cancel_all
        ks.on_stop_trading = stop_trading

        await ks.activate("Test")
        assert len(closed) == 1
        assert len(cancelled) == 1
        assert len(stopped) == 1

    @pytest.mark.asyncio
    async def test_event_emitted(self, bus, ks):
        received = []
        async def handler(reason): received.append(reason)
        bus.subscribe("emergency_stop", handler)

        await ks.activate("Test reason")
        assert len(received) == 1
        assert "Test reason" in received[0]

    def test_reset(self, ks):
        ks._activated = True
        ks.reset()
        assert ks.is_activated is False
