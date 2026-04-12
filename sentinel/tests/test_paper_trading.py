"""Тесты Phase 7 — Paper Executor + Position Manager."""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.events import EventBus
from core.models import Direction, Order, OrderStatus, OrderType, PositionStatus, Signal
from execution.paper_executor import PaperExecutor
from position.manager import PaperWallet, PositionManager


# ──────────────────────────────────────────────
# PaperWallet
# ──────────────────────────────────────────────

class TestPaperWallet:
    def test_default_balance(self):
        w = PaperWallet()
        assert w.initial_balance == 500.0
        assert w.usdt_balance == 500.0
        assert w.available_usd() == 500.0

    def test_custom_balance(self):
        w = PaperWallet(initial_balance=1000, usdt_balance=1000)
        assert w.available_usd() == 1000.0


# ──────────────────────────────────────────────
# PaperExecutor
# ──────────────────────────────────────────────

class TestPaperExecutor:
    @pytest.fixture
    def bus(self):
        return EventBus()

    @pytest.fixture
    def executor(self, bus):
        return PaperExecutor(event_bus=bus, commission_pct=0.1)

    def _make_signal(self, direction: Direction = Direction.BUY, symbol: str = "BTCUSDT") -> Signal:
        return Signal(
            timestamp=1700000000000,
            symbol=symbol,
            direction=direction,
            confidence=0.85,
            strategy_name="ema_crossover_rsi",
            reason="Test signal",
        )

    @pytest.mark.asyncio
    async def test_buy_order(self, executor):
        signal = self._make_signal(Direction.BUY)
        order = await executor.execute_order(signal, quantity=0.001, current_price=67000.0)
        assert order is not None
        assert order.status == OrderStatus.FILLED
        assert order.side == Direction.BUY
        assert order.is_paper is True
        assert order.fill_price > 0
        assert order.fill_quantity == 0.001
        assert order.commission > 0
        assert order.strategy_name == signal.strategy_name
        assert order.signal_id == signal.signal_id
        assert order.signal_reason == signal.reason
        # Проскальзывание: fill_price примерно = 67000
        assert abs(order.fill_price - 67000) / 67000 < 0.001

    @pytest.mark.asyncio
    async def test_buy_order_carries_stop_levels(self, executor):
        signal = self._make_signal(Direction.BUY)
        signal.stop_loss_price = 65000.0
        signal.take_profit_price = 70350.0

        order = await executor.execute_order(signal, quantity=0.001, current_price=67000.0)

        assert order is not None
        assert order.stop_loss_price == 65000.0
        assert order.take_profit_price == 70350.0

    @pytest.mark.asyncio
    async def test_sell_order(self, executor):
        signal = self._make_signal(Direction.SELL)
        order = await executor.execute_order(signal, quantity=0.001, current_price=67000.0)
        assert order is not None
        assert order.side == Direction.SELL
        assert order.fill_price > 0
        # При продаже проскальзывание чуть вниз
        assert order.fill_price <= 67000.0 * 1.001

    @pytest.mark.asyncio
    async def test_order_too_small(self, executor):
        signal = self._make_signal()
        order = await executor.execute_order(signal, quantity=0.0001, current_price=67000.0)
        # $6.70 < $10 min
        assert order is None

    @pytest.mark.asyncio
    async def test_invalid_quantity(self, executor):
        signal = self._make_signal()
        order = await executor.execute_order(signal, quantity=0, current_price=67000.0)
        assert order is None

    @pytest.mark.asyncio
    async def test_commission_calculated(self, executor):
        signal = self._make_signal()
        order = await executor.execute_order(signal, quantity=0.01, current_price=67000.0)
        assert order is not None
        # Commission = 0.01 * ~67000 * 0.1% ≈ $0.67
        assert 0.5 < order.commission < 0.8

    @pytest.mark.asyncio
    async def test_event_emitted(self, bus, executor):
        received = []
        async def handler(order):
            received.append(order)
        bus.subscribe("order_filled", handler)

        signal = self._make_signal()
        await executor.execute_order(signal, quantity=0.001, current_price=67000.0)
        assert len(received) == 1
        assert received[0].status == OrderStatus.FILLED


# ──────────────────────────────────────────────
# PositionManager
# ──────────────────────────────────────────────

class TestPositionManager:
    @pytest.fixture
    def bus(self):
        return EventBus()

    @pytest.fixture
    def pm(self, bus):
        return PositionManager(event_bus=bus, initial_balance=500.0, max_open_positions=4)

    def _make_buy_order(self, symbol: str = "BTCUSDT", qty: float = 0.001, price: float = 67000.0) -> Order:
        return Order(
            timestamp=1700000000000,
            symbol=symbol,
            side=Direction.BUY,
            order_type=OrderType.MARKET,
            quantity=qty,
            fill_price=price,
            fill_quantity=qty,
            commission=qty * price * 0.001,
            status=OrderStatus.FILLED,
            is_paper=True,
        )

    def _make_sell_order(self, symbol: str = "BTCUSDT", qty: float = 0.001, price: float = 68000.0) -> Order:
        return Order(
            timestamp=1700000000000,
            symbol=symbol,
            side=Direction.SELL,
            order_type=OrderType.MARKET,
            quantity=qty,
            fill_price=price,
            fill_quantity=qty,
            commission=qty * price * 0.001,
            status=OrderStatus.FILLED,
            is_paper=True,
        )

    @pytest.mark.asyncio
    async def test_open_position(self, pm):
        order = self._make_buy_order()
        pos = await pm.open_position(order, stop_loss_price=64990.0, take_profit_price=70350.0)
        assert pos is not None
        assert pos.symbol == "BTCUSDT"
        assert pos.entry_price == 67000.0
        assert pos.stop_loss_price == 64990.0
        assert pos.take_profit_price == 70350.0
        assert pos.status == PositionStatus.OPEN
        assert pm.open_positions_count == 1
        assert pm.has_position("BTCUSDT")

    @pytest.mark.asyncio
    async def test_open_position_uses_order_signal_context(self, pm):
        order = self._make_buy_order()
        order.strategy_name = "ema_crossover_rsi"
        order.signal_id = "sig-123"
        order.signal_reason = "EMA crossover"
        order.stop_loss_price = 64990.0
        order.take_profit_price = 70350.0

        pos = await pm.open_position(order)

        assert pos is not None
        assert pos.strategy_name == "ema_crossover_rsi"
        assert pos.signal_id == "sig-123"
        assert pos.signal_reason == "EMA crossover"
        assert pos.stop_loss_price == 64990.0
        assert pos.take_profit_price == 70350.0

    @pytest.mark.asyncio
    async def test_balance_reduced_on_open(self, pm):
        initial = pm.wallet.usdt_balance
        order = self._make_buy_order(qty=0.001, price=67000.0)
        await pm.open_position(order)
        cost = 0.001 * 67000.0 + order.commission  # ~$67.067
        assert pm.wallet.usdt_balance == pytest.approx(initial - cost, abs=0.01)

    @pytest.mark.asyncio
    async def test_close_position(self, pm):
        buy = self._make_buy_order(qty=0.001, price=67000.0)
        await pm.open_position(buy)

        sell = self._make_sell_order(qty=0.001, price=68000.0)
        pos = await pm.close_position(sell)
        assert pos is not None
        assert pos.status == PositionStatus.CLOSED
        # PnL = (68000 - 67000) * 0.001 - commission = $1.0 - ~$0.068 ≈ $0.932
        assert pos.realized_pnl > 0
        assert pm.open_positions_count == 0
        assert not pm.has_position("BTCUSDT")

    @pytest.mark.asyncio
    async def test_close_nonexistent_position(self, pm):
        sell = self._make_sell_order()
        pos = await pm.close_position(sell)
        assert pos is None

    @pytest.mark.asyncio
    async def test_max_positions(self, pm):
        for i, symbol in enumerate(["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]):
            order = self._make_buy_order(symbol=symbol, qty=0.001, price=100.0)
            await pm.open_position(order)
        assert pm.open_positions_count == 4

        # 5-й должен быть отклонён
        order5 = self._make_buy_order(symbol="ADAUSDT", qty=0.001, price=100.0)
        pos5 = await pm.open_position(order5)
        assert pos5 is None

    @pytest.mark.asyncio
    async def test_duplicate_position_rejected(self, pm):
        order1 = self._make_buy_order()
        await pm.open_position(order1)

        order2 = self._make_buy_order()
        pos2 = await pm.open_position(order2)
        assert pos2 is None

    @pytest.mark.asyncio
    async def test_insufficient_balance(self, pm):
        # Попытка купить на $600 при балансе $500
        order = self._make_buy_order(qty=0.01, price=60000.0)  # $600
        pos = await pm.open_position(order)
        assert pos is None

    def test_update_price(self, pm):
        # Нет позиции — не падает
        pm.update_price("BTCUSDT", 68000.0)

    @pytest.mark.asyncio
    async def test_update_price_unrealized_pnl(self, pm):
        buy = self._make_buy_order(qty=0.001, price=67000.0)
        await pm.open_position(buy)

        pm.update_price("BTCUSDT", 68000.0)
        pos = pm.get_position("BTCUSDT")
        assert pos is not None
        assert pos.current_price == 68000.0
        # unrealized_pnl = (68000 - 67000) * 0.001 = $1.0
        assert pos.unrealized_pnl == pytest.approx(1.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_stop_loss_trigger(self, pm):
        buy = self._make_buy_order(qty=0.001, price=67000.0)
        await pm.open_position(buy, stop_loss_price=64990.0, take_profit_price=70350.0)

        # Цена выше SL — нет триггера
        pm.update_price("BTCUSDT", 66000.0)
        result = pm.check_stop_loss_take_profit("BTCUSDT")
        assert result is None

        # Цена ниже SL
        pm.update_price("BTCUSDT", 64900.0)
        result = pm.check_stop_loss_take_profit("BTCUSDT")
        assert result == "stop_loss"

    @pytest.mark.asyncio
    async def test_take_profit_trigger(self, pm):
        buy = self._make_buy_order(qty=0.001, price=67000.0)
        await pm.open_position(buy, stop_loss_price=64990.0, take_profit_price=70350.0)

        pm.update_price("BTCUSDT", 70400.0)
        result = pm.check_stop_loss_take_profit("BTCUSDT")
        assert result == "take_profit"

    @pytest.mark.asyncio
    async def test_daily_stats(self, pm):
        buy = self._make_buy_order(qty=0.001, price=67000.0)
        await pm.open_position(buy)

        sell = self._make_sell_order(qty=0.001, price=68000.0)
        await pm.close_position(sell)

        stats = pm.get_daily_stats()
        assert stats["trades_today"] == 1
        assert stats["wins"] == 1
        assert stats["losses"] == 0
        assert stats["win_rate"] == 100.0

    @pytest.mark.asyncio
    async def test_reset_daily_stats(self, pm):
        buy = self._make_buy_order(qty=0.001, price=67000.0)
        await pm.open_position(buy)
        sell = self._make_sell_order(qty=0.001, price=68000.0)
        await pm.close_position(sell)

        pm.reset_daily_stats()
        stats = pm.get_daily_stats()
        assert stats["trades_today"] == 0

    @pytest.mark.asyncio
    async def test_get_state(self, pm):
        state = pm.get_state()
        assert "balance" in state
        assert "pnl_today" in state
        assert "positions" in state
        assert "win_rate" in state
        assert "pnl_history" in state
        assert state["open_positions"] == 0

    @pytest.mark.asyncio
    async def test_runtime_metrics_track_drawdown(self, pm):
        buy = self._make_buy_order(qty=1.0, price=100.0)
        await pm.open_position(buy)

        pm.update_price("BTCUSDT", 80.0)
        state = pm.get_state()

        assert state["current_drawdown_pct"] > 0
        assert state["max_drawdown_pct"] > 0
        assert state["exposure_pct"] > 0
        assert len(state["pnl_history"]) >= 1

    @pytest.mark.asyncio
    async def test_position_events_emitted(self, bus, pm):
        opened = []
        closed = []
        async def on_open(pos): opened.append(pos)
        async def on_close(pos): closed.append(pos)

        bus.subscribe("position_opened", on_open)
        bus.subscribe("position_closed", on_close)

        buy = self._make_buy_order(qty=0.001, price=67000.0)
        await pm.open_position(buy)
        assert len(opened) == 1

        sell = self._make_sell_order(qty=0.001, price=68000.0)
        await pm.close_position(sell)
        assert len(closed) == 1

    @pytest.mark.asyncio
    async def test_losing_trade(self, pm):
        buy = self._make_buy_order(qty=0.001, price=67000.0)
        await pm.open_position(buy)

        sell = self._make_sell_order(qty=0.001, price=65000.0)  # Убыточная
        pos = await pm.close_position(sell)
        assert pos.realized_pnl < 0

        stats = pm.get_daily_stats()
        assert stats["losses"] == 1
