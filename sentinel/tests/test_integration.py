"""
Integration tests — full pipeline: data → features → strategy → risk → execution → position.

Tests the complete trading flow end-to-end with real module instances.
"""

import asyncio
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.events import EventBus
from core.models import (
    Candle, Direction, FeatureVector, MarketRegimeType, Order,
    OrderStatus, OrderType, Position, Signal, StrategyTrade,
)
from execution.paper_executor import PaperExecutor
from features.feature_builder import FeatureBuilder
from position.manager import PositionManager
from risk.circuit_breakers import CircuitBreakers
from risk.sentinel import RiskLimits, RiskSentinel
from risk.state_machine import RiskStateMachine
from strategy.ema_crossover_rsi import EMACrossoverRSI, EMAConfig
from strategy.bollinger_breakout import BollingerBreakout
from strategy.mean_reversion import MeanReversion
from strategy.macd_divergence import MACDDivergence

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def make_features(**overrides) -> FeatureVector:
    """Create FeatureVector with reasonable defaults."""
    defaults = dict(
        timestamp=int(time.time() * 1000), symbol="BTCUSDT",
        ema_9=100.0, ema_21=99.0, ema_50=97.0, adx=30.0,
        macd=0.5, macd_signal=0.3, macd_histogram=0.2,
        rsi_14=50.0, stoch_rsi=0.5,
        bb_upper=105.0, bb_middle=100.0, bb_lower=95.0, bb_bandwidth=0.10,
        atr=2.0, volume=1000.0, volume_sma_20=800.0, volume_ratio=1.25,
        obv=50000.0, close=100.0, momentum=0.5, spread=0.01,
        price_change_1m=0.1, price_change_5m=0.3, price_change_15m=-0.5,
        market_regime="trending_up",
    )
    defaults.update(overrides)
    return FeatureVector(**defaults)


# ──────────────────────────────────────────────
# Full Pipeline: Strategy → Risk → Execution → Position
# ──────────────────────────────────────────────


class TestFullPipeline:
    """Integration test: complete buy → hold → sell cycle."""

    @pytest.fixture
    def event_bus(self):
        return EventBus()

    @pytest.fixture
    def position_manager(self, event_bus):
        return PositionManager(event_bus, initial_balance=500.0)

    @pytest.fixture
    def executor(self, event_bus):
        return PaperExecutor(event_bus=event_bus, commission_pct=0.1)

    @pytest.fixture
    def risk_sentinel(self, event_bus):
        sm = RiskStateMachine(event_bus)
        return RiskSentinel(limits=RiskLimits(), state_machine=sm)

    @pytest.fixture
    def circuit_breakers(self):
        return CircuitBreakers()

    @pytest.mark.asyncio
    async def test_buy_sell_cycle(self, event_bus, position_manager, executor, risk_sentinel):
        """Test complete BUY → position open → SELL → position close chain."""
        # 1. Strategy generates BUY signal
        strat = EMACrossoverRSI()

        # Warm up crossover detection
        f1 = make_features(ema_9=99.0, ema_21=100.0, close=99.0)
        sig1 = strat.generate_signal(f1, has_open_position=False)
        assert sig1 is None

        # Trigger crossover — use large enough diff vs ATR
        f2 = make_features(ema_9=101.5, ema_21=100.0, close=101.0, atr=2.0,
                           volume_ratio=1.5, macd_histogram=0.3, adx=28)
        signal = strat.generate_signal(f2, has_open_position=False)
        assert signal is not None
        assert signal.direction == Direction.BUY
        # Threshold relaxed to 0.72 — grouped_confidence now applies
        # correlation-penalty attenuation across groups (more honest score)
        assert signal.confidence >= 0.72

        # Override signal with explicit SL/TP to pass risk checks
        signal = Signal(
            timestamp=signal.timestamp, symbol=signal.symbol,
            direction=signal.direction, confidence=signal.confidence,
            strategy_name=signal.strategy_name, reason=signal.reason,
            stop_loss_price=99.0,       # ~2% SL (< 3% max)
            take_profit_price=106.0,
            suggested_quantity=0.5,
        )

        # 2. Risk check
        check = risk_sentinel.check_signal(
            signal=signal, daily_pnl=0.0,
            open_positions_count=0, total_exposure_pct=0.0,
            balance=500.0, current_market_price=101.0,
        )
        assert check.approved, f"Risk rejected: {check.reason}"

        # 3. Execute order
        order = await executor.execute_order(signal, quantity=0.5, current_price=101.0)
        assert order is not None
        assert order.status == OrderStatus.FILLED

        # 4. Open position
        pos = await position_manager.open_position(order, signal.stop_loss_price, signal.take_profit_price)
        assert pos is not None
        assert position_manager.has_position("BTCUSDT")
        assert position_manager.open_positions_count == 1

        # 5. Update price and check PnL
        await position_manager.update_price("BTCUSDT", 105.0)
        assert position_manager.total_unrealized_pnl > 0

        # 6. Strategy generates SELL signal (take profit scenario)
        f_sell = make_features(ema_9=99.0, ema_21=100.0, close=110.0)
        sell_signal = strat.generate_signal(
            f_sell, has_open_position=True, entry_price=pos.entry_price
        )
        assert sell_signal is not None
        assert sell_signal.direction == Direction.SELL

        # 7. Execute sell
        sell_order = await executor.execute_order(sell_signal, quantity=0.5, current_price=110.0)
        assert sell_order is not None

        # 8. Close position
        closed = await position_manager.close_position(sell_order)
        assert closed is not None
        assert not position_manager.has_position("BTCUSDT")
        assert closed.realized_pnl > 0

    @pytest.mark.asyncio
    async def test_risk_rejection_blocks_execution(self, event_bus, position_manager, executor, risk_sentinel):
        """Test that risk rejection prevents order execution."""
        signal = Signal(
            timestamp=int(time.time() * 1000),
            symbol="BTCUSDT",
            direction=Direction.BUY,
            confidence=0.80,
            strategy_name="ema_crossover_rsi",
            reason="test",
            stop_loss_price=95.0,
            take_profit_price=110.0,
        )

        # Inject heavy daily loss
        check = risk_sentinel.check_signal(
            signal=signal, daily_pnl=-50.0,  # Heavy loss
            open_positions_count=0, total_exposure_pct=0.0,
            balance=450.0, current_market_price=100.0,
        )
        assert not check.approved

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_after_losses(self, circuit_breakers):
        """Test that 5 consecutive losses trip CB-2."""
        for _ in range(4):
            result = circuit_breakers.record_trade_result(False)
            assert result is None  # Not yet tripped
        result = circuit_breakers.record_trade_result(False)
        assert result is not None
        assert not circuit_breakers.is_trading_allowed()

    @pytest.mark.asyncio
    async def test_circuit_breaker_resets_on_win(self, circuit_breakers):
        """Test that wins reset the consecutive loss counter."""
        circuit_breakers.record_trade_result(False)
        circuit_breakers.record_trade_result(False)
        circuit_breakers.record_trade_result(False)
        circuit_breakers.record_trade_result(True)
        circuit_breakers.record_trade_result(False)
        assert circuit_breakers.is_trading_allowed()

    @pytest.mark.asyncio
    async def test_max_positions_respected(self, event_bus, position_manager, executor):
        """Test that max open positions limit is enforced."""
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
        opened = 0

        for sym in symbols:
            signal = Signal(
                timestamp=int(time.time() * 1000), symbol=sym,
                direction=Direction.BUY, confidence=0.80,
                strategy_name="test", reason="test",
                stop_loss_price=95.0, take_profit_price=110.0,
            )
            order = await executor.execute_order(signal, quantity=0.5, current_price=100.0)
            if order:
                pos = await position_manager.open_position(order)
                if pos:
                    opened += 1

        assert opened == 4  # max_open_positions = 4

    @pytest.mark.asyncio
    async def test_stop_loss_detection(self, event_bus, position_manager, executor):
        """Test SL/TP detection in position manager."""
        signal = Signal(
            timestamp=int(time.time() * 1000), symbol="BTCUSDT",
            direction=Direction.BUY, confidence=0.80,
            strategy_name="test", reason="test",
            stop_loss_price=95.0, take_profit_price=110.0,
        )
        order = await executor.execute_order(signal, quantity=0.5, current_price=100.0)
        assert order is not None, "Order should fill (qty*price >= $10)"
        await position_manager.open_position(order, 95.0, 110.0)

        # Price above SL → no trigger
        await position_manager.update_price("BTCUSDT", 98.0)
        assert position_manager.check_stop_loss_take_profit("BTCUSDT") is None

        # Price hits SL
        await position_manager.update_price("BTCUSDT", 94.0)
        assert position_manager.check_stop_loss_take_profit("BTCUSDT") == "stop_loss"

    @pytest.mark.asyncio
    async def test_take_profit_detection(self, event_bus, position_manager, executor):
        """Test TP detection."""
        signal = Signal(
            timestamp=int(time.time() * 1000), symbol="BTCUSDT",
            direction=Direction.BUY, confidence=0.80,
            strategy_name="test", reason="test",
            stop_loss_price=95.0, take_profit_price=110.0,
        )
        order = await executor.execute_order(signal, quantity=0.5, current_price=100.0)
        assert order is not None
        await position_manager.open_position(order, 95.0, 110.0)

        await position_manager.update_price("BTCUSDT", 112.0)
        assert position_manager.check_stop_loss_take_profit("BTCUSDT") == "take_profit"


# ──────────────────────────────────────────────
# Strategy Selection Integration
# ──────────────────────────────────────────────


class TestMultiStrategyIntegration:
    """Test multiple strategies don't conflict."""

    def test_each_strategy_returns_none_on_neutral_market(self):
        """All strategies should return None (HOLD) on neutral data."""
        strategies = [
            EMACrossoverRSI(),
            BollingerBreakout(),
            MeanReversion(),
            MACDDivergence(),
        ]
        neutral = make_features(
            ema_9=100.0, ema_21=100.0, ema_50=100.0,
            rsi_14=50.0, adx=15.0, volume_ratio=1.0,
            close=100.0, bb_upper=105.0, bb_lower=95.0,
        )
        for strat in strategies:
            sig = strat.generate_signal(neutral, has_open_position=False)
            assert sig is None, f"{strat.NAME} should HOLD on neutral data"

    def test_strategies_have_unique_names(self):
        """No two strategies share the same NAME."""
        strategies = [
            EMACrossoverRSI(),
            BollingerBreakout(),
            MeanReversion(),
            MACDDivergence(),
        ]
        names = [s.NAME for s in strategies]
        assert len(names) == len(set(names)), f"Duplicate strategy names: {names}"


# ──────────────────────────────────────────────
# Risk Pipeline Integration
# ──────────────────────────────────────────────


class TestRiskPipelineIntegration:
    """Test risk modules work together."""

    @pytest.mark.asyncio
    async def test_state_machine_transitions(self):
        sm = RiskStateMachine(EventBus())
        assert sm.state.value == "NORMAL"

        # Simulate daily loss
        await sm.update(daily_pnl=-3.0)
        # After -3 USD daily loss, should still be NORMAL (threshold is 50 USD)

    def test_circuit_breakers_independence(self):
        """Circuit breakers should work independently."""
        cbs = CircuitBreakers()

        # Trip CB-1 (price anomaly)
        cbs.check_price_anomaly(6.0)
        assert not cbs.is_trading_allowed()

        # CB-2 should still be untouched
        breakers = cbs.get_active_breakers()
        assert "CB-1" in breakers
        assert "CB-2" not in breakers


# ──────────────────────────────────────────────
# ML Pipeline Integration
# ──────────────────────────────────────────────


class TestMLPipelineIntegration:
    """Test ML predictor with proper data flow."""

    def test_extract_features_chronological_safety(self):
        """Verify that out-of-order trades are properly filtered."""
        from analyzer.ml_predictor import MLPredictor

        ml = MLPredictor()

        # Create trades with explicit timestamps
        trade_past = StrategyTrade(
            trade_id="t1", symbol="BTCUSDT", strategy_name="ema_crossover_rsi",
            market_regime="trending_up",
            timestamp_open="2026-01-01T08:00:00", timestamp_close="2026-01-01T09:00:00",
            entry_price=100.0, exit_price=101.0, quantity=0.01,
            pnl_usd=0.01, pnl_pct=1.0, is_win=True, confidence=0.80,
            rsi_at_entry=45.0, adx_at_entry=30.0, volume_ratio_at_entry=1.5,
        )
        trade_future = StrategyTrade(
            trade_id="t2", symbol="BTCUSDT", strategy_name="ema_crossover_rsi",
            market_regime="trending_up",
            timestamp_open="2026-01-01T15:00:00", timestamp_close="2026-01-01T16:00:00",
            entry_price=102.0, exit_price=100.0, quantity=0.01,
            pnl_usd=-0.02, pnl_pct=-2.0, is_win=False, confidence=0.75,
            rsi_at_entry=60.0, adx_at_entry=25.0, volume_ratio_at_entry=1.2,
        )
        current = StrategyTrade(
            trade_id="t3", symbol="BTCUSDT", strategy_name="ema_crossover_rsi",
            market_regime="trending_up",
            timestamp_open="2026-01-01T12:00:00", timestamp_close="2026-01-01T13:00:00",
            entry_price=101.0, exit_price=102.0, quantity=0.01,
            pnl_usd=0.01, pnl_pct=1.0, is_win=True, confidence=0.82,
            rsi_at_entry=48.0, adx_at_entry=28.0, volume_ratio_at_entry=1.3,
        )

        # Pass unordered list — future trade should be filtered out
        features = ml.extract_features(current, [trade_past, trade_future])
        assert len(features) == 32

        # recent_win_rate_10 is at index 13 in 32-feature vector
        # With only 1 prior trade, extract_features uses default 0.5 at idx=0
        assert features[13] == pytest.approx(0.5, abs=0.01)

    def test_prediction_without_model_allows(self):
        """ML predictor without trained model should allow all signals."""
        from analyzer.ml_predictor import MLPredictor

        ml = MLPredictor()
        pred = ml.predict([0.0] * 32)
        assert pred.decision == "allow"


# ──────────────────────────────────────────────
# Concurrency Safety
# ──────────────────────────────────────────────


class TestConcurrencySafety:
    """Test concurrent access patterns."""

    @pytest.mark.asyncio
    async def test_concurrent_position_opens_blocked(self):
        """Two concurrent opens for the same symbol — only one should succeed."""
        bus = EventBus()
        pm = PositionManager(bus, initial_balance=1000.0, max_open_positions=4)
        executor = PaperExecutor(event_bus=bus, commission_pct=0.1)

        signal = Signal(
            timestamp=int(time.time() * 1000), symbol="BTCUSDT",
            direction=Direction.BUY, confidence=0.80,
            strategy_name="test", reason="test",
        )

        order1 = await executor.execute_order(signal, quantity=0.5, current_price=100.0)
        order2 = await executor.execute_order(signal, quantity=0.5, current_price=100.0)

        results = await asyncio.gather(
            pm.open_position(order1),
            pm.open_position(order2),
        )

        successful = [r for r in results if r is not None]
        assert len(successful) == 1, "Only one position should open for same symbol"

    @pytest.mark.asyncio
    async def test_position_manager_balance_consistency(self):
        """Balance should never go negative after operations."""
        bus = EventBus()
        pm = PositionManager(bus, initial_balance=500.0, max_open_positions=4)
        executor = PaperExecutor(event_bus=bus, commission_pct=0.1)

        for sym in ["BTCUSDT", "ETHUSDT", "BNBUSDT"]:
            signal = Signal(
                timestamp=int(time.time() * 1000), symbol=sym,
                direction=Direction.BUY, confidence=0.80,
                strategy_name="test", reason="test",
            )
            order = await executor.execute_order(signal, quantity=0.5, current_price=100.0)
            assert order is not None
            await pm.open_position(order)

        assert pm.wallet.usdt_balance >= 0
        assert pm.balance > 0
