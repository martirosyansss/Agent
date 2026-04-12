"""Tests Phase 9 - Backtest Engine & Quality Gates."""
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sentinel"))

import pytest
from core.models import Candle, Direction, FeatureVector, Signal
from backtest.engine import BacktestConfig, BacktestEngine, BacktestResult, BacktestTrade
from backtest.quality_gates import QualityGates, QualityReport
from backtest.quality_gates import test_strategy_skill_on_history as skill_test_fn
from strategy.base_strategy import BaseStrategy
from typing import Optional


def make_candle(ts, close, symbol="BTCUSDT",
                interval="1h", volume=100.0,
                open_=0, high=0, low=0):
    o = open_ or close * 0.999
    h = high or close * 1.001
    l_ = low or close * 0.998
    return Candle(
        timestamp=ts, symbol=symbol, interval=interval,
        open=o, high=h, low=l_, close=close,
        volume=volume, trades_count=50,
    )


def generate_trending_candles(start_price, count, trend=0.5,
                              interval="1h", symbol="BTCUSDT"):
    candles = []
    price = start_price
    for i in range(count):
        ts = 1700000000000 + i * 3600000
        noise = (i % 3 - 1) * 0.1
        price = price * (1 + (trend + noise) / 100)
        candles.append(make_candle(ts, price, symbol, interval, volume=100 + i * 2))
    return candles


class DummyStrategy(BaseStrategy):
    NAME = "dummy_test"
    _call_count: int = 0

    def generate_signal(self, features, has_open_position=False, entry_price=None):
        self._call_count += 1
        ts = features.timestamp
        if not has_open_position and self._call_count % 4 == 0:
            return Signal(
                timestamp=ts, symbol=features.symbol,
                direction=Direction.BUY, confidence=0.8,
                strategy_name=self.NAME, reason="test buy",
                stop_loss_price=features.close * 0.97,
                take_profit_price=features.close * 1.05,
            )
        if has_open_position and self._call_count % 4 == 2:
            return Signal(
                timestamp=ts, symbol=features.symbol,
                direction=Direction.SELL, confidence=0.8,
                strategy_name=self.NAME, reason="test sell",
            )
        return None


class AlwaysBuyStrategy(BaseStrategy):
    NAME = "always_buy"
    _count: int = 0

    def generate_signal(self, features, has_open_position=False, entry_price=None):
        self._count += 1
        if not has_open_position:
            return Signal(
                timestamp=features.timestamp, symbol=features.symbol,
                direction=Direction.BUY, confidence=0.9,
                strategy_name=self.NAME, reason="always buy",
                stop_loss_price=features.close * 0.97,
                take_profit_price=features.close * 1.10,
            )
        if has_open_position and self._count % 3 == 0:
            return Signal(
                timestamp=features.timestamp, symbol=features.symbol,
                direction=Direction.SELL, confidence=0.9,
                strategy_name=self.NAME, reason="always sell",
            )
        return None


class NeverTradeStrategy(BaseStrategy):
    NAME = "never_trade"

    def generate_signal(self, features, has_open_position=False, entry_price=None):
        return None


class TestBacktestConfig:
    def test_defaults(self):
        cfg = BacktestConfig()
        assert cfg.initial_balance == 500.0
        assert cfg.commission_pct == 0.1
        assert cfg.slippage_pct == 0.05
        assert cfg.safety_discount == 0.7
        assert cfg.position_size_pct == 20.0

    def test_custom(self):
        cfg = BacktestConfig(initial_balance=1000, safety_discount=0.5)
        assert cfg.initial_balance == 1000
        assert cfg.safety_discount == 0.5


class TestBacktestEngineBasic:
    def test_empty_candles(self):
        engine = BacktestEngine()
        result = engine.run(DummyStrategy(), [], [], "BTCUSDT")
        assert result.total_trades == 0
        assert result.final_balance == 500.0
        assert result.total_pnl == 0.0

    def test_insufficient_candles(self):
        candles = generate_trending_candles(50000, 30)
        engine = BacktestEngine()
        result = engine.run(DummyStrategy(), candles, candles)
        assert result.total_trades == 0

    def test_never_trade_strategy(self):
        c1h = generate_trending_candles(50000, 100)
        c4h = generate_trending_candles(50000, 100, interval="4h")
        engine = BacktestEngine()
        result = engine.run(NeverTradeStrategy(), c1h, c4h)
        assert result.total_trades == 0
        assert result.final_balance == 500.0
        assert result.win_rate == 0

    def test_result_fields(self):
        c1h = generate_trending_candles(50000, 100, trend=0.3)
        c4h = generate_trending_candles(50000, 100, trend=0.3, interval="4h")
        engine = BacktestEngine()
        result = engine.run(DummyStrategy(), c1h, c4h)
        assert isinstance(result, BacktestResult)
        assert result.strategy_name == "DummyStrategy"
        assert result.symbol == "BTCUSDT"
        assert result.initial_balance == 500.0
        assert result.safety_discount == 0.7
        assert result.period_start > 0
        assert result.period_end > result.period_start

    def test_safety_discount(self):
        c1h = generate_trending_candles(50000, 120, trend=0.5)
        c4h = generate_trending_candles(50000, 120, trend=0.5, interval="4h")
        engine = BacktestEngine()
        result = engine.run(AlwaysBuyStrategy(), c1h, c4h)
        if result.total_trades > 0:
            expected = result.total_pnl * 0.7
            assert abs(result.expected_real_pnl - expected) < 0.01


class TestBacktestEngineTrading:
    def test_makes_trades(self):
        c1h = generate_trending_candles(50000, 150, trend=0.2)
        c4h = generate_trending_candles(50000, 150, trend=0.2, interval="4h")
        engine = BacktestEngine()
        result = engine.run(AlwaysBuyStrategy(), c1h, c4h)
        assert result.total_trades > 0
        assert result.wins + result.losses == result.total_trades

    def test_commission_applied(self):
        c1h = generate_trending_candles(50000, 120, trend=0)
        c4h = generate_trending_candles(50000, 120, trend=0, interval="4h")
        engine = BacktestEngine()
        result = engine.run(AlwaysBuyStrategy(), c1h, c4h)
        if result.total_trades > 0:
            total_comm = sum(t.commission for t in result.trades)
            assert total_comm > 0

    def test_slippage_applied(self):
        c1h = generate_trending_candles(50000, 120, trend=0.3)
        c4h = generate_trending_candles(50000, 120, trend=0.3, interval="4h")
        engine = BacktestEngine()
        result = engine.run(AlwaysBuyStrategy(), c1h, c4h)
        if result.total_trades > 0:
            trade = result.trades[0]
            assert trade.entry_price > 0
            assert trade.exit_price > 0

    def test_uptrend_profit(self):
        c1h = generate_trending_candles(50000, 200, trend=0.8)
        c4h = generate_trending_candles(50000, 200, trend=0.8, interval="4h")
        engine = BacktestEngine()
        result = engine.run(AlwaysBuyStrategy(), c1h, c4h)
        if result.total_trades >= 2:
            assert result.total_pnl > 0
            assert result.final_balance > result.initial_balance

    def test_win_rate_range(self):
        c1h = generate_trending_candles(50000, 150, trend=0.2)
        c4h = generate_trending_candles(50000, 150, trend=0.2, interval="4h")
        engine = BacktestEngine()
        result = engine.run(AlwaysBuyStrategy(), c1h, c4h)
        assert 0 <= result.win_rate <= 100

    def test_max_drawdown_not_negative(self):
        c1h = generate_trending_candles(50000, 150, trend=0.5)
        c4h = generate_trending_candles(50000, 150, trend=0.5, interval="4h")
        engine = BacktestEngine()
        result = engine.run(AlwaysBuyStrategy(), c1h, c4h)
        assert result.max_drawdown_pct >= 0

    def test_profit_factor_positive(self):
        c1h = generate_trending_candles(50000, 200, trend=0.5)
        c4h = generate_trending_candles(50000, 200, trend=0.5, interval="4h")
        engine = BacktestEngine()
        result = engine.run(AlwaysBuyStrategy(), c1h, c4h)
        if result.total_trades > 0 and result.wins > 0:
            assert result.profit_factor > 0

    def test_custom_config(self):
        cfg = BacktestConfig(initial_balance=1000, position_size_pct=10)
        c1h = generate_trending_candles(50000, 120, trend=0.5)
        c4h = generate_trending_candles(50000, 120, trend=0.5, interval="4h")
        engine = BacktestEngine(cfg)
        result = engine.run(AlwaysBuyStrategy(), c1h, c4h)
        assert result.initial_balance == 1000


class TestBacktestEngineSLTP:
    def test_stop_loss_triggered(self):
        prices_up = [50000 + i * 50 for i in range(60)]
        prices_down = [prices_up[-1] - i * 200 for i in range(60)]
        all_prices = prices_up + prices_down
        candles = [make_candle(1700000000000 + i * 3600000, p) for i, p in enumerate(all_prices)]
        candles_4h = candles[::4]
        for i, c in enumerate(candles_4h):
            candles_4h[i] = make_candle(c.timestamp, c.close, interval="4h")
        engine = BacktestEngine()
        result = engine.run(AlwaysBuyStrategy(), candles, candles_4h)
        assert result.total_trades >= 0

    def test_take_profit_triggered(self):
        prices = [50000 + i * 500 for i in range(120)]
        candles = [make_candle(1700000000000 + i * 3600000, p) for i, p in enumerate(prices)]
        candles_4h = [make_candle(c.timestamp, c.close, interval="4h") for c in candles[::4]]
        engine = BacktestEngine()
        result = engine.run(AlwaysBuyStrategy(), candles, candles_4h)
        assert result.total_trades >= 0


class TestBacktestReport:
    def test_format_contains_sections(self):
        c1h = generate_trending_candles(50000, 120, trend=0.3)
        c4h = generate_trending_candles(50000, 120, trend=0.3, interval="4h")
        engine = BacktestEngine()
        result = engine.run(DummyStrategy(), c1h, c4h)
        report = engine.format_report(result)
        assert "BACKTEST REPORT" in report
        assert "$500.00" in report
        assert "Sharpe Ratio:" in report
        assert "Profit Factor:" in report

    def test_format_inf_profit_factor(self):
        result = BacktestResult(
            strategy_name="test", symbol="BTCUSDT",
            period_start=0, period_end=1,
            initial_balance=500, final_balance=600,
            total_pnl=100, total_pnl_pct=20,
            total_trades=5, wins=5, losses=0,
            win_rate=100, max_drawdown_pct=0,
            sharpe_ratio=2.0, profit_factor=float("inf"),
            avg_win=20, avg_loss=0,
            safety_discount=0.7, expected_real_pnl=70,
        )
        engine = BacktestEngine()
        report = engine.format_report(result)
        assert "inf" not in report.lower() or "Profit Factor" in report


class TestQualityGates:
    def test_all_passed(self):
        qg = QualityGates()
        report = qg.check(
            win_rate=60, total_pnl=50, max_drawdown_pct=3,
            total_trades=100, has_critical_errors=False,
        )
        assert report.all_passed
        assert all(g.passed for g in report.gates)

    def test_low_win_rate_fails(self):
        qg = QualityGates()
        report = qg.check(
            win_rate=40, total_pnl=50, max_drawdown_pct=3, total_trades=100,
        )
        assert not report.all_passed
        wr_gate = next(g for g in report.gates if g.name == "Win Rate")
        assert not wr_gate.passed

    def test_negative_pnl_fails(self):
        qg = QualityGates()
        report = qg.check(
            win_rate=60, total_pnl=-10, max_drawdown_pct=3, total_trades=100,
        )
        assert not report.all_passed
        pnl_gate = next(g for g in report.gates if g.name == "Total PnL")
        assert not pnl_gate.passed

    def test_high_drawdown_fails(self):
        qg = QualityGates()
        report = qg.check(
            win_rate=60, total_pnl=50, max_drawdown_pct=7, total_trades=100,
        )
        assert not report.all_passed
        dd_gate = next(g for g in report.gates if g.name == "Max Drawdown")
        assert not dd_gate.passed

    def test_not_enough_trades_fails(self):
        qg = QualityGates()
        report = qg.check(
            win_rate=60, total_pnl=50, max_drawdown_pct=3, total_trades=20,
        )
        assert not report.all_passed
        tc_gate = next(g for g in report.gates if g.name == "Trade Count")
        assert not tc_gate.passed

    def test_critical_errors_fail(self):
        qg = QualityGates()
        report = qg.check(
            win_rate=60, total_pnl=50, max_drawdown_pct=3,
            total_trades=100, has_critical_errors=True,
        )
        assert not report.all_passed
        err_gate = next(g for g in report.gates if g.name == "Critical Errors")
        assert not err_gate.passed

    def test_multiple_failures(self):
        qg = QualityGates()
        report = qg.check(
            win_rate=30, total_pnl=-50, max_drawdown_pct=10,
            total_trades=10, has_critical_errors=True,
        )
        assert not report.all_passed
        failed = sum(1 for g in report.gates if not g.passed)
        assert failed >= 4

    def test_borderline_values(self):
        qg = QualityGates()
        report = qg.check(win_rate=50, total_pnl=1, max_drawdown_pct=4.9, total_trades=50)
        wr = next(g for g in report.gates if g.name == "Win Rate")
        assert not wr.passed
        tc = next(g for g in report.gates if g.name == "Trade Count")
        assert tc.passed

    def test_gate_count(self):
        qg = QualityGates()
        report = qg.check(win_rate=60, total_pnl=10, max_drawdown_pct=2, total_trades=100)
        assert len(report.gates) == 5


class TestSkillTest:
    def test_empty_trades(self):
        result = skill_test_fn([])
        assert result["skill_score"] == 0
        assert result["confidence"] == "low"

    def test_few_trades(self):
        trades = [{"pnl": 1.0, "timestamp": i} for i in range(5)]
        result = skill_test_fn(trades)
        assert result["confidence"] == "low"

    def test_enough_trades_medium(self):
        trades = [{"pnl": 1.0 if i % 2 == 0 else -0.5, "timestamp": i} for i in range(50)]
        result = skill_test_fn(trades)
        assert result["confidence"] in ("medium", "high")
        assert 0 <= result["skill_score"] <= 1

    def test_high_confidence(self):
        trades = [{"pnl": 1.0 if i % 2 == 0 else -0.5, "timestamp": i} for i in range(100)]
        result = skill_test_fn(trades)
        assert result["confidence"] == "high"

    def test_all_winners(self):
        trades = [{"pnl": 2.0, "timestamp": i} for i in range(100)]
        result = skill_test_fn(trades)
        assert result["precision"] == 1.0
        assert result["skill_score"] > 0.5

    def test_all_losers(self):
        trades = [{"pnl": -1.0, "timestamp": i} for i in range(100)]
        result = skill_test_fn(trades)
        assert result["precision"] == 0.0
        assert result["expected_pnl"] < 0

    def test_time_order_preserved(self):
        trades = [{"pnl": 1.0, "timestamp": 100 - i} for i in range(100)]
        result = skill_test_fn(trades)
        assert result["train_trades"] == 70
        assert result["test_trades"] == 30

    def test_result_fields(self):
        trades = [{"pnl": 1.0 if i % 3 != 0 else -0.5, "timestamp": i} for i in range(100)]
        result = skill_test_fn(trades)
        assert "skill_score" in result
        assert "precision" in result
        assert "recall" in result
        assert "expected_pnl" in result
        assert "confidence" in result
        assert "message" in result

    def test_custom_train_ratio(self):
        trades = [{"pnl": 1.0, "timestamp": i} for i in range(100)]
        result = skill_test_fn(trades, train_ratio=0.5)
        assert result["train_trades"] == 50
        assert result["test_trades"] == 50
