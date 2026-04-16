"""Tests for new improvement modules: position_sizer, dynamic_sltp, alerts, walk-forward."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ══════════════════════════════════════════════
# POSITION SIZER
# ══════════════════════════════════════════════

from risk.position_sizer import (
    SizingInput,
    SizingResult,
    calculate_position_size,
    kelly_fraction,
    regime_dampener,
    volatility_factor,
)


class TestKellyFraction:
    def test_edge_case_no_wins(self):
        assert kelly_fraction(0.0, 3.0, 2.0) == 0.0

    def test_edge_case_no_loss(self):
        assert kelly_fraction(0.5, 3.0, 0.0) == 0.0

    def test_typical_case(self):
        f = kelly_fraction(0.55, 3.0, 2.0)
        assert 0 < f <= 0.25

    def test_high_win_rate(self):
        f = kelly_fraction(0.8, 4.0, 1.0)
        # Should be capped at 0.25
        assert f == 0.25 or f <= 0.25

    def test_low_edge(self):
        # Win rate 0.4, ratio 1:1 = negative edge → returns 0
        f = kelly_fraction(0.4, 1.0, 1.0)
        assert f == 0.0


class TestVolatilityFactor:
    def test_zero_price(self):
        assert volatility_factor(100, 0) == 1.0

    def test_zero_atr(self):
        assert volatility_factor(0, 50000) == 1.0

    def test_normal_volatility(self):
        # ATR = 750 on price 50000 = 1.5%
        f = volatility_factor(750, 50000, base_atr_pct=1.5)
        assert 0.9 < f < 1.1  # roughly 1.0

    def test_high_volatility_shrinks(self):
        # ATR = 1500 on price 50000 = 3%
        f = volatility_factor(1500, 50000, base_atr_pct=1.5)
        assert f < 1.0

    def test_low_volatility_grows(self):
        # ATR = 375 on price 50000 = 0.75%
        f = volatility_factor(375, 50000, base_atr_pct=1.5)
        assert f > 1.0


class TestRegimeDampener:
    def test_strong_trend(self):
        assert regime_dampener(35) == 1.0

    def test_moderate(self):
        d = regime_dampener(25)
        assert 0.7 < d < 1.0

    def test_weak(self):
        assert regime_dampener(15) == 0.5


class TestCalculatePositionSize:
    def test_zero_balance(self):
        r = calculate_position_size(SizingInput(balance=0, price=50000, atr=750))
        assert r.quantity == 0.0
        assert r.method == "minimum"

    def test_normal_sizing(self):
        inp = SizingInput(
            balance=1000,
            price=50000,
            atr=750,
            win_rate=0.55,
            avg_win_pct=3.0,
            avg_loss_pct=2.0,
            regime_adx=30,
            max_position_pct=20.0,
            max_order_usd=100.0,
        )
        r = calculate_position_size(inp)
        assert r.quantity > 0
        assert r.budget_usd <= 100.0  # max_order_usd cap
        assert r.budget_pct <= 20.0  # max_position_pct cap
        assert r.method in ("kelly_atr", "fixed")

    def test_max_order_cap(self):
        inp = SizingInput(
            balance=10000,
            price=100,
            atr=1.5,
            max_order_usd=50.0,
        )
        r = calculate_position_size(inp)
        assert r.budget_usd <= 50.0


# ══════════════════════════════════════════════
# DYNAMIC SL/TP
# ══════════════════════════════════════════════

from risk.dynamic_sltp import (
    SLTPResult,
    calculate_dynamic_sltp,
    STRATEGY_SLTP_DEFAULTS,
)


class TestDynamicSLTP:
    def test_zero_price(self):
        r = calculate_dynamic_sltp(0, 750, "ema_crossover_rsi")
        assert r.stop_loss_price == 0
        assert r.method == "fixed_fallback"

    def test_no_atr_fallback(self):
        r = calculate_dynamic_sltp(50000, 0, "ema_crossover_rsi",
                                    fallback_sl_pct=3.0, fallback_tp_pct=5.0)
        assert r.method == "fixed_fallback"
        assert r.stop_loss_pct == 3.0
        assert r.take_profit_pct == 5.0

    def test_dynamic_calculation(self):
        r = calculate_dynamic_sltp(50000, 750, "ema_crossover_rsi")
        assert r.method == "atr_dynamic"
        assert r.stop_loss_price < 50000
        assert r.take_profit_price > 50000
        assert r.stop_loss_pct > 0
        assert r.take_profit_pct > 0

    def test_mean_reversion_wider_sl(self):
        r_ema = calculate_dynamic_sltp(50000, 750, "ema_crossover_rsi")
        r_mr = calculate_dynamic_sltp(50000, 750, "mean_reversion")
        # Mean reversion has higher atr_sl_mult (2.5 vs 2.0) → wider SL
        assert r_mr.stop_loss_pct >= r_ema.stop_loss_pct

    def test_unknown_strategy_uses_defaults(self):
        r = calculate_dynamic_sltp(50000, 750, "unknown_strategy")
        assert r.method == "atr_dynamic"
        assert r.stop_loss_price > 0

    def test_sl_clamped_to_limits(self):
        # Very high ATR → should clamp to max SL
        r = calculate_dynamic_sltp(50000, 5000, "ema_crossover_rsi")
        cfg = STRATEGY_SLTP_DEFAULTS["ema_crossover_rsi"]
        assert r.stop_loss_pct <= cfg.max_sl_pct
        assert r.take_profit_pct <= cfg.max_tp_pct


# ══════════════════════════════════════════════
# ALERT MONITOR
# ══════════════════════════════════════════════

from monitoring.alerts import AlertMonitor, Alert


class TestAlertMonitor:
    def test_price_gap_detected(self):
        m = AlertMonitor(price_gap_pct=2.0)
        m._last_price = 50000
        alert = m.check_price_gap(51100)  # 2.2% gap
        assert alert is not None
        assert alert.category == "price_gap"
        assert "UP" in alert.message

    def test_price_gap_not_triggered(self):
        m = AlertMonitor(price_gap_pct=2.0)
        m._last_price = 50000
        alert = m.check_price_gap(50500)  # 1.0% gap
        assert alert is None

    def test_first_price_no_alert(self):
        m = AlertMonitor()
        alert = m.check_price_gap(50000)
        assert alert is None

    def test_execution_latency_alert(self):
        m = AlertMonitor(max_latency_sec=2.0)
        alert = m.check_execution_latency(1000, 4000)  # 3s > 2s
        assert alert is not None
        assert alert.category == "latency"

    def test_execution_latency_ok(self):
        m = AlertMonitor(max_latency_sec=5.0)
        alert = m.check_execution_latency(1000, 3000)  # 2s < 5s
        assert alert is None

    def test_rejection_streak(self):
        m = AlertMonitor(rejection_threshold=3)
        assert m.record_signal_rejection("risk") is None
        assert m.record_signal_rejection("risk") is None
        alert = m.record_signal_rejection("risk")
        assert alert is not None
        assert alert.category == "rejection"

    def test_rejection_reset(self):
        m = AlertMonitor(rejection_threshold=3)
        m.record_signal_rejection("risk")
        m.record_signal_rejection("risk")
        m.record_signal_accepted()
        assert m._rejection_count == 0

    def test_loss_streak(self):
        m = AlertMonitor(loss_streak_threshold=2)
        m.record_trade_result(False)
        alert = m.record_trade_result(False)
        assert alert is not None
        assert alert.category == "loss_streak"

    def test_loss_streak_reset_on_win(self):
        m = AlertMonitor(loss_streak_threshold=3)
        m.record_trade_result(False)
        m.record_trade_result(True)
        assert m._loss_streak == 0

    def test_get_recent_alerts(self):
        m = AlertMonitor(price_gap_pct=0.1)
        m._last_price = 100
        m.check_price_gap(110)
        m.check_price_gap(90)
        alerts = m.get_recent_alerts()
        assert len(alerts) >= 2
        assert all("message" in a for a in alerts)

    def test_stats(self):
        m = AlertMonitor()
        s = m.stats
        assert "total_alerts" in s
        assert "rejection_count" in s


# ══════════════════════════════════════════════
# WALK-FORWARD
# ══════════════════════════════════════════════

from backtest.engine import BacktestEngine, BacktestConfig
from core.models import Candle


class TestWalkForward:
    def _make_candles(self, n: int, interval: str = "1h", base_price: float = 50000) -> list[Candle]:
        """Generate N synthetic candles."""
        candles = []
        for i in range(n):
            ts = 1700000000000 + i * 3600000
            p = base_price + (i % 20 - 10) * 50
            candles.append(Candle(
                timestamp=ts, symbol="BTCUSDT", interval=interval,
                open=p - 20, high=p + 30, low=p - 40, close=p, volume=100,
            ))
        return candles

    def test_insufficient_data(self):
        engine = BacktestEngine(BacktestConfig())
        result = engine.run_walk_forward(
            strategy=None,  # won't be called with insufficient data
            candles_1h=self._make_candles(50),
            candles_4h=self._make_candles(20, "4h"),
        )
        assert "error" in result

    def test_walk_forward_structure(self):
        """Test that walk-forward returns proper structure (won't run real strategy)."""
        engine = BacktestEngine(BacktestConfig())
        # We need 100+ candles and valid strategy for real run
        # Just validate the structure with small data
        result = engine.run_walk_forward(
            strategy=None,
            candles_1h=self._make_candles(50),
            candles_4h=self._make_candles(20, "4h"),
            n_splits=1,
        )
        assert "error" in result or "folds" in result


# ══════════════════════════════════════════════
# REPOSITORY — strategy_performance
# ══════════════════════════════════════════════

from database.db import Database
from database.repository import Repository


class TestRepositoryNewMethods:
    @pytest.fixture
    def repo(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.connect()
        return Repository(db)

    def test_get_strategy_performance_empty(self, repo):
        result = repo.get_strategy_performance()
        assert result == []

    def test_get_all_trades_for_export_empty(self, repo):
        result = repo.get_all_trades_for_export()
        assert result == []

    def test_insert_signal_execution(self, repo):
        rid = repo.insert_signal_execution(
            timestamp=1700000000000,
            symbol="BTCUSDT",
            strategy_name="ema_crossover_rsi",
            direction="BUY",
            confidence=0.82,
            outcome="filled",
            reason="test",
            latency_ms=150,
        )
        assert rid > 0

    def test_get_signal_execution_stats(self, repo):
        repo.insert_signal_execution(
            timestamp=int(__import__('time').time() * 1000),
            symbol="BTCUSDT",
            strategy_name="ema",
            direction="BUY",
            confidence=0.8,
            outcome="filled",
        )
        repo.insert_signal_execution(
            timestamp=int(__import__('time').time() * 1000),
            symbol="BTCUSDT",
            strategy_name="ema",
            direction="BUY",
            confidence=0.6,
            outcome="rejected",
            reason="risk",
        )
        stats = repo.get_signal_execution_stats(hours=1)
        assert stats["total"] == 2
        assert stats["filled"] == 1
        assert stats["rejected"] == 1


# ══════════════════════════════════════════════
# FORMATTERS — format_portfolio
# ══════════════════════════════════════════════

from telegram_bot.formatters import format_portfolio


class TestFormatPortfolio:
    def test_empty_portfolio(self):
        text = format_portfolio([], 1000.0)
        assert "Нет данных" in text

    def test_with_strategies(self):
        perf = [
            {"strategy_name": "ema_crossover_rsi", "total_trades": 10, "win_rate": 60.0,
             "total_pnl": 25.50, "avg_pnl": 2.55},
            {"strategy_name": "mean_reversion", "total_trades": 5, "win_rate": 40.0,
             "total_pnl": -8.20, "avg_pnl": -1.64},
        ]
        text = format_portfolio(perf, 500.0)
        assert "Portfolio" in text
        assert "ema_crossover_rsi" in text
        assert "mean_reversion" in text
        assert "ИТОГО" in text or "Итого" in text
