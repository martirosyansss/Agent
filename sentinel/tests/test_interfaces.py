"""Тесты Phase 6 — Telegram formatters + Dashboard API."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.models import Direction, Order, OrderStatus, OrderType, Position, RiskState, Signal
from telegram_bot.formatters import (
    format_config_summary,
    format_daily_report,
    format_error,
    format_order_filled,
    format_pnl,
    format_positions,
    format_risk_state_changed,
    format_signal,
    format_status,
    format_stop_loss,
    format_take_profit,
    format_trades,
    fmt_pnl,
    fmt_price,
)


# ──────────────────────────────────────────────
# Formatter helpers
# ──────────────────────────────────────────────

class TestFormatHelpers:
    def test_fmt_price(self):
        assert "$67,234.00" == fmt_price(67234)
        assert "$3.4120" == fmt_price(3.412)

    def test_fmt_pnl_positive(self):
        assert fmt_pnl(5.23) == "+$5.23"

    def test_fmt_pnl_negative(self):
        assert fmt_pnl(-1.57) == "-$1.57"


# ──────────────────────────────────────────────
# Signal formatters
# ──────────────────────────────────────────────

class TestSignalFormatter:
    def test_buy_signal(self):
        sig = Signal(
            timestamp=1700000000000,
            symbol="BTCUSDT",
            direction=Direction.BUY,
            confidence=0.82,
            strategy_name="ema_crossover_rsi",
            reason="EMA9 crossed EMA21 up, RSI=45",
            stop_loss_price=65660.0,
            take_profit_price=69010.0,
        )
        text = format_signal(sig)
        assert "📈" in text
        assert "BUY" in text
        assert "BTCUSDT" in text
        assert "0.82" in text
        assert "ema_crossover_rsi" in text
        assert "Stop-Loss" in text
        assert "Take-Profit" in text

    def test_sell_signal(self):
        sig = Signal(
            timestamp=1700000000000,
            symbol="ETHUSDT",
            direction=Direction.SELL,
            confidence=0.90,
            strategy_name="ema_crossover_rsi",
            reason="Stop-loss triggered",
        )
        text = format_signal(sig)
        assert "📉" in text
        assert "SELL" in text


class TestOrderFormatter:
    def test_buy_order(self):
        order = Order(
            timestamp=1700000000000,
            symbol="BTCUSDT",
            side=Direction.BUY,
            order_type=OrderType.MARKET,
            quantity=0.001,
            fill_price=67234.0,
            fill_quantity=0.001,
            is_paper=True,
        )
        text = format_order_filled(order)
        assert "✅" in text
        assert "КУПЛЕНО" in text
        assert "Paper" in text

    def test_sell_order_live(self):
        order = Order(
            timestamp=1700000000000,
            symbol="ETHUSDT",
            side=Direction.SELL,
            order_type=OrderType.LIMIT,
            quantity=0.01,
            price=3445.0,
            fill_price=3445.0,
            is_paper=False,
        )
        text = format_order_filled(order)
        assert "ПРОДАНО" in text
        assert "LIVE" in text


class TestPositionFormatters:
    def test_stop_loss(self):
        pos = Position(symbol="BTCUSDT", current_price=65660.0, entry_price=67000)
        text = format_stop_loss(pos, -1.57)
        assert "🛑" in text
        assert "STOP-LOSS" in text
        assert "-$1.57" in text

    def test_take_profit(self):
        pos = Position(symbol="BTCUSDT", current_price=69010.0, entry_price=67000)
        text = format_take_profit(pos, 1.78)
        assert "🎯" in text
        assert "+$1.78" in text


class TestRiskStateFormatter:
    def test_state_change(self):
        text = format_risk_state_changed(
            RiskState.NORMAL, RiskState.REDUCED, "Daily loss > $16"
        )
        assert "NORMAL" in text
        assert "REDUCED" in text
        assert "Daily loss" in text


class TestStatusFormatter:
    def test_basic_status(self):
        text = format_status(
            mode="paper",
            risk_state="NORMAL",
            uptime="2д 14ч",
            pnl_today=3.27,
            pnl_total=18.42,
            open_positions=2,
            trades_today=8,
        )
        assert "🟢" in text
        assert "PAPER" in text
        assert "+$3.27" in text
        assert "+$18.42" in text


class TestPnlFormatter:
    def test_pnl_report(self):
        text = format_pnl(pnl_day=5.0, pnl_week=20.0, pnl_month=50.0, balance=550.0)
        assert "+$5.00" in text
        assert "+$20.00" in text
        assert "$550.00" in text


class TestPositionsFormatter:
    def test_no_positions(self):
        text = format_positions([])
        assert "Нет открытых позиций" in text

    def test_with_positions(self):
        pos = [
            Position(symbol="BTCUSDT", side="LONG", entry_price=67000.0,
                     current_price=67500.0, unrealized_pnl=1.5),
        ]
        text = format_positions(pos)
        assert "BTCUSDT" in text
        assert "LONG" in text


class TestTradesFormatter:
    def test_no_trades(self):
        text = format_trades([])
        assert "Нет сделок" in text

    def test_with_trades(self):
        trades = [{"side": "BUY", "symbol": "BTCUSDT", "price": 67234.0, "pnl": None}]
        text = format_trades(trades)
        assert "BUY" in text
        assert "BTCUSDT" in text


class TestDailyReport:
    def test_basic(self):
        text = format_daily_report(pnl=5.23, win_rate=58.0, trades_count=12, wins=5, losses=3)
        assert "+$5.23" in text
        assert "58%" in text
        assert "12" in text


class TestErrorFormatter:
    def test_error(self):
        text = format_error("Потеряно соединение с Binance")
        assert "🚨" in text
        assert "Потеряно" in text


class TestConfigFormatter:
    def test_config(self):
        text = format_config_summary({"mode": "paper", "symbols": "BTCUSDT, ETHUSDT"})
        assert "mode" in text
        assert "paper" in text


# ──────────────────────────────────────────────
# Telegram Bot unit tests (no real API)
# ──────────────────────────────────────────────

class TestTelegramBotAuthorization:
    def test_enabled_check(self):
        """Бот должен быть disabled без токена."""
        from unittest.mock import MagicMock
        from core.events import EventBus

        mock_settings = MagicMock()
        mock_settings.telegram_bot_token = ""
        mock_settings.telegram_chat_id = ""
        mock_settings.telegram_pin = ""

        from telegram_bot.bot import TelegramBot
        bot = TelegramBot(mock_settings, EventBus())
        assert bot.enabled is False

    def test_enabled_with_token(self):
        from unittest.mock import MagicMock
        from core.events import EventBus

        mock_settings = MagicMock()
        mock_settings.telegram_bot_token = "test_token"
        mock_settings.telegram_chat_id = "12345"
        mock_settings.telegram_pin = ""

        from telegram_bot.bot import TelegramBot
        bot = TelegramBot(mock_settings, EventBus())
        assert bot.enabled is True

    def test_authorization(self):
        from unittest.mock import MagicMock
        from core.events import EventBus

        mock_settings = MagicMock()
        mock_settings.telegram_bot_token = "test_token"
        mock_settings.telegram_chat_id = "12345"
        mock_settings.telegram_pin = ""

        from telegram_bot.bot import TelegramBot
        bot = TelegramBot(mock_settings, EventBus())
        assert bot._is_authorized(12345) is True
        assert bot._is_authorized("12345") is True
        assert bot._is_authorized(99999) is False


# ──────────────────────────────────────────────
# Dashboard API tests
# ──────────────────────────────────────────────

class TestDashboardAPI:
    @pytest.fixture
    def client(self):
        from unittest.mock import MagicMock
        from core.events import EventBus
        from dashboard.app import Dashboard

        mock_settings = MagicMock()
        mock_settings.dashboard_port = 8080
        mock_settings.dashboard_password = ""

        dashboard = Dashboard(mock_settings, EventBus(), state_provider=lambda: {
            "mode": "paper",
            "risk_state": "NORMAL",
            "uptime": "1д 2ч",
            "pnl_today": 5.0,
            "pnl_total": 20.0,
            "open_positions": 1,
            "trades_today": 3,
            "balance": 505.0,
            "positions": [],
            "recent_trades": [],
            "pnl_history": [],
        })

        app = dashboard._create_app()

        from starlette.testclient import TestClient
        return TestClient(app)

    def test_health(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_status(self, client):
        r = client.get("/api/status")
        assert r.status_code == 200
        data = r.json()
        assert data["mode"] == "paper"
        assert data["risk_state"] == "NORMAL"
        assert data["pnl_today"] == 5.0

    def test_positions_empty(self, client):
        r = client.get("/api/positions")
        assert r.status_code == 200
        assert r.json() == []

    def test_trades_empty(self, client):
        r = client.get("/api/trades")
        assert r.status_code == 200
        assert r.json() == []

    def test_pnl_history(self, client):
        r = client.get("/api/pnl-history")
        assert r.status_code == 200

    def test_control_stop_no_handler(self, client):
        r = client.post("/api/control/stop")
        assert r.status_code == 503

    def test_dashboard_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "SENTINEL" in r.text
        assert "Chart" in r.text
