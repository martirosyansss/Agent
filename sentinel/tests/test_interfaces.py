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
# Dashboard helpers
# ──────────────────────────────────────────────

class TestDashboardHelpers:
    def test_format_uptime(self):
        from dashboard.app import _format_uptime
        result = _format_uptime()
        # Should return a string containing 'm' and 's'
        assert isinstance(result, str)
        assert "m" in result or "h" in result or "d" in result


# ──────────────────────────────────────────────
# Dashboard API tests
# ──────────────────────────────────────────────

class TestDashboardAPI:
    @pytest.fixture
    def client(self):
        from types import SimpleNamespace
        from core.events import EventBus
        from dashboard.app import Dashboard

        mock_settings = SimpleNamespace(
            dashboard_port=8080,
            dashboard_password="",
            trading_mode="paper",
            trading_symbols=["BTCUSDT", "ETHUSDT"],
            signal_timeframe="1h",
            trend_timeframe="4h",
            min_confidence=0.75,
            stop_loss_pct=3.0,
            take_profit_pct=5.0,
            max_trades_per_day=6,
            grid_enabled=True,
            grid_num_levels=8,
            grid_capital_pct=30.0,
            grid_auto_range=True,
            grid_min_profit_pct=0.3,
            grid_max_loss_pct=5.0,
            meanrev_enabled=False,
            meanrev_rsi_oversold=25.0,
            meanrev_rsi_overbought=75.0,
            meanrev_capital_pct=15.0,
            meanrev_stop_loss_pct=4.0,
            meanrev_take_profit_pct=6.0,
            bb_breakout_enabled=False,
            bb_period=20,
            bb_std_dev=2.0,
            bb_volume_confirm_mult=1.5,
            bb_trailing_stop_pct=2.0,
            bb_take_profit_pct=6.0,
            dca_enabled=True,
            dca_base_amount_usd=10.0,
            dca_interval_hours=24,
            dca_max_daily_buys=3,
            dca_max_invested_pct=40.0,
            dca_take_profit_pct=8.0,
            macd_div_enabled=False,
            macd_fast=12,
            macd_slow=26,
            macd_signal_period=9,
            macd_lookback_candles=30,
            macd_require_rsi_confirm=True,
            macd_require_vol_confirm=True,
            auto_strategy_selection=True,
            max_daily_loss_usd=50.0,
            max_daily_loss_pct=10.0,
            max_position_pct=20.0,
            max_total_exposure_pct=60.0,
            max_open_positions=2,
            max_order_usd=100.0,
            max_trades_per_hour=2,
            resume_cooldown_min=30,
            paper_initial_balance=500.0,
            paper_commission_pct=0.1,
            paper_slippage_pct=0.05,
            max_data_age_sec=30,
            price_cross_validation_interval=300,
            watchdog_heartbeat_interval=10,
            watchdog_timeout=120,
            db_backup_interval_hours=6,
            max_ram_mb=2048,
            analyzer_stats_enabled=True,
            analyzer_ml_shadow_mode=True,
        )

        dashboard = Dashboard(mock_settings, EventBus(), state_provider=lambda: {
            "mode": "paper",
            "risk_state": "NORMAL",
            "uptime": "1д 2ч",
            "pnl_today": 5.0,
            "pnl_total": 20.0,
            "open_positions": 1,
            "trades_today": 3,
            "balance": 505.0,
            "win_rate": 62.5,
            "risk_details": {
                "daily_loss": -2.5,
                "max_drawdown": 0.08,
                "exposure": 0.12,
                "trade_freq": 1,
                "daily_commission": 0.34,
                "market_data_age_sec": 1.5,
            },
            "positions": [],
            "recent_trades": [{
                "time": "2026-04-12 14:30:00",
                "symbol": "BTCUSDT",
                "strategy_name": "ema_crossover_rsi",
                "signal_id": "sig-42",
                "signal_reason": "EMA crossover with volume confirmation and RSI below 50",
                "side": "SELL",
                "price": 68000.0,
                "pnl": 1.25,
            }],
            "pnl_history": [],
            "market_chart": {
                "symbol": "BTCUSDT",
                "interval": "1m",
                "source": "candles_1m",
                "candles": [
                    {"t": 1712930998000, "label": "14:29", "o": 67950.0, "h": 67990.0, "l": 67940.0, "c": 67980.0, "v": 12.5},
                    {"t": 1712931058000, "label": "14:30", "o": 67980.0, "h": 68020.0, "l": 67970.0, "c": 68000.0, "v": 8.3},
                ],
            },
            "backtest_results": {"sharpe": 1.2, "win_rate": 55.0},
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
        assert "uptime" in data
        assert "timestamp" in data

    def test_status(self, client):
        r = client.get("/api/status")
        assert r.status_code == 200
        data = r.json()
        assert data["mode"] == "paper"
        assert data["risk_state"] == "NORMAL"
        assert data["pnl_today"] == 5.0
        assert data["balance"] == 505.0
        assert data["win_rate"] == 62.5
        assert data["risk_details"]["trade_freq"] == 1
        assert data["risk_details"]["daily_commission"] == 0.34

    def test_positions_empty(self, client):
        r = client.get("/api/positions")
        assert r.status_code == 200
        assert r.json() == []

    def test_trades_empty(self, client):
        r = client.get("/api/trades")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 1
        assert data[0]["strategy_name"] == "ema_crossover_rsi"
        assert data[0]["signal_id"] == "sig-42"
        assert "volume confirmation" in data[0]["signal_reason"]

    def test_pnl_history(self, client):
        r = client.get("/api/pnl-history")
        assert r.status_code == 200

    def test_market_chart(self, client):
        r = client.get("/api/market-chart")
        assert r.status_code == 200
        data = r.json()
        assert data["symbol"] == "BTCUSDT"
        assert data["source"] == "candles_1m"
        assert len(data["candles"]) == 2
        assert data["candles"][0]["o"] == 67950.0

    def test_market_chart_interval(self, client):
        r = client.get("/api/market-chart?interval=1h")
        assert r.status_code == 200
        data = r.json()
        # Falls back to state since no market_chart_provider set
        assert "candles" in data

    def test_backtest_results(self, client):
        r = client.get("/api/backtest-results")
        assert r.status_code == 200
        data = r.json()
        assert data["sharpe"] == 1.2
        assert data["win_rate"] == 55.0

    def test_config_snapshot(self, client):
        r = client.get("/api/config")
        assert r.status_code == 200
        data = r.json()
        assert data["control_center"]["mode"] == "paper"
        assert data["control_center"]["symbols"] == ["BTCUSDT", "ETHUSDT"]
        assert len(data["risk_limits"]) >= 5
        assert len(data["execution_profile"]) >= 4
        assert len(data["system_profile"]) >= 4
        assert any(strategy["name"] == "Grid Trading" for strategy in data["strategies"])

    def test_control_stop_no_handler(self, client):
        r = client.post("/api/control/stop")
        assert r.status_code == 503

    def test_control_resume_no_handler(self, client):
        r = client.post("/api/control/resume")
        assert r.status_code == 503

    def test_control_kill_no_handler(self, client):
        r = client.post("/api/control/kill")
        assert r.status_code == 503

    def test_dashboard_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "SENTINEL" in r.text
        assert "Chart" in r.text

    def test_dashboard_html_no_emojis(self, client):
        """UI/UX: no emojis used as icons."""
        r = client.get("/")
        html = r.text
        for emoji in ["🛡️", "🟢", "🔴", "☠️", "📈", "📉", "🎨", "🚀"]:
            assert emoji not in html, f"Emoji {emoji} found in dashboard HTML"

    def test_dashboard_html_svg_icons(self, client):
        """UI/UX: SVG icons used instead of emojis."""
        html = client.get("/").text
        assert "<svg" in html
        assert "viewBox" in html

    def test_dashboard_html_cursor_pointer(self, client):
        """UI/UX: cursor-pointer on clickable elements."""
        html = client.get("/").text
        assert "cursor: pointer" in html or "cursor:pointer" in html

    def test_dashboard_html_focus_visible(self, client):
        """UI/UX: focus-visible for keyboard navigation."""
        html = client.get("/").text
        assert "focus-visible" in html

    def test_dashboard_html_reduced_motion(self, client):
        """UI/UX: prefers-reduced-motion support."""
        html = client.get("/").text
        assert "prefers-reduced-motion" in html

    def test_dashboard_html_aria_labels(self, client):
        """UI/UX: aria labels for accessibility."""
        html = client.get("/").text
        assert "aria-label" in html

    def test_dashboard_html_inter_font(self, client):
        """UI/UX: Inter font for professional fintech look."""
        html = client.get("/").text
        assert "Inter" in html

    def test_dashboard_html_responsive(self, client):
        """UI/UX: responsive breakpoints."""
        html = client.get("/").text
        assert "max-width: 768px" in html or "max-width:768px" in html

    def test_dashboard_html_websocket(self, client):
        """Dashboard uses WebSocket for real-time updates."""
        html = client.get("/").text
        assert "WebSocket" in html
        assert "/ws" in html

    def test_dashboard_html_risk_panel(self, client):
        """Dashboard shows risk overview panel."""
        html = client.get("/").text
        assert "Risk Overview" in html
        assert "risk-grid" in html
        assert 'id="risk-data-age"' in html
        assert 'id="risk-commission"' in html

    def test_dashboard_html_balance_card(self, client):
        """Dashboard shows balance card."""
        html = client.get("/").text
        assert 'id="balance"' in html

    def test_dashboard_html_positions_context_columns(self, client):
        """Dashboard positions table shows strategy and protective levels."""
        html = client.get("/").text
        assert "Strategy" in html
        assert "SL / TP" in html

    def test_dashboard_html_trades_strategy_column(self, client):
        """Dashboard recent trades table shows strategy column."""
        html = client.get("/").text
        assert "Recent Trades" in html
        assert "Strategy" in html

    def test_dashboard_html_trades_signal_column(self, client):
        """Dashboard recent trades table shows compact signal context column."""
        html = client.get("/").text
        assert "Signal" in html
        assert "signal_reason" in html
        assert "table-cell-meta" in html

    def test_dashboard_html_uptime(self, client):
        """Dashboard shows uptime indicator."""
        html = client.get("/").text
        assert 'id="uptime-label"' in html

    def test_dashboard_html_connection_indicator(self, client):
        """Dashboard shows WebSocket connection status."""
        html = client.get("/").text
        assert "conn-indicator" in html
        assert "conn-dot" in html

    def test_dashboard_html_operator_panels(self, client):
        """Dashboard settings page shows operator-facing config panels."""
        html = client.get("/settings").text
        assert "Control Center" in html
        assert "Execution Profile" in html
        assert "Risk Limits" in html
        assert "System Profile" in html
        assert "Strategy Stack" in html
        assert "Reset section" in html
