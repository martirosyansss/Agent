"""Тесты Database и Repository — Phase 4."""

import os
import tempfile
import time

import pytest

# Чтобы импорты работали из корня sentinel/
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.models import (
    Candle,
    Direction,
    MarketTrade,
    Order,
    OrderStatus,
    OrderType,
    Position,
    Signal,
)
from database.db import Database
from database.repository import Repository


@pytest.fixture
def db_and_repo(tmp_path):
    """Создаёт временную БД и репозиторий."""
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.connect()
    repo = Repository(db)
    yield db, repo
    db.close()


class TestDatabase:
    def test_connect_and_schema(self, db_and_repo):
        db, repo = db_and_repo
        # Проверяем что таблицы созданы
        tables = db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        table_names = {r["name"] for r in tables}
        expected = {
            "trades", "candles", "signals", "orders", "positions",
            "daily_stats", "strategy_trades", "ml_model_registry",
        }
        assert expected.issubset(table_names)

    def test_wal_mode(self, db_and_repo):
        db, _ = db_and_repo
        row = db.fetchone("PRAGMA journal_mode")
        assert row[0] == "wal"

    def test_integrity_check(self, db_and_repo):
        db, _ = db_and_repo
        assert db.integrity_check() is True


class TestTradesRepository:
    def test_insert_and_get(self, db_and_repo):
        _, repo = db_and_repo
        t = MarketTrade(
            timestamp=1700000000000,
            symbol="BTCUSDT",
            price=67000.0,
            quantity=0.001,
            is_buyer_maker=False,
        )
        assert repo.insert_trade(t) is True
        trades = repo.get_recent_trades("BTCUSDT", limit=10)
        assert len(trades) == 1
        assert trades[0]["price"] == 67000.0

    def test_deduplication(self, db_and_repo):
        _, repo = db_and_repo
        t = MarketTrade(
            timestamp=1700000000000,
            symbol="BTCUSDT",
            price=67000.0,
            quantity=0.001,
            is_buyer_maker=False,
        )
        repo.insert_trade(t)
        repo.insert_trade(t)  # дубликат
        trades = repo.get_recent_trades("BTCUSDT")
        assert len(trades) == 1

    def test_batch_insert(self, db_and_repo):
        _, repo = db_and_repo
        trades = [
            MarketTrade(1700000000000 + i, "BTCUSDT", 67000.0 + i, 0.001, False)
            for i in range(100)
        ]
        count = repo.insert_trades_batch(trades)
        assert count == 100
        result = repo.get_recent_trades("BTCUSDT", limit=200)
        assert len(result) == 100

    def test_delete_old_trades(self, db_and_repo):
        _, repo = db_and_repo
        old_ts = 1600000000000
        new_ts = 1700000000000
        repo.insert_trade(MarketTrade(old_ts, "BTCUSDT", 30000.0, 0.01, False))
        repo.insert_trade(MarketTrade(new_ts, "BTCUSDT", 67000.0, 0.01, False))
        deleted = repo.delete_old_trades(1650000000000)
        assert deleted == 1
        remaining = repo.get_recent_trades("BTCUSDT")
        assert len(remaining) == 1


class TestCandlesRepository:
    def test_upsert_candle(self, db_and_repo):
        _, repo = db_and_repo
        c = Candle(1700000000000, "BTCUSDT", "1h", 67000, 67500, 66900, 67200, 100.0, 500)
        repo.upsert_candle(c)

        result = repo.get_candles("BTCUSDT", "1h")
        assert len(result) == 1
        assert result[0]["close"] == 67200

        # Upsert обновляет
        c2 = Candle(1700000000000, "BTCUSDT", "1h", 67000, 67800, 66900, 67500, 120.0, 600)
        repo.upsert_candle(c2)
        result = repo.get_candles("BTCUSDT", "1h")
        assert len(result) == 1
        assert result[0]["close"] == 67500

    def test_get_latest_candle(self, db_and_repo):
        _, repo = db_and_repo
        repo.upsert_candle(Candle(1700000000000, "BTCUSDT", "1h", 67000, 67500, 66900, 67200, 100, 0))
        repo.upsert_candle(Candle(1700003600000, "BTCUSDT", "1h", 67200, 67800, 67100, 67600, 110, 0))
        latest = repo.get_latest_candle("BTCUSDT", "1h")
        assert latest is not None
        assert latest["timestamp"] == 1700003600000


class TestSignalsRepository:
    def test_insert_signal(self, db_and_repo):
        _, repo = db_and_repo
        sig = Signal(
            timestamp=1700000000000,
            symbol="BTCUSDT",
            direction=Direction.BUY,
            confidence=0.82,
            strategy_name="ema_crossover_rsi",
            reason="EMA crossover + RSI filter",
        )
        rowid = repo.insert_signal(sig)
        assert rowid > 0

        signals = repo.get_recent_signals("BTCUSDT")
        assert len(signals) == 1
        assert signals[0]["confidence"] == 0.82


class TestOrdersRepository:
    def test_insert_and_update(self, db_and_repo):
        _, repo = db_and_repo
        o = Order(
            timestamp=1700000000000,
            symbol="BTCUSDT",
            side=Direction.BUY,
            order_type=OrderType.MARKET,
            quantity=0.001,
            status=OrderStatus.PENDING,
        )
        rowid = repo.insert_order(o)
        assert rowid > 0

        repo.update_order_status(rowid, "FILLED", fill_price=67000.0, fill_quantity=0.001, commission=0.067)
        orders = repo.get_recent_orders("BTCUSDT")
        assert orders[0]["status"] == "FILLED"
        assert orders[0]["fill_price"] == 67000.0

    def test_count_orders_since(self, db_and_repo):
        _, repo = db_and_repo
        for i in range(5):
            o = Order(
                timestamp=1700000000000 + i * 1000,
                symbol="BTCUSDT",
                side=Direction.BUY,
                order_type=OrderType.MARKET,
                quantity=0.001,
                status=OrderStatus.FILLED,
            )
            repo.insert_order(o)
        count = repo.count_orders_since(1700000002000)
        assert count == 3


class TestPositionsRepository:
    def test_insert_open_close(self, db_and_repo):
        _, repo = db_and_repo
        p = Position(symbol="BTCUSDT", side="LONG", entry_price=67000.0, quantity=0.001)
        pid = repo.insert_position(p)
        assert pid > 0

        open_pos = repo.get_open_positions("BTCUSDT")
        assert len(open_pos) == 1

        repo.close_position(pid, realized_pnl=3.27, closed_at="2026-04-12 14:30:00")
        open_pos = repo.get_open_positions("BTCUSDT")
        assert len(open_pos) == 0


class TestDailyStatsRepository:
    def test_upsert_daily_stats(self, db_and_repo):
        _, repo = db_and_repo
        repo.upsert_daily_stats("2026-04-12", 10, 6, 4, 3.27, 1.5, 0.67)
        stats = repo.get_daily_stats(7)
        assert len(stats) == 1
        assert stats[0]["total_pnl"] == 3.27

        # Upsert обновляет
        repo.upsert_daily_stats("2026-04-12", 12, 7, 5, 5.00, 2.0, 0.80)
        stats = repo.get_daily_stats(7)
        assert len(stats) == 1
        assert stats[0]["total_pnl"] == 5.00

    def test_get_daily_pnl(self, db_and_repo):
        _, repo = db_and_repo
        assert repo.get_daily_pnl("2026-04-12") == 0.0
        repo.upsert_daily_stats("2026-04-12", 5, 3, 2, 2.50, 1.0, 0.5)
        assert repo.get_daily_pnl("2026-04-12") == 2.50


class TestDataValidator:
    def test_valid_trade(self):
        from collector.data_validator import validate_trade
        t = MarketTrade(int(time.time() * 1000), "BTCUSDT", 67000.0, 0.001, False)
        assert validate_trade(t) is True

    def test_reject_negative_price(self):
        from collector.data_validator import validate_trade
        t = MarketTrade(int(time.time() * 1000), "BTCUSDT", -100.0, 0.001, False)
        assert validate_trade(t) is False

    def test_reject_future_timestamp(self):
        from collector.data_validator import validate_trade
        future = int(time.time() * 1000) + 60_000  # 1 minute in future
        t = MarketTrade(future, "BTCUSDT", 67000.0, 0.001, False)
        assert validate_trade(t) is False

    def test_reject_price_out_of_range(self):
        from collector.data_validator import validate_trade
        t = MarketTrade(int(time.time() * 1000), "BTCUSDT", 500.0, 0.001, False)
        assert validate_trade(t) is False  # BTC < $1000 is suspicious

    def test_valid_candle(self):
        from collector.data_validator import validate_candle
        c = Candle(int(time.time() * 1000), "ETHUSDT", "1h", 3400, 3500, 3350, 3450, 1000.0, 100)
        assert validate_candle(c) is True

    def test_reject_high_less_than_low(self):
        from collector.data_validator import validate_candle
        c = Candle(int(time.time() * 1000), "ETHUSDT", "1h", 3400, 3300, 3500, 3450, 1000.0, 100)
        assert validate_candle(c) is False
