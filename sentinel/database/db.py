"""
SQLite подключение с WAL mode, integrity check и автоматическим созданием схемы.

Используется aiosqlite для async-доступа из asyncio-event loop.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from loguru import logger

log = logger.bind(module="database")

# ──────────────────────────────────────────────
# Схема таблиц (DDL)
# ──────────────────────────────────────────────

SCHEMA_SQL = """
-- trades — сырые сделки с биржи
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    price REAL NOT NULL,
    quantity REAL NOT NULL,
    is_buyer_maker INTEGER NOT NULL,
    UNIQUE(timestamp, symbol, price, quantity)
);
CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades(symbol, timestamp);

-- candles — OHLCV свечи
CREATE TABLE IF NOT EXISTS candles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    trades_count INTEGER DEFAULT 0,
    UNIQUE(timestamp, symbol, interval)
);
CREATE INDEX IF NOT EXISTS idx_candles_symbol_time ON candles(symbol, interval, timestamp);

-- signals — сигналы стратегий
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    confidence REAL NOT NULL,
    strategy TEXT NOT NULL,
    features TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- orders — ордера (paper / live)
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    order_type TEXT NOT NULL,
    quantity REAL NOT NULL,
    price REAL,
    status TEXT NOT NULL,
    exchange_order_id TEXT,
    fill_price REAL,
    fill_quantity REAL,
    commission REAL DEFAULT 0,
    is_paper INTEGER DEFAULT 1,
    created_at TEXT DEFAULT (datetime('now'))
);

-- positions — текущие позиции
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_price REAL NOT NULL,
    quantity REAL NOT NULL,
    current_price REAL,
    unrealized_pnl REAL DEFAULT 0,
    realized_pnl REAL DEFAULT 0,
    status TEXT DEFAULT 'OPEN',
    opened_at TEXT DEFAULT (datetime('now')),
    closed_at TEXT,
    is_paper INTEGER DEFAULT 1
);

-- daily_stats — дневная статистика
CREATE TABLE IF NOT EXISTS daily_stats (
    date TEXT PRIMARY KEY,
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    total_pnl REAL DEFAULT 0,
    max_drawdown REAL DEFAULT 0,
    total_commission REAL DEFAULT 0,
    is_paper INTEGER DEFAULT 1
);

-- strategy_trades — завершённые сделки для Trade Analyzer
CREATE TABLE IF NOT EXISTS strategy_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT NOT NULL UNIQUE,
    signal_id INTEGER,
    symbol TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    market_regime TEXT,
    timestamp_open TEXT NOT NULL,
    timestamp_close TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL NOT NULL,
    quantity REAL NOT NULL,
    pnl_usd REAL NOT NULL,
    pnl_pct REAL NOT NULL,
    is_win INTEGER NOT NULL,
    confidence REAL,
    hour_of_day INTEGER,
    day_of_week INTEGER,
    rsi_at_entry REAL,
    adx_at_entry REAL,
    volume_ratio_at_entry REAL,
    exit_reason TEXT,
    hold_duration_hours REAL,
    max_drawdown_during_trade REAL,
    max_profit_during_trade REAL,
    commission_usd REAL DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_strategy_trades_strategy_time
    ON strategy_trades(strategy_name, timestamp_close);
CREATE INDEX IF NOT EXISTS idx_strategy_trades_regime_time
    ON strategy_trades(market_regime, timestamp_close);

-- ml_model_registry — история ML-моделей
CREATE TABLE IF NOT EXISTS ml_model_registry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version TEXT NOT NULL,
    trained_from TEXT NOT NULL,
    trained_to TEXT NOT NULL,
    test_from TEXT NOT NULL,
    test_to TEXT NOT NULL,
    train_samples INTEGER NOT NULL,
    test_samples INTEGER NOT NULL,
    precision REAL,
    recall REAL,
    roc_auc REAL,
    uplift_profit_factor REAL,
    uplift_drawdown REAL,
    rollout_mode TEXT NOT NULL,
    is_active INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_ml_model_registry_active
    ON ml_model_registry(is_active, created_at);
"""


class Database:
    """Синхронная обёртка вокруг SQLite с WAL mode."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Открыть соединение, включить WAL, создать схему."""
        log.info("Подключаюсь к SQLite: {}", self._db_path)
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            timeout=30,
        )
        self._conn.row_factory = sqlite3.Row
        # WAL mode для параллельного чтения/записи
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._apply_schema()
        log.info("SQLite инициализирована (WAL mode)")

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
            log.info("SQLite соединение закрыто")

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._conn

    # ------------------------------------------------------------------
    # Schema & integrity
    # ------------------------------------------------------------------

    def _apply_schema(self) -> None:
        """Создать таблицы и индексы если не существуют."""
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def integrity_check(self) -> bool:
        """PRAGMA integrity_check — вернуть True если БД целая."""
        try:
            result = self.conn.execute("PRAGMA integrity_check").fetchone()
            ok = result[0] == "ok"
            if ok:
                log.info("Integrity check: OK")
            else:
                log.error("Integrity check FAILED: {}", result[0])
            return ok
        except sqlite3.Error as e:
            log.error("Integrity check error: {}", e)
            return False

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def executemany(self, sql: str, params_seq) -> sqlite3.Cursor:
        return self.conn.executemany(sql, params_seq)

    def commit(self) -> None:
        self.conn.commit()

    def fetchone(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        return self.conn.execute(sql, params).fetchone()

    def fetchall(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()
