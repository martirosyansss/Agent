"""
SQLite подключение с WAL mode, integrity check и автоматическим созданием схемы.

Используется aiosqlite для async-доступа из asyncio-event loop.
"""

from __future__ import annotations

import sqlite3
import threading
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
    position_id TEXT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_price REAL NOT NULL,
    quantity REAL NOT NULL,
    current_price REAL,
    unrealized_pnl REAL DEFAULT 0,
    realized_pnl REAL DEFAULT 0,
    stop_loss_price REAL DEFAULT 0,
    take_profit_price REAL DEFAULT 0,
    strategy_name TEXT DEFAULT '',
    signal_id TEXT DEFAULT '',
    signal_reason TEXT DEFAULT '',
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

-- signal_executions — audit trail for signal processing
CREATE TABLE IF NOT EXISTS signal_executions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    direction TEXT NOT NULL,
    confidence REAL,
    outcome TEXT NOT NULL,
    reason TEXT,
    latency_ms INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_signal_exec_ts
    ON signal_executions(timestamp);
CREATE INDEX IF NOT EXISTS idx_signal_exec_strategy
    ON signal_executions(strategy_name, outcome);

-- news_cache — кэш LLM-анализа новостей (чтобы не переанализировать)
CREATE TABLE IF NOT EXISTS news_cache (
    url TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    published_at INTEGER NOT NULL,
    sentiment_score REAL DEFAULT 0,
    impact_pct REAL DEFAULT 0,
    direction TEXT DEFAULT 'neutral',
    coins_mentioned TEXT DEFAULT '[]',
    llm_reasoning TEXT DEFAULT '',
    urgency TEXT DEFAULT 'low',
    confidence REAL DEFAULT 0.5,
    category TEXT DEFAULT 'other',
    impact_timeframe TEXT DEFAULT 'hours',
    cached_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_news_cache_published
    ON news_cache(published_at);
"""


class Database:
    """Синхронная обёртка вокруг SQLite с WAL mode."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Открыть соединение, включить WAL, создать схему."""
        log.info("Подключаюсь к SQLite: {}", self._db_path)
        conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            timeout=30,
        )
        try:
            conn.row_factory = sqlite3.Row
            # WAL mode для параллельного чтения/записи
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=5000")
            self._conn = conn
            self._apply_schema()
            log.info("SQLite инициализирована (WAL mode)")
        except Exception:
            conn.close()
            raise

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
        self._migrate()

    def _migrate(self) -> None:
        """Добавить новые колонки (safe: ALTER TABLE ADD IF NOT EXISTS)."""
        migrations = [
            ("strategy_trades", "news_sentiment", "REAL DEFAULT 0"),
            ("strategy_trades", "fear_greed_index", "INTEGER DEFAULT 50"),
            ("positions", "position_id", "TEXT"),
            ("positions", "stop_loss_price", "REAL DEFAULT 0"),
            ("positions", "take_profit_price", "REAL DEFAULT 0"),
            ("positions", "strategy_name", "TEXT DEFAULT ''"),
            ("positions", "signal_id", "TEXT DEFAULT ''"),
            ("positions", "signal_reason", "TEXT DEFAULT ''"),
        ]
        for table, column, col_type in migrations:
            try:
                self.conn.execute(f"ALTER TABLE [{table}] ADD COLUMN [{column}] {col_type}")
                self.conn.commit()
                log.info("Migration: added {}.{}", table, column)
            except sqlite3.OperationalError as e:
                if "duplicate column" in str(e).lower() or "already exists" in str(e).lower():
                    pass  # column already exists — expected
                else:
                    log.warning("Unexpected migration error for {}.{}: {}", table, column, e)

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
        with self._lock:
            return self.conn.execute(sql, params)

    def executemany(self, sql: str, params_seq) -> sqlite3.Cursor:
        with self._lock:
            return self.conn.executemany(sql, params_seq)

    def commit(self) -> None:
        with self._lock:
            self.conn.commit()

    def fetchone(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        with self._lock:
            return self.conn.execute(sql, params).fetchone()

    def fetchall(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(sql, params).fetchall()
