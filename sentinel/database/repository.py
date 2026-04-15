"""
Repository — CRUD-операции для всех таблиц SENTINEL.

Все методы работают через Database.conn (синхронный sqlite3).
При необходимости вызываются из asyncio через asyncio.to_thread().
"""

from __future__ import annotations

import json
import time
from typing import Optional

from loguru import logger

from core.models import (
    Candle,
    Direction,
    MarketTrade,
    Order,
    OrderStatus,
    OrderType,
    Position,
    PositionStatus,
    Signal,
    StrategyTrade,
)
from database.db import Database

log = logger.bind(module="repository")


class Repository:
    """Единый CRUD-репозиторий для всех сущностей SENTINEL."""

    def __init__(self, db: Database) -> None:
        self._db = db

    # ==================================================================
    # TRADES
    # ==================================================================

    def insert_trade(self, t: MarketTrade) -> bool:
        """Вставить сырую рыночную сделку. Возвращает False при дубликате."""
        try:
            self._db.execute(
                "INSERT OR IGNORE INTO trades (timestamp, symbol, price, quantity, is_buyer_maker) "
                "VALUES (?, ?, ?, ?, ?)",
                (t.timestamp, t.symbol, t.price, t.quantity, int(t.is_buyer_maker)),
            )
            self._db.commit()
            return True
        except Exception as e:
            log.warning("insert_trade error: {}", e)
            return False

    def insert_trades_batch(self, trades: list[MarketTrade]) -> int:
        """Пакетная вставка сделок. Возвращает количество вставленных."""
        if not trades:
            return 0
        params = [
            (t.timestamp, t.symbol, t.price, t.quantity, int(t.is_buyer_maker))
            for t in trades
        ]
        self._db.executemany(
            "INSERT OR IGNORE INTO trades (timestamp, symbol, price, quantity, is_buyer_maker) "
            "VALUES (?, ?, ?, ?, ?)",
            params,
        )
        self._db.commit()
        return len(params)

    def get_recent_trades(self, symbol: str, limit: int = 100) -> list[dict]:
        rows = self._db.fetchall(
            "SELECT * FROM trades WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?",
            (symbol, limit),
        )
        return [dict(r) for r in rows]

    def delete_old_trades(self, before_ts: int) -> int:
        """Удалить сделки старше timestamp. Возвращает количество."""
        cur = self._db.execute("DELETE FROM trades WHERE timestamp < ?", (before_ts,))
        self._db.commit()
        return cur.rowcount

    # ==================================================================
    # CANDLES
    # ==================================================================

    def upsert_candle(self, c: Candle) -> None:
        """Вставить или обновить свечу (UPSERT по unique constraint)."""
        self._db.execute(
            "INSERT INTO candles (timestamp, symbol, interval, open, high, low, close, volume, trades_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(timestamp, symbol, interval) DO UPDATE SET "
            "open=excluded.open, high=excluded.high, low=excluded.low, "
            "close=excluded.close, volume=excluded.volume, trades_count=excluded.trades_count",
            (c.timestamp, c.symbol, c.interval, c.open, c.high, c.low, c.close, c.volume, c.trades_count),
        )
        self._db.commit()

    def upsert_candles_batch(self, candles: list[Candle]) -> int:
        if not candles:
            return 0
        params = [
            (c.timestamp, c.symbol, c.interval, c.open, c.high, c.low, c.close, c.volume, c.trades_count)
            for c in candles
        ]
        self._db.executemany(
            "INSERT INTO candles (timestamp, symbol, interval, open, high, low, close, volume, trades_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(timestamp, symbol, interval) DO UPDATE SET "
            "open=excluded.open, high=excluded.high, low=excluded.low, "
            "close=excluded.close, volume=excluded.volume, trades_count=excluded.trades_count",
            params,
        )
        self._db.commit()
        return len(params)

    def get_candles(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        since_ts: int = 0,
    ) -> list[dict]:
        """Получить свечи, отсортированные по времени (старые → новые).
        Возвращает последние `limit` свечей (не первые), отсортированные ASC.
        """
        rows = self._db.fetchall(
            "SELECT * FROM (SELECT * FROM candles WHERE symbol = ? AND interval = ? "
            "AND timestamp >= ? ORDER BY timestamp DESC LIMIT ?) ORDER BY timestamp ASC",
            (symbol, interval, since_ts, limit),
        )
        return [dict(r) for r in rows]

    def get_latest_candle(self, symbol: str, interval: str) -> dict | None:
        row = self._db.fetchone(
            "SELECT * FROM candles WHERE symbol = ? AND interval = ? "
            "ORDER BY timestamp DESC LIMIT 1",
            (symbol, interval),
        )
        return dict(row) if row else None

    def delete_old_candles(self, before_ts: int) -> int:
        cur = self._db.execute("DELETE FROM candles WHERE timestamp < ?", (before_ts,))
        self._db.commit()
        return cur.rowcount

    # ==================================================================
    # SIGNALS
    # ==================================================================

    def insert_signal(self, s: Signal) -> int:
        """Вставить сигнал. Возвращает rowid."""
        features_json = None
        if s.features:
            from dataclasses import asdict
            features_json = json.dumps(asdict(s.features), ensure_ascii=False)
        cur = self._db.execute(
            "INSERT INTO signals (timestamp, symbol, direction, confidence, strategy, features) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (s.timestamp, s.symbol, s.direction.value, s.confidence, s.strategy_name, features_json),
        )
        self._db.commit()
        return cur.lastrowid

    def get_recent_signals(self, symbol: str, limit: int = 50) -> list[dict]:
        rows = self._db.fetchall(
            "SELECT * FROM signals WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?",
            (symbol, limit),
        )
        return [dict(r) for r in rows]

    # ==================================================================
    # ORDERS
    # ==================================================================

    def insert_order(self, o: Order) -> int:
        cur = self._db.execute(
            "INSERT INTO orders (timestamp, symbol, side, order_type, quantity, price, "
            "status, exchange_order_id, fill_price, fill_quantity, commission, is_paper) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                o.timestamp, o.symbol, o.side.value, o.order_type.value,
                o.quantity, o.price, o.status.value, o.exchange_order_id,
                o.fill_price, o.fill_quantity, o.commission, int(o.is_paper),
            ),
        )
        self._db.commit()
        return cur.lastrowid

    def update_order_status(self, order_db_id: int, status: str,
                            fill_price: float | None = None,
                            fill_quantity: float | None = None,
                            commission: float = 0.0) -> None:
        self._db.execute(
            "UPDATE orders SET status = ?, fill_price = ?, fill_quantity = ?, commission = ? "
            "WHERE id = ?",
            (status, fill_price, fill_quantity, commission, order_db_id),
        )
        self._db.commit()

    def get_recent_orders(self, symbol: str | None = None, limit: int = 50) -> list[dict]:
        if symbol:
            rows = self._db.fetchall(
                "SELECT * FROM orders WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?",
                (symbol, limit),
            )
        else:
            rows = self._db.fetchall(
                "SELECT * FROM orders ORDER BY timestamp DESC LIMIT ?", (limit,)
            )
        return [dict(r) for r in rows]

    def count_orders_since(self, since_ts: int, symbol: str | None = None) -> int:
        if symbol:
            row = self._db.fetchone(
                "SELECT COUNT(*) as cnt FROM orders WHERE timestamp >= ? AND symbol = ?",
                (since_ts, symbol),
            )
        else:
            row = self._db.fetchone(
                "SELECT COUNT(*) as cnt FROM orders WHERE timestamp >= ?", (since_ts,)
            )
        return row["cnt"] if row else 0

    # ==================================================================
    # POSITIONS
    # ==================================================================

    def insert_position(self, p: Position) -> int:
        # Guard: prevent duplicate OPEN positions for the same symbol
        existing = self._db.fetchone(
            "SELECT id FROM positions WHERE symbol = ? AND status = 'OPEN'",
            (p.symbol,),
        )
        if existing:
            log.warning("Duplicate OPEN position for {} — closing stale row id={}", p.symbol, existing["id"])
            self._db.execute(
                "UPDATE positions SET status = 'STALE' WHERE id = ?", (existing["id"],)
            )
            self._db.commit()
        cur = self._db.execute(
            "INSERT INTO positions (position_id, symbol, side, entry_price, quantity, "
            "current_price, unrealized_pnl, realized_pnl, stop_loss_price, take_profit_price, "
            "strategy_name, signal_id, signal_reason, status, opened_at, is_paper) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                p.position_id, p.symbol, p.side, p.entry_price, p.quantity,
                p.current_price, p.unrealized_pnl, p.realized_pnl,
                p.stop_loss_price, p.take_profit_price,
                p.strategy_name, p.signal_id, p.signal_reason,
                p.status.value, p.opened_at, int(p.is_paper),
            ),
        )
        self._db.commit()
        return cur.lastrowid

    def get_open_positions(self, symbol: str | None = None) -> list[dict]:
        if symbol:
            rows = self._db.fetchall(
                "SELECT * FROM positions WHERE status = 'OPEN' AND symbol = ?", (symbol,)
            )
        else:
            rows = self._db.fetchall("SELECT * FROM positions WHERE status = 'OPEN'")
        return [dict(r) for r in rows]

    def close_position(self, position_id: int, realized_pnl: float, closed_at: str) -> None:
        self._db.execute(
            "UPDATE positions SET status = 'CLOSED', realized_pnl = ?, closed_at = ? WHERE id = ?",
            (realized_pnl, closed_at, position_id),
        )
        self._db.commit()

    def update_position_price(self, position_id: int, current_price: float, unrealized_pnl: float) -> None:
        self._db.execute(
            "UPDATE positions SET current_price = ?, unrealized_pnl = ? WHERE id = ?",
            (current_price, unrealized_pnl, position_id),
        )
        self._db.commit()

    # ==================================================================
    # DAILY STATS
    # ==================================================================

    def upsert_daily_stats(
        self,
        date: str,
        total_trades: int,
        winning: int,
        losing: int,
        total_pnl: float,
        max_dd: float,
        total_commission: float,
        is_paper: bool = True,
    ) -> None:
        self._db.execute(
            "INSERT INTO daily_stats (date, total_trades, winning_trades, losing_trades, "
            "total_pnl, max_drawdown, total_commission, is_paper) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(date) DO UPDATE SET "
            "total_trades=excluded.total_trades, winning_trades=excluded.winning_trades, "
            "losing_trades=excluded.losing_trades, total_pnl=excluded.total_pnl, "
            "max_drawdown=excluded.max_drawdown, total_commission=excluded.total_commission",
            (date, total_trades, winning, losing, total_pnl, max_dd, total_commission, int(is_paper)),
        )
        self._db.commit()

    def get_daily_stats(self, days: int = 30) -> list[dict]:
        rows = self._db.fetchall(
            "SELECT * FROM daily_stats ORDER BY date DESC LIMIT ?", (days,)
        )
        return [dict(r) for r in rows]

    def get_daily_pnl(self, date: str) -> float:
        row = self._db.fetchone(
            "SELECT total_pnl FROM daily_stats WHERE date = ?", (date,)
        )
        return row["total_pnl"] if row else 0.0

    # ==================================================================
    # STRATEGY TRADES
    # ==================================================================

    def insert_strategy_trade(self, st: StrategyTrade) -> int:
        # Check for duplicates before inserting
        existing = self._db.fetchone(
            "SELECT id FROM strategy_trades WHERE trade_id = ?", (st.trade_id,)
        )
        if existing:
            log.warning("Duplicate strategy_trade ignored: trade_id={}", st.trade_id)
            return existing["id"]

        cur = self._db.execute(
            "INSERT INTO strategy_trades "
            "(trade_id, signal_id, symbol, strategy_name, market_regime, "
            "timestamp_open, timestamp_close, entry_price, exit_price, quantity, "
            "pnl_usd, pnl_pct, is_win, confidence, hour_of_day, day_of_week, "
            "rsi_at_entry, adx_at_entry, volume_ratio_at_entry, exit_reason, "
            "hold_duration_hours, max_drawdown_during_trade, max_profit_during_trade, "
            "commission_usd, news_sentiment, fear_greed_index) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                st.trade_id, st.signal_id, st.symbol, st.strategy_name, st.market_regime,
                st.timestamp_open, st.timestamp_close, st.entry_price, st.exit_price, st.quantity,
                st.pnl_usd, st.pnl_pct, int(st.is_win), st.confidence,
                st.hour_of_day, st.day_of_week,
                st.rsi_at_entry, st.adx_at_entry, st.volume_ratio_at_entry,
                st.exit_reason, st.hold_duration_hours,
                st.max_drawdown_during_trade, st.max_profit_during_trade, st.commission_usd,
                st.news_sentiment, st.fear_greed_index,
            ),
        )
        self._db.commit()
        return cur.lastrowid

    def get_strategy_trades(
        self,
        strategy_name: str | None = None,
        limit: int = 500,
    ) -> list[dict]:
        if strategy_name:
            rows = self._db.fetchall(
                "SELECT * FROM strategy_trades WHERE strategy_name = ? "
                "ORDER BY timestamp_close DESC LIMIT ?",
                (strategy_name, limit),
            )
        else:
            rows = self._db.fetchall(
                "SELECT * FROM strategy_trades ORDER BY timestamp_close DESC LIMIT ?",
                (limit,),
            )
        return [dict(r) for r in rows]

    def count_strategy_trades(self, strategy_name: str | None = None) -> int:
        if strategy_name:
            row = self._db.fetchone(
                "SELECT COUNT(*) as cnt FROM strategy_trades WHERE strategy_name = ?",
                (strategy_name,),
            )
        else:
            row = self._db.fetchone("SELECT COUNT(*) as cnt FROM strategy_trades")
        return row["cnt"] if row else 0

    def get_strategy_performance(self) -> list[dict]:
        """Aggregate per-strategy PnL statistics."""
        rows = self._db.fetchall(
            "SELECT strategy_name, "
            "  COUNT(*) as total_trades, "
            "  SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as wins, "
            "  SUM(CASE WHEN is_win = 0 THEN 1 ELSE 0 END) as losses, "
            "  ROUND(SUM(pnl_usd), 2) as total_pnl, "
            "  ROUND(AVG(pnl_usd), 2) as avg_pnl, "
            "  ROUND(AVG(CASE WHEN is_win = 1 THEN pnl_usd END), 2) as avg_win, "
            "  ROUND(AVG(CASE WHEN is_win = 0 THEN pnl_usd END), 2) as avg_loss, "
            "  ROUND(AVG(pnl_pct), 2) as avg_pnl_pct, "
            "  ROUND(MAX(pnl_usd), 2) as best_trade, "
            "  ROUND(MIN(pnl_usd), 2) as worst_trade, "
            "  ROUND(SUM(commission_usd), 2) as total_commission "
            "FROM strategy_trades GROUP BY strategy_name ORDER BY total_pnl DESC"
        )
        result = []
        for r in rows:
            d = dict(r)
            total = d.get("total_trades", 0)
            w = d.get("wins", 0)
            d["win_rate"] = round(w / total * 100, 1) if total > 0 else 0.0
            result.append(d)
        return result

    def get_all_trades_for_export(self, limit: int = 10000) -> list[dict]:
        """Get all trades for CSV export (strategy_trades + orders)."""
        rows = self._db.fetchall(
            "SELECT trade_id, strategy_name, symbol, market_regime, "
            "  timestamp_open, timestamp_close, entry_price, exit_price, "
            "  quantity, pnl_usd, pnl_pct, is_win, confidence, "
            "  exit_reason, hold_duration_hours, commission_usd "
            "FROM strategy_trades ORDER BY timestamp_close DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in rows]

    # ==================================================================
    # SIGNAL EXECUTIONS (audit trail)
    # ==================================================================

    def insert_signal_execution(
        self,
        timestamp: int,
        symbol: str,
        strategy_name: str,
        direction: str,
        confidence: float,
        outcome: str,
        reason: str = "",
        latency_ms: int = 0,
    ) -> int:
        """Record signal processing outcome for audit trail."""
        cur = self._db.execute(
            "INSERT INTO signal_executions "
            "(timestamp, symbol, strategy_name, direction, confidence, outcome, reason, latency_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (timestamp, symbol, strategy_name, direction, confidence, outcome, reason, latency_ms),
        )
        self._db.commit()
        return cur.lastrowid

    def get_signal_execution_stats(self, hours: int = 24) -> dict:
        """Get signal execution statistics for the last N hours."""
        cutoff = int(time.time() * 1000) - hours * 3600_000
        row = self._db.fetchone(
            "SELECT "
            "  COUNT(*) as total, "
            "  SUM(CASE WHEN outcome = 'filled' THEN 1 ELSE 0 END) as filled, "
            "  SUM(CASE WHEN outcome = 'rejected' THEN 1 ELSE 0 END) as rejected, "
            "  SUM(CASE WHEN outcome = 'error' THEN 1 ELSE 0 END) as errors, "
            "  ROUND(AVG(latency_ms), 0) as avg_latency_ms "
            "FROM signal_executions WHERE timestamp >= ?",
            (cutoff,),
        )
        if not row:
            return {"total": 0, "filled": 0, "rejected": 0, "errors": 0, "avg_latency_ms": 0}
        return dict(row)

    # ==================================================================
    # STATE RESTORATION (for strategy cold-start)
    # ==================================================================

    def get_last_filled_buy_ts(self, strategy_name: str, symbol: str) -> int:
        """Timestamp последнего исполненного BUY для стратегии+символа.

        Используется для восстановления cooldown-таймеров стратегий
        после перезапуска (например, DCA Bot _last_buy_time).
        """
        row = self._db.fetchone(
            "SELECT MAX(timestamp) as ts FROM signal_executions "
            "WHERE strategy_name = ? AND symbol = ? "
            "AND direction = 'BUY' AND outcome = 'filled'",
            (strategy_name, symbol),
        )
        return row["ts"] if row and row["ts"] else 0

    def count_buys_today(self, strategy_name: str, symbol: str) -> int:
        """Количество исполненных BUY сегодня для стратегии+символа.

        Используется для восстановления дневных лимитов (DCA max_daily_buys).
        """
        import datetime
        today_start_ms = int(
            datetime.datetime.now()
            .replace(hour=0, minute=0, second=0, microsecond=0)
            .timestamp() * 1000
        )
        row = self._db.fetchone(
            "SELECT COUNT(*) as cnt FROM signal_executions "
            "WHERE strategy_name = ? AND symbol = ? "
            "AND direction = 'BUY' AND outcome = 'filled' AND timestamp >= ?",
            (strategy_name, symbol, today_start_ms),
        )
        return row["cnt"] if row else 0

    # ==================================================================
    # ML MODEL REGISTRY
    # ==================================================================

    def insert_ml_model(self, **kwargs) -> int:
        cols = ", ".join(kwargs.keys())
        placeholders = ", ".join(["?"] * len(kwargs))
        cur = self._db.execute(
            f"INSERT INTO ml_model_registry ({cols}) VALUES ({placeholders})",
            tuple(kwargs.values()),
        )
        self._db.commit()
        return cur.lastrowid

    def get_active_ml_model(self) -> dict | None:
        row = self._db.fetchone(
            "SELECT * FROM ml_model_registry WHERE is_active = 1 "
            "ORDER BY created_at DESC LIMIT 1"
        )
        return dict(row) if row else None

    # ==================================================================
    # Retention / cleanup
    # ==================================================================

    def cleanup_old_data(self, trades_retention_days: int = 7, candles_retention_days: int = 90) -> dict:
        """Удалить устаревшие данные по политике хранения."""
        now_ms = int(time.time() * 1000)
        trades_cutoff = now_ms - trades_retention_days * 86_400_000
        candles_cutoff = now_ms - candles_retention_days * 86_400_000

        deleted_trades = self.delete_old_trades(trades_cutoff)
        deleted_candles = self.delete_old_candles(candles_cutoff)

        if deleted_trades or deleted_candles:
            log.info(
                "Cleanup: удалено {} trades (>{} дней), {} candles (>{} дней)",
                deleted_trades, trades_retention_days,
                deleted_candles, candles_retention_days,
            )
        return {"trades": deleted_trades, "candles": deleted_candles}
