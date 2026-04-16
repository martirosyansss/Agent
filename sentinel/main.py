"""
SENTINEL V1.5 — Точка входа.

Последовательность запуска (21 шаг) описана в ТЗ §26.2.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import signal
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

from loguru import logger

# Корень проекта — директория, где лежит main.py
BASE_DIR = Path(__file__).resolve().parent

# Добавляем корень проекта в sys.path, чтобы импорты работали
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import load_settings, Settings  # noqa: E402
from core.absolute_limits import FORBIDDEN_API_PERMISSIONS  # noqa: E402
from core.constants import (  # noqa: E402
    APP_NAME,
    EVENT_NEW_CANDLE,
    EVENT_NEW_SIGNAL,
    EVENT_NEW_TRADE,
    EVENT_ORDER_FILLED,
    VERSION,
    LOG_ROTATION_SIZE,
    LOG_ROTATION_COUNT,
    PID_FILE,
    STATE_FILE,
    HEARTBEAT_FILE,
)
from core.events import EventBus  # noqa: E402
from core.models import Direction, RiskState  # noqa: E402


BINANCE_REST_BASE = "https://api.binance.com"


# ──────────────────────────────────────────────
# Logging (loguru)
# ──────────────────────────────────────────────

def setup_logging(settings: Settings) -> None:
    """Настройка loguru: файлы + консоль."""
    logger.remove()  # убираем default stderr handler

    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)

    fmt = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | "
        "{extra[module]:<12} | {message}"
    )

    # Основной лог
    logger.add(
        log_dir / "sentinel.log",
        format=fmt,
        level=settings.log_level,
        rotation=LOG_ROTATION_SIZE,
        retention=LOG_ROTATION_COUNT,
        encoding="utf-8",
    )
    # Только ошибки
    logger.add(
        log_dir / "errors.log",
        format=fmt,
        level="ERROR",
        rotation=LOG_ROTATION_SIZE,
        retention=LOG_ROTATION_COUNT,
        encoding="utf-8",
    )
    # Сделки
    logger.add(
        log_dir / "trades.log",
        format=fmt,
        level="INFO",
        rotation=LOG_ROTATION_SIZE,
        retention=LOG_ROTATION_COUNT,
        filter=lambda r: r["extra"].get("module") in ("executor", "position"),
        encoding="utf-8",
    )
    # Risk
    logger.add(
        log_dir / "risk.log",
        format=fmt,
        level="INFO",
        rotation=LOG_ROTATION_SIZE,
        retention=LOG_ROTATION_COUNT,
        filter=lambda r: r["extra"].get("module") in ("risk", "circuit_breaker"),
        encoding="utf-8",
    )
    # Консоль (кратко)
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        level=settings.log_level,
        colorize=True,
    )

    logger.configure(extra={"module": "main"})


# ──────────────────────────────────────────────
# PID Lock
# ──────────────────────────────────────────────

def acquire_pid_lock() -> bool:
    """Проверяет, не запущен ли уже инстанс. Создаёт PID-файл."""
    pid_path = BASE_DIR / PID_FILE
    if pid_path.exists():
        try:
            old_pid = int(pid_path.read_text().strip())
            # Проверяем жив ли процесс (Windows-совместимо)
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x100000, False, old_pid)  # SYNCHRONIZE
            if handle:
                kernel32.CloseHandle(handle)
                logger.error("Другой инстанс уже запущен (PID {})", old_pid)
                return False
        except (ValueError, OSError, AttributeError) as exc:
            logger.warning("PID файл будет перезаписан: {}", exc)

    pid_path.write_text(str(os.getpid()))
    return True


def release_pid_lock() -> None:
    pid_path = BASE_DIR / PID_FILE
    pid_path.unlink(missing_ok=True)


# ──────────────────────────────────────────────
# State persistence
# ──────────────────────────────────────────────

def save_state(state: dict) -> None:
    """Сохранить текущее состояние системы в JSON."""
    state_path = BASE_DIR / STATE_FILE
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def load_state() -> dict:
    """Загрузить последнее состояние системы."""
    state_path = BASE_DIR / STATE_FILE
    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Не удалось загрузить state.json — стартуем с чистого состояния")
    return {}


# ──────────────────────────────────────────────
# Heartbeat
# ──────────────────────────────────────────────

async def heartbeat_writer(settings: Settings) -> None:
    """Пишет timestamp в heartbeat-файл каждые N секунд для Watchdog."""
    hb_path = BASE_DIR / HEARTBEAT_FILE
    hb_path.parent.mkdir(parents=True, exist_ok=True)
    interval = settings.watchdog_heartbeat_interval
    while True:
        # Atomic write: write to temp file, then rename
        tmp_path = hb_path.with_suffix(".tmp")
        tmp_path.write_text(str(int(time.time())), encoding="utf-8")
        tmp_path.replace(hb_path)
        await asyncio.sleep(interval)


def _build_signed_binance_params(
    settings: Settings,
    extra_params: dict[str, str | int] | None = None,
) -> list[tuple[str, str | int]]:
    """Собирает signed query params для Binance REST API."""
    params: list[tuple[str, str | int]] = list((extra_params or {}).items())
    params.extend([
        ("timestamp", int(time.time() * 1000)),
        ("recvWindow", 5000),
    ])
    query = urlencode(params)
    signature = hmac.new(
        settings.binance_api_secret.encode("utf-8"),
        query.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    params.append(("signature", signature))
    return params


async def _binance_get_json(
    client,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    params: list[tuple[str, str | int]] | None = None,
) -> dict:
    response = await client.get(path, headers=headers, params=params)
    response.raise_for_status()
    return response.json()


def _evaluate_binance_permissions(
    account_data: dict | None,
    restrictions_data: dict | None,
) -> tuple[bool, str]:
    """Проверяет, что API-ключ ограничен только разрешёнными правами."""
    if account_data and account_data.get("canTrade") is False:
        return False, "spot trading disabled for API key"

    if restrictions_data:
        forbidden_enabled: list[str] = []
        if restrictions_data.get("enableWithdrawals"):
            forbidden_enabled.append("withdraw")
        if restrictions_data.get("enableFutures"):
            forbidden_enabled.append("futures")
        if restrictions_data.get("enableMargin"):
            forbidden_enabled.append("margin")
        if forbidden_enabled:
            return False, f"forbidden API rights enabled: {', '.join(forbidden_enabled)}"
        return True, "api restrictions verified"

    permissions = {
        str(permission).lower()
        for permission in (account_data or {}).get("permissions", [])
    }
    forbidden = sorted(permissions.intersection(FORBIDDEN_API_PERMISSIONS))
    if forbidden:
        return False, f"forbidden permissions present: {', '.join(forbidden)}"

    if account_data and account_data.get("canWithdraw") is True:
        return False, "withdrawal access is enabled"

    return True, "account permissions verified"


async def _run_binance_preflight(settings: Settings) -> dict[str, int | bool | str]:
    """Выполняет сетевые pre-flight проверки Binance REST API."""
    import httpx

    result: dict[str, int | bool | str] = {
        "ping_ok": False,
        "auth_ok": False,
        "permissions_ok": False,
        "time_sync_ok": False,
        "time_drift_ms": -1,
        "permissions_reason": "not checked",
    }

    headers = {"X-MBX-APIKEY": settings.binance_api_key}
    timeout = httpx.Timeout(10.0, connect=5.0)
    async with httpx.AsyncClient(base_url=BINANCE_REST_BASE, timeout=timeout) as client:
        await _binance_get_json(client, "/api/v3/ping")
        result["ping_ok"] = True

        time_payload = await _binance_get_json(client, "/api/v3/time")
        server_time = int(time_payload.get("serverTime", 0))
        drift_ms = abs(server_time - int(time.time() * 1000)) if server_time else -1
        result["time_drift_ms"] = drift_ms
        result["time_sync_ok"] = 0 <= drift_ms <= 5_000

        account_payload = await _binance_get_json(
            client,
            "/api/v3/account",
            headers=headers,
            params=_build_signed_binance_params(settings),
        )
        result["auth_ok"] = True

        restrictions_payload: dict | None = None
        try:
            restrictions_payload = await _binance_get_json(
                client,
                "/sapi/v1/account/apiRestrictions",
                headers=headers,
                params=_build_signed_binance_params(settings),
            )
        except Exception as _restr_err:
            log.debug("API restrictions query failed (non-critical): {}", _restr_err)
            restrictions_payload = None

        permissions_ok, permissions_reason = _evaluate_binance_permissions(
            account_payload,
            restrictions_payload,
        )
        result["permissions_ok"] = permissions_ok
        result["permissions_reason"] = permissions_reason

    return result


def _check_sqlite_integrity(settings: Settings) -> bool:
    """Запускает реальную integrity-check для SQLite."""
    from database.db import Database

    db = Database(BASE_DIR / settings.db_path)
    try:
        db.connect()
        return db.integrity_check()
    finally:
        db.close()


# ──────────────────────────────────────────────
# Pre-flight checklist
# ──────────────────────────────────────────────

async def preflight_check(settings: Settings) -> bool:
    """
    Pre-flight checklist (ТЗ §26.3): 10 проверок.
    Возвращает True если все критичные прошли.
    """
    log = logger.bind(module="preflight")
    ok = True

    # [1/10] .env существует
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        log.info("[1/10] .env файл — ✅")
    else:
        log.error("[1/10] .env файл — ❌ (не найден)")
        ok = False

    # [2/10] API ключи формат
    if len(settings.binance_api_key) >= 10 and len(settings.binance_api_secret) >= 10:
        log.info("[2/10] API ключи формат — ✅")
    else:
        log.error("[2/10] API ключи формат — ❌ (слишком короткие)")
        ok = False

    try:
        binance_result = await _run_binance_preflight(settings)
    except Exception as exc:
        binance_result = {
            "ping_ok": False,
            "auth_ok": False,
            "permissions_ok": False,
            "time_sync_ok": False,
            "time_drift_ms": -1,
            "permissions_reason": str(exc),
        }

    if binance_result["ping_ok"]:
        log.info("[3/10] Binance API ping — ✅")
    else:
        log.error("[3/10] Binance API ping — ❌")
        ok = False

    if binance_result["auth_ok"]:
        log.info("[4/10] API auth test — ✅")
    else:
        log.error("[4/10] API auth test — ❌")
        ok = False

    if binance_result["permissions_ok"]:
        log.info("[5/10] API permissions — ✅ ({})", binance_result["permissions_reason"])
    else:
        log.error("[5/10] API permissions — ❌ ({})", binance_result["permissions_reason"])
        ok = False

    if binance_result["time_sync_ok"]:
        log.info("[6/10] Time sync — ✅ (drift={} ms)", binance_result["time_drift_ms"])
    else:
        log.error("[6/10] Time sync — ❌ (drift={} ms)", binance_result["time_drift_ms"])
        ok = False

    try:
        if _check_sqlite_integrity(settings):
            log.info("[7/10] SQLite integrity — ✅")
        else:
            log.error("[7/10] SQLite integrity — ❌")
            ok = False
    except Exception as exc:
        log.error("[7/10] SQLite integrity — ❌ ({})", exc)
        ok = False

    # [8/10] Свободное место на диске
    import shutil
    usage = shutil.disk_usage(BASE_DIR)
    free_gb = usage.free / (1024**3)
    if free_gb > 1.0:
        log.info("[8/10] Диск свободно {:.1f} GB — ✅", free_gb)
    else:
        log.error("[8/10] Диск свободно {:.1f} GB — ❌ (нужно > 1 GB)", free_gb)
        ok = False

    # [9/10] Telegram — не блокирует
    if settings.telegram_bot_token:
        log.info("[9/10] Telegram token задан — ✅")
    else:
        log.warning("[9/10] Telegram token не задан — ⚠️")

    # [10/10] PID lock
    if acquire_pid_lock():
        log.info("[10/10] PID lock — ✅")
    else:
        log.error("[10/10] PID lock — ❌ (другой инстанс?)")
        ok = False

    return ok


# ──────────────────────────────────────────────
# Graceful Shutdown
# ──────────────────────────────────────────────

class GracefulShutdown:
    """Обработчик Ctrl+C / SIGTERM."""

    def __init__(self) -> None:
        self._shutdown_event = asyncio.Event()

    @property
    def event(self) -> asyncio.Event:
        return self._shutdown_event

    def trigger(self) -> None:
        self._shutdown_event.set()

    def install_signal_handlers(self, loop: asyncio.AbstractEventLoop) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.trigger)
            except NotImplementedError:
                # Windows не поддерживает add_signal_handler
                signal.signal(sig, lambda s, f: self.trigger())


# ──────────────────────────────────────────────
# Uptime
# ──────────────────────────────────────────────

_BOOT_TIME = time.time()


def _format_uptime() -> str:
    elapsed = int(time.time() - _BOOT_TIME)
    days, rem = divmod(elapsed, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    return f"{minutes}m {seconds}s"


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

async def run() -> None:
    """Основной цикл SENTINEL."""
    settings = load_settings()
    setup_logging(settings)

    log = logger.bind(module="main")
    log.info("=== {} v{} ===", APP_NAME, VERSION)
    log.info("Режим: {}", settings.trading_mode.upper())

    # Pre-flight
    if not await preflight_check(settings):
        log.critical("Pre-flight check FAILED — система НЕ запускается")
        return

    # Загрузка состояния
    state = load_state()
    if state:
        log.info("Загружено предыдущее состояние: {} ключей", len(state))

    # EventBus
    bus = EventBus()

    # Graceful shutdown
    shutdown = GracefulShutdown()
    loop = asyncio.get_running_loop()
    shutdown.install_signal_handlers(loop)

    # ------------------------------------------------------------------
    # Инициализация модулей
    # ------------------------------------------------------------------

    # 6. SQLite (WAL mode)
    from database.db import Database
    from database.repository import Repository
    db = Database(BASE_DIR / settings.db_path)
    db.connect()
    repo = Repository(db)
    log.info("[Module] Database initialized")

    # 9. Data Collector (WebSocket)
    collector = None
    collector_task: asyncio.Task | None = None
    try:
        from collector.binance_ws import BinanceWebSocketCollector
        collector = BinanceWebSocketCollector(
            symbols=settings.trading_symbols,
            repo=repo,
            bus=bus,
        )
        collector_task = asyncio.create_task(collector.start())
        log.info("[Module] Collector started for {}", settings.trading_symbols)
    except Exception as e:
        log.warning("[Module] Collector failed to start: {}", e)

    # 15. Execution Engine (paper mode)
    executor = None
    try:
        from execution.paper_executor import PaperExecutor
        executor = PaperExecutor(
            event_bus=bus,
            commission_pct=settings.paper_commission_pct,
        )
        log.info("[Module] Paper Executor initialized")
    except Exception as e:
        log.warning("[Module] Paper Executor failed: {}", e)

    # 16. Position / Risk runtime
    position_manager = None
    risk_state_machine = None
    risk_sentinel = None
    try:
        from position.manager import PositionManager
        from risk.sentinel import RiskLimits, RiskSentinel
        from risk.state_machine import RiskStateMachine

        position_manager = PositionManager(
            event_bus=bus,
            initial_balance=settings.paper_initial_balance,
            max_open_positions=settings.max_open_positions,
        )

        # Restore open positions from DB after restart (atomic)
        if repo:
            try:
                from core.models import Position as _Pos, PositionStatus as _PS
                saved_positions = await asyncio.to_thread(repo.get_open_positions)
                _positions_to_restore: list[_Pos] = []
                _total_cost = 0.0
                for row in saved_positions:
                    _ep = row["entry_price"]
                    _qty = row["quantity"]
                    if _ep <= 0 or _qty <= 0:
                        log.warning("Skipping invalid DB position: entry={} qty={}", _ep, _qty)
                        continue
                    pos = _Pos(
                        symbol=row["symbol"],
                        side=row["side"],
                        entry_price=_ep,
                        quantity=_qty,
                        current_price=row.get("current_price") or _ep,
                        unrealized_pnl=row.get("unrealized_pnl", 0.0),
                        realized_pnl=row.get("realized_pnl", 0.0),
                        stop_loss_price=row.get("stop_loss_price", 0.0),
                        take_profit_price=row.get("take_profit_price", 0.0),
                        strategy_name=row.get("strategy_name", ""),
                        signal_id=row.get("signal_id", ""),
                        signal_reason=row.get("signal_reason", ""),
                        status=_PS.OPEN,
                        opened_at=row.get("opened_at", ""),
                        is_paper=bool(row.get("is_paper", 1)),
                    )
                    if row.get("position_id"):
                        pos.position_id = row["position_id"]
                    pos.db_id = row["id"]
                    _positions_to_restore.append(pos)
                    _total_cost += _qty * _ep

                # Validate: total cost must not exceed initial balance
                if _total_cost > position_manager.wallet.initial_balance:
                    log.error(
                        "Restored positions cost ${:.2f} > initial balance ${:.2f} — skipping restore",
                        _total_cost, position_manager.wallet.initial_balance,
                    )
                else:
                    # Atomic: apply all positions at once
                    for pos in _positions_to_restore:
                        position_manager._positions[pos.symbol] = pos
                        sl = pos.stop_loss_price
                        tp = pos.take_profit_price
                        if sl > 0 or tp > 0:
                            position_manager._sl_tp[pos.symbol] = (sl, tp)
                    position_manager.wallet.usdt_balance -= _total_cost
                    if _positions_to_restore:
                        log.info("Restored {} open position(s) from DB (cost=${:.2f})",
                                 len(_positions_to_restore), _total_cost)
            except Exception as _restore_err:
                log.warning("Failed to restore positions from DB: {}", _restore_err)

        risk_state_machine = RiskStateMachine(
            event_bus=bus,
            max_daily_loss=settings.max_daily_loss_usd,
        )
        risk_sentinel = RiskSentinel(
            limits=RiskLimits(
                max_daily_loss_usd=settings.max_daily_loss_usd,
                max_daily_loss_pct=settings.max_daily_loss_pct,
                max_daily_trades=settings.max_trades_per_day,
                max_position_pct=settings.max_position_pct,
                max_total_exposure_pct=settings.max_total_exposure_pct,
                max_open_positions=settings.max_open_positions,
                max_trades_per_hour=settings.max_trades_per_hour,
                min_trade_interval_sec=settings.resume_cooldown_min * 60,
                max_order_usd=settings.max_order_usd,
                max_loss_per_trade_pct=settings.stop_loss_pct,
                max_daily_commission_pct=settings.cb_commission_alert_pct,
            ),
            state_machine=risk_state_machine,
        )

        # Stores {(symbol, strategy): (probability, entry_price)} for open positions
        # so we can feed the outcome back to LivePerformanceTracker when the trade closes.
        _ml_prob_at_entry: dict[tuple, tuple] = {}

        async def _on_order_filled(order):
            if not position_manager:
                return

            # Persist order to DB — critical for trade audit trail
            if repo:
                try:
                    await asyncio.to_thread(repo.insert_order, order)
                except Exception as _db_err:
                    log.critical("Order DB write FAILED: {} — halting trading", _db_err)
                    nonlocal trading_paused
                    trading_paused = True
                    return

            if order.side == Direction.BUY:
                opened = await position_manager.open_position(order)
                if opened and risk_sentinel:
                    risk_sentinel.record_trade(order.commission, increment_trade=True)
                # Setup trailing stop for strategies that support it
                if opened and order.strategy_name == "ema_crossover_rsi":
                    position_manager.set_trailing_stop(order.symbol, 2.5, 1.5)
                elif opened and order.strategy_name == "bollinger_breakout":
                    position_manager.set_trailing_stop(order.symbol, 3.0, 2.0)
                # Persist opened position to DB
                if opened and repo:
                    try:
                        db_id = await asyncio.to_thread(repo.insert_position, opened)
                        opened.db_id = db_id
                        log.info("Position persisted to DB: {} id={}", opened.symbol, db_id)
                    except Exception as _db_err:
                        log.error("Position DB write FAILED for {}: {}", order.symbol, _db_err)
            elif order.side == Direction.SELL:
                # Capture position data BEFORE close (close deletes from dict)
                _pos_before = position_manager.get_position(order.symbol)
                _entry_px_pos = _pos_before.entry_price if _pos_before else 0.0
                _opened_at = _pos_before.opened_at if _pos_before else ""
                _strat_name = _pos_before.strategy_name if _pos_before else order.strategy_name
                _signal_id = _pos_before.signal_id if _pos_before else ""
                _signal_reason = _pos_before.signal_reason if _pos_before else ""
                _pos_db_id = _pos_before.db_id if _pos_before else None

                closed = await position_manager.close_position(order)
                if closed and risk_sentinel:
                    risk_sentinel.record_trade(order.commission, increment_trade=False)

                # Update position status in DB
                if closed and repo and _pos_db_id:
                    try:
                        await asyncio.to_thread(
                            repo.close_position,
                            _pos_db_id, closed.realized_pnl, closed.closed_at or "",
                        )
                    except Exception as _db_err:
                        log.warning("Position DB close failed: {}", _db_err)

                # Persist completed trade to strategy_trades DB
                if closed and repo:
                    try:
                        from core.models import StrategyTrade as _ST
                        import uuid as _uuid
                        _fill_px = order.fill_price or order.price or 0
                        _fill_qty = order.fill_quantity or order.quantity
                        _pnl = closed.realized_pnl
                        _cost_basis = _entry_px_pos * _fill_qty
                        _pnl_pct = (_pnl / _cost_basis * 100) if _cost_basis > 0 else 0.0
                        _hold_ms = (int(closed.closed_at or 0) - int(_opened_at or 0)) if _opened_at else 0
                        _hold_hours = max(_hold_ms / 3_600_000, 0)
                        _regime_name = _current_regime.regime.value if _current_regime else "unknown"

                        # Look up entry confidence from signal_executions
                        _entry_conf = 0.0
                        if _signal_id:
                            try:
                                _sig_row = await asyncio.to_thread(
                                    repo._db.fetchone,
                                    "SELECT confidence FROM signal_executions "
                                    "WHERE symbol = ? AND outcome = 'filled' "
                                    "ORDER BY timestamp DESC LIMIT 1",
                                    (order.symbol,),
                                )
                                if _sig_row and _sig_row["confidence"]:
                                    _entry_conf = float(_sig_row["confidence"])
                            except Exception:
                                pass

                        _st = _ST(
                            trade_id=_uuid.uuid4().hex[:12],
                            signal_id=_signal_id,
                            symbol=order.symbol,
                            strategy_name=_strat_name,
                            market_regime=_regime_name,
                            timestamp_open=_opened_at,
                            timestamp_close=closed.closed_at or "",
                            entry_price=_entry_px_pos,
                            exit_price=_fill_px,
                            quantity=_fill_qty,
                            pnl_usd=round(_pnl, 4),
                            pnl_pct=round(_pnl_pct, 4),
                            is_win=_pnl > 0,
                            confidence=_entry_conf,
                            hour_of_day=int(time.strftime("%H")),
                            day_of_week=int(time.strftime("%w")),
                            exit_reason=order.signal_reason or "",
                            hold_duration_hours=round(_hold_hours, 2),
                            commission_usd=order.commission,
                        )
                        await asyncio.to_thread(repo.insert_strategy_trade, _st)
                        log.info("Trade saved to DB: {} {} PnL=${:.2f} ({:.2f}%)",
                                 order.symbol, _strat_name, _pnl, _pnl_pct)
                    except Exception as _db_err:
                        log.warning("Strategy trade DB write failed: {}", _db_err)

                # Feed realized outcome back to ML tracker for concept drift detection
                if closed:
                    # Find matching ML entry by symbol (any strategy key)
                    _matched_key = None
                    for _k in list(_ml_prob_at_entry.keys()):
                        if _k[0] == order.symbol:
                            _matched_key = _k
                            break
                    if _matched_key is not None:
                        _entry_prob, _entry_px = _ml_prob_at_entry.pop(_matched_key, (0.5, 0.0))
                        # Use realized PnL as source of truth (works for both LONG and SHORT)
                        _actual_win = closed.realized_pnl > 0
                        # Record to per-symbol model if available, and always to unified
                        _sym_ml = _ml_predictors.get(order.symbol)
                        if _sym_ml:
                            _sym_ml.record_outcome(_entry_prob, _actual_win)
                        if _ml_predictor:
                            _ml_predictor.record_outcome(_entry_prob, _actual_win)

                        # ML auto-promote: shadow → block after 30+ trades with live precision >= training precision * 0.9
                        for _auto_ml in [_sym_ml, _ml_predictor]:
                            if _auto_ml and _auto_ml.rollout_mode == "shadow" and _auto_ml._live_tracker.n_recorded >= 30:
                                _live = _auto_ml._live_tracker.live_metrics()
                                if "live_precision" in _live and _auto_ml.metrics:
                                    _train_prec = _auto_ml.metrics.precision
                                    if _live["live_precision"] >= _train_prec * 0.9 and _live["live_auc"] >= 0.60:
                                        _auto_ml.rollout_mode = "block"
                                        log.info("ML AUTO-PROMOTE: shadow → block (live_prec={:.3f} >= {:.3f}, auc={:.3f}, n={})",
                                                 _live["live_precision"], _train_prec * 0.9, _live["live_auc"], _live["n"])

            if risk_state_machine:
                await risk_state_machine.update(position_manager.total_realized_pnl)

        async def _on_market_trade(trade):
            if not position_manager:
                return
            await position_manager.update_price(trade.symbol, trade.price)
            if risk_state_machine:
                await risk_state_machine.update(position_manager.total_realized_pnl)

        bus.subscribe(EVENT_ORDER_FILLED, _on_order_filled)
        bus.subscribe(EVENT_NEW_TRADE, _on_market_trade)
        log.info("[Module] Position & Risk runtime initialized")
    except Exception as e:
        log.warning("[Module] Position/Risk runtime failed: {}", e)

    # ------------------------------------------------------------------
    # 17. Trading Loop — стратегии, фичи, сигналы, исполнение
    # ------------------------------------------------------------------
    trading_paused = False  # управляется через Start/Stop в Dashboard

    # Инициализация стратегий
    strategies = {}
    feature_builder = None
    try:
        from features.feature_builder import FeatureBuilder
        from strategy.ema_crossover_rsi import EMACrossoverRSI, EMAConfig
        from strategy.bollinger_breakout import BollingerBreakout
        from strategy.mean_reversion import MeanReversion
        from strategy.dca_bot import DCABot
        from strategy.macd_divergence import MACDDivergence
        from strategy.grid_trading import GridTrading
        from strategy.market_regime import detect_regime
        from strategy.strategy_selector import get_active_strategies, get_strategy_budget_pct, AdaptiveAllocator

        feature_builder = FeatureBuilder()

        strategies["ema_crossover_rsi"] = EMACrossoverRSI(EMAConfig(
            stop_loss_pct=settings.stop_loss_pct,
            take_profit_pct=settings.take_profit_pct,
            min_confidence=settings.min_confidence,
        ))
        strategies["bollinger_breakout"] = BollingerBreakout()
        strategies["mean_reversion"] = MeanReversion()
        strategies["dca_bot"] = DCABot()
        strategies["macd_divergence"] = MACDDivergence()
        strategies["grid_trading"] = GridTrading()

        # ── Restore DCA Bot state from DB (prevent duplicate orders on restart) ──
        if repo:
            dca: DCABot = strategies["dca_bot"]
            _last_buys: dict[str, int] = {}
            _today_buys: dict[str, int] = {}
            for sym in settings.trading_symbols:
                _last_buys[sym] = repo.get_last_filled_buy_ts(DCABot.NAME, sym)
                _today_buys[sym] = repo.count_buys_today(DCABot.NAME, sym)
            dca.restore_state(_last_buys, _today_buys)
            _restored = {s: t for s, t in _last_buys.items() if t > 0}
            if _restored:
                log.info("[DCA] State restored from DB: last_buys={}, daily={}", _restored, _today_buys)
            else:
                log.info("[DCA] No prior buy history found — fresh start")

        log.info("[Module] Strategies initialized: {}", list(strategies.keys()))
    except Exception as e:
        log.warning("[Module] Strategies failed to initialize: {}", e)

    _current_regime = None
    _last_features = None  # последний FeatureVector для дашборда (backward compat)
    _last_features_per_symbol: dict = {}  # symbol → FeatureVector
    _adaptive_allocator = None
    try:
        _adaptive_allocator = AdaptiveAllocator(lookback_trades=50)
        log.info("[Module] AdaptiveAllocator initialized")
    except Exception as e:
        log.warning("[Module] AdaptiveAllocator failed: {}", e)

    # Position sizer & dynamic SL/TP & alerts
    _position_sizer = None
    _alert_monitor = None
    try:
        from risk.position_sizer import calculate_position_size, SizingInput
        from risk.dynamic_sltp import calculate_dynamic_sltp
        from monitoring.alerts import AlertMonitor
        _position_sizer = True  # flag — functions imported
        _alert_monitor = AlertMonitor()
        log.info("[Module] Position sizer + Dynamic SL/TP + Alert monitor initialized")
    except Exception as e:
        log.warning("[Module] Advanced risk modules failed: {}", e)

    # ML Predictor — фильтрация сигналов (per-symbol models + unified fallback)
    _ml_predictor = None          # unified fallback (backward compat)
    _ml_predictors: dict = {}     # symbol → MLPredictor
    try:
        from analyzer.ml_predictor import MLPredictor, MLConfig
        _ml_cfg = MLConfig(
            block_threshold=settings.analyzer_ml_block_threshold,
            min_precision=settings.analyzer_ml_min_precision,
            min_recall=settings.analyzer_ml_min_recall,
            min_roc_auc=settings.analyzer_ml_min_roc_auc,
            min_skill_score=settings.analyzer_ml_min_skill_score,
            min_trades=settings.analyzer_min_trades_ml,
            retrain_days=settings.analyzer_ml_retrain_days,
            test_window_days=settings.analyzer_ml_test_window_days,
        )
        _rollout = "shadow" if settings.analyzer_ml_shadow_mode else ("block" if settings.analyzer_ml_enabled else "off")
        _ml_models_dir = Path(__file__).parent / "data" / "ml_models"

        # Load per-symbol models
        for _sym in (settings.trading_symbols or []):
            _sym_path = _ml_models_dir / f"ml_predictor_{_sym}.pkl"
            if _sym_path.exists():
                _sym_predictor = MLPredictor(config=_ml_cfg)
                _sym_predictor.rollout_mode = _rollout
                if _sym_predictor.load_from_file(_sym_path):
                    _ml_predictors[_sym] = _sym_predictor
                    log.info("[Module] ML model for {} loaded (version={})", _sym, _sym_predictor._model_version)
                else:
                    log.warning("[Module] ML model load failed for {}", _sym)

        # Load unified fallback model
        _ml_predictor = MLPredictor(config=_ml_cfg)
        _ml_predictor.rollout_mode = _rollout
        _ml_model_path = _ml_models_dir / "ml_predictor.pkl"
        if _ml_model_path.exists():
            loaded = _ml_predictor.load_from_file(_ml_model_path)
            if loaded:
                log.info("[Module] ML unified model loaded (version={})", _ml_predictor._model_version)
            else:
                log.warning("[Module] ML unified model load failed")
        else:
            log.warning("[Module] No saved ML model found at {}", _ml_model_path)

        log.info("[Module] ML Predictors: {} per-symbol + unified fallback (mode={})",
                 len(_ml_predictors), _ml_predictor.rollout_mode)
    except Exception as e:
        log.warning("[Module] ML Predictor failed: {}", e)

    # ── ML retrain function (defined early so get_system_state can reference it) ──
    _ml_model_path_unified = Path(__file__).parent / "data" / "ml_models" / "ml_predictor.pkl"
    _ml_retrain_lock = asyncio.Lock()

    async def _run_ml_training() -> bool:
        """Load trades from DB, train per-symbol + unified ML models, save to disk."""
        if not _ml_predictor or not repo:
            return False
        try:
            import sys as _sys
            _scripts_dir = str(Path(__file__).parent / "scripts")
            if _scripts_dir not in _sys.path:
                _sys.path.insert(0, _scripts_dir)
            from scripts.train_ml import build_trades_per_symbol
            log.info("ML auto-retrain: collecting training data...")
            trades_by_sym = await asyncio.to_thread(build_trades_per_symbol, repo, settings)
            if not trades_by_sym:
                log.warning("ML auto-retrain: no trades available for training")
                return False

            _ml_models_dir = Path(__file__).parent / "data" / "ml_models"
            _ml_models_dir.mkdir(parents=True, exist_ok=True)
            any_saved = False

            # Train per-symbol models
            for sym, sym_trades in trades_by_sym.items():
                if len(sym_trades) < 50:
                    log.info("ML auto-retrain: {} — too few trades ({}), skip", sym, len(sym_trades))
                    continue
                # Use existing per-symbol predictor or create new one
                sym_predictor = _ml_predictors.get(sym)
                if sym_predictor is None:
                    from analyzer.ml_predictor import MLPredictor as _MLP, MLConfig as _MLC
                    sym_predictor = _MLP(config=_ml_predictor._cfg)
                    sym_predictor.rollout_mode = _ml_predictor.rollout_mode
                log.info("ML auto-retrain: training {} on {} trades...", sym, len(sym_trades))
                metrics = await asyncio.to_thread(sym_predictor.train, sym_trades)
                if metrics is None or not sym_predictor.is_ready:
                    log.warning("ML auto-retrain: {} — training failed or below threshold", sym)
                    continue
                sym_path = _ml_models_dir / f"ml_predictor_{sym}.pkl"
                saved = await asyncio.to_thread(sym_predictor.save_to_file, sym_path)
                if saved:
                    _ml_predictors[sym] = sym_predictor
                    log.info("ML auto-retrain: ✅ {} saved (skill={:.3f})", sym, metrics.skill_score)
                    any_saved = True

            # Train unified fallback
            all_trades = []
            for sym_trades in trades_by_sym.values():
                all_trades.extend(sym_trades)
            all_trades.sort(key=lambda t: t.timestamp_open)
            if all_trades:
                log.info("ML auto-retrain: training unified on {} trades...", len(all_trades))
                metrics = await asyncio.to_thread(_ml_predictor.train, all_trades)
                if metrics is not None and _ml_predictor.is_ready:
                    saved = await asyncio.to_thread(_ml_predictor.save_to_file, _ml_model_path_unified)
                    if saved:
                        log.info("ML auto-retrain: ✅ unified saved (skill={:.3f})", metrics.skill_score)
                        any_saved = True

            return any_saved
        except Exception as exc:
            log.error("ML auto-retrain error: {}", exc)
            return False

    # ── Strategy Decision Log (ring buffer) ──
    from collections import deque
    _strategy_log: deque[dict] = deque(maxlen=50)

    # news_collector будет инициализирован позже (вместе с Dashboard)
    news_collector = None

    async def _on_new_candle(candle):
        """Главный торговый цикл — вызывается при закрытии каждой свечи."""
        nonlocal _current_regime, trading_paused, _last_features, _last_features_per_symbol

        if trading_paused:
            return

        if not feature_builder or not executor or not position_manager or not risk_sentinel:
            return

        # Только 1h свечи запускают стратегии (signal_timeframe)
        if candle.interval != settings.signal_timeframe:
            return

        symbol = candle.symbol
        ts_now = int(time.time() * 1000)
        log.info("Trading loop triggered: {} {} candle closed @ {:.2f}", symbol, candle.interval, candle.close)

        # 1. Собрать свечи из БД
        try:
            candles_1h_raw = await asyncio.to_thread(
                repo.get_candles, symbol, "1h", limit=60
            )
            candles_4h_raw = await asyncio.to_thread(
                repo.get_candles, symbol, "4h", limit=60
            )
            candles_1d_raw = await asyncio.to_thread(
                repo.get_candles, symbol, "1d", limit=60
            )
        except Exception as e:
            log.warning("Failed to fetch candles for {}: {}", symbol, e)
            _strategy_log.append({"ts": ts_now, "symbol": symbol, "event": "error", "msg": f"Failed to fetch candles: {e}"})
            return

        if not candles_1h_raw or not candles_4h_raw:
            log.info("Not enough candles for {} (1h={}, 4h={})", symbol, len(candles_1h_raw or []), len(candles_4h_raw or []))
            _strategy_log.append({"ts": ts_now, "symbol": symbol, "event": "skip", "msg": f"Not enough candles (1h={len(candles_1h_raw or [])}, 4h={len(candles_4h_raw or [])})"})
            return

        # Конвертировать dict → Candle
        from core.models import Candle as CandleModel
        candles_1h = [
            CandleModel(
                timestamp=c["timestamp"], symbol=c["symbol"], interval=c["interval"],
                open=float(c["open"]), high=float(c["high"]), low=float(c["low"]),
                close=float(c["close"]), volume=float(c.get("volume", 0)),
                trades_count=int(c.get("trades_count", 0)),
            ) for c in candles_1h_raw
        ]
        candles_4h = [
            CandleModel(
                timestamp=c["timestamp"], symbol=c["symbol"], interval=c["interval"],
                open=float(c["open"]), high=float(c["high"]), low=float(c["low"]),
                close=float(c["close"]), volume=float(c.get("volume", 0)),
                trades_count=int(c.get("trades_count", 0)),
            ) for c in candles_4h_raw
        ]

        # Convert daily candles (may be empty, feature_builder handles None)
        candles_1d = [
            CandleModel(
                timestamp=c["timestamp"], symbol=c["symbol"], interval=c["interval"],
                open=float(c["open"]), high=float(c["high"]), low=float(c["low"]),
                close=float(c["close"]), volume=float(c.get("volume", 0)),
                trades_count=int(c.get("trades_count", 0)),
            ) for c in (candles_1d_raw or [])
        ] or None

        # 2. FeatureBuilder (with daily candles for multi-TF)
        features = feature_builder.build(symbol, candles_1h, candles_4h, candles_1d)
        if features is None:
            log.info("FeatureBuilder returned None for {} (1h={}, 4h={} candles)", symbol, len(candles_1h), len(candles_4h))
            _strategy_log.append({"ts": ts_now, "symbol": symbol, "event": "skip", "msg": "FeatureBuilder: not enough data to compute indicators"})
            return

        _last_features = features
        _last_features_per_symbol[symbol] = features

        # 2.5. Обогащение FeatureVector данными из NewsCollector
        if news_collector:
            try:
                sentiment_data = news_collector.get_sentiment()
                impact_data = news_collector.get_impact_summary()
                news_signal = news_collector.get_news_signal()
                features.news_sentiment = sentiment_data.get("overall_score", 0.0)
                features.fear_greed_index = sentiment_data.get("fear_greed_index", 50)
                features.news_impact_pct = impact_data.get("avg_impact_pct", 0.0)
                features.high_impact_news = impact_data.get("high_impact_count", 0)
                # Pro fields
                features.news_composite_score = news_signal.get("composite_score", 0.0)
                features.news_signal_strength = news_signal.get("signal_strength", 0.0)
                features.news_critical_alert = news_signal.get("critical_alert", False)
                features.news_actionable = news_signal.get("actionable", False)
                features.news_dominant_category = news_signal.get("dominant_category", "")
                # Update news timestamp for time decay in confidence adjustment
                from strategy.base_strategy import update_news_timestamp
                update_news_timestamp()
            except Exception as e:
                log.debug("News sentiment enrichment failed: {}", e)

        # 3. Market Regime Detection
        try:
            _current_regime = detect_regime(features)
            log.debug("Regime for {}: {}", symbol, _current_regime.regime.value)
        except Exception as e:
            log.warning("Regime detection failed: {}", e)

        regime_name = _current_regime.regime.value if _current_regime else "unknown"

        # 4. Определить активные стратегии
        if settings.auto_strategy_selection and _current_regime:
            if _adaptive_allocator and _adaptive_allocator._skill_scores:
                allocs = _adaptive_allocator.get_adaptive_allocations(_current_regime)
                active_names = [a.strategy_name for a in allocs if a.is_active]
            else:
                active_names = get_active_strategies(_current_regime)
        else:
            # Запускаем все стратегии, у которых есть инициализированный объект
            # (стратегии фильтруются через .env флаги при инициализации)
            active_names = list(strategies.keys())

        # 5. Проверить позицию
        has_position = position_manager.has_position(symbol)
        pos = position_manager.get_position(symbol)
        entry_price = pos.entry_price if pos else None

        # 6. Генерировать сигналы от каждой активной стратегии
        # Sort: strategy that opened current position goes first (SELL priority),
        # so exit signals are evaluated before new entries from other strategies.
        _pos_strategy = pos.strategy_name if pos else None
        if _pos_strategy and _pos_strategy in active_names:
            active_names = [_pos_strategy] + [n for n in active_names if n != _pos_strategy]
        signal_found = False
        strat_results = []
        for strat_name in active_names:
            strat = strategies.get(strat_name)
            if not strat:
                continue

            try:
                signal = strat.generate_signal(
                    features,
                    has_open_position=has_position,
                    entry_price=entry_price,
                )
            except Exception as e:
                log.warning("Strategy {} error: {}", strat_name, e)
                strat_results.append({"strategy": strat_name, "result": "error", "detail": str(e)})
                continue

            if signal is None:
                strat_results.append({"strategy": strat_name, "result": "no_signal"})
                continue

            # For SELL signals, use current position quantity
            if signal.suggested_quantity <= 0 and signal.direction == Direction.SELL:
                if pos and pos.quantity > 0:
                    signal.suggested_quantity = pos.quantity
                else:
                    log.warning("SELL signal from {} but no open position for {}", strat_name, symbol)
                    strat_results.append({"strategy": strat_name, "result": "no_position"})
                    continue

            # Compute position size with real Kelly params from recent trades
            if signal.suggested_quantity <= 0 and signal.direction == Direction.BUY:
                balance = position_manager.balance
                if _position_sizer and features.atr > 0:
                    _kelly_win_rate = 0.45
                    _kelly_avg_win = 2.5
                    _kelly_avg_loss = 2.5
                    try:
                        _recent = await asyncio.to_thread(
                            repo.get_strategy_trades, strat_name, limit=50
                        ) if repo else []
                        if len(_recent) >= 30:
                            _wins = [t for t in _recent if t.get("pnl_pct", 0) > 0]
                            _losses = [t for t in _recent if t.get("pnl_pct", 0) <= 0]
                            _kelly_win_rate = len(_wins) / len(_recent) if _recent else 0.5
                            _kelly_avg_win = sum(t.get("pnl_pct", 0) for t in _wins) / len(_wins) if _wins else 3.0
                            _kelly_avg_loss = abs(sum(t.get("pnl_pct", 0) for t in _losses) / len(_losses)) if _losses else 2.0
                    except Exception as _kelly_err:
                        log.debug("Kelly stats fetch failed: {}", _kelly_err)

                    # Collect currently open position symbols for correlation correction
                    _open_syms = [p.symbol for p in position_manager.open_positions] if position_manager else []
                    # Count consecutive losses from recent trades (for loss streak dampener)
                    _consec_losses = 0
                    if _recent:
                        for _t in reversed(_recent):
                            if _t.get("pnl_pct", 0) <= 0:
                                _consec_losses += 1
                            else:
                                break
                    sizing = calculate_position_size(SizingInput(
                        balance=balance,
                        price=features.close,
                        atr=features.atr,
                        win_rate=_kelly_win_rate,
                        avg_win_pct=_kelly_avg_win,
                        avg_loss_pct=max(_kelly_avg_loss, 0.1),
                        regime_adx=features.adx,
                        max_position_pct=settings.max_position_pct,
                        max_order_usd=settings.max_order_usd,
                        symbol=symbol,
                        open_symbols=_open_syms,
                        consecutive_losses=_consec_losses,
                    ))
                    signal.suggested_quantity = sizing.quantity
                    log.debug("Position sizer: {} budget=${:.2f} ({:.1f}%) kelly={:.3f}",
                              sizing.method, sizing.budget_usd, sizing.budget_pct, sizing.kelly_fraction)
                else:
                    if _current_regime and settings.auto_strategy_selection:
                        budget_pct = get_strategy_budget_pct(_current_regime, strat_name)
                    else:
                        budget_pct = settings.max_position_pct
                    budget_usd = balance * budget_pct / 100
                    budget_usd = min(budget_usd, settings.max_order_usd)
                    if features.close > 0:
                        signal.suggested_quantity = budget_usd / features.close

            # Dynamic SL/TP — merge with strategy-calculated values (news-adjusted)
            # Use tighter SL (higher price = closer to entry) and wider TP (higher price)
            if _position_sizer and signal.direction == Direction.BUY:
                sltp = calculate_dynamic_sltp(
                    entry_price=features.close,
                    atr=features.atr,
                    strategy_name=strat_name,
                    fallback_sl_pct=settings.stop_loss_pct,
                    fallback_tp_pct=settings.take_profit_pct,
                )
                # Merge: keep the tighter (more protective) SL and wider TP
                if signal.stop_loss_price > 0:
                    signal.stop_loss_price = max(signal.stop_loss_price, sltp.stop_loss_price)
                else:
                    signal.stop_loss_price = sltp.stop_loss_price
                if signal.take_profit_price > 0:
                    signal.take_profit_price = max(signal.take_profit_price, sltp.take_profit_price)
                else:
                    signal.take_profit_price = sltp.take_profit_price
                log.debug("Dynamic SL/TP [{}]: SL={:.2f} ({:.1f}%) TP={:.2f} ({:.1f}%) (merged with strategy)",
                          sltp.method, signal.stop_loss_price, sltp.stop_loss_pct,
                          signal.take_profit_price, sltp.take_profit_pct)

            # 6.5 ML Predictor filter (per-symbol model with unified fallback)
            _active_ml = _ml_predictors.get(symbol, _ml_predictor)
            if _active_ml and settings.analyzer_ml_enabled and _active_ml.rollout_mode != "off":
                try:
                    from core.models import StrategyTrade as _ST
                    _ml_trade = _ST(
                        trade_id="pending",
                        symbol=symbol,
                        strategy_name=strat_name,
                        market_regime=regime_name,
                        entry_price=features.close,
                        confidence=signal.confidence,
                        hour_of_day=int(time.strftime("%H")),
                        day_of_week=int(time.strftime("%w")),
                        rsi_at_entry=features.rsi_14,
                        adx_at_entry=features.adx,
                        volume_ratio_at_entry=features.volume_ratio,
                        news_sentiment=features.news_sentiment,
                        fear_greed_index=features.fear_greed_index,
                    )
                    _prev_trades_for_ml = []
                    if repo:
                        try:
                            _raw = await asyncio.to_thread(repo.get_strategy_trades, strat_name, limit=20)
                            _prev_trades_for_ml = [_ST(**t) if isinstance(t, dict) else t for t in _raw]
                        except Exception as _ml_hist_err:
                            log.debug("ML history fetch failed: {}", _ml_hist_err)
                    _ml_features = _active_ml.extract_features(_ml_trade, _prev_trades_for_ml)
                    _ml_pred = _active_ml.predict(_ml_features)
                    log.info("ML prediction: {} prob={:.2f} decision={} mode={}",
                             strat_name, _ml_pred.probability, _ml_pred.decision, _ml_pred.rollout_mode)
                    # Track ML probability at entry so we can record outcome on close.
                    # Store (prob, entry_price) tuple — used in _on_order_filled SELL path.
                    if signal.direction.value == "BUY":
                        _ml_prob_at_entry[(symbol, strat_name)] = (_ml_pred.probability, features.close)
                    if _ml_pred.decision == "block":
                        log.info("ML BLOCKED signal: {} {} {} prob={:.2f}",
                                 strat_name, signal.direction.value, symbol, _ml_pred.probability)
                        strat_results.append({"strategy": strat_name, "result": "ml_blocked",
                                              "direction": signal.direction.value,
                                              "detail": f"ML blocked: prob={_ml_pred.probability:.2f}"})
                        if repo:
                            try:
                                repo.insert_signal_execution(
                                    timestamp=ts_now, symbol=symbol, strategy_name=strat_name,
                                    direction=signal.direction.value, confidence=signal.confidence,
                                    outcome="ml_blocked", reason=f"ML prob={_ml_pred.probability:.2f}",
                                )
                            except Exception as _ml_audit_err:
                                log.debug("ML block audit write failed: {}", _ml_audit_err)
                        continue
                except Exception as e:
                    log.debug("ML prediction failed: {}", e)

            # 6.7 Global min_confidence enforcement (catches edge cases)
            if signal.direction == Direction.BUY and signal.confidence < settings.min_confidence:
                log.debug("Signal SKIPPED: {} conf={:.2f} < min {:.2f}",
                          strat_name, signal.confidence, settings.min_confidence)
                strat_results.append({"strategy": strat_name, "result": "low_confidence",
                                      "detail": f"conf={signal.confidence:.2f} < {settings.min_confidence}"})
                continue

            # 7. Risk check
            daily_pnl = position_manager.total_realized_pnl
            check = risk_sentinel.check_signal(
                signal=signal,
                daily_pnl=daily_pnl,
                open_positions_count=position_manager.open_positions_count,
                total_exposure_pct=(position_manager.total_exposure_usd / position_manager.balance * 100)
                    if position_manager.balance > 0 else 0.0,
                balance=position_manager.balance,
                current_market_price=features.close,
            )

            if not check.approved:
                log.info("Signal REJECTED by Risk: {} {} {} — {}",
                         strat_name, signal.direction.value, symbol, check.reason)
                strat_results.append({"strategy": strat_name, "result": "rejected", "direction": signal.direction.value, "detail": f"{signal.direction.value} rejected: {check.reason}"})
                # Audit trail
                if repo:
                    try:
                        repo.insert_signal_execution(
                            timestamp=ts_now, symbol=symbol, strategy_name=strat_name,
                            direction=signal.direction.value, confidence=signal.confidence,
                            outcome="rejected", reason=check.reason,
                        )
                    except Exception as _rej_audit_err:
                        log.debug("Rejection audit write failed: {}", _rej_audit_err)
                if _alert_monitor:
                    _alert_monitor.record_signal_rejection(check.reason)
                continue

            # 8. Emit signal event
            await bus.emit(EVENT_NEW_SIGNAL, signal)
            if _alert_monitor:
                _alert_monitor.record_signal_accepted()

            # 9. Execute order
            _exec_start = time.time()
            log.info("Signal APPROVED: {} {} {} conf={:.2f} reason={}",
                     strat_name, signal.direction.value, symbol, signal.confidence, signal.reason)
            order_msg = f"{signal.direction.value} conf={signal.confidence:.2f}"
            try:
                order = await executor.execute_order(
                    signal=signal,
                    quantity=signal.suggested_quantity,
                    current_price=features.close,
                )
                if order:
                    _exec_ms = int((time.time() - _exec_start) * 1000)
                    log.info("Order FILLED: {} {} qty={:.6f} @ {:.2f}",
                             order.side.value, order.symbol, order.fill_quantity, order.fill_price)
                    order_msg += f" → FILLED @ ${order.fill_price:.2f}"
                    # Audit trail — filled
                    if repo:
                        try:
                            repo.insert_signal_execution(
                                timestamp=ts_now, symbol=symbol, strategy_name=strat_name,
                                direction=signal.direction.value, confidence=signal.confidence,
                                outcome="filled", reason=signal.reason, latency_ms=_exec_ms,
                            )
                        except Exception as _fill_audit_err:
                            log.debug("Fill audit write failed: {}", _fill_audit_err)
                    if _alert_monitor:
                        _alert_monitor.check_execution_latency(ts_now, ts_now + _exec_ms)
            except Exception as e:
                log.error("Execution error: {}", e)
                order_msg += f" → exec error: {e}"
                # Audit trail — error
                if repo:
                    try:
                        repo.insert_signal_execution(
                            timestamp=ts_now, symbol=symbol, strategy_name=strat_name,
                            direction=signal.direction.value, confidence=signal.confidence,
                            outcome="error", reason=str(e),
                        )
                    except Exception as _err_audit_err:
                        log.debug("Error audit write failed: {}", _err_audit_err)

            strat_results.append({"strategy": strat_name, "result": "signal", "direction": signal.direction.value, "detail": order_msg})
            signal_found = True
            # Только один сигнал на символ за свечу
            break

        # ── Запись в strategy log ──
        _strategy_log.append({
            "ts": ts_now,
            "symbol": symbol,
            "event": "signal" if signal_found else "scan",
            "regime": regime_name,
            "price": round(features.close, 2),
            "active_strategies": active_names,
            "strategies": strat_results,
            "msg": f"Signal found via {strat_results[-1]['strategy']}" if signal_found else f"Scanned {len(active_names)} strategies — no trade",
        })

    # SL/TP проверка на каждый тик
    async def _check_sl_tp(trade):
        """Проверить stop-loss / take-profit на каждый маркет-трейд."""
        if trading_paused or not position_manager or not executor:
            return

        symbol = trade.symbol
        if not position_manager.has_position(symbol):
            return

        trigger = position_manager.check_stop_loss_take_profit(symbol)
        if trigger is None:
            return

        # Re-check position after trigger — it may have been closed by another signal
        pos = position_manager.get_position(symbol)
        if not pos or pos.quantity <= 0:
            return

        # Snapshot values before async execution (guard against concurrent close)
        _qty = pos.quantity
        _strategy = pos.strategy_name or "sl_tp"
        _sl = pos.stop_loss_price
        _tp = pos.take_profit_price

        direction = Direction.SELL
        reason = f"SL/TP triggered: {trigger}"
        signal = Signal(
            timestamp=int(time.time() * 1000),
            symbol=symbol,
            direction=direction,
            confidence=1.0,
            strategy_name=_strategy,
            reason=reason,
            suggested_quantity=_qty,
            stop_loss_price=_sl,
            take_profit_price=_tp,
        )

        try:
            # Final check right before execution
            if not position_manager.has_position(symbol):
                return
            order = await executor.execute_order(
                signal=signal,
                quantity=_qty,
                current_price=trade.price,
            )
            if order:
                log.info("{} {} @ {:.2f} — {}", trigger, symbol, trade.price, reason)
        except Exception as e:
            log.error("SL/TP execution error: {}", e)

    # Импорт Signal для SL/TP
    from core.models import Signal

    # Подписываемся на события
    if feature_builder and strategies:
        bus.subscribe(EVENT_NEW_CANDLE, _on_new_candle)
        bus.subscribe(EVENT_NEW_TRADE, _check_sl_tp)
        log.info("[Module] Trading loop active — listening for {} candles", settings.signal_timeframe)

        # ── Startup prediction: сразу прогноз по последней закрытой свече ──
        async def _startup_prediction():
            import time as _time
            from core.models import Candle as CandleModel

            _INTERVAL_MS = {
                "1m": 60_000, "5m": 300_000, "15m": 900_000,
                "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
            }

            for sym in settings.trading_symbols:
                last = await asyncio.to_thread(repo.get_latest_candle, sym, settings.signal_timeframe)
                if not last:
                    log.info("[Startup] No historical {} candles for {} — waiting for first close", settings.signal_timeframe, sym)
                    continue

                # Проверяем, что свеча гарантированно закрыта:
                # timestamp + длительность интервала <= текущее время
                ivl_ms = _INTERVAL_MS.get(settings.signal_timeframe, 3_600_000)
                candle_close_time = last["timestamp"] + ivl_ms
                now_ms = int(_time.time() * 1000)

                if candle_close_time > now_ms:
                    log.info("[Startup] Latest {} candle for {} is still OPEN (closes in {:.0f}s) — skipping",
                             settings.signal_timeframe, sym, (candle_close_time - now_ms) / 1000)
                    continue

                candle = CandleModel(
                    timestamp=last["timestamp"], symbol=last["symbol"],
                    interval=last["interval"],
                    open=float(last["open"]), high=float(last["high"]),
                    low=float(last["low"]), close=float(last["close"]),
                    volume=float(last.get("volume", 0)),
                    trades_count=int(last.get("trades_count", 0)),
                )
                log.info("[Startup] Running prediction on last closed {} candle for {}", settings.signal_timeframe, sym)
                await _on_new_candle(candle)

        asyncio.ensure_future(_startup_prediction())
    else:
        log.warning("[Module] Trading loop DISABLED — strategies not initialized")

    # 18. Web Dashboard
    dashboard = None

    _ALLOWED_INTERVALS = {"1m", "5m", "15m", "1h", "4h", "1d"}
    _INTERVAL_LIMITS = {"1m": 120, "5m": 120, "15m": 96, "1h": 96, "4h": 120, "1d": 90}

    def build_market_chart(interval: str = "1m", symbol: str = "") -> dict:
        primary_symbol = symbol.upper() if symbol else (settings.trading_symbols[0] if settings.trading_symbols else "BTCUSDT")
        # Validate symbol is in configured list
        if settings.trading_symbols and primary_symbol not in [s.upper() for s in settings.trading_symbols]:
            primary_symbol = settings.trading_symbols[0]
        if interval not in _ALLOWED_INTERVALS:
            interval = "1m"
        limit = _INTERVAL_LIMITS.get(interval, 120)

        try:
            candles = repo.get_candles(primary_symbol, interval, limit=limit)
        except Exception as _candle_err:
            log.debug("Chart candle fetch failed: {}", _candle_err)
            candles = []

        time_fmt = "%H:%M" if interval in ("1m", "5m", "15m") else (
            "%d %b %H:%M" if interval in ("1h", "4h") else "%d %b"
        )

        # Fallback: aggregate 1m candles into 5m/15m when native candles not yet available
        if len(candles) < 2 and interval in ("5m", "15m"):
            bucket_mins = 5 if interval == "5m" else 15
            raw_limit = bucket_mins * limit
            try:
                raw = repo.get_candles(primary_symbol, "1m", limit=raw_limit)
            except Exception as _agg_err:
                log.debug("Aggregation candle fetch failed: {}", _agg_err)
                raw = []
            if len(raw) >= bucket_mins:
                aggregated = []
                bucket_ms = bucket_mins * 60_000
                for i in range(0, len(raw) - bucket_mins + 1, bucket_mins):
                    chunk = raw[i:i + bucket_mins]
                    t0 = chunk[0]["timestamp"]
                    # align to bucket boundary
                    aligned = (t0 // bucket_ms) * bucket_ms
                    aggregated.append({
                        "timestamp": aligned,
                        "open": chunk[0]["open"],
                        "high": max(float(c["high"]) for c in chunk),
                        "low": min(float(c["low"]) for c in chunk),
                        "close": chunk[-1]["close"],
                        "volume": sum(float(c.get("volume", 0)) for c in chunk),
                    })
                candles = aggregated

        if len(candles) >= 2:
            # ── Calculate EMA 9 & EMA 21 overlays ──
            closes = [float(c["close"]) for c in candles]

            def _ema(data, period):
                if len(data) < period:
                    return [None] * len(data)
                k = 2.0 / (period + 1)
                result = [None] * (period - 1)
                sma = sum(data[:period]) / period
                result.append(round(sma, 6))
                for i in range(period, len(data)):
                    val = data[i] * k + result[-1] * (1 - k)
                    result.append(round(val, 6))
                return result

            ema9_vals = _ema(closes, 9)
            ema21_vals = _ema(closes, 21)

            built = []
            for i, c in enumerate(candles):
                entry = {
                    "t": c["timestamp"],
                    "label": time.strftime(time_fmt, time.localtime(c["timestamp"] / 1000)),
                    "o": float(c["open"]),
                    "h": float(c["high"]),
                    "l": float(c["low"]),
                    "c": float(c["close"]),
                    "v": float(c.get("volume", 0)),
                }
                if ema9_vals[i] is not None:
                    entry["ema9"] = ema9_vals[i]
                if ema21_vals[i] is not None:
                    entry["ema21"] = ema21_vals[i]
                built.append(entry)

            return {
                "symbol": primary_symbol,
                "interval": interval,
                "source": f"candles_{interval}",
                "candles": built,
            }

        # Fallback to trades for 1m only
        if interval == "1m":
            try:
                trades = repo.get_recent_trades(primary_symbol, limit=120)
            except Exception as _trade_err:
                log.debug("Chart trades fetch failed: {}", _trade_err)
                trades = []
            trades = list(reversed(trades[-60:])) if trades else []
            if len(trades) >= 2:
                return {
                    "symbol": primary_symbol,
                    "interval": interval,
                    "source": "trades",
                    "candles": [
                        {
                            "t": t["timestamp"],
                            "label": time.strftime("%H:%M:%S", time.localtime(t["timestamp"] / 1000)),
                            "o": float(t["price"]),
                            "h": float(t["price"]),
                            "l": float(t["price"]),
                            "c": float(t["price"]),
                            "v": 0,
                        }
                        for t in trades
                    ],
                }

        return {
            "symbol": primary_symbol,
            "interval": interval,
            "source": "none",
            "candles": [],
        }

    # Cache: {symbol: (timestamp_ms, result_dict)} — invalidated when _last_features_per_symbol updates
    _indicators_cache: dict[str, tuple[int, dict]] = {}
    _INDICATORS_CACHE_TTL_MS = 5_000  # 5 seconds

    def _build_indicators_snapshot(target_symbol: str = "") -> dict:
        """Собирает текущие значения индикаторов для дашборда (cached)."""
        primary_symbol = target_symbol.upper() if target_symbol else (settings.trading_symbols[0] if settings.trading_symbols else "BTCUSDT")

        # Check cache — features change only on new candle (hourly), so 5s TTL is safe
        _now_ms = int(time.time() * 1000)
        _cached = _indicators_cache.get(primary_symbol)
        if _cached and (_now_ms - _cached[0]) < _INDICATORS_CACHE_TTL_MS:
            return _cached[1]

        result = _build_indicators_snapshot_uncached(primary_symbol)
        _indicators_cache[primary_symbol] = (_now_ms, result)
        return result

    def _build_indicators_snapshot_uncached(primary_symbol: str) -> dict:
        """Uncached impl — called by _build_indicators_snapshot."""
        f = _last_features_per_symbol.get(primary_symbol, _last_features)
        # Если ещё нет features от торгового цикла — вычислить напрямую из 1h свечей
        if f is None and repo:
            try:
                from features import indicators as ind
                c1r = repo.get_candles(primary_symbol, "1h", limit=60)
                if not c1r or len(c1r) < 30:
                    return {}
                closes = [float(c["close"]) for c in c1r]
                highs = [float(c["high"]) for c in c1r]
                lows = [float(c["low"]) for c in c1r]
                volumes = [float(c.get("volume", 0)) for c in c1r]

                ema_9 = ind.ema(closes, 9)
                ema_21 = ind.ema(closes, 21)
                ema_50 = ind.ema(closes, 50)
                rsi_val = ind.rsi(closes, 14)
                macd_r = ind.macd(closes, 12, 26, 9)
                adx_val = ind.adx(highs, lows, closes, 14)
                bb = ind.bollinger_bands(closes, 20, 2.0)
                atr_val = ind.atr(highs, lows, closes, 14)
                stoch_rsi = ind.stochastic_rsi(closes, 14, 14)
                vol_ratio = ind.volume_ratio(volumes, 20)
                mom = ind.momentum(closes, 10)

                if ema_9 is None or ema_21 is None or rsi_val is None:
                    return {}

                trend = "neutral"
                if ema_9 > ema_21:
                    trend = "bullish"
                elif ema_9 < ema_21:
                    trend = "bearish"
                adx_v = adx_val or 0.0
                trend_strength = "strong" if adx_v >= 40 else "moderate" if adx_v >= 25 else "weak"

                return {
                    "symbol": primary_symbol,
                    "close": round(closes[-1], 2),
                    "trend": trend,
                    "trend_strength": trend_strength,
                    "ema_9": round(ema_9, 2),
                    "ema_21": round(ema_21, 2),
                    "ema_50": round(ema_50, 2) if ema_50 else 0.0,
                    "rsi_14": round(rsi_val, 2),
                    "macd": round(macd_r[0], 4) if macd_r else 0.0,
                    "macd_signal": round(macd_r[1], 4) if macd_r else 0.0,
                    "macd_histogram": round(macd_r[2], 4) if macd_r else 0.0,
                    "adx": round(adx_v, 2),
                    "bb_upper": round(bb[0], 2) if bb else 0.0,
                    "bb_middle": round(bb[1], 2) if bb else 0.0,
                    "bb_lower": round(bb[2], 2) if bb else 0.0,
                    "bb_bandwidth": round(bb[3], 4) if bb else 0.0,
                    "atr": round(atr_val, 2) if atr_val else 0.0,
                    "volume_ratio": round(vol_ratio, 2) if vol_ratio else 0.0,
                    "stoch_rsi": round(stoch_rsi, 2) if stoch_rsi else 0.0,
                    "momentum": round(mom, 4) if mom else 0.0,
                }
            except Exception as e:
                log.debug("Indicators on-demand calc failed: {}", e)
                return {}
        if f is None:
            return {}
        # Определяем тренд по EMA
        trend = "neutral"
        if f.ema_9 > 0 and f.ema_21 > 0:
            if f.ema_9 > f.ema_21:
                trend = "bullish"
            elif f.ema_9 < f.ema_21:
                trend = "bearish"
        # Сила тренда по ADX
        trend_strength = "weak"
        if f.adx >= 40:
            trend_strength = "strong"
        elif f.adx >= 25:
            trend_strength = "moderate"
        # Crossover state from EMA strategy
        ema_diff = f.ema_9 - f.ema_21
        prev_ema_diff = None
        has_crossover = False
        strat_ema = strategies.get("ema_crossover_rsi")
        if strat_ema and hasattr(strat_ema, '_prev_ema_diff'):
            prev_ema_diff = strat_ema._prev_ema_diff.get(f.symbol)
            if prev_ema_diff is not None:
                has_crossover = prev_ema_diff <= 0 and ema_diff > 0

        # Min crossover threshold
        min_cross_threshold = f.atr * 0.1 if f.atr > 0 else f.close * 0.0003
        crossover_strong = has_crossover and ema_diff >= min_cross_threshold

        # News data
        news = {
            "sentiment": round(f.news_sentiment, 3),
            "composite_score": round(f.news_composite_score, 3),
            "signal_strength": round(f.news_signal_strength, 3),
            "fear_greed_index": f.fear_greed_index,
            "critical_alert": f.news_critical_alert,
            "actionable": f.news_actionable,
            "dominant_category": f.news_dominant_category,
            "impact_pct": round(f.news_impact_pct, 3),
        }

        # Confidence breakdown (simulating EMA crossover strategy logic)
        conf_base = 0.50
        conf_rsi = 0.10 if f.rsi_14 < 50 else (0.05 if f.rsi_14 < 60 else 0.0)
        conf_volume = 0.10 if f.volume_ratio > 2.0 else (0.05 if f.volume_ratio > 1.5 else 0.0)
        conf_ema50 = 0.10 if (f.ema_50 > 0 and f.close > f.ema_50) else 0.0
        conf_macd = 0.10 if f.macd_histogram > 0 else 0.0
        conf_adx = 0.05 if f.adx > 25 else 0.0
        conf_trend_align = 0.10 if f.trend_alignment >= 0.8 else (0.05 if f.trend_alignment >= 0.6 else (-0.10 if f.trend_alignment <= 0.2 else 0.0))
        conf_total = min(conf_base + conf_rsi + conf_volume + conf_ema50 + conf_macd + conf_adx + conf_trend_align, 0.95)

        # BUY conditions checklist
        buy_conditions = {
            "ema_crossover": has_crossover,
            "crossover_strong": crossover_strong,
            "rsi_below_70": f.rsi_14 < 70,
            "volume_above_1x": f.volume_ratio >= 1.0,
            "price_above_ema50": f.close > f.ema_50 if f.ema_50 > 0 else True,
            "no_critical_news": not (f.news_critical_alert and f.news_composite_score < -0.3 and f.news_signal_strength > 0.3),
            "confidence_above_min": conf_total >= 0.60,
        }
        all_conditions_met = all(buy_conditions.values())

        return {
            "symbol": f.symbol,
            "close": round(f.close, 2),
            "trend": trend,
            "trend_strength": trend_strength,
            "ema_9": round(f.ema_9, 2),
            "ema_21": round(f.ema_21, 2),
            "ema_50": round(f.ema_50, 2),
            "ema_diff": round(ema_diff, 4),
            "prev_ema_diff": round(prev_ema_diff, 4) if prev_ema_diff is not None else None,
            "has_crossover": has_crossover,
            "crossover_strong": crossover_strong,
            "min_cross_threshold": round(min_cross_threshold, 4),
            "rsi_14": round(f.rsi_14, 2),
            "macd": round(f.macd, 4),
            "macd_signal": round(f.macd_signal, 4),
            "macd_histogram": round(f.macd_histogram, 4),
            "adx": round(f.adx, 2),
            "bb_upper": round(f.bb_upper, 2),
            "bb_middle": round(f.bb_middle, 2),
            "bb_lower": round(f.bb_lower, 2),
            "bb_bandwidth": round(f.bb_bandwidth, 4),
            "atr": round(f.atr, 2),
            "volume_ratio": round(f.volume_ratio, 2),
            "stoch_rsi": round(f.stoch_rsi, 2),
            "momentum": round(f.momentum, 4),
            "trend_alignment": round(f.trend_alignment, 3),
            "ema_50_daily": round(f.ema_50_daily, 2),
            "rsi_14_daily": round(f.rsi_14_daily, 2),
            "news": news,
            "confidence_breakdown": {
                "base": conf_base,
                "rsi_bonus": conf_rsi,
                "volume_bonus": conf_volume,
                "ema50_bonus": conf_ema50,
                "macd_bonus": conf_macd,
                "adx_bonus": conf_adx,
                "trend_align_bonus": conf_trend_align,
                "total": round(conf_total, 3),
            },
            "buy_conditions": buy_conditions,
            "all_buy_conditions_met": all_conditions_met,
        }

    def _calc_win_rate_per_symbol(recent_trades: list) -> dict:
        """Compute win rate per symbol from recent closed trades."""
        by_sym: dict[str, dict] = {}
        for t in recent_trades:
            sym = t.get("symbol", "") if isinstance(t, dict) else getattr(t, "symbol", "")
            if not sym:
                continue
            if sym not in by_sym:
                by_sym[sym] = {"wins": 0, "total": 0}
            by_sym[sym]["total"] += 1
            pnl = t.get("pnl", 0) if isinstance(t, dict) else getattr(t, "realized_pnl", 0)
            if pnl > 0:
                by_sym[sym]["wins"] += 1
        result = {}
        for sym, stats in by_sym.items():
            result[sym] = round(stats["wins"] / stats["total"] * 100, 1) if stats["total"] > 0 else 0.0
        return result

    def get_system_state() -> dict:
        position_state = position_manager.get_state() if position_manager else {}
        balance = float(position_state.get("balance", settings.paper_initial_balance))
        pnl_today = float(position_state.get("pnl_today", 0.0))
        risk_metrics = (
            risk_sentinel.get_runtime_metrics(balance=balance)
            if risk_sentinel else {}
        )

        market_data_age_sec = -1.0
        if collector:
            age = collector.last_data_age_sec
            if age != float("inf"):
                market_data_age_sec = round(age, 1)

        # Readiness — прогресс сбора данных для торговли
        readiness = {"ready": False, "pct": 0, "steps": []}
        try:
            from features.feature_builder import MIN_CANDLES_1H, MIN_CANDLES_4H
            primary = settings.trading_symbols[0] if settings.trading_symbols else "BTCUSDT"
            n_1h = len(repo.get_candles(primary, "1h", limit=MIN_CANDLES_1H)) if repo else 0
            n_4h = len(repo.get_candles(primary, "4h", limit=MIN_CANDLES_4H)) if repo else 0
            pct_1h = min(n_1h / MIN_CANDLES_1H * 100, 100)
            pct_4h = min(n_4h / MIN_CANDLES_4H * 100, 100)
            ws_ok = collector._ws is not None and collector._running if collector else False
            strats_ok = len(strategies) > 0
            overall = (pct_1h * 0.4 + pct_4h * 0.4 + (100 if ws_ok else 0) * 0.1 + (100 if strats_ok else 0) * 0.1)
            readiness = {
                "ready": pct_1h >= 100 and pct_4h >= 100 and ws_ok and strats_ok,
                "pct": round(overall, 1),
                "symbol": primary,
                "timeframe": settings.signal_timeframe,
                "steps": [
                    {"name": "Binance WS", "done": ws_ok, "detail": "connected" if ws_ok else "disconnected"},
                    {"name": "Strategies", "done": strats_ok, "detail": f"{len(strategies)} loaded"},
                    {"name": f"1h candles", "done": pct_1h >= 100, "detail": f"{n_1h}/{MIN_CANDLES_1H}", "pct": round(pct_1h, 1)},
                    {"name": f"4h candles", "done": pct_4h >= 100, "detail": f"{n_4h}/{MIN_CANDLES_4H}", "pct": round(pct_4h, 1)},
                ],
            }
        except Exception as _ready_err:
            log.debug("Readiness check failed: {}", _ready_err)

        return {
            "mode": settings.trading_mode,
            "risk_state": risk_state_machine.state.value if risk_state_machine else "NORMAL",
            "trading_paused": trading_paused,
            "uptime": _format_uptime(),
            "pnl_today": pnl_today,
            "pnl_total": float(position_state.get("pnl_total", 0.0)),
            "open_positions": int(position_state.get("open_positions", 0)),
            "trades_today": int(position_state.get("trades_today", 0)),
            "balance": balance,
            "win_rate": float(position_state.get("win_rate", 0.0)),
            "profit_factor": float(position_state.get("profit_factor", 0.0)),
            "avg_rr_ratio": float(position_state.get("avg_rr_ratio", 0.0)),
            "max_drawdown_pct": float(position_state.get("max_drawdown_pct", 0.0)),
            "current_drawdown_pct": float(position_state.get("current_drawdown_pct", 0.0)),
            "peak_balance": float(position_state.get("peak_balance", balance)),
            "total_wins": int(position_state.get("total_wins", 0)),
            "total_losses": int(position_state.get("total_losses", 0)),
            "positions": position_state.get("positions", []),
            "recent_trades": position_state.get("recent_trades", []),
            "pnl_history": position_state.get("pnl_history", []),
            # market_chart omitted — dashboard fetches via /api/market-chart with symbol param
            "activity": {
                "collector": collector.stats if collector else {},
                "events": bus.get_event_stats(),
                "strategies_loaded": list(strategies.keys()) if strategies else [],
                "current_regime": _current_regime.regime.value if _current_regime else None,
            },
            "readiness": readiness,
            "strategy_log": list(_strategy_log),
            "risk_details": {
                "daily_loss": min(pnl_today, 0.0),
                "max_drawdown": float(position_state.get("max_drawdown_pct", 0.0)) / 100,
                "exposure": float(position_state.get("exposure_pct", 0.0)) / 100,
                "trade_freq": int(risk_metrics.get("trades_last_hour", 0)),
                "daily_commission": float(risk_metrics.get("daily_commission", 0.0)),
                "market_data_age_sec": market_data_age_sec,
                "cooldown_remaining_sec": int(risk_metrics.get("cooldown_remaining_sec", 0)),
                "daily_trades": int(risk_metrics.get("daily_trades", 0)),
                "limits": {
                    "max_daily_loss_usd": risk_sentinel._limits.max_daily_loss_usd if risk_sentinel else 50.0,
                    "max_open_positions": risk_sentinel._limits.max_open_positions if risk_sentinel else 2,
                    "max_total_exposure_pct": risk_sentinel._limits.max_total_exposure_pct if risk_sentinel else 60.0,
                    "max_daily_trades": risk_sentinel._limits.max_daily_trades if risk_sentinel else 6,
                    "max_trades_per_hour": risk_sentinel._limits.max_trades_per_hour if risk_sentinel else 2,
                    "min_trade_interval_sec": risk_sentinel._limits.min_trade_interval_sec if risk_sentinel else 1800,
                    "min_order_usd": risk_sentinel._limits.min_order_usd if risk_sentinel else 10.0,
                    "max_order_usd": risk_sentinel._limits.max_order_usd if risk_sentinel else 100.0,
                    "max_loss_per_trade_pct": risk_sentinel._limits.max_loss_per_trade_pct if risk_sentinel else 3.0,
                },
                "risk_checks": {
                    "state_ok": risk_state_machine.state.value != "STOP" if risk_state_machine else True,
                    "daily_loss_ok": pnl_today > -(risk_sentinel._limits.max_daily_loss_usd if risk_sentinel else 50.0),
                    "positions_ok": int(position_state.get("open_positions", 0)) < (risk_sentinel._limits.max_open_positions if risk_sentinel else 2),
                    "exposure_ok": float(position_state.get("exposure_pct", 0.0)) < (risk_sentinel._limits.max_total_exposure_pct if risk_sentinel else 60.0),
                    "daily_trades_ok": int(risk_metrics.get("daily_trades", 0)) < (risk_sentinel._limits.max_daily_trades if risk_sentinel else 6),
                    "hourly_trades_ok": int(risk_metrics.get("trades_last_hour", 0)) < (risk_sentinel._limits.max_trades_per_hour if risk_sentinel else 2),
                    "cooldown_ok": int(risk_metrics.get("cooldown_remaining_sec", 0)) <= 0,
                },
            },
            "ml_status": {
                "enabled": settings.analyzer_ml_enabled if hasattr(settings, 'analyzer_ml_enabled') else False,
                "is_ready": _ml_predictor.is_ready if _ml_predictor else False,
                "mode": _ml_predictor.rollout_mode if _ml_predictor else "off",
                "block_threshold": _ml_predictor._cfg.block_threshold if _ml_predictor else 0.55,
                "reduce_threshold": _ml_predictor._cfg.reduce_threshold if _ml_predictor else 0.65,
                "model_version": _ml_predictor._model_version if _ml_predictor and hasattr(_ml_predictor, '_model_version') else "",
                "per_symbol_models": {
                    sym: {"ready": p.is_ready, "version": getattr(p, '_model_version', '')}
                    for sym, p in _ml_predictors.items()
                },
            },
            "indicators": _build_indicators_snapshot(),
            "indicators_per_symbol": {
                sym: _build_indicators_snapshot(sym)
                for sym in (settings.trading_symbols or [])
            },
            "trading_symbols": settings.trading_symbols or [],
            "win_rate_per_symbol": _calc_win_rate_per_symbol(position_state.get("recent_trades", [])),
            "strategy_performance": repo.get_strategy_performance() if repo else [],
            "trades_export": repo.get_all_trades_for_export() if repo else [],
            "trades_export_full": repo.get_strategy_trades() if repo else [],
            "alerts": _alert_monitor.get_recent_alerts() if _alert_monitor else [],
            "signal_exec_stats": repo.get_signal_execution_stats() if repo else {},
            "ml_predictor": _ml_predictor,
            "ml_retrain_fn": _run_ml_training,
        }

    # Control handlers — shared by Dashboard and Telegram bot
    async def _handle_stop():
        nonlocal trading_paused
        log.warning("STOP requested — trading paused")
        trading_paused = True

    async def _handle_resume():
        nonlocal trading_paused
        log.info("RESUME requested — trading resumed")
        trading_paused = False
        if risk_state_machine:
            risk_state_machine.reset()

    async def _handle_kill():
        log.warning("KILL requested — shutting down")
        shutdown.trigger()

    def _handle_settings_update(new_settings):
        """Propagate risk-limit changes to runtime objects without restart."""
        nonlocal settings
        settings = new_settings
        if risk_sentinel:
            risk_sentinel._limits.max_open_positions = new_settings.max_open_positions
            risk_sentinel._limits.max_daily_loss_usd = new_settings.max_daily_loss_usd
            risk_sentinel._limits.max_daily_loss_pct = new_settings.max_daily_loss_pct
            risk_sentinel._limits.max_position_pct = new_settings.max_position_pct
            risk_sentinel._limits.max_total_exposure_pct = new_settings.max_total_exposure_pct
            risk_sentinel._limits.max_trades_per_hour = new_settings.max_trades_per_hour
            risk_sentinel._limits.max_order_usd = new_settings.max_order_usd
            risk_sentinel._limits.max_loss_per_trade_pct = new_settings.stop_loss_pct
            risk_sentinel._limits.max_daily_commission_pct = new_settings.cb_commission_alert_pct
            log.info("Risk limits updated at runtime: max_open_positions={}", new_settings.max_open_positions)
        if position_manager:
            position_manager._max_open_positions = new_settings.max_open_positions
            log.info("PositionManager max_open_positions updated to {}", new_settings.max_open_positions)

    try:
        from dashboard.app import Dashboard
        dashboard = Dashboard(settings, bus, state_provider=get_system_state)

        dashboard.on_stop = _handle_stop
        dashboard.on_resume = _handle_resume
        dashboard.on_kill = _handle_kill
        dashboard.on_settings_update = _handle_settings_update
        dashboard.market_chart_provider = build_market_chart

        # News collector
        from collector.news_collector import NewsCollector
        news_collector = NewsCollector(
            update_interval=300,
            groq_api_key=settings.groq_api_key,
            openrouter_api_key=settings.openrouter_api_key,
            db=db,
        )
        dashboard.news_collector = news_collector
        await news_collector.start()
        log.info("[Module] NewsCollector started (5min interval)")

        await dashboard.start()
        log.info("[Module] Dashboard started on http://localhost:{}", settings.dashboard_port)
    except Exception as e:
        log.warning("[Module] Dashboard failed: {}", e)

    # 19. Telegram Bot
    telegram_bot = None
    try:
        from telegram_bot.bot import TelegramBot
        telegram_bot = TelegramBot(settings, bus, state_provider=get_system_state)
        telegram_bot.on_stop = _handle_stop
        telegram_bot.on_resume = _handle_resume
        telegram_bot.on_kill = _handle_kill
        await telegram_bot.start()
        if telegram_bot.enabled and telegram_bot._running:
            log.info("[Module] Telegram bot started")
            # Отправить уведомление о старте
            await telegram_bot.send_message(
                f"🟢 <b>SENTINEL v{VERSION}</b> запущен\n"
                f"Режим: {settings.trading_mode.upper()}\n"
                f"Символы: {', '.join(settings.trading_symbols)}"
            )
        else:
            log.warning("[Module] Telegram bot disabled (token/chat_id not set)")
    except Exception as e:
        log.warning("[Module] Telegram bot failed: {}", e)
        telegram_bot = None

    # ── ML Auto-Retrain loop ─────────────────────────────────────────────────
    async def _ml_retrain_loop():
        """Background task: retrain ML model every ANALYZER_ML_RETRAIN_DAYS days."""
        # Small delay so system fully starts before first retrain check
        await asyncio.sleep(60)
        while True:
            try:
                async with _ml_retrain_lock:
                    if _ml_predictor:
                        if not _ml_predictor.is_ready:
                            log.info("ML auto-retrain: no model loaded — triggering initial training")
                            await _run_ml_training()
                        elif _ml_predictor.needs_retrain():
                            log.info("ML auto-retrain: model is stale — retraining")
                            await _run_ml_training()
            except Exception as exc:
                log.error("ML retrain loop error: {}", exc)
            # Check every 6 hours
            await asyncio.sleep(6 * 3600)

    ml_retrain_task = asyncio.create_task(_ml_retrain_loop())

    # Periodic equity snapshot (для графика PnL)
    async def _equity_snapshot_loop():
        while True:
            await asyncio.sleep(30)
            if position_manager:
                position_manager._record_equity_snapshot(force=True)

    equity_task = asyncio.create_task(_equity_snapshot_loop())

    # Periodic position price sync to DB (every 60s)
    async def _position_db_sync_loop():
        while True:
            await asyncio.sleep(60)
            if not position_manager or not repo:
                continue
            for pos in position_manager.open_positions:
                if pos.db_id:
                    try:
                        await asyncio.to_thread(
                            repo.update_position_price,
                            pos.db_id, pos.current_price, pos.unrealized_pnl,
                        )
                    except Exception as _sync_err:
                        log.warning("Position price sync failed for {}: {}", pos.symbol, _sync_err)

    pos_sync_task = asyncio.create_task(_position_db_sync_loop())

    # Heartbeat writer (шаг 19)
    heartbeat_task = asyncio.create_task(heartbeat_writer(settings))

    # Monitor background tasks for unexpected failures
    _background_tasks: list[tuple[str, asyncio.Task]] = [
        ("equity_snapshot", equity_task),
        ("heartbeat", heartbeat_task),
        ("ml_retrain", ml_retrain_task),
        ("pos_db_sync", pos_sync_task),
    ]
    if collector_task:
        _background_tasks.append(("collector", collector_task))

    async def _task_watchdog():
        """Monitor background tasks and log if any fail unexpectedly."""
        while True:
            await asyncio.sleep(10)
            for name, task in _background_tasks:
                if task.done() and not task.cancelled():
                    exc = task.exception()
                    if exc:
                        log.error("Background task '{}' crashed: {} — system may be inconsistent!", name, exc)
                        if _alert_monitor:
                            _alert_monitor.send_critical_alert(f"Task '{name}' crashed: {exc}")

    watchdog_task = asyncio.create_task(_task_watchdog())

    log.info("🟢 Система запущена. Режим: {} TRADING", settings.trading_mode.upper())

    # Ожидание сигнала остановки
    await shutdown.event.wait()

    # Graceful shutdown sequence
    log.info("Остановка системы...")
    watchdog_task.cancel()
    heartbeat_task.cancel()
    equity_task.cancel()
    ml_retrain_task.cancel()
    pos_sync_task.cancel()
    await asyncio.gather(watchdog_task, heartbeat_task, equity_task, ml_retrain_task, pos_sync_task, return_exceptions=True)
    if collector:
        await collector.stop()
    if collector_task:
        collector_task.cancel()
        await asyncio.gather(collector_task, return_exceptions=True)
    if telegram_bot:
        try:
            await telegram_bot.send_message("🔴 <b>SENTINEL</b> останавливается...")
            await telegram_bot.stop()
        except Exception:
            pass
    if dashboard:
        await dashboard.stop()
    # Final sync: persist latest position prices to DB before closing
    if position_manager and repo:
        for _pos in position_manager.open_positions:
            if _pos.db_id:
                try:
                    repo.update_position_price(_pos.db_id, _pos.current_price, _pos.unrealized_pnl)
                    log.info("Shutdown sync: {} @ {:.2f} uPnL={:.4f}", _pos.symbol, _pos.current_price, _pos.unrealized_pnl)
                except Exception as _e:
                    log.warning("Shutdown position sync failed for {}: {}", _pos.symbol, _e)
    db.close()
    save_state({"stopped_at": int(time.time()), "mode": settings.trading_mode})
    release_pid_lock()
    log.info("🔴 Система остановлена. Состояние сохранено.")


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Остановка по Ctrl+C")
    finally:
        release_pid_lock()


if __name__ == "__main__":
    main()
