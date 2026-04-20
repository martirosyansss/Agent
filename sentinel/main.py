"""
SENTINEL V1.5 — Точка входа.

Последовательность запуска (21 шаг) описана в ТЗ §26.2.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode

from loguru import logger


# Маппинг std-logging logger.name -> loguru extra[module],
# чтобы фильтры trades.log / risk.log заполнялись (см. setup_logging).
_STDLOG_MODULE_MAP = (
    ("sentinel.risk", "risk"),
    ("risk.", "risk"),
    ("sentinel.guards", "risk"),
    ("sentinel.execution", "executor"),
    ("execution.", "executor"),
    ("sentinel.position", "position"),
    ("position.", "position"),
    ("sentinel.dashboard", "dashboard"),
    ("dashboard.", "dashboard"),
)

# UI-флаги *_enabled из Settings → имена ключей в словаре `strategies`.
# Используется в hot-path фильтре (см. шаг "4. Определить активные стратегии"),
# чтобы выключение тоггла в дашборде реально снимало стратегию с ротации,
# а не работало как косметика. EMA Crossover RSI флага не имеет — это
# базовая стратегия и в UI не выводится.
_STRATEGY_ENABLE_FLAGS: dict[str, str] = {
    "grid_trading": "grid_enabled",
    "mean_reversion": "meanrev_enabled",
    "bollinger_breakout": "bb_breakout_enabled",
    "dca_bot": "dca_enabled",
    "macd_divergence": "macd_div_enabled",
}


class _InterceptHandler(logging.Handler):
    """Перенаправляет stdlib logging → loguru с авто-биндом extra[module]."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except (ValueError, AttributeError):
            level = record.levelno

        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        module = "stdlib"
        name = record.name or ""
        for prefix, mod in _STDLOG_MODULE_MAP:
            if name.startswith(prefix) or f".{prefix}" in name:
                module = mod
                break

        logger.opt(depth=depth, exception=record.exc_info).bind(module=module).log(
            level, record.getMessage()
        )

# Корень проекта — директория, где лежит main.py
BASE_DIR = Path(__file__).resolve().parent

# Добавляем корень проекта в sys.path, чтобы импорты работали
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import load_settings, Settings  # noqa: E402
from core.absolute_limits import FORBIDDEN_API_PERMISSIONS  # noqa: E402
from core.constants import (  # noqa: E402
    APP_NAME,
    EVENT_EXECUTION_DEGRADED,
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
        enqueue=True,
    )
    # Ошибки и важные предупреждения
    logger.add(
        log_dir / "errors.log",
        format=fmt,
        level="WARNING",
        rotation=LOG_ROTATION_SIZE,
        retention=LOG_ROTATION_COUNT,
        encoding="utf-8",
        enqueue=True,
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
        filter=lambda r: r["extra"].get("module") in ("risk", "circuit_breaker", "circuit_breakers"),
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

    # Мост std logging -> loguru. Без него модули risk/, execution/, position/
    # (использующие logging.getLogger) не попадают ни в risk.log, ни в trades.log.
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    for noisy in (
        "urllib3", "asyncio", "websockets",
        "httpx", "httpcore", "httpx._client",  # Telegram-bot polling спам
        "telegram", "telegram.ext", "telegram.ext.Updater",
        "apscheduler",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Structured JSONL event log: machine-parseable record for post-mortem
    # analytics. Every signal decision, guard trip, regime change, etc. is
    # appended here as a single JSON line. Pandas can ingest the whole file
    # with pd.read_json("events.jsonl", lines=True).
    try:
        from monitoring.event_log import EventLog, set_event_log
        set_event_log(EventLog(
            path=log_dir / "events.jsonl",
            max_bytes=50 * 1024 * 1024,
            backup_count=5,
        ))
        logger.info("Structured event log enabled: logs/events.jsonl")
    except Exception as _ev_err:
        logger.warning("Event log setup failed: {}", _ev_err)


# ──────────────────────────────────────────────
# PID Lock
# ──────────────────────────────────────────────

def acquire_pid_lock() -> bool:
    """Проверяет, не запущен ли уже инстанс. Создаёт PID-файл."""
    pid_path = BASE_DIR / PID_FILE
    if pid_path.exists():
        try:
            raw = pid_path.read_bytes()
            if raw.startswith(b"\xff\xfe") or b"\x00" in raw[:8]:
                text = raw.decode("utf-16", errors="ignore")
            else:
                text = raw.decode("utf-8", errors="ignore")
            old_pid = int(text.strip().lstrip("\ufeff"))
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

    pid_path.write_text(str(os.getpid()), encoding="utf-8")
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
    # Idempotent migration for the structured decision-audit table.
    try:
        repo.ensure_decision_audit_table()
    except Exception as _da_table_err:
        log.warning("decision_audit table init failed: {}", _da_table_err)
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

        # Startup reconciliation: cross-check DB-restored positions against
        # live exchange state. Flags two drift classes:
        #   1. DB has OPEN position but exchange has no protective OCO →
        #      entry filled before crash, protection never placed / got
        #      cancelled. Operator must re-attach or close manually.
        #   2. Exchange has an open order for a symbol we don't track →
        #      orphan from a prior run. Not auto-cancelled (might be manual
        #      user order); surfaced for review.
        if executor and hasattr(executor, "reconcile_with_exchange"):
            try:
                tracked_syms = {p.symbol for p in position_manager.open_positions}
                all_syms = list(set(settings.trading_symbols or []) | tracked_syms)
                recon = await executor.reconcile_with_exchange(all_syms)
                from monitoring.event_log import emit_component_error as _emit_cerr
                for sym, info in (recon or {}).items():
                    has_pos = sym in tracked_syms
                    has_oco = info.get("has_protective_oco", False)
                    n_open = len(info.get("open_orders", []) or [])
                    if has_pos and not has_oco:
                        _emit_cerr(
                            "main.reconcile",
                            f"DB position {sym} restored but no protective OCO on exchange",
                            severity="critical",
                            symbol=sym,
                            open_orders=n_open,
                        )
                        log.critical(
                            "RECONCILE: {} has restored DB position but NO protective OCO on exchange",
                            sym,
                        )
                    elif not has_pos and n_open > 0:
                        _emit_cerr(
                            "main.reconcile",
                            f"exchange has {n_open} open order(s) for {sym} but no tracked position",
                            severity="warning",
                            symbol=sym,
                            open_orders=n_open,
                        )
                        log.warning(
                            "RECONCILE: {} — {} open order(s) on exchange but no tracked position",
                            sym, n_open,
                        )
            except Exception as _recon_err:
                log.warning("Startup reconciliation raised: {}", _recon_err)

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

        # Restore RiskSentinel counters and RiskStateMachine state across restarts.
        # Without this, a mid-day restart forgets daily_trades/daily_commission
        # and always starts in NORMAL even if the prior run hit SAFE/STOP — the
        # bot would happily re-enter trades against a real risk ceiling.
        if repo is not None:
            try:
                import json as _json
                _rs_blob = repo.load_system_state("risk_sentinel")
                if _rs_blob:
                    risk_sentinel.restore_state(_json.loads(_rs_blob))
                _rsm_blob = repo.load_system_state("risk_state_machine")
                if _rsm_blob:
                    risk_state_machine.restore_state(_json.loads(_rsm_blob))
            except Exception as _risk_restore_err:
                log.debug("Risk state restore skipped: {}", _risk_restore_err)

        # ──────────────────────────────────────────────
        # Pro risk guards: drawdown breaker, correlation, asset-class caps.
        # All three are best-effort: if construction fails we keep trading
        # without them rather than crashing the whole runtime — but log loud.
        # ──────────────────────────────────────────────
        try:
            from risk.drawdown_breaker import DrawdownBreaker, DrawdownThresholds
            from risk.correlation_guard import CorrelationConfig, CorrelationGuard
            from risk.exposure_caps import ExposureCapGuard
            from risk.price_history_cache import PriceHistoryCache
            from risk.regime_gate import RegimeGate
            from risk.news_cooldown import NewsCooldownGuard
            from risk.liquidity_gate import LiquidityGate, LiquidityGateConfig
            from risk.stale_data_gate import StaleDataGate, StaleDataGateConfig
            from risk.circuit_breakers import CircuitBreakers
            from strategy.multi_tf_gate import MultiTFGate, MultiTFGateConfig

            dd_breaker = DrawdownBreaker(DrawdownThresholds(
                daily_pct=settings.max_daily_loss_pct / 100.0,
                weekly_pct=min(0.10, settings.max_daily_loss_pct / 100.0 * 2.0),
                monthly_pct=min(0.15, settings.max_daily_loss_pct / 100.0 * 3.0),
            ))
            # Restore tripped-window state across restarts. Without this,
            # a bot crashing mid-drawdown forgets it was blocking entries and
            # starts BUYing again the moment it's back up, against a real DD.
            if repo is not None:
                try:
                    _dd_blob = repo.load_system_state("dd_breaker")
                    if _dd_blob:
                        import json as _json
                        dd_breaker.restore_state(_json.loads(_dd_blob))
                except Exception as _dd_restore_err:
                    log.debug("DD breaker restore skipped: {}", _dd_restore_err)
            corr_guard = CorrelationGuard(CorrelationConfig(
                threshold=0.85,
                min_observations=30,
                max_cluster_size=2,
                min_effective_positions=1.05,
            ))
            exposure_cap = ExposureCapGuard()
            price_history_cache = PriceHistoryCache(max_history=240)
            # Multi-TF gate: fail-OPEN on missing data — startup phase often
            # lacks enough 1d candles to compute EMA50_daily. Once daily data
            # is present, the gate enforces strict 4h + 1d trend confluence.
            multi_tf_gate = MultiTFGate(MultiTFGateConfig(
                require_4h_alignment=True,
                require_1d_alignment=True,
                min_trend_alignment_score=0.6,
                fail_closed_on_missing_data=False,
            ))
            regime_gate = RegimeGate()
            news_cooldown = NewsCooldownGuard()
            liquidity_gate = LiquidityGate(LiquidityGateConfig(
                min_volume_ratio_buy=0.4,
                max_pct_of_recent_volume=0.05,
                blocked_utc_hours=(),  # leave empty until backtest justifies
            ))
            # Stale-data gate: block BUY when WS messages stop arriving.
            # Threshold slightly above collector's forced-reconnect window
            # (MAX_DATA_AGE_SEC=30 default) so reconnect gets a chance first.
            stale_data_gate = StaleDataGate(StaleDataGateConfig(
                max_age_sec=max(90.0, settings.max_data_age_sec * 3.0),
            ))
            # 8 Circuit Breakers: fast-anomaly guardrails. Until now instantiated
            # only in tests — real trading had no price/spread/latency/API
            # protection. Feeds come from collector and executor.
            circuit_breakers = CircuitBreakers(
                consecutive_loss_threshold=settings.cb_consecutive_losses,
                strategy_cooldown_sec=settings.cb_strategy_cooldown_sec,
                strategy_cooldown_overrides=settings.cb_strategy_cooldown_overrides,
            )

            risk_sentinel.attach_drawdown_breaker(dd_breaker)
            risk_sentinel.attach_correlation_guard(corr_guard)
            risk_sentinel.attach_exposure_cap_guard(exposure_cap)
            risk_sentinel.attach_multi_tf_gate(multi_tf_gate)
            risk_sentinel.attach_regime_gate(regime_gate)
            risk_sentinel.attach_news_cooldown(news_cooldown)
            risk_sentinel.attach_liquidity_gate(liquidity_gate)
            risk_sentinel.attach_stale_data_gate(stale_data_gate)
            risk_sentinel.attach_circuit_breakers(circuit_breakers)
            log.info("Pro risk guards attached: DD + correlation + exposure + multi-TF + regime + news cooldown + liquidity + stale-data + 8CBs")
        except Exception as _guards_err:
            log.error("Failed to attach pro risk guards: {} — continuing without them", _guards_err)
            dd_breaker = None
            corr_guard = None
            exposure_cap = None
            price_history_cache = None
            multi_tf_gate = None
            regime_gate = None
            news_cooldown = None
            liquidity_gate = None
            stale_data_gate = None
            circuit_breakers = None

        # Stores {(symbol, strategy): (probability, entry_price)} for open positions
        # so we can feed the outcome back to LivePerformanceTracker when the trade closes.
        _ml_prob_at_entry: dict[tuple, tuple] = {}

        async def _on_order_filled(order):
            nonlocal trading_paused
            if not position_manager:
                return

            if order.side == Direction.BUY:
                opened = await position_manager.open_position(order)
                # Persist order + position atomically so a crash between the
                # two writes cannot produce half-state. On DB failure we halt
                # trading: in-memory position exists, exchange has the fill —
                # operator must inspect before new trades are opened.
                if opened and repo:
                    try:
                        order_id, position_id = await asyncio.to_thread(
                            repo.insert_order_and_position, order, opened,
                        )
                        opened.db_id = position_id
                        log.info("Order+Position persisted atomically: {} order_id={} pos_id={}",
                                 opened.symbol, order_id, position_id)
                    except Exception as _db_err:
                        log.critical(
                            "Atomic order+position DB write FAILED for {}: {} — halting trading",
                            order.symbol, _db_err,
                        )
                        from monitoring.event_log import emit_component_error as _emit_cerr
                        _emit_cerr(
                            "main.on_order_filled",
                            f"atomic DB write failed for {order.symbol}",
                            exc=_db_err,
                            severity="critical",
                            symbol=order.symbol,
                        )
                        trading_paused = True
                        return
                elif repo and not opened:
                    # Position did not open (duplicate / invalid fill / insufficient
                    # funds): still persist the order alone for audit trail.
                    try:
                        await asyncio.to_thread(repo.insert_order, order)
                    except Exception as _db_err:
                        log.error("Order-only DB write failed for {}: {}", order.symbol, _db_err)

                if opened and risk_sentinel:
                    risk_sentinel.record_trade(order.commission, increment_trade=True)
                if opened and order.strategy_name == "ema_crossover_rsi":
                    await position_manager.set_trailing_stop(order.symbol, 2.5, 1.5)
                elif opened and order.strategy_name == "bollinger_breakout":
                    await position_manager.set_trailing_stop(order.symbol, 3.0, 2.0)
                if opened:
                    await position_manager.setup_tp_levels(order.symbol)
                if opened and order.features is not None and order.features.atr > 0:
                    await position_manager.setup_chandelier(
                        order.symbol,
                        atr=order.features.atr,
                        strategy_name=order.strategy_name,
                    )
            elif order.side == Direction.SELL:
                # Audit trail for the SELL order itself. Kept separate from the
                # atomic BUY path because close_position + repo.close_position
                # handles the position-side write below.
                if repo:
                    try:
                        await asyncio.to_thread(repo.insert_order, order)
                    except Exception as _db_err:
                        log.error("SELL order DB write failed for {}: {}", order.symbol, _db_err)

                # Capture position data BEFORE close (close deletes from dict)
                _pos_before = position_manager.get_position(order.symbol)
                _entry_px_pos = _pos_before.entry_price if _pos_before else 0.0
                _opened_at = _pos_before.opened_at if _pos_before else ""
                _strat_name = _pos_before.strategy_name if _pos_before else order.strategy_name
                _signal_id = _pos_before.signal_id if _pos_before else ""
                _signal_reason = _pos_before.signal_reason if _pos_before else ""
                _pos_db_id = _pos_before.db_id if _pos_before else None
                _entry_features = _pos_before.entry_features if _pos_before else None
                _max_price_hold = _pos_before.max_price_during_hold if _pos_before else 0.0
                _min_price_hold = _pos_before.min_price_during_hold if _pos_before else 0.0

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

                        _trade_id = _uuid.uuid4().hex[:12]
                        _hour = int(time.strftime("%H"))
                        _dow = int(time.strftime("%w"))
                        if _entry_features is not None:
                            # Capture ALL indicator/sentiment fields from the entry-time
                            # feature vector so downstream ML training sees real values
                            # instead of zeros.
                            _st = _ST.from_feature_vector(
                                _entry_features,
                                trade_id=_trade_id,
                                strategy_name=_strat_name,
                                market_regime=_regime_name,
                                confidence=_entry_conf,
                                hour_of_day=_hour,
                                day_of_week=_dow,
                            )
                            _st.symbol = order.symbol
                            _st.signal_id = _signal_id
                            _st.timestamp_open = _opened_at
                            _st.timestamp_close = closed.closed_at or ""
                            _st.entry_price = _entry_px_pos
                            _st.exit_price = _fill_px
                            _st.quantity = _fill_qty
                            _st.pnl_usd = round(_pnl, 4)
                            _st.pnl_pct = round(_pnl_pct, 4)
                            _st.is_win = _pnl > 0
                            _st.exit_reason = order.signal_reason or ""
                            _st.hold_duration_hours = round(_hold_hours, 2)
                            _st.commission_usd = order.commission
                        else:
                            _st = _ST(
                                trade_id=_trade_id,
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
                                hour_of_day=_hour,
                                day_of_week=_dow,
                                exit_reason=order.signal_reason or "",
                                hold_duration_hours=round(_hold_hours, 2),
                                commission_usd=order.commission,
                            )
                        # Drawdown/profit excursion during hold (MAE / MFE in %).
                        if _entry_px_pos > 0:
                            if _max_price_hold > 0:
                                _st.max_profit_during_trade = round(
                                    (_max_price_hold - _entry_px_pos) / _entry_px_pos * 100, 4,
                                )
                            if _min_price_hold > 0:
                                _st.max_drawdown_during_trade = round(
                                    (_min_price_hold - _entry_px_pos) / _entry_px_pos * 100, 4,
                                )
                        await asyncio.to_thread(repo.insert_strategy_trade, _st)
                        log.info("Trade saved to DB: {} {} PnL=${:.2f} ({:.2f}%)",
                                 order.symbol, _strat_name, _pnl, _pnl_pct)
                    except Exception as _db_err:
                        log.warning("Strategy trade DB write failed: {}", _db_err)

                    # Feed CB-2 (consecutive losses per strategy) and CB-8
                    # (daily commission spike vs balance) — both trigger
                    # on closed-trade side effects, not on entry signals.
                    if circuit_breakers is not None:
                        try:
                            circuit_breakers.record_trade_result(
                                is_win=_pnl > 0, strategy_name=_strat_name or "",
                            )
                            if risk_sentinel is not None and position_manager is not None:
                                circuit_breakers.check_commission_spike(
                                    daily_commission=risk_sentinel.daily_commission,
                                    balance=position_manager.balance,
                                )
                        except Exception as _cb_err:
                            log.debug("CB feed on trade close failed: {}", _cb_err)

                # Feed realized outcome back to ML tracker for concept drift detection
                if closed:
                    # Find matching ML entry by symbol (any strategy key)
                    _matched_key = None
                    for _k in list(_ml_prob_at_entry.keys()):
                        if _k[0] == order.symbol:
                            _matched_key = _k
                            break
                    if _matched_key is not None:
                        # _ml_prob_at_entry stores either (prob, entry_px) or
                        # (prob, entry_px, challenger_prob) depending on
                        # whether A/B is wired. Unpack defensively to keep
                        # both shapes working without a flag day.
                        _stash = _ml_prob_at_entry.pop(_matched_key, (0.5, 0.0))
                        if len(_stash) == 3:
                            _entry_prob, _entry_px, _challenger_prob = _stash
                        else:
                            _entry_prob, _entry_px = _stash
                            _challenger_prob = None
                        # Use realized PnL as source of truth (works for both LONG and SHORT)
                        _actual_win = closed.realized_pnl > 0
                        # Record to per-symbol model if available, and always to unified
                        _sym_ml = _ml_predictors.get(order.symbol)
                        if _sym_ml:
                            _sym_ml.record_outcome(_entry_prob, _actual_win)
                        if _ml_predictor:
                            _ml_predictor.record_outcome(_entry_prob, _actual_win)

                        # A/B Champion-Challenger: feed paired observations
                        # into the comparator. McNemar's test in evaluate()
                        # decides when the challenger has earned promotion;
                        # we just collect ground-truth here. Threshold for
                        # binary "would have entered" matches the calibrated
                        # decision threshold of each model.
                        if (_ml_ab_comparator is not None
                                and _challenger_prob is not None
                                and _ml_predictor is not None
                                and _ml_challenger is not None):
                            try:
                                _champ_thr = _ml_predictor._calibrated_threshold
                                _chall_thr = _ml_challenger._calibrated_threshold
                                _ml_ab_comparator.record(
                                    champion_pred=(_entry_prob >= _champ_thr),
                                    challenger_pred=(_challenger_prob >= _chall_thr),
                                    actual_win=_actual_win,
                                )
                                # When we hit a meaningful sample size, log
                                # the verdict so operators know whether the
                                # challenger is winning. McNemar handles small-N
                                # gracefully; the log is a daily diagnostic.
                                if _ml_ab_comparator.n_pairs % 25 == 0:
                                    _verdict = _ml_ab_comparator.evaluate()
                                    if _verdict is not None:
                                        log.info(
                                            "ML A/B: n={} verdict={} lift={:+.3f} p={:.4f} ({})",
                                            _verdict.n_pairs, _verdict.verdict,
                                            _verdict.precision_lift, _verdict.mcnemar_p_value,
                                            _verdict.reason,
                                        )
                            except Exception as _ab_err:
                                log.debug("A/B comparator record failed: {}", _ab_err)

                        # Drift detection → Telegram/alert bus. Before this
                        # change drift was only logged; operators missed it.
                        # We debounce via _ml_drift_last_alert_ts so a single
                        # drift episode doesn't spam.
                        for _drift_ml, _drift_tag in [(_sym_ml, order.symbol), (_ml_predictor, "unified")]:
                            if _drift_ml is None or not _drift_ml.metrics:
                                continue
                            try:
                                if _drift_ml._live_tracker.is_drifting(_drift_ml.metrics.precision):
                                    _last_alert = getattr(_drift_ml, "_last_drift_alert_ts", 0)
                                    if (time.time() - _last_alert) > 3600:  # 1h cooldown
                                        _drift_ml._last_drift_alert_ts = time.time()
                                        _live_m = _drift_ml._live_tracker.live_metrics()
                                        await bus.emit("ml_drift_detected", {
                                            "model": _drift_tag,
                                            "train_precision": _drift_ml.metrics.precision,
                                            "live_precision": _live_m.get("live_precision", 0.0),
                                            "n_pred_win": _live_m.get("n_pred_win", 0),
                                            "version": _drift_ml._model_version,
                                        })
                                        log.warning(
                                            "ML DRIFT DETECTED: {} — train_prec={:.3f} live_prec={:.3f} n={}",
                                            _drift_tag, _drift_ml.metrics.precision,
                                            _live_m.get("live_precision", 0.0),
                                            _live_m.get("n_pred_win", 0),
                                        )
                            except Exception as _drift_err:
                                log.warning("ML drift check failed for {}: {}", _drift_tag, _drift_err)

                        # ML auto-promote: shadow → block when live performance is
                        # statistically non-inferior to training. Uses Wilson-score
                        # lower bound on live precision — requires both enough samples
                        # (N≥100) and a 95% CI above a tolerance-adjusted training mark.
                        from analyzer.ml_predictor import wilson_lower_bound as _wilson
                        _MIN_PROMOTE_N = 100
                        _PRECISION_TOLERANCE = 0.05
                        _MIN_AUC_FOR_PROMOTE = 0.60
                        for _auto_ml in [_sym_ml, _ml_predictor]:
                            if _auto_ml is None or _auto_ml.rollout_mode != "shadow":
                                continue
                            if _auto_ml._live_tracker.n_recorded < _MIN_PROMOTE_N:
                                continue
                            _live = _auto_ml._live_tracker.live_metrics()
                            if "live_precision" not in _live or not _auto_ml.metrics:
                                continue
                            # Wilson CI is computed on the precision proportion, whose
                            # denominator is n_pred_win (events the model allowed), not
                            # total window size. Skip if the model has taken too few
                            # positive actions to estimate precision meaningfully.
                            _n_pred_win = int(_live.get("n_pred_win", 0))
                            if _n_pred_win < 30:
                                continue
                            _train_prec = _auto_ml.metrics.precision
                            _live_prec = _live["live_precision"]
                            _n_tp = max(int(round(_live_prec * _n_pred_win)), 0)
                            _wilson_low = _wilson(_n_tp, _n_pred_win)
                            _required = max(_train_prec - _PRECISION_TOLERANCE, 0.55)
                            if _wilson_low >= _required and _live["live_auc"] >= _MIN_AUC_FOR_PROMOTE:
                                _auto_ml.rollout_mode = "block"
                                log.info(
                                    "ML AUTO-PROMOTE: shadow → block "
                                    "(live_prec={:.3f} wilson_lo={:.3f} ≥ {:.3f}, auc={:.3f}, n_pos={})",
                                    _live_prec, _wilson_low, _required, _live["live_auc"], _n_pred_win,
                                )

            if risk_state_machine:
                await risk_state_machine.update(position_manager.realized_pnl_today)

        async def _on_market_trade(trade):
            if not position_manager:
                return
            await position_manager.update_price(trade.symbol, trade.price)
            if risk_state_machine:
                await risk_state_machine.update(position_manager.realized_pnl_today)

        async def _on_execution_degraded(payload):
            """Executor lost safety invariants (unprotected fill, orphan).

            Halts trading so no new exposure is added until an operator
            inspects and resumes. Event content is already written to
            events.jsonl by the executor.
            """
            nonlocal trading_paused
            trading_paused = True
            try:
                _sym = (payload or {}).get("symbol", "?")
                _reason = (payload or {}).get("reason", "unknown")
            except Exception:
                _sym, _reason = "?", "unknown"
            log.critical(
                "TRADING HALTED by execution_degraded: symbol={} reason={} — operator action required",
                _sym, _reason,
            )

        bus.subscribe(EVENT_ORDER_FILLED, _on_order_filled)
        bus.subscribe(EVENT_NEW_TRADE, _on_market_trade)
        bus.subscribe(EVENT_EXECUTION_DEGRADED, _on_execution_degraded)
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
    # Standing ML score per symbol — evaluated every cycle regardless of
    # whether a strategy signal fired. Gives the dashboard a live, always-fresh
    # probability + timestamp (instead of the stale "last ml_prob from strategy_log").
    _standing_ml_per_symbol: dict = {}  # symbol → {"prob", "decision", "ref_strategy", "ts_ms", "model_version", "mode"}
    # Per-symbol "last processed" timestamp — set each time a symbol finishes its
    # trading cycle. Surfaces data-freshness per symbol on the dashboard so pro users
    # can spot stale indicators (e.g., one symbol failing while others update).
    _last_cycle_ts_per_symbol: dict = {}  # symbol → ts_ms
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
        # Restore prior alerts + streak counters across restarts — otherwise
        # the Alerts tab on the dashboard starts empty on every boot, and a
        # loss-streak warning can fire again after a short restart.
        if repo is not None:
            try:
                import json as _json
                _am_blob = repo.load_system_state("alert_monitor")
                if _am_blob:
                    _alert_monitor.restore_state(_json.loads(_am_blob))
            except Exception as _am_restore_err:
                log.debug("AlertMonitor restore skipped: {}", _am_restore_err)
        log.info("[Module] Position sizer + Dynamic SL/TP + Alert monitor initialized")
    except Exception as e:
        log.warning("[Module] Advanced risk modules failed: {}", e)

    # ML Predictor — фильтрация сигналов (per-symbol models + unified fallback)
    _ml_predictor = None          # unified fallback (backward compat)
    _ml_predictors: dict = {}     # symbol → MLPredictor
    # Guards concurrent read/write between inference path, record_outcome, and
    # the auto-retrain task. Without this, a retrain task replacing a predictor
    # while another coroutine is mid-predict can corrupt the dict (rare but
    # catastrophic: mixed-version results or attribute errors).
    _ml_lock = asyncio.Lock()
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
            use_walk_forward=settings.analyzer_ml_use_walk_forward,
            use_bootstrap_ci=settings.analyzer_ml_use_bootstrap_ci,
            use_stacking=settings.analyzer_ml_use_stacking,
            use_regime_routing=settings.analyzer_ml_use_regime_routing,
        )
        _rollout = "shadow" if settings.analyzer_ml_shadow_mode else ("block" if settings.analyzer_ml_enabled else "off")
        _ml_models_dir = Path(__file__).parent / "data" / "ml_models"

        # Round-8 §4.4: rollout_mode reconciliation rule.
        # - Pre-load: seed from env (_rollout)
        # - load_from_file() restores the persisted mode (schema ≥ 5) and
        #   thereby may override the seed.
        # - Post-load: if env says "off" (operator disabled ML), force off
        #   regardless of pickled value — env is always-authoritative for
        #   the kill switch. Otherwise keep whatever was loaded, so an
        #   auto-promoted ``block`` survives a restart.
        def _reconcile_rollout(predictor) -> None:
            if _rollout == "off":
                predictor.rollout_mode = "off"

        # Load per-symbol models
        for _sym in (settings.trading_symbols or []):
            _sym_path = _ml_models_dir / f"ml_predictor_{_sym}.pkl"
            if _sym_path.exists():
                _sym_predictor = MLPredictor(config=_ml_cfg)
                _sym_predictor.rollout_mode = _rollout
                if _sym_predictor.load_from_file(_sym_path):
                    _reconcile_rollout(_sym_predictor)
                    _ml_predictors[_sym] = _sym_predictor
                    log.info("[Module] ML model for {} loaded (version={}, mode={})",
                             _sym, _sym_predictor._model_version, _sym_predictor.rollout_mode)
                else:
                    log.warning("[Module] ML model load failed for {}", _sym)

        # Load unified fallback model
        _ml_predictor = MLPredictor(config=_ml_cfg)
        _ml_predictor.rollout_mode = _rollout
        _ml_model_path = _ml_models_dir / "ml_predictor.pkl"
        _ml_unified_load_ok = False
        if _ml_model_path.exists():
            _ml_unified_load_ok = _ml_predictor.load_from_file(_ml_model_path)
            _reconcile_rollout(_ml_predictor)
            if _ml_unified_load_ok:
                log.info("[Module] ML unified model loaded (version={}, mode={})",
                         _ml_predictor._model_version, _ml_predictor.rollout_mode)
            else:
                log.warning("[Module] ML unified model load failed")
        else:
            log.warning("[Module] No saved ML model found at {}", _ml_model_path)

        # Round-8 §2.1: surface a loud failure when rollout_mode=block expects
        # ML protection but nothing actually loaded. Silent fail-open here
        # means the bot trades without the filter operators thought they'd
        # installed — at the exact configuration that said "don't trade
        # without ML". We emit to the event bus so Telegram / alert pipelines
        # notice. Shadow / off modes tolerate no-model startup (first boot).
        _expected_ml_active = settings.analyzer_ml_enabled and not settings.analyzer_ml_shadow_mode
        _have_any_model = bool(_ml_predictors) or _ml_unified_load_ok
        if _expected_ml_active and not _have_any_model:
            log.error(
                "[Module] ML rollout_mode=block but NO model loaded — bot will "
                "FAIL-OPEN (trades without ML filter). Either populate "
                "data/ml_models/ or set analyzer_ml_shadow_mode=true.",
            )
            try:
                await bus.emit("ml_load_failure", {
                    "rollout_mode": _rollout,
                    "reason": "no_model_loaded",
                    "per_symbol_loaded": list(_ml_predictors.keys()),
                    "unified_loaded": _ml_unified_load_ok,
                })
            except Exception as _bus_err:
                log.warning("[Module] ml_load_failure bus emit failed: {}", _bus_err)

        log.info("[Module] ML Predictors: {} per-symbol + unified fallback (mode={})",
                 len(_ml_predictors), _ml_predictor.rollout_mode)

        # ── A/B challenger (optional) ─────────────────────────────────
        # When ``data/ml_models/ml_predictor_challenger.pkl`` exists, load
        # it as a side-by-side challenger. Every entry signal then gets
        # scored by BOTH the production champion and the challenger; the
        # outcome is fed into a ChampionChallengerComparator so McNemar's
        # test can decide whether to promote. Absent the challenger pkl
        # the comparator stays None and the hot path is unchanged.
        _ml_challenger = None
        _ml_ab_comparator = None
        _challenger_path = _ml_models_dir / "ml_predictor_challenger.pkl"
        if _challenger_path.exists() and _ml_predictor and _ml_predictor.is_ready:
            try:
                _ml_challenger = MLPredictor(config=_ml_cfg)
                _ml_challenger.rollout_mode = "shadow"  # challenger is always shadow-only
                if _ml_challenger.load_from_file(_challenger_path):
                    from analyzer.ml.ab import ChampionChallengerComparator
                    _ml_ab_comparator = ChampionChallengerComparator(
                        champion_id=_ml_predictor._model_version,
                        challenger_id=_ml_challenger._model_version,
                        window=500, min_samples=50,
                    )
                    log.info(
                        "[Module] A/B challenger loaded: champion={}, challenger={}",
                        _ml_predictor._model_version, _ml_challenger._model_version,
                    )
                else:
                    _ml_challenger = None
            except Exception as _ab_err:
                log.warning("[Module] A/B challenger load failed: {}", _ab_err)
                _ml_challenger = None
                _ml_ab_comparator = None
    except ImportError as e:
        log.error("[Module] ML Predictor disabled — missing dependency: {}", e)
        _ml_predictor = None
        _ml_predictors = {}
    except FileNotFoundError as e:
        log.error("[Module] ML Predictor disabled — model file missing: {}", e)
        _ml_predictor = None
        _ml_predictors = {}
    except Exception as e:
        # Unknown failure — fail-open means bot trades WITHOUT ML filter, so
        # elevate to ERROR + include full traceback so operator notices.
        log.error("[Module] ML Predictor disabled (unexpected): {}", e, exc_info=True)
        _ml_predictor = None
        _ml_predictors = {}

    # ── ML retrain function (defined early so get_system_state can reference it) ──
    _ml_model_path_unified = Path(__file__).parent / "data" / "ml_models" / "ml_predictor.pkl"
    _ml_retrain_lock = asyncio.Lock()

    # Dashboard-facing progress snapshot. Mutated only by _run_ml_training;
    # read by GET /api/ml/training-progress. Kept as a plain dict so the
    # dashboard state provider can return it without locking — the coarse
    # fields (phase / percent) are updated atomically per assignment.
    _ml_training_progress: dict = {
        "active": False,
        "phase": "idle",
        "message": "",
        "symbols_total": 0,
        "symbols_done": 0,
        "current_symbol": None,
        "percent": 0,
        "started_at": None,
        "finished_at": None,
        "ok": None,
        "metrics": None,
    }

    def _ml_progress_set(**kwargs) -> None:
        _ml_training_progress.update(kwargs)
        t = _ml_training_progress.get("symbols_total") or 0
        d = _ml_training_progress.get("symbols_done") or 0
        if t > 0:
            _ml_training_progress["percent"] = min(100, int(round(d / t * 100)))

    def _get_ml_training_progress() -> dict:
        return dict(_ml_training_progress)

    async def _run_ml_training() -> bool:
        """Load trades from DB, train per-symbol + unified ML models, save to disk."""
        # Open MLRegistry run for full-pipeline traceability. Every retrain
        # now appears in the SQL registry timeline alongside per-symbol model
        # versions, with parameters, metrics, and any failure mode captured.
        # This replaces ad-hoc JSON writes with a queryable history that the
        # dashboard reads via /api/ml/registry endpoints.
        _retrain_run_id: Optional[str] = None
        _registry: Any = None
        try:
            from analyzer.ml.registry import MLRegistry
            _registry = MLRegistry(BASE_DIR / "data" / "sentinel.db")
            _retrain_run = _registry.start_run(
                "auto_retrain",
                params={
                    "symbols": list(settings.trading_symbols or []),
                    "use_walk_forward": getattr(_ml_predictor._cfg, "use_walk_forward", False)
                                         if _ml_predictor else False,
                    "use_regime_routing": getattr(_ml_predictor._cfg, "use_regime_routing", False)
                                           if _ml_predictor else False,
                    "use_stacking": getattr(_ml_predictor._cfg, "use_stacking", False)
                                     if _ml_predictor else False,
                },
                tags={"trigger": "scheduled", "kind": "training"},
            )
            _retrain_run_id = _retrain_run.run_id
        except Exception as _reg_err:
            log.warning("ML registry start_run failed (non-fatal): {}", _reg_err)

        _ml_progress_set(
            active=True,
            phase="starting",
            message="Подготовка к обучению…",
            symbols_total=0,
            symbols_done=0,
            current_symbol=None,
            percent=0,
            started_at=_ml_training_progress.get("started_at") or time.time(),
            finished_at=None,
            ok=None,
            metrics=None,
        )
        if not _ml_predictor or not repo:
            _ml_progress_set(
                phase="failed",
                message="ML-предиктор или БД не инициализированы",
                ok=False,
                active=False,
                finished_at=time.time(),
            )
            return False
        try:
            import sys as _sys
            _scripts_dir = str(Path(__file__).parent / "scripts")
            if _scripts_dir not in _sys.path:
                _sys.path.insert(0, _scripts_dir)
            from scripts.train_ml import build_trades_per_symbol
            _ml_progress_set(
                phase="collecting_data",
                message="Собираем сделки из БД…",
                symbols_total=0,
                symbols_done=0,
            )
            log.info("ML auto-retrain: collecting training data...")

            # Per-backtest callback — fires from the worker thread, mutates
            # the same dict the dashboard reads. Keeps the operator informed
            # during the 3–6 min data-collection phase instead of the bar
            # sitting at 0% for the whole window.
            def _collecting_progress(msg: str, done: int, total: int) -> None:
                # During this phase the bar fills 0→100% on its own scale;
                # when training starts it resets to 0 and fills again. The
                # phase label on the UI makes the two steps distinguishable.
                pct_collect = int(round((done / total) * 100)) if total else 0
                _ml_training_progress.update({
                    "message": f"Сбор данных: {msg}",
                    "percent": pct_collect,
                })

            trades_by_sym = await asyncio.to_thread(
                build_trades_per_symbol, repo, settings, _collecting_progress,
            )
            if not trades_by_sym:
                log.warning("ML auto-retrain: no trades available for training")
                _ml_progress_set(
                    phase="failed",
                    message="Нет сделок для обучения",
                    ok=False,
                )
                return False

            _ml_models_dir = Path(__file__).parent / "data" / "ml_models"
            _ml_models_dir.mkdir(parents=True, exist_ok=True)
            any_saved = False

            # +1 for unified step at the end
            _ml_progress_set(
                phase="training",
                message="Обучение моделей…",
                symbols_total=len(trades_by_sym) + 1,
                symbols_done=0,
            )

            from analyzer.ml_predictor import MLPredictor as _MLP

            # Train per-symbol models. IMPORTANT: train into a FRESH predictor
            # instance, then atomically swap. Calling .train() on the live
            # predictor mutates its internals mid-flight, which would race with
            # concurrent .predict() / .record_outcome() calls.
            # Skill regression tolerance: don't replace a working model if the
            # newly trained one is materially worse.
            SKILL_REGRESSION_TOLERANCE = 0.05

            # Helper that runs the configured training pipeline for one
            # symbol (or the unified fallback, same code path). Respects the
            # MLConfig feature flags introduced in phases 1-5:
            #   - use_regime_routing → train_with_regime_routing (calls
            #     .train() internally, then builds per-regime specialists)
            #   - use_walk_forward / use_stacking → train_walk_forward (runs
            #     WF validation, optionally fits stacking head on OOF)
            #   - neither flag → plain .train()
            # Called inside asyncio.to_thread so the trading loop stays
            # responsive while sklearn/lightgbm/xgboost release the GIL.
            def _run_configured_training(predictor, trades):
                cfg = predictor._cfg
                if getattr(cfg, "use_regime_routing", False):
                    predictor.train_with_regime_routing(trades)
                    m = predictor.metrics
                else:
                    m = predictor.train(trades)
                if (
                    m is not None
                    and predictor.is_ready
                    and (getattr(cfg, "use_walk_forward", False)
                         or getattr(cfg, "use_stacking", False))
                ):
                    try:
                        predictor.train_walk_forward(trades)
                    except Exception as wf_err:
                        log.warning("ML walk-forward step failed (model still usable): {}", wf_err)
                return m

            # Round-8 §4.2: emit bus alert on walk-forward instability so
            # operators catch regime-shift / overfitting early. Triggers
            # only when std_auc > 0.10 across folds; debounced 6h per
            # predictor via _last_wf_instability_alert_ts so persistent
            # instability doesn't spam Telegram.
            async def _maybe_emit_wf_instability(predictor, label: str) -> None:
                # walk_forward_report is a defined @property on MLPredictor —
                # direct attribute access is fine. std_auc / mean_auc are
                # dataclass fields on WFReport, also non-defensive reads.
                report = predictor.walk_forward_report
                if report is None:
                    return
                std_auc = report.std_auc
                mean_auc = report.mean_auc
                if std_auc <= 0.10:
                    return
                last_alert = getattr(predictor, "_last_wf_instability_alert_ts", 0)
                if (time.time() - last_alert) < 6 * 3600:
                    return
                predictor._last_wf_instability_alert_ts = time.time()
                try:
                    await bus.emit("ml_wf_instability", {
                        "model": label,
                        "std_auc": round(std_auc, 4),
                        "mean_auc": round(mean_auc, 4),
                        "min_auc": round(report.min_auc, 4),
                        "n_folds": report.n_folds_completed,
                        "version": predictor.model_version,
                    })
                    log.warning(
                        "ML WF INSTABILITY: {} — mean_auc={:.3f} std_auc={:.3f} "
                        "(threshold 0.10) — consider retraining more often",
                        label, mean_auc, std_auc,
                    )
                except Exception as _err:
                    log.warning("ml_wf_instability bus emit failed: {}", _err)

            for sym, sym_trades in trades_by_sym.items():
                _ml_progress_set(
                    current_symbol=sym,
                    message=f"{sym}: {len(sym_trades)} сделок",
                )
                if len(sym_trades) < 50:
                    log.info("ML auto-retrain: {} — too few trades ({}), skip", sym, len(sym_trades))
                    _ml_progress_set(symbols_done=_ml_training_progress["symbols_done"] + 1)
                    continue
                old_predictor = _ml_predictors.get(sym)
                old_skill = (
                    old_predictor.metrics.skill_score
                    if old_predictor is not None and old_predictor.metrics is not None
                    else 0.0
                )
                sym_predictor = _MLP(config=_ml_predictor._cfg)
                # Round-9 C1: inherit rollout_mode from the EXISTING per-symbol
                # predictor when one exists — not from the unified model. The
                # auto-promote path at ``_on_order_filled`` can move an
                # individual symbol from shadow → block when its own live
                # metrics pass the bar, independent of the unified predictor.
                # Seeding from unified would demote that symbol back to
                # shadow on the next retrain, silently erasing the promotion.
                sym_predictor.rollout_mode = (
                    old_predictor.rollout_mode
                    if old_predictor is not None
                    else _ml_predictor.rollout_mode
                )
                log.info("ML auto-retrain: training {} on {} trades...", sym, len(sym_trades))
                _ml_progress_set(message=f"Обучение {sym}…")
                metrics = await asyncio.to_thread(_run_configured_training, sym_predictor, sym_trades)
                if metrics is None or not sym_predictor.is_ready:
                    log.warning("ML auto-retrain: {} — training failed or below threshold", sym)
                    _ml_progress_set(symbols_done=_ml_training_progress["symbols_done"] + 1)
                    continue
                if old_skill > 0 and metrics.skill_score < old_skill - SKILL_REGRESSION_TOLERANCE:
                    log.warning(
                        "ML auto-retrain: {} — new skill={:.3f} < old skill={:.3f} - {:.2f} → keeping old model",
                        sym, metrics.skill_score, old_skill, SKILL_REGRESSION_TOLERANCE,
                    )
                    _ml_progress_set(symbols_done=_ml_training_progress["symbols_done"] + 1)
                    continue
                sym_path = _ml_models_dir / f"ml_predictor_{sym}.pkl"
                saved = await asyncio.to_thread(sym_predictor.save_to_file, sym_path)
                if saved:
                    async with _ml_lock:
                        _ml_predictors[sym] = sym_predictor
                    log.info(
                        "ML auto-retrain: ✅ {} saved (skill={:.3f}, was {:.3f})",
                        sym, metrics.skill_score, old_skill,
                    )
                    await _maybe_emit_wf_instability(sym_predictor, sym)
                    any_saved = True
                _ml_progress_set(symbols_done=_ml_training_progress["symbols_done"] + 1)

            # Train unified fallback — same pattern (new instance, swap)
            all_trades = []
            for sym_trades in trades_by_sym.values():
                all_trades.extend(sym_trades)
            all_trades.sort(key=lambda t: t.timestamp_open)
            _ml_progress_set(
                current_symbol="unified",
                message=f"Обучение unified на {len(all_trades)} сделках…",
            )
            if all_trades:
                old_unified_skill = (
                    _ml_predictor.metrics.skill_score
                    if _ml_predictor.metrics is not None
                    else 0.0
                )
                log.info("ML auto-retrain: training unified on {} trades...", len(all_trades))
                new_unified = _MLP(config=_ml_predictor._cfg)
                new_unified.rollout_mode = _ml_predictor.rollout_mode
                metrics = await asyncio.to_thread(_run_configured_training, new_unified, all_trades)
                if metrics is not None and new_unified.is_ready:
                    if old_unified_skill > 0 and metrics.skill_score < old_unified_skill - SKILL_REGRESSION_TOLERANCE:
                        log.warning(
                            "ML auto-retrain: unified — new skill={:.3f} < old skill={:.3f} - {:.2f} → keeping old model",
                            metrics.skill_score, old_unified_skill, SKILL_REGRESSION_TOLERANCE,
                        )
                    else:
                        saved = await asyncio.to_thread(new_unified.save_to_file, _ml_model_path_unified)
                        if saved:
                            async with _ml_lock:
                                # In-place field swap: callers everywhere kept a
                                # reference to _ml_predictor, so we keep the object
                                # but replace its trained fields. Live tracker is
                                # reset — past predictions were from the old model
                                # and would poison drift detection for the new one.
                                # Phase-1/2/4/5 artifacts (round-8 §2.4): copy them
                                # too so a retrain that rolls FROM stacking-on TO
                                # stacking-off clears the stale WF/regime/bootstrap
                                # state on the live predictor.
                                #
                                # Round-8 §1.1 defense: use __dict__.update() rather
                                # than 13 separate `=` assignments. In CPython the
                                # update is a single C call holding the GIL, so a
                                # hypothetical future caller that invokes predict()
                                # from a worker thread can never observe a half-
                                # swapped predictor (some fields new, some old).
                                # Today predict only runs on the event-loop thread
                                # and this is strictly defense-in-depth.
                                _ml_predictor.__dict__.update({
                                    "_ensemble": new_unified._ensemble,
                                    "_scaler": new_unified._scaler,
                                    "_feature_selector": new_unified._feature_selector,
                                    "_calibrated_threshold": new_unified._calibrated_threshold,
                                    "_model": new_unified._model,
                                    "_model_version": new_unified._model_version,
                                    "_metrics": new_unified._metrics,
                                    "_last_train_ts": new_unified._last_train_ts,
                                    "_live_tracker": new_unified._live_tracker,
                                    "_wf_report": new_unified._wf_report,
                                    "_regime_router": new_unified._regime_router,
                                    "_bootstrap_ci": dict(new_unified._bootstrap_ci),
                                    "_member_error_correlation": dict(
                                        new_unified._member_error_correlation
                                    ),
                                })
                            log.info(
                                "ML auto-retrain: ✅ unified saved (skill={:.3f}, was {:.3f})",
                                metrics.skill_score, old_unified_skill,
                            )
                            await _maybe_emit_wf_instability(_ml_predictor, "unified")
                            any_saved = True

            _ml_progress_set(symbols_done=_ml_training_progress["symbols_total"])
            final_metrics = None
            if _ml_predictor and _ml_predictor.metrics:
                m = _ml_predictor.metrics
                final_metrics = {
                    "precision": round(m.precision, 4),
                    "recall": round(m.recall, 4),
                    "roc_auc": round(m.roc_auc, 4),
                    "skill_score": round(m.skill_score, 4),
                    "train_samples": m.train_samples,
                }
            _ml_progress_set(
                phase="done" if any_saved else "failed",
                message=(
                    "Готово — модели обновлены"
                    if any_saved
                    else "Обучение завершилось без сохранения (skill-регрессия)"
                ),
                current_symbol=None,
                ok=any_saved,
                metrics=final_metrics,
            )

            # Register new model version in MLRegistry + atomic promote.
            # All metrics persisted to ml_runs.metrics_json; the model row
            # in ml_model_versions tags this run as the artifact source.
            # Failure here is non-fatal — the pickle on disk is still
            # the source of truth for the predict() path.
            if _registry is not None and _retrain_run_id is not None and any_saved:
                try:
                    if final_metrics is not None:
                        _registry.log_metrics(_retrain_run_id, {
                            **final_metrics,
                            "ece": float(_ml_predictor.metrics.ece) if _ml_predictor.metrics else 0.0,
                            "brier_score": float(_ml_predictor.metrics.brier_score) if _ml_predictor.metrics else 0.0,
                        })
                    _registry.log_artifact(
                        _retrain_run_id, "unified_pkl",
                        str(_ml_model_path_unified.resolve()),
                    )
                    _registry.finish_run(_retrain_run_id, status="finished")

                    mv = _registry.register_model(
                        name="ensemble_unified",
                        run_id=_retrain_run_id,
                        artifact_path=str(_ml_model_path_unified.resolve()),
                        description=f"Auto-retrain at {int(time.time())}, "
                                    f"skill={final_metrics.get('skill_score') if final_metrics else 0:.3f}",
                        metrics=final_metrics or {},
                        tags={"source": "auto_retrain"},
                    )
                    from analyzer.ml.registry import STAGE_PRODUCTION
                    _registry.transition_stage("ensemble_unified", mv.version, STAGE_PRODUCTION)
                    log.info(
                        "ML registry: registered ensemble_unified v{} → production "
                        "(prev versions auto-archived)",
                        mv.version,
                    )
                except Exception as _reg_err:
                    log.warning("ML registry post-train update failed: {}", _reg_err)
            elif _registry is not None and _retrain_run_id is not None:
                # Training failed or skill regression — record failure status.
                try:
                    _registry.finish_run(
                        _retrain_run_id,
                        status="failed",
                        error="no_model_saved (skill regression or training error)",
                    )
                except Exception:
                    pass

            return any_saved
        except Exception as exc:
            log.error("ML auto-retrain error: {}", exc)
            if _registry is not None and _retrain_run_id is not None:
                try:
                    _registry.finish_run(_retrain_run_id, status="failed",
                                          error=str(exc) or type(exc).__name__)
                except Exception:
                    pass
            try:
                from monitoring.event_log import emit_component_error
                emit_component_error(
                    "ml_auto_retrain",
                    str(exc) or type(exc).__name__,
                    exc=exc,
                    severity="error",
                    phase=_ml_training_progress.get("phase"),
                    current_symbol=_ml_training_progress.get("current_symbol"),
                )
            except Exception as _emit_err:
                log.debug("event_log emit failed: {}", _emit_err)
            _ml_progress_set(
                phase="failed",
                message=f"Ошибка: {exc}",
                ok=False,
                current_symbol=None,
            )
            return False
        finally:
            _ml_progress_set(active=False, finished_at=time.time())

    # ── Strategy Decision Log (ring buffer) ──
    from collections import deque
    _strategy_log: deque[dict] = deque(maxlen=50)

    # Restore strategy log across restarts — the dashboard "Strategy Log"
    # panel is otherwise empty on every boot until the next evaluation tick.
    if repo is not None:
        try:
            import json as _json
            _sl_blob = repo.load_system_state("strategy_log")
            if _sl_blob:
                for entry in _json.loads(_sl_blob) or []:
                    if isinstance(entry, dict):
                        _strategy_log.append(entry)
                log.info("Strategy log restored: {} entries", len(_strategy_log))
        except Exception as _sl_restore_err:
            log.debug("Strategy log restore skipped: {}", _sl_restore_err)

    # news_collector будет инициализирован позже (вместе с Dashboard)
    news_collector = None

    # Serialises the risk-check → execute path across concurrent candle events.
    # Without this lock, two symbols whose candles close in the same scheduler
    # tick each read ``open_symbols`` / ``open_positions_count`` before the
    # first trade is recorded, so both pass the correlation guard and the max
    # positions cap simultaneously — doubling exposure beyond configured caps.
    _trade_decision_lock = asyncio.Lock()

    async def _on_new_candle(candle):
        """Главный торговый цикл — вызывается при закрытии каждой свечи."""
        nonlocal _current_regime, trading_paused, _last_features, _last_features_per_symbol, _standing_ml_per_symbol, _last_cycle_ts_per_symbol

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
        _last_cycle_ts_per_symbol[symbol] = int(time.time() * 1000)

        # Refresh Chandelier Exit ATR for the open position (if any).
        # Cheap no-op when no position is armed; when one is, a volatility
        # regime shift after entry now propagates into the trailing stop
        # without waiting for the next open. Regime-independent, so safe
        # to run before regime detection.
        if position_manager.has_position(symbol) and features.atr > 0:
            await position_manager.update_chandelier_atr(symbol, features.atr)

        # 2.5 Update price-history cache for the correlation guard.
        # Cheap: replaces the deque each tick with the same closes feature_builder used.
        if price_history_cache is not None and candles_1h:
            try:
                price_history_cache.update_from_candles(symbol, [c.close for c in candles_1h])
            except Exception as _ph_err:
                log.debug("Price history cache update failed for {}: {}", symbol, _ph_err)

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

        # 3.5. Inject regime into FeatureVector for adaptive confidence thresholds
        features.market_regime = regime_name

        # 3.6. Post-regime risk updates for open positions.
        # Placed here so both hooks see the freshly detected regime.
        if position_manager.has_position(symbol) and features.atr > 0:
            # Phase 2: re-evaluate SL under current regime + ATR.
            # Ratchets the SL upward when either (a) position got deep
            # into profit (breakeven raise), or (b) regime flipped
            # adverse while in profit (protective tighten).
            _reeval = await position_manager.reevaluate_sl_tp(
                symbol,
                current_atr=features.atr,
                regime=regime_name,
            )
            _reeval_actions = _reeval.get("actions", [])
            if _reeval_actions:
                from monitoring.event_log import emit_guard_tripped
                _pos_for_evt = position_manager.get_position(symbol)
                _strat_for_evt = _pos_for_evt.strategy_name if _pos_for_evt else ""
                for _act in _reeval_actions:
                    emit_guard_tripped(
                        guard="sl_reeval",
                        name=_act["type"],
                        reason=f"SL adjusted: {_act['from']:.4f} → {_act['to']:.4f}",
                        severity="info",
                        symbol=symbol,
                        strategy=_strat_for_evt,
                        **{k: v for k, v in _act.items() if k not in ("type", "from", "to")},
                    )
                    log.info(
                        "SL re-evaluated {} ({}): {:.4f} → {:.4f}",
                        symbol, _act["type"], _act["from"], _act["to"],
                    )

            # Phase 3: hard regime-flip exit. Narrower criteria than
            # Phase 2 tighten — only fires on confirmed bearish trend
            # (trending_down + ADX ≥ 25). A tightened stop gives the
            # market a chance to recover; by the time it hits, most of
            # the open profit is already gone. This cut locks in the
            # remainder immediately.
            _pos_regime = position_manager.get_position(symbol)
            if _pos_regime is not None:
                from risk.regime_flip_exit import should_exit_on_regime_flip
                _should_flip, _flip_reason = should_exit_on_regime_flip(
                    strategy_name=_pos_regime.strategy_name,
                    current_regime=regime_name,
                    adx=features.adx,
                )
                if _should_flip:
                    await _force_exit_position(
                        _pos_regime, features, reason=_flip_reason,
                        guard_name="regime_flip",
                    )

            # Phase 4: weekend exit guard (opt-in).
            # Crypto weekend liquidity is thin and flash-crash-prone;
            # operators who leave a bot unattended over the weekend
            # can flip settings.weekend_exit_enabled on to get
            # cash-out-by-cutoff behaviour. Default-off — existing
            # deployments see no behaviour change.
            if settings.weekend_exit_enabled:
                _pos_weekend = position_manager.get_position(symbol)
                if _pos_weekend is not None:
                    from datetime import datetime, timezone
                    from risk.weekend_exit import should_exit_before_weekend
                    _should_we, _we_reason = should_exit_before_weekend(
                        datetime.now(timezone.utc),
                        enabled=settings.weekend_exit_enabled,
                        cutoff_day_of_week=settings.weekend_exit_cutoff_day_of_week,
                        cutoff_hour_utc=settings.weekend_exit_cutoff_hour_utc,
                        reopen_day_of_week=settings.weekend_exit_reopen_day_of_week,
                        reopen_hour_utc=settings.weekend_exit_reopen_hour_utc,
                    )
                    if _should_we:
                        await _force_exit_position(
                            _pos_weekend, features, reason=_we_reason,
                            guard_name="weekend_exit",
                        )

        # 3.6. Standing ML evaluation — runs every cycle, independent of signals.
        # Gives the dashboard a fresh probability per symbol without waiting for
        # a strategy to fire. We bypass MLPredictor.predict() (which short-circuits
        # to 0.5 when rollout_mode == "off") and call the underlying ensemble
        # directly, so the dashboard shows real model output even in shadow/off mode.
        _active_ml_sym = _ml_predictors.get(symbol, _ml_predictor)
        if _active_ml_sym and _active_ml_sym.is_ready and settings.analyzer_ml_enabled:
            try:
                from core.models import StrategyTrade as _ST_standing
                _ref_strat = "ema_crossover_rsi" if "ema_crossover_rsi" in strategies else (next(iter(strategies.keys())) if strategies else "reference")
                _standing_trade = _ST_standing.from_feature_vector(
                    features,
                    trade_id="standing",
                    strategy_name=_ref_strat,
                    market_regime=regime_name,
                    confidence=0.5,
                    hour_of_day=int(time.strftime("%H")),
                    day_of_week=int(time.strftime("%w")),
                )
                _standing_feats = _active_ml_sym.extract_features(_standing_trade, [])
                # Call predict() first — honours calibrated threshold. In "off" mode
                # it returns prob=0.5 (short-circuit), so we detect that and fall
                # back to raw ensemble inference for genuine probability surface.
                _standing_pred = _active_ml_sym.predict(_standing_feats)
                _standing_prob = float(_standing_pred.probability)
                _standing_decision = str(_standing_pred.decision)
                if _active_ml_sym.rollout_mode == "off":
                    # Bypass short-circuit: run the ensemble directly for a real reading.
                    try:
                        import numpy as _np_standing
                        _fv_arr = _np_standing.array([_standing_feats], dtype=_np_standing.float64)
                        _fv_arr = _np_standing.nan_to_num(_fv_arr, nan=0.0, posinf=0.0, neginf=0.0)
                        if _active_ml_sym._feature_selector.is_fitted and _active_ml_sym._feature_selector.dropped_names:
                            _fv_arr = _active_ml_sym._feature_selector.transform(_fv_arr)
                        if _active_ml_sym._scaler is not None:
                            _fv_arr = _active_ml_sym._scaler.transform(_fv_arr)
                        if _active_ml_sym._ensemble is not None and _active_ml_sym._ensemble.is_ready:
                            _standing_prob = float(_active_ml_sym._ensemble.predict_proba_calibrated(_fv_arr)[0])
                        elif _active_ml_sym._model is not None:
                            _standing_prob = float(_active_ml_sym._model.predict_proba(_fv_arr)[0][1])
                        _thr = max(_active_ml_sym._calibrated_threshold or 0.0, _active_ml_sym._cfg.block_threshold or 0.0) or 0.5
                        _standing_decision = "block" if _standing_prob < _thr * 0.85 else ("reduce" if _standing_prob < _thr else "allow")
                    except Exception as _raw_err:
                        log.debug("Standing raw ensemble call failed for {}: {}", symbol, _raw_err)
                _standing_ml_per_symbol[symbol] = {
                    "prob": _standing_prob,
                    "decision": _standing_decision,
                    "ref_strategy": _ref_strat,
                    "ts_ms": int(time.time() * 1000),
                    "model_version": str(_standing_pred.model_version or ""),
                    "mode": str(_active_ml_sym.rollout_mode or "off"),
                }
            except Exception as _standing_err:
                log.warning("Standing ML eval failed for {}: {}", symbol, _standing_err)

        # 4. Определить активные стратегии
        if settings.auto_strategy_selection and _current_regime:
            if _adaptive_allocator and _adaptive_allocator._skill_scores:
                allocs = _adaptive_allocator.get_adaptive_allocations(_current_regime)
                active_names = [a.strategy_name for a in allocs if a.is_active]
            else:
                active_names = get_active_strategies(_current_regime)
        else:
            active_names = list(strategies.keys())

        # Hot-path фильтр по UI-тогглам *_enabled. Применяется к обеим веткам
        # выше, чтобы отключённая в дашборде стратегия гарантированно не
        # попадала в ротацию — раньше эти флаги читались только UI и были
        # косметикой. settings уже подменяется в _handle_settings_update,
        # поэтому изменение тоггла подхватывается без рестарта.
        active_names = [
            n for n in active_names
            if getattr(settings, _STRATEGY_ENABLE_FLAGS.get(n, ""), True)
        ]

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

            # Pre-compute dynamic SL/TP BEFORE position sizing (SL% feeds risk-based sizing)
            _actual_sl_pct = 0.0
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
                _actual_sl_pct = sltp.stop_loss_pct
                log.debug("Dynamic SL/TP [{}]: SL={:.2f} ({:.1f}%) TP={:.2f} ({:.1f}%) (merged with strategy)",
                          sltp.method, signal.stop_loss_price, sltp.stop_loss_pct,
                          signal.take_profit_price, sltp.take_profit_pct)

            # Compute position size with real Kelly params + risk-based SL sizing
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
                    # Inject correlation-aware lookup so the Kelly sizer
                    # downsizes against ANY correlated open position, not
                    # just the legacy BTC/ETH cluster.
                    _ph_for_sizer = price_history_cache.snapshot() if price_history_cache else None
                    def _corr_lookup(a: str, b: str, _ph=_ph_for_sizer, _g=corr_guard) -> bool:
                        if _g is None or _ph is None:
                            return False
                        try:
                            from risk.correlation_guard import _log_returns, _pearson
                            xs = _log_returns(_ph.get(a, []))
                            ys = _log_returns(_ph.get(b, []))
                            n = min(len(xs), len(ys))
                            if n < 30:
                                return False
                            rho = _pearson(xs[-n:], ys[-n:])
                            return rho is not None and abs(rho) >= 0.70
                        except Exception:
                            return False
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
                        stop_loss_pct=_actual_sl_pct,
                        max_risk_per_trade_pct=risk_sentinel._limits.max_risk_per_trade_pct if risk_sentinel else 3.0,
                        corr_lookup=_corr_lookup,
                    ))
                    signal.suggested_quantity = sizing.quantity
                    log.debug("Position sizer: {} budget=${:.2f} ({:.1f}%) kelly={:.3f} sl={:.1f}%",
                              sizing.method, sizing.budget_usd, sizing.budget_pct, sizing.kelly_fraction, _actual_sl_pct)
                else:
                    if _current_regime and settings.auto_strategy_selection:
                        budget_pct = get_strategy_budget_pct(_current_regime, strat_name)
                    else:
                        budget_pct = settings.max_position_pct
                    budget_usd = balance * budget_pct / 100
                    budget_usd = min(budget_usd, settings.max_order_usd)
                    if features.close > 0:
                        signal.suggested_quantity = budget_usd / features.close

            # 6.5 ML Predictor filter (per-symbol model with unified fallback)
            # Captured regardless of decision so we can attach it to any
            # downstream signal/rejected result for dashboard visibility.
            _last_ml_prob_for_strategy = None
            _last_ml_decision_for_strategy = None
            _active_ml = _ml_predictors.get(symbol, _ml_predictor)
            if _active_ml and settings.analyzer_ml_enabled and _active_ml.rollout_mode != "off":
                try:
                    from core.models import StrategyTrade as _ST
                    # Use factory that guarantees 30/30 feature coverage (N-1 fix).
                    # Previous manual construction left 17 indicator fields at default 0,
                    # causing severe training/serving skew and bimodal predictions.
                    _ml_trade = _ST.from_feature_vector(
                        features,
                        trade_id="pending",
                        strategy_name=strat_name,
                        market_regime=regime_name,
                        confidence=signal.confidence,
                        hour_of_day=int(time.strftime("%H")),
                        day_of_week=int(time.strftime("%w")),
                    )
                    _prev_trades_for_ml = []
                    if repo:
                        try:
                            _raw = await asyncio.to_thread(repo.get_strategy_trades, strat_name, limit=20)
                            _prev_trades_for_ml = [
                                _ST.from_db_row(t) if isinstance(t, dict) else t
                                for t in _raw
                            ]
                        except Exception as _ml_hist_err:
                            # Empty history forces recent_win_rate / consecutive_losses
                            # to defaults — model loses ~5 features of signal. Visible
                            # log so we notice if this starts failing regularly.
                            log.warning("ML history fetch failed — using empty history: {}", _ml_hist_err)
                    _ml_features = _active_ml.extract_features(_ml_trade, _prev_trades_for_ml)
                    _ml_pred = _active_ml.predict(_ml_features)
                    log.info("ML prediction: {} prob={:.2f} decision={} mode={}",
                             strat_name, _ml_pred.probability, _ml_pred.decision, _ml_pred.rollout_mode)

                    # A/B challenger: when configured, score the same input
                    # with the challenger model and stash both probabilities.
                    # Cheap (one extra inference) and only fires when the
                    # operator dropped a challenger pkl into ml_models/.
                    _challenger_prob: Optional[float] = None
                    if _ml_challenger is not None and _ml_challenger.is_ready:
                        try:
                            _ch_pred = _ml_challenger.predict(_ml_features)
                            _challenger_prob = float(_ch_pred.probability)
                        except Exception as _ch_err:
                            log.debug("A/B challenger predict failed: {}", _ch_err)

                    # Track ML probability at entry so we can record outcome on close.
                    # When A/B is on, store (champ_prob, entry_price, challenger_prob);
                    # otherwise (champ_prob, entry_price). The 3-tuple shape is
                    # detected on the SELL path so existing 2-tuple consumers
                    # keep working.
                    if signal.direction.value == "BUY":
                        if _challenger_prob is not None:
                            _ml_prob_at_entry[(symbol, strat_name)] = (
                                _ml_pred.probability, features.close, _challenger_prob,
                            )
                        else:
                            _ml_prob_at_entry[(symbol, strat_name)] = (
                                _ml_pred.probability, features.close,
                            )
                    # Propagate the probability onto the downstream strat_results entry so
                    # the dashboard can show the *actual* live ML output, not only blocks.
                    _last_ml_prob_for_strategy = float(_ml_pred.probability)
                    _last_ml_decision_for_strategy = str(_ml_pred.decision)
                    if _ml_pred.decision == "block":
                        log.info("ML BLOCKED signal: {} {} {} prob={:.2f}",
                                 strat_name, signal.direction.value, symbol, _ml_pred.probability)
                        strat_results.append({"strategy": strat_name, "result": "ml_blocked",
                                              "direction": signal.direction.value,
                                              "ml_prob": _last_ml_prob_for_strategy,
                                              "ml_decision": _last_ml_decision_for_strategy,
                                              "detail": f"ML blocked: prob={_ml_pred.probability:.2f}"})
                        if repo:
                            try:
                                repo.insert_signal_execution(
                                    timestamp=ts_now, symbol=symbol, strategy_name=strat_name,
                                    direction=signal.direction.value, confidence=signal.confidence,
                                    outcome="ml_blocked", reason=f"ML prob={_ml_pred.probability:.2f}",
                                )
                            except Exception as _ml_audit_err:
                                log.warning("ML block audit write failed: {}", _ml_audit_err)
                        continue
                except Exception as e:
                    # ML prediction failure should be visible: without this, a
                    # broken feature selector or corrupt model silently lets
                    # every signal through (fail-open). Log at WARNING with
                    # traceback so operators can fix the root cause.
                    log.warning("ML prediction failed (fail-open — allowing signal): {}", e, exc_info=True)

            # 6.7 Global min_confidence enforcement (catches edge cases)
            if signal.direction == Direction.BUY and signal.confidence < settings.min_confidence:
                log.debug("Signal SKIPPED: {} conf={:.2f} < min {:.2f}",
                          strat_name, signal.confidence, settings.min_confidence)
                strat_results.append({"strategy": strat_name, "result": "low_confidence",
                                      "detail": f"conf={signal.confidence:.2f} < {settings.min_confidence}"})
                continue

            # 7. Risk check — use per-day realized PnL, not lifetime total.
            # Feeding total_realized_pnl here turns max_daily_loss_usd into a
            # lifetime cap: once cumulative PnL goes below -cap, every BUY is
            # rejected forever until manual reset.
            #
            # The whole critical section (read shared state → risk decision →
            # execute_order) is serialised through ``_trade_decision_lock``.
            # Without the lock, two symbols' candle events that arrive in the
            # same scheduler tick both read ``open_positions`` before the
            # first execute_order completes, so both pass the correlation
            # guard and max-open-positions cap — doubling real exposure.
            async with _trade_decision_lock:
                daily_pnl = position_manager.realized_pnl_today
                # Pro guards inputs (None when guards aren't attached → check_signal skips them).
                _ph_snap = price_history_cache.snapshot() if price_history_cache else None
                _open_exp = None
                if exposure_cap is not None:
                    from risk.exposure_caps import OpenPositionExposure as _OPE
                    _open_exp = [
                        _OPE(symbol=p.symbol, notional_usd=p.quantity * p.current_price)
                        for p in position_manager.open_positions
                    ]
                # Current WS data age (None if collector not ready) — gates BUY if
                # stale. Using the live reading at decision time catches a freeze
                # that started between the signal event and this evaluation.
                _data_age = None
                if collector is not None:
                    try:
                        _age = collector.last_data_age_sec
                        if _age != float("inf"):
                            _data_age = float(_age)
                    except Exception:
                        _data_age = None
                check, decision_trace = risk_sentinel.evaluate_with_trace(
                    signal=signal,
                    daily_pnl=daily_pnl,
                    open_positions_count=position_manager.open_positions_count,
                    total_exposure_pct=(position_manager.total_exposure_usd / position_manager.balance * 100)
                        if position_manager.balance > 0 else 0.0,
                    balance=position_manager.balance,
                    current_market_price=features.close,
                    open_symbols={p.symbol for p in position_manager.open_positions},
                    price_history=_ph_snap,
                    open_positions_exposure=_open_exp,
                    shadow_mode=False,  # production path: short-circuit on first reject
                    market_data_age_sec=_data_age,
                )

                # Persist structured decision trace + emit JSONL event.
                try:
                    _trace_dict = decision_trace.to_dict()
                    if repo:
                        try:
                            await asyncio.to_thread(repo.insert_decision_audit, _trace_dict)
                        except Exception as _da_err:
                            log.debug("decision_audit insert failed: {}", _da_err)
                    from monitoring.event_log import EventType, get_event_log
                    get_event_log().emit(
                        EventType.SIGNAL_DECISION,
                        **_trace_dict,
                    )
                except Exception as _trace_emit_err:
                    log.debug("Decision trace emit failed: {}", _trace_emit_err)

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
                        # Feed executor latency into CB-6. Three consecutive >5s
                        # fills trip the breaker (one-shot slowness won't).
                        if circuit_breakers is not None:
                            try:
                                circuit_breakers.check_latency(_exec_ms / 1000.0)
                            except Exception as _cb_err:
                                log.debug("CB-6 latency feed failed: {}", _cb_err)
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

            strat_results.append({
                "strategy": strat_name,
                "result": "signal",
                "direction": signal.direction.value,
                "confidence": float(signal.confidence),
                "ml_prob": _last_ml_prob_for_strategy,
                "ml_decision": _last_ml_decision_for_strategy,
                "detail": order_msg,
            })
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

    async def _force_exit_position(pos, features, *, reason: str, guard_name: str) -> None:
        """Hard-close a position outside the tick-level SL/TP loop.

        Used by guards that fire on candle-close events (regime flip,
        weekend cut-off, …) rather than price-tick triggers. Builds a
        synthetic SELL signal at the close price, runs it through the
        executor, then finalises the position. Emits a ``guard_tripped``
        event so the transition is reconstructable from events.jsonl.
        """
        nonlocal trading_paused
        if trading_paused or not executor:
            return
        if not position_manager.has_position(pos.symbol):
            return

        from monitoring.event_log import emit_guard_tripped
        signal = Signal(
            timestamp=int(time.time() * 1000),
            symbol=pos.symbol,
            direction=Direction.SELL,
            confidence=1.0,
            strategy_name=pos.strategy_name or guard_name,
            reason=reason,
            suggested_quantity=pos.quantity,
            stop_loss_price=pos.stop_loss_price,
            take_profit_price=pos.take_profit_price,
            close_pct=100.0,
        )
        # Close is safety-critical: one retry with short backoff, then halt
        # trading so the position is visible to the operator instead of
        # silently lingering open on the exchange.
        order = None
        last_exc: Optional[BaseException] = None
        for _attempt in (1, 2):
            try:
                order = await executor.execute_order(
                    signal=signal,
                    quantity=pos.quantity,
                    current_price=features.close,
                )
                if order:
                    break
                last_exc = None
            except Exception as e:  # noqa: BLE001 — retry path
                last_exc = e
                log.warning("{} force-exit attempt {} failed for {}: {}",
                            guard_name, _attempt, pos.symbol, e)
            if _attempt == 1:
                await asyncio.sleep(1.0)

        if order:
            closed = await position_manager.close_position(order)
            if closed:
                emit_guard_tripped(
                    guard=guard_name,
                    name=guard_name,
                    reason=reason,
                    severity="warning",
                    symbol=pos.symbol,
                    strategy=pos.strategy_name,
                    exit_price=float(features.close),
                    pnl=float(closed.realized_pnl),
                )
                log.warning(
                    "{} FORCED EXIT {} @ {:.4f} — {} (pnl={:.2f})",
                    guard_name.upper(), pos.symbol, features.close, reason, closed.realized_pnl,
                )
            return

        # Close failed twice: surface it to events.jsonl and pause trading.
        from monitoring.event_log import emit_component_error as _emit_cerr
        _emit_cerr(
            "main.force_exit",
            f"{guard_name} force-exit failed for {pos.symbol} after retry",
            exc=last_exc,
            severity="critical",
            symbol=pos.symbol,
            strategy=pos.strategy_name,
            reason=reason,
            guard=guard_name,
        )
        trading_paused = True
        log.critical(
            "TRADING HALTED: {} force-exit failed twice for {} — position stays open, operator action required",
            guard_name, pos.symbol,
        )

    # SL/TP проверка на каждый тик
    async def _check_sl_tp(trade):
        """Проверить stop-loss / take-profit на каждый маркет-трейд."""
        # `nonlocal` must precede every use of the name in the function
        # body — Python raises SyntaxError otherwise. The halt-on-failed-
        # close branch later in this function assigns `trading_paused = True`,
        # so the declaration belongs at the very top.
        nonlocal trading_paused
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

        # Determine if this is a partial or full close.
        # Phase 5: TP triggers are f"tp{N}_partial" for arbitrary ladder
        # depth; the per-stage close% and the post-fill trailing config
        # come from the ladder state, not hard-coded here.
        _is_partial = trigger.startswith("tp") and trigger.endswith("_partial")
        _stage_num = 0
        _stage_info = None
        if _is_partial:
            try:
                _stage_num = int(trigger[len("tp"):-len("_partial")])
            except ValueError:
                _stage_num = 0
            _stage_info = position_manager.get_current_tp_stage(symbol)
            _close_pct = float(_stage_info.close_pct_of_remaining) if _stage_info else 50.0
        else:
            _close_pct = 100.0  # full close

        _qty = pos.quantity if not _is_partial else pos.quantity * _close_pct / 100
        _strategy = pos.strategy_name or "sl_tp"
        _sl = pos.stop_loss_price
        _tp = pos.take_profit_price

        # Guard against partial slices below the exchange min-notional. A
        # dust-sized tp_partial used to hit executor → None → retry → HALT
        # (see risk/tp_splits.evaluate_partial_notional docstring).
        if _is_partial:
            from risk.tp_splits import (
                PartialNotionalDecision,
                evaluate_partial_notional,
            )
            from monitoring.event_log import emit_guard_tripped as _emit_guard
            _min_notional = float(getattr(executor, "MIN_ORDER_USD", 10.0))
            _decision = evaluate_partial_notional(
                remaining_qty=pos.quantity,
                close_pct=_close_pct,
                price=trade.price,
                min_notional_usd=_min_notional,
            )
            _slice_usd = round(_qty * trade.price, 2)
            _full_usd = round(pos.quantity * trade.price, 2)
            if _decision is PartialNotionalDecision.ESCALATE:
                log.warning(
                    "Partial {} slice ${:.2f} < ${:.2f} min-notional — escalating to full close ({})",
                    trigger, _slice_usd, _min_notional, symbol,
                )
                _emit_guard(
                    guard="sl_tp_close",
                    name="partial_escalated_to_full",
                    reason=f"partial ${_slice_usd:.2f} < ${_min_notional:.2f} min — closing 100%",
                    severity="info",
                    symbol=symbol,
                    strategy=_strategy,
                    trigger=trigger,
                    partial_notional_usd=_slice_usd,
                    full_notional_usd=_full_usd,
                )
                _is_partial = False
                _close_pct = 100.0
                _qty = pos.quantity
            elif _decision is PartialNotionalDecision.SKIP:
                log.warning(
                    "Partial {} slice ${:.2f} and full position ${:.2f} both < ${:.2f} min-notional — skipping trigger ({})",
                    trigger, _slice_usd, _full_usd, _min_notional, symbol,
                )
                _emit_guard(
                    guard="sl_tp_close",
                    name="partial_skipped_dust",
                    reason=f"partial ${_slice_usd:.2f} and full ${_full_usd:.2f} below min — trigger skipped",
                    severity="warning",
                    symbol=symbol,
                    strategy=_strategy,
                    trigger=trigger,
                    partial_notional_usd=_slice_usd,
                    full_notional_usd=_full_usd,
                )
                return

        direction = Direction.SELL
        reason = f"SL/TP triggered: {trigger}" + (f" ({_close_pct:.0f}%)" if _is_partial else "")
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
            close_pct=_close_pct,
        )

        # Final check right before execution
        if not position_manager.has_position(symbol):
            return

        # SL/TP close is safety-critical — one retry, then halt + event.
        # `nonlocal trading_paused` already declared at the top of _check_sl_tp.
        order = None
        last_exc: Optional[BaseException] = None
        for _attempt in (1, 2):
            try:
                order = await executor.execute_order(
                    signal=signal,
                    quantity=_qty,
                    current_price=trade.price,
                )
                if order:
                    break
                last_exc = None
            except Exception as e:  # noqa: BLE001
                last_exc = e
                log.warning("SL/TP close attempt {} failed for {}: {}", _attempt, symbol, e)
            if _attempt == 1:
                await asyncio.sleep(1.0)

        if not order:
            from monitoring.event_log import emit_component_error as _emit_cerr
            _emit_cerr(
                "main.sl_tp_close",
                f"SL/TP close failed for {symbol} after retry (trigger={trigger})",
                exc=last_exc,
                severity="critical",
                symbol=symbol,
                strategy=_strategy,
                trigger=trigger,
            )
            trading_paused = True
            log.critical(
                "TRADING HALTED: SL/TP close failed twice for {} (trigger={}) — position stays open",
                symbol, trigger,
            )
            return

        if _is_partial:
            closed_pos = await position_manager.partial_close_position(order, _close_pct)
            if closed_pos and position_manager.has_position(symbol):
                _trailing = _stage_info.trailing_after if _stage_info else None
                await position_manager.apply_tp_stage_transition(
                    symbol,
                    stage=_stage_num,
                    move_to_breakeven=(_stage_num == 1),
                    trailing=_trailing,
                )
            log.info("Partial close {} {}: {}% @ {:.2f}", trigger, symbol, _close_pct, trade.price)
        else:
            log.info("{} {} @ {:.2f} — {}", trigger, symbol, trade.price, reason)

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
                    log.info("[Startup] Latest {} candle for {} is still OPEN (closes in {:.0f}s) — running standing ML warmup",
                             settings.signal_timeframe, sym, (candle_close_time - now_ms) / 1000)
                    # Candle is open — run standing ML warmup using last closed candle data
                    # so dashboard shows ML forecast immediately without waiting for next close.
                    try:
                        candles_1h_raw = await asyncio.to_thread(repo.get_candles, sym, "1h", limit=60)
                        candles_4h_raw = await asyncio.to_thread(repo.get_candles, sym, "4h", limit=60)
                        candles_1d_raw = await asyncio.to_thread(repo.get_candles, sym, "1d", limit=60)
                        if candles_1h_raw and candles_4h_raw:
                            _mk_candle = lambda c: CandleModel(
                                timestamp=c["timestamp"], symbol=c["symbol"], interval=c["interval"],
                                open=float(c["open"]), high=float(c["high"]), low=float(c["low"]),
                                close=float(c["close"]), volume=float(c.get("volume", 0)),
                                trades_count=int(c.get("trades_count", 0)),
                            )
                            _c1h = [_mk_candle(c) for c in candles_1h_raw]
                            _c4h = [_mk_candle(c) for c in candles_4h_raw]
                            _c1d = [_mk_candle(c) for c in candles_1d_raw] or None
                            _wup_feats = feature_builder.build(sym, _c1h, _c4h, _c1d)
                            if _wup_feats is not None:
                                _last_features_per_symbol[sym] = _wup_feats
                                _last_cycle_ts_per_symbol[sym] = int(_time.time() * 1000)
                                _wup_ml = _ml_predictors.get(sym, _ml_predictor)
                                if _wup_ml and _wup_ml.is_ready and settings.analyzer_ml_enabled:
                                    _wup_regime = detect_regime(_wup_feats)
                                    _wup_regime_name = _wup_regime.regime.value if _wup_regime else "unknown"
                                    from core.models import StrategyTrade as _ST_wup
                                    _wup_ref = "ema_crossover_rsi" if "ema_crossover_rsi" in strategies else (next(iter(strategies.keys())) if strategies else "reference")
                                    _wup_trade = _ST_wup.from_feature_vector(
                                        _wup_feats, trade_id="warmup", strategy_name=_wup_ref,
                                        market_regime=_wup_regime_name, confidence=0.5,
                                        hour_of_day=int(_time.strftime("%H")),
                                        day_of_week=int(_time.strftime("%w")),
                                    )
                                    _wup_feat_vec = _wup_ml.extract_features(_wup_trade, [])
                                    _wup_pred = _wup_ml.predict(_wup_feat_vec)
                                    _wup_prob = float(_wup_pred.probability)
                                    if _wup_ml.rollout_mode == "off":
                                        try:
                                            import numpy as _np_wup
                                            _fv = _np_wup.nan_to_num(_np_wup.array([_wup_feat_vec], dtype=_np_wup.float64), nan=0.0, posinf=0.0, neginf=0.0)
                                            if _wup_ml._feature_selector.is_fitted and _wup_ml._feature_selector.dropped_names:
                                                _fv = _wup_ml._feature_selector.transform(_fv)
                                            if _wup_ml._scaler is not None:
                                                _fv = _wup_ml._scaler.transform(_fv)
                                            if _wup_ml._ensemble is not None and _wup_ml._ensemble.is_ready:
                                                _wup_prob = float(_wup_ml._ensemble.predict_proba_calibrated(_fv)[0])
                                            elif _wup_ml._model is not None:
                                                _wup_prob = float(_wup_ml._model.predict_proba(_fv)[0][1])
                                        except Exception:
                                            pass
                                    _thr_wup = max(_wup_ml._calibrated_threshold or 0.0, _wup_ml._cfg.block_threshold or 0.0) or 0.5
                                    _wup_dec = "block" if _wup_prob < _thr_wup * 0.85 else ("reduce" if _wup_prob < _thr_wup else "allow")
                                    _standing_ml_per_symbol[sym] = {
                                        "prob": _wup_prob,
                                        "decision": _wup_dec,
                                        "ref_strategy": _wup_ref,
                                        "ts_ms": int(_time.time() * 1000),
                                        "model_version": str(_wup_pred.model_version or ""),
                                        "mode": str(_wup_ml.rollout_mode or "off"),
                                    }
                                    log.info("[Startup] Standing ML warmup for {}: prob={:.2f} decision={}", sym, _wup_prob, _wup_dec)
                    except Exception as _wup_err:
                        log.warning("[Startup] Standing ML warmup failed for {}: {}", sym, _wup_err)
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

    def build_market_chart(interval: str = "1m", symbol: str = "", end_ts: int = 0) -> dict:
        primary_symbol = symbol.upper() if symbol else (settings.trading_symbols[0] if settings.trading_symbols else "BTCUSDT")
        # Validate symbol is in configured list
        if settings.trading_symbols and primary_symbol not in [s.upper() for s in settings.trading_symbols]:
            primary_symbol = settings.trading_symbols[0]
        if interval not in _ALLOWED_INTERVALS:
            interval = "1m"
        limit = _INTERVAL_LIMITS.get(interval, 120)
        # Historical pagination: end_ts (ms) shifts the window to end at that time
        try:
            end_ts_int = int(end_ts) if end_ts else 0
        except (TypeError, ValueError):
            end_ts_int = 0

        try:
            candles = repo.get_candles(primary_symbol, interval, limit=limit, before_ts=end_ts_int)
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
                raw = repo.get_candles(primary_symbol, "1m", limit=raw_limit, before_ts=end_ts_int)
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

        # Fallback to trades for 1m only — skip in historical mode (end_ts set)
        if interval == "1m" and not end_ts_int:
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
                "blocked_strategies": (
                    circuit_breakers.get_blocked_strategies() if circuit_breakers is not None else {}
                ),
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
            "standing_ml_per_symbol": dict(_standing_ml_per_symbol),
            "last_cycle_ts_per_symbol": dict(_last_cycle_ts_per_symbol),
            "win_rate_per_symbol": _calc_win_rate_per_symbol(position_state.get("recent_trades", [])),
            "strategy_performance": repo.get_strategy_performance() if repo else [],
            "trades_export": repo.get_all_trades_for_export() if repo else [],
            "trades_export_full": repo.get_strategy_trades() if repo else [],
            "alerts": _alert_monitor.get_recent_alerts() if _alert_monitor else [],
            "signal_exec_stats": repo.get_signal_execution_stats() if repo else {},
            "ml_predictor": _ml_predictor,
            "ml_retrain_fn": _run_ml_training,
            "ml_training_progress_fn": _get_ml_training_progress,
            "ml_training_progress_set_fn": _ml_progress_set,
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

    # Kill Switch: wired to the dashboard / telegram kill buttons. Runs the
    # three-step shutdown protocol (cancel open orders → close positions →
    # halt trading) before triggering the process shutdown signal.
    from risk.kill_switch import KillSwitch as _KillSwitch
    kill_switch = _KillSwitch(bus)

    async def _kill_cancel_all_orders() -> None:
        if not executor:
            return
        syms = list(settings.trading_symbols or [])
        if position_manager:
            syms = list(set(syms) | {p.symbol for p in position_manager.open_positions})
        await executor.cancel_all_open_orders(syms)

    async def _kill_close_all_positions() -> None:
        if not position_manager:
            return
        # Snapshot the symbols first — close_position mutates the dict.
        syms = [p.symbol for p in position_manager.open_positions]
        for _sym in syms:
            try:
                result = await _handle_manual_close(_sym)
                if not result.get("ok"):
                    log.error("Kill: close {} failed — {}", _sym, result.get("error"))
            except Exception as _close_err:
                log.error("Kill: close {} raised: {}", _sym, _close_err)
                from monitoring.event_log import emit_component_error as _emit_cerr
                _emit_cerr(
                    "main.kill_close",
                    f"kill-switch close failed for {_sym}",
                    exc=_close_err, severity="critical", symbol=_sym,
                )

    async def _kill_stop_trading() -> None:
        nonlocal trading_paused
        trading_paused = True

    kill_switch.on_cancel_all_orders = _kill_cancel_all_orders
    kill_switch.on_close_all_positions = _kill_close_all_positions
    kill_switch.on_stop_trading = _kill_stop_trading

    async def _handle_kill():
        log.warning("KILL requested — running kill-switch protocol")
        try:
            await kill_switch.activate("Manual kill")
        except Exception as _k_err:
            log.critical("Kill switch activation raised: {}", _k_err)
            from monitoring.event_log import emit_component_error as _emit_cerr
            _emit_cerr(
                "main.handle_kill",
                f"kill-switch activate raised: {_k_err}",
                exc=_k_err, severity="critical",
            )
        shutdown.trigger()

    async def _handle_manual_close(symbol: str) -> dict:
        """Ручное закрытие одной позиции по символу.

        Строит SELL-сигнал и прогоняет его через executor; дальнейшая
        обработка (обновление БД, уведомления) идёт стандартным путём
        через _on_order_filled → position_manager.close_position.
        """
        from core.models import Signal as _Signal
        if not position_manager or not executor:
            return {"ok": False, "error": "execution engine not initialised"}
        pos = position_manager.get_position(symbol)
        if not pos:
            return {"ok": False, "error": f"no open position for {symbol}"}
        if pos.quantity <= 0 or pos.current_price <= 0:
            return {"ok": False, "error": "invalid position state (qty/price)"}
        signal = _Signal(
            timestamp=int(time.time() * 1000),
            symbol=symbol,
            direction=Direction.SELL,
            confidence=1.0,
            strategy_name=pos.strategy_name or "manual",
            reason="manual_close",
            suggested_quantity=pos.quantity,
            stop_loss_price=pos.stop_loss_price,
            take_profit_price=pos.take_profit_price,
            close_pct=100.0,
        )
        order = None
        last_exc: Optional[BaseException] = None
        for _attempt in (1, 2):
            try:
                order = await executor.execute_order(
                    signal=signal,
                    quantity=pos.quantity,
                    current_price=pos.current_price,
                )
                if order:
                    break
                last_exc = None
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                log.warning("Manual close attempt {} failed for {}: {}", _attempt, symbol, exc)
            if _attempt == 1:
                await asyncio.sleep(1.0)
        if not order:
            from monitoring.event_log import emit_component_error as _emit_cerr
            _emit_cerr(
                "main.manual_close",
                f"manual close failed for {symbol} after retry",
                exc=last_exc,
                severity="error",
                symbol=symbol,
            )
            return {"ok": False, "error": str(last_exc) if last_exc else "executor rejected the order"}
        log.warning("Manual close submitted: {} qty={:.6f} @ {:.2f}",
                    symbol, pos.quantity, pos.current_price)
        return {"ok": True, "symbol": symbol, "price": pos.current_price,
                "quantity": pos.quantity}

    def _handle_settings_update(new_settings):
        """Propagate risk-limit changes to runtime objects without restart."""
        nonlocal settings
        # Snapshot strategy-enable flags BEFORE swapping settings so we can
        # diff old → new and emit one transition event per actually-changed
        # toggle. Per-tick logging would spam events.jsonl; this fires only
        # when the user clicks Save and a flag flipped.
        prev_strategy_flags = {
            strat: bool(getattr(settings, flag, False))
            for strat, flag in _STRATEGY_ENABLE_FLAGS.items()
        }
        settings = new_settings
        try:
            from monitoring.event_log import get_event_log, EventType
            evlog = get_event_log()
            for strat, flag in _STRATEGY_ENABLE_FLAGS.items():
                new_val = bool(getattr(new_settings, flag, False))
                if new_val != prev_strategy_flags[strat]:
                    evlog.emit(
                        EventType.STRATEGY_TOGGLED,
                        strategy=strat,
                        flag=flag,
                        enabled=new_val,
                        source="settings_update",
                    )
                    log.info(
                        "Strategy {} toggled via UI: {} -> {}",
                        strat, prev_strategy_flags[strat], new_val,
                    )
        except Exception as _toggle_emit_err:
            log.debug("strategy_toggled emit failed: {}", _toggle_emit_err)
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
        if circuit_breakers:
            try:
                if new_settings.cb_consecutive_losses >= 1:
                    circuit_breakers._loss_threshold = new_settings.cb_consecutive_losses
                if new_settings.cb_strategy_cooldown_sec >= 0:
                    circuit_breakers._default_strategy_cooldown_sec = new_settings.cb_strategy_cooldown_sec
                circuit_breakers._strategy_cooldown_overrides = dict(
                    new_settings.cb_strategy_cooldown_overrides or {}
                )
                log.info(
                    "CB-2 updated at runtime: threshold={}, cooldown={}s, overrides={}",
                    new_settings.cb_consecutive_losses,
                    new_settings.cb_strategy_cooldown_sec,
                    new_settings.cb_strategy_cooldown_overrides,
                )
            except Exception as _cb_upd_err:
                log.warning("CB-2 runtime update failed: {}", _cb_upd_err)
        if position_manager:
            position_manager._max_open_positions = new_settings.max_open_positions
            log.info("PositionManager max_open_positions updated to {}", new_settings.max_open_positions)

    try:
        from dashboard.app import Dashboard
        dashboard = Dashboard(settings, bus, state_provider=get_system_state)

        dashboard.on_stop = _handle_stop
        dashboard.on_resume = _handle_resume
        dashboard.on_kill = _handle_kill
        dashboard.on_manual_close = _handle_manual_close
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
        telegram_bot.on_manual_close = _handle_manual_close
        await telegram_bot.start()
        if telegram_bot.enabled and telegram_bot._running:
            log.info("[Module] Telegram bot started")
            # Отправить уведомление о старте
            mode_icon = "📄" if settings.trading_mode.lower() == "paper" else "💰"
            symbols_str = ", ".join(settings.trading_symbols)
            strats = list(strategies.keys()) if strategies else []
            strats_str = ", ".join(strats) if strats else "нет"
            await telegram_bot.send_message(
                f"🟢 <b>SENTINEL v{VERSION} — ЗАПУЩЕН</b>\n"
                f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
                f"\n"
                f"{mode_icon} Режим торговли: <b>{settings.trading_mode.upper()}</b>\n"
                f"📊 Торговые пары: <b>{symbols_str}</b>\n"
                f"⚙️ Стратегии: <b>{strats_str}</b>\n"
                f"💵 Начальный баланс: <b>${settings.paper_initial_balance:,.2f}</b>\n"
                f"\n"
                f"Бот активен. Используйте /help для управления."
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
        last_retrain_ts = 0.0
        MIN_RETRAIN_INTERVAL_SEC = 24 * 3600  # cooldown so persistent drift doesn't spam train every 6h
        while True:
            try:
                async with _ml_retrain_lock:
                    if _ml_predictor:
                        now = time.time()
                        # Determine retrain trigger reason. Three signals:
                        #   1. Model never trained (cold start) → train.
                        #   2. Schedule says model is stale (`needs_retrain`).
                        #   3. PSI feature-drift monitor reports `major_drift`
                        #      (live distribution materially diverged from
                        #      training reference). This catches regime shifts
                        #      faster than waiting for live-precision drop —
                        #      drift in inputs precedes drift in outcomes.
                        psi_drift_triggered = False
                        psi_summary = None
                        if (_ml_predictor.is_ready
                                and getattr(_ml_predictor, "_feature_drift_monitor", None) is not None):
                            try:
                                psi_summary = _ml_predictor._feature_drift_monitor.summary()
                                psi_drift_triggered = psi_summary.get("status") == "major_drift"
                            except Exception as _psi_err:
                                log.debug("PSI summary fetch failed: {}", _psi_err)

                        if not _ml_predictor.is_ready:
                            log.info("ML auto-retrain: no model loaded — triggering initial training")
                            await _run_ml_training()
                            last_retrain_ts = now
                        elif psi_drift_triggered:
                            elapsed = now - last_retrain_ts
                            if elapsed < MIN_RETRAIN_INTERVAL_SEC:
                                log.warning(
                                    "ML auto-retrain: PSI MAJOR DRIFT detected (max_psi={:.3f}, "
                                    "drifting features={}) — but cooldown active ({:.1f}h remaining)",
                                    psi_summary.get("max_psi", 0.0),
                                    psi_summary.get("drifting", []),
                                    (MIN_RETRAIN_INTERVAL_SEC - elapsed) / 3600,
                                )
                            else:
                                log.warning(
                                    "ML auto-retrain: PSI MAJOR DRIFT — retraining "
                                    "(max_psi={:.3f}, drifting={})",
                                    psi_summary.get("max_psi", 0.0),
                                    psi_summary.get("drifting", []),
                                )
                                await _run_ml_training()
                                last_retrain_ts = now
                        elif _ml_predictor.needs_retrain():
                            elapsed = now - last_retrain_ts
                            if elapsed < MIN_RETRAIN_INTERVAL_SEC:
                                log.info(
                                    "ML auto-retrain: needs retrain but cooldown active ({:.1f}h remaining)",
                                    (MIN_RETRAIN_INTERVAL_SEC - elapsed) / 3600,
                                )
                            else:
                                log.info("ML auto-retrain: model is stale — retraining")
                                await _run_ml_training()
                                last_retrain_ts = now
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
                # Feed live equity into the drawdown breaker even when no signal
                # is being checked — without this the breaker only ticks on
                # signal evaluation and could miss a fast intraday drawdown.
                if dd_breaker is not None:
                    try:
                        dd_breaker.update(position_manager.balance)
                    except Exception as _dd_err:
                        log.debug("DD breaker tick failed: {}", _dd_err)
                    # Persist state every snapshot cycle (~30s) so restart
                    # loses at most one tick of drawdown context.
                    if repo is not None:
                        try:
                            import json as _json
                            await asyncio.to_thread(
                                repo.save_system_state,
                                "dd_breaker",
                                _json.dumps(dd_breaker.export_state()),
                            )
                        except Exception as _dd_save_err:
                            log.debug("DD breaker save failed: {}", _dd_save_err)

            # Persist risk counters / state-machine / alerts / strategy log so
            # a mid-day restart preserves daily trade caps, cooldowns, the
            # risk-state (NORMAL/REDUCED/SAFE/STOP), the dashboard Strategy Log
            # panel, and recent alerts. Best-effort — a failing save should
            # never take down the snapshot loop.
            if repo is not None:
                try:
                    import json as _json
                    if risk_sentinel is not None:
                        await asyncio.to_thread(
                            repo.save_system_state,
                            "risk_sentinel",
                            _json.dumps(risk_sentinel.export_state()),
                        )
                    if risk_state_machine is not None:
                        await asyncio.to_thread(
                            repo.save_system_state,
                            "risk_state_machine",
                            _json.dumps(risk_state_machine.export_state()),
                        )
                    if _alert_monitor is not None:
                        await asyncio.to_thread(
                            repo.save_system_state,
                            "alert_monitor",
                            _json.dumps(_alert_monitor.export_state()),
                        )
                    await asyncio.to_thread(
                        repo.save_system_state,
                        "strategy_log",
                        _json.dumps(list(_strategy_log)),
                    )
                except Exception as _state_save_err:
                    log.debug("Runtime state save failed: {}", _state_save_err)

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

    # Daily reset on UTC midnight rollover.
    # Without this, RiskSentinel._daily_trades and _daily_commission accumulate
    # indefinitely: max_daily_trades and max_daily_commission_pct become
    # lifetime caps rather than daily caps, permanently blocking new entries.
    async def _daily_reset_loop():
        last_day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        while True:
            await asyncio.sleep(60)
            now_day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if now_day != last_day:
                try:
                    if risk_sentinel is not None:
                        risk_sentinel.reset_daily()
                    if position_manager is not None:
                        position_manager.reset_daily_stats()
                    if circuit_breakers is not None:
                        circuit_breakers.reset_daily()
                    log.info("Daily reset: {} -> {} (UTC rollover)", last_day, now_day)
                except Exception as _reset_err:
                    log.error("Daily reset failed: {}", _reset_err)
                last_day = now_day

    daily_reset_task = asyncio.create_task(_daily_reset_loop())

    # Heartbeat writer (шаг 19)
    heartbeat_task = asyncio.create_task(heartbeat_writer(settings))

    # Throttled Telegram summary of risk-rejected signals.
    # Without this, rejections only appear in logs/events.jsonl — operators
    # miss that signals are being produced but filtered. Per-rejection DMs
    # would spam (dca_bot can fire hourly); a windowed top-N digest surfaces
    # the pattern without the noise.
    async def _rejection_summary_loop():
        interval_hours = settings.telegram_rejection_summary_hours
        if interval_hours <= 0 or not telegram_bot or not _alert_monitor:
            return
        interval_sec = interval_hours * 3600
        while True:
            await asyncio.sleep(interval_sec)
            try:
                summary = _alert_monitor.drain_rejection_summary(top_n=5)
                if not summary:
                    continue
                from telegram_bot.formatters import format_rejection_summary
                elapsed_hours = (summary["window_end"] - summary["window_start"]) / 3600
                text = format_rejection_summary(
                    total=summary["total"],
                    top=summary["top"],
                    window_hours=elapsed_hours,
                )
                await telegram_bot.send_message(text)
            except Exception as _rej_sum_err:
                log.warning("Rejection summary task failed: {}", _rej_sum_err)

    rejection_summary_task = asyncio.create_task(_rejection_summary_loop())

    # Monitor background tasks for unexpected failures
    _background_tasks: list[tuple[str, asyncio.Task]] = [
        ("equity_snapshot", equity_task),
        ("heartbeat", heartbeat_task),
        ("ml_retrain", ml_retrain_task),
        ("pos_db_sync", pos_sync_task),
        ("daily_reset", daily_reset_task),
        ("rejection_summary", rejection_summary_task),
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
    daily_reset_task.cancel()
    rejection_summary_task.cancel()
    await asyncio.gather(watchdog_task, heartbeat_task, equity_task, ml_retrain_task, pos_sync_task, daily_reset_task, rejection_summary_task, return_exceptions=True)
    if collector:
        await collector.stop()
    if collector_task:
        collector_task.cancel()
        await asyncio.gather(collector_task, return_exceptions=True)
    if telegram_bot:
        try:
            state = get_system_state()
            pnl_today = state.get("pnl_today", 0.0)
            pnl_total = state.get("pnl_total", 0.0)
            trades_today = state.get("trades_today", 0)
            balance = state.get("balance", 0.0)
            wins = state.get("total_wins", 0)
            losses_count = state.get("total_losses", 0)
            uptime = state.get("uptime", "N/A")
            pnl_icon = "📈" if pnl_today >= 0 else "📉"
            pnl_total_icon = "📈" if pnl_total >= 0 else "📉"
            wr = wins / (wins + losses_count) * 100 if (wins + losses_count) > 0 else 0
            shutdown_msg = (
                f"🔴 <b>SENTINEL — ОСТАНОВКА</b>\n"
                f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
                f"⏱️ Uptime: {uptime}\n"
                f"\n"
                f"💵 <b>P&amp;L за сегодня: {'+'if pnl_today >= 0 else ''}{pnl_today:.2f}$</b>  {pnl_icon}\n"
                f"💰 <b>P&amp;L всего: {'+'if pnl_total >= 0 else ''}{pnl_total:.2f}$</b>  {pnl_total_icon}\n"
            )
            if trades_today > 0:
                shutdown_msg += f"📊 Сделок сегодня: <b>{trades_today}</b>  (W:{wins} / L:{losses_count})  WR:{wr:.0f}%\n"
            if balance > 0:
                shutdown_msg += f"💵 Финальный баланс: <b>${balance:,.2f}</b>\n"
            shutdown_msg += "\nСистема остановлена."
            await telegram_bot.send_message(shutdown_msg)
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
    # Final persistence of in-memory runtime state (risk counters, alerts,
    # strategy log, dd_breaker). Without this, a clean shutdown between
    # snapshot cycles loses up to 30s of the newest state on next boot.
    if repo is not None:
        try:
            import json as _json
            if risk_sentinel is not None:
                repo.save_system_state("risk_sentinel", _json.dumps(risk_sentinel.export_state()))
            if risk_state_machine is not None:
                repo.save_system_state("risk_state_machine", _json.dumps(risk_state_machine.export_state()))
            if _alert_monitor is not None:
                repo.save_system_state("alert_monitor", _json.dumps(_alert_monitor.export_state()))
            repo.save_system_state("strategy_log", _json.dumps(list(_strategy_log)))
            if dd_breaker is not None:
                repo.save_system_state("dd_breaker", _json.dumps(dd_breaker.export_state()))
            log.info("Shutdown: runtime state persisted")
        except Exception as _final_save_err:
            log.warning("Shutdown runtime state save failed: {}", _final_save_err)
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
