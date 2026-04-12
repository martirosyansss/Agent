"""
SENTINEL V1.5 — Точка входа.

Последовательность запуска (21 шаг) описана в ТЗ §26.2.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path

from loguru import logger

# Корень проекта — директория, где лежит main.py
BASE_DIR = Path(__file__).resolve().parent

# Добавляем корень проекта в sys.path, чтобы импорты работали
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import load_settings, Settings  # noqa: E402
from core.constants import (  # noqa: E402
    APP_NAME,
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
from core.models import Direction  # noqa: E402


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
        except (ValueError, OSError, AttributeError):
            pass  # PID файл повреждён или процесс мёртв — перехватываем

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
        hb_path.write_text(str(int(time.time())), encoding="utf-8")
        await asyncio.sleep(interval)


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

    # [3/10] Binance API ping — будет реализован в collector
    log.info("[3/10] Binance API ping — ⏳ (будет при подключении)")

    # [4/10] Auth test — будет реализован в collector
    log.info("[4/10] API auth test — ⏳ (будет при подключении)")

    # [5/10] Нет withdrawal прав — будет реализован в collector
    log.info("[5/10] API permissions — ⏳ (будет при подключении)")

    # [6/10] Синхронизация времени — будет реализован в collector
    log.info("[6/10] Time sync — ⏳ (будет при подключении)")

    # [7/10] SQLite integrity — будет реализован в database
    log.info("[7/10] SQLite integrity — ⏳ (будет при инициализации DB)")

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
    # TODO: Инициализация модулей (Phase 4+)
    # 6.  SQLite (WAL mode, integrity check)
    # 7.  State recovery
    # 9.  Data Collector (WebSocket)
    # 10. Data Integrity Guard
    # 11. Feature Engine
    # 12. Strategy Engine
    # 13. Risk Sentinel
    # 14. Circuit Breakers
    # 15. Execution Engine (paper mode)
    # 16. Position Manager
    # 17. Telegram Bot
    # 18. Web Dashboard
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

        async def _on_order_filled(order):
            if not position_manager:
                return

            if order.side == Direction.BUY:
                opened = await position_manager.open_position(order)
                if opened and risk_sentinel:
                    risk_sentinel.record_trade(order.commission, increment_trade=True)
            elif order.side == Direction.SELL:
                closed = await position_manager.close_position(order)
                if closed and risk_sentinel:
                    risk_sentinel.record_trade(order.commission, increment_trade=False)

            if risk_state_machine:
                await risk_state_machine.update(position_manager.total_realized_pnl)

        async def _on_market_trade(trade):
            if not position_manager:
                return
            position_manager.update_price(trade.symbol, trade.price)
            if risk_state_machine:
                await risk_state_machine.update(position_manager.total_realized_pnl)

        bus.subscribe(EVENT_ORDER_FILLED, _on_order_filled)
        bus.subscribe(EVENT_NEW_TRADE, _on_market_trade)
        log.info("[Module] Position & Risk runtime initialized")
    except Exception as e:
        log.warning("[Module] Position/Risk runtime failed: {}", e)

    # 18. Web Dashboard
    dashboard = None

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

        return {
            "mode": settings.trading_mode,
            "risk_state": risk_state_machine.state.value if risk_state_machine else "NORMAL",
            "uptime": _format_uptime(),
            "pnl_today": pnl_today,
            "pnl_total": float(position_state.get("pnl_total", 0.0)),
            "open_positions": int(position_state.get("open_positions", 0)),
            "trades_today": int(position_state.get("trades_today", 0)),
            "balance": balance,
            "win_rate": float(position_state.get("win_rate", 0.0)),
            "positions": position_state.get("positions", []),
            "recent_trades": position_state.get("recent_trades", []),
            "pnl_history": position_state.get("pnl_history", []),
            "risk_details": {
                "daily_loss": min(pnl_today, 0.0),
                "max_drawdown": float(position_state.get("max_drawdown_pct", 0.0)) / 100,
                "exposure": float(position_state.get("exposure_pct", 0.0)) / 100,
                "trade_freq": int(risk_metrics.get("trades_last_hour", 0)),
                "daily_commission": float(risk_metrics.get("daily_commission", 0.0)),
                "market_data_age_sec": market_data_age_sec,
                "cooldown_remaining_sec": int(risk_metrics.get("cooldown_remaining_sec", 0)),
            },
        }

    try:
        from dashboard.app import Dashboard
        dashboard = Dashboard(settings, bus, state_provider=get_system_state)

        async def _handle_stop():
            log.warning("STOP requested from dashboard")
            shutdown.trigger()

        async def _handle_resume():
            log.info("RESUME requested from dashboard")
            if risk_state_machine:
                risk_state_machine.reset()

        dashboard.on_stop = _handle_stop
        dashboard.on_resume = _handle_resume
        dashboard.on_kill = _handle_stop
        await dashboard.start()
        log.info("[Module] Dashboard started on http://localhost:{}", settings.dashboard_port)
    except Exception as e:
        log.warning("[Module] Dashboard failed: {}", e)

    # Heartbeat writer (шаг 19)
    heartbeat_task = asyncio.create_task(heartbeat_writer(settings))

    log.info("🟢 Система запущена. Режим: {} TRADING", settings.trading_mode.upper())

    # Ожидание сигнала остановки
    await shutdown.event.wait()

    # Graceful shutdown sequence
    log.info("Остановка системы...")
    heartbeat_task.cancel()
    await asyncio.gather(heartbeat_task, return_exceptions=True)
    if collector:
        await collector.stop()
    if collector_task:
        collector_task.cancel()
        await asyncio.gather(collector_task, return_exceptions=True)
    if dashboard:
        await dashboard.stop()
    db.close()
    save_state({"stopped_at": int(time.time()), "mode": settings.trading_mode})
    release_pid_lock()
    log.info("🔴 Система остановлена. Состояние сохранено.")


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
    finally:
        release_pid_lock()


if __name__ == "__main__":
    main()
