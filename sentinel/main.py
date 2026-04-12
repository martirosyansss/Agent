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
    VERSION,
    LOG_ROTATION_SIZE,
    LOG_ROTATION_COUNT,
    PID_FILE,
    STATE_FILE,
    HEARTBEAT_FILE,
)
from core.events import EventBus  # noqa: E402


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

    # Heartbeat writer (шаг 19)
    heartbeat_task = asyncio.create_task(heartbeat_writer(settings))

    log.info("🟢 Система запущена. Режим: {} TRADING", settings.trading_mode.upper())

    # Ожидание сигнала остановки
    await shutdown.event.wait()

    # Graceful shutdown sequence
    log.info("Остановка системы...")
    heartbeat_task.cancel()
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
