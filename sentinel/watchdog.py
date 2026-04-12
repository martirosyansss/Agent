"""
SENTINEL Watchdog — независимый сторожевой процесс.

Читает heartbeat-файл основного процесса.
Если heartbeat устарел — логирует, шлёт alert, может принять emergency-меры.
Запускается в ОТДЕЛЬНОМ терминале: python watchdog.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from loguru import logger

BASE_DIR = Path(__file__).resolve().parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import load_settings  # noqa: E402
from core.constants import HEARTBEAT_FILE, VERSION  # noqa: E402


def setup_watchdog_logging() -> None:
    logger.remove()
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    logger.add(
        log_dir / "watchdog.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | watchdog | {message}",
        level="INFO",
        rotation="10 MB",
        retention=5,
        encoding="utf-8",
    )
    logger.add(
        sys.stderr,
        format="<cyan>{time:HH:mm:ss}</cyan> | <level>{level:<8}</level> | 🐕 {message}",
        level="INFO",
        colorize=True,
    )


def read_heartbeat() -> int | None:
    """Возвращает Unix timestamp последнего heartbeat или None."""
    hb_path = BASE_DIR / HEARTBEAT_FILE
    if not hb_path.exists():
        return None
    try:
        return int(hb_path.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return None


def main() -> None:
    setup_watchdog_logging()
    settings = load_settings()
    timeout = settings.watchdog_timeout
    interval = settings.watchdog_heartbeat_interval

    logger.info("🐕 Watchdog v{} запущен. Timeout: {}с, Интервал проверки: {}с",
                VERSION, timeout, interval)

    consecutive_misses = 0

    while True:
        time.sleep(interval)

        last_hb = read_heartbeat()
        now = int(time.time())

        if last_hb is None:
            consecutive_misses += 1
            logger.warning("Heartbeat файл не найден (miss #{})", consecutive_misses)
        else:
            age = now - last_hb
            if age > timeout:
                consecutive_misses += 1
                logger.error(
                    "⚠️ Heartbeat устарел на {} сек (лимит {}с) — miss #{}",
                    age, timeout, consecutive_misses,
                )
            else:
                if consecutive_misses > 0:
                    logger.info("✅ Heartbeat восстановлен (возраст {}с)", age)
                consecutive_misses = 0

        # Emergency: если N пропусков подряд
        if consecutive_misses >= 3:
            logger.critical(
                "🚨 SENTINEL не отвечает {} раз подряд! "
                "Требуется ручное вмешательство.",
                consecutive_misses,
            )
            # TODO: Telegram alert, emergency close (Phase 8)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("🐕 Watchdog остановлен.")
