"""
SENTINEL Watchdog — независимый сторожевой процесс.

Читает heartbeat-файл основного процесса.
Если heartbeat устарел — логирует, шлёт alert, может принять emergency-меры.
Запускается в ОТДЕЛЬНОМ терминале: python watchdog.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from urllib import error, parse, request

from loguru import logger

BASE_DIR = Path(__file__).resolve().parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import load_settings  # noqa: E402
from core.constants import HEARTBEAT_FILE, VERSION  # noqa: E402


WATCHDOG_ALERT_FILE = BASE_DIR / "data" / "watchdog_alert.json"


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


def write_alert_report(consecutive_misses: int, last_heartbeat: int | None, now: int) -> Path:
    """Сохраняет локальный аварийный отчёт watchdog."""
    WATCHDOG_ALERT_FILE.parent.mkdir(parents=True, exist_ok=True)
    WATCHDOG_ALERT_FILE.write_text(
        json.dumps(
            {
                "generated_at": now,
                "consecutive_misses": consecutive_misses,
                "last_heartbeat": last_heartbeat,
                "age_sec": None if last_heartbeat is None else max(now - last_heartbeat, 0),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return WATCHDOG_ALERT_FILE


def send_watchdog_alert(settings, message: str) -> bool:
    """Отправляет Telegram alert, если токен и chat_id заданы."""
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        return False

    body = parse.urlencode(
        {
            "chat_id": settings.telegram_chat_id,
            "text": message,
        }
    ).encode("utf-8")
    req = request.Request(
        url=f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage",
        data=body,
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=10) as response:
            return response.status == 200
    except (error.URLError, TimeoutError):
        return False


def main() -> None:
    setup_watchdog_logging()
    settings = load_settings()
    timeout = settings.watchdog_timeout
    interval = settings.watchdog_heartbeat_interval

    logger.info("🐕 Watchdog v{} запущен. Timeout: {}с, Интервал проверки: {}с",
                VERSION, timeout, interval)

    consecutive_misses = 0
    alert_sent = False

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
                alert_sent = False

        # Emergency: если N пропусков подряд
        if consecutive_misses >= 3 and not alert_sent:
            report_path = write_alert_report(consecutive_misses, last_hb, now)
            message = (
                f"SENTINEL watchdog alert: нет heartbeat {consecutive_misses} проверок подряд. "
                f"Локальный отчёт: {report_path.name}"
            )
            logger.critical(
                "🚨 SENTINEL не отвечает {} раз подряд! "
                "Требуется ручное вмешательство. Отчёт: {}",
                consecutive_misses,
                report_path,
            )
            if send_watchdog_alert(settings, message):
                logger.info("Telegram alert отправлен")
            else:
                logger.warning("Telegram alert не отправлен")
            alert_sent = True


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("🐕 Watchdog остановлен.")
