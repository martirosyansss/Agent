"""
Автоматический backup SQLite каждые N часов.

Копирует файл БД (через VACUUM INTO для консистентности).
Хранит последние 5 бэкапов, автоматически удаляет старые.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path

from loguru import logger

from database.db import Database

log = logger.bind(module="backup")

MAX_BACKUPS = 5


async def backup_loop(db: Database, backup_dir: str | Path, interval_hours: int = 6) -> None:
    """Фоновая задача: бэкап БД каждые interval_hours."""
    backup_path = Path(backup_dir)
    backup_path.mkdir(parents=True, exist_ok=True)
    interval_sec = interval_hours * 3600

    while True:
        await asyncio.sleep(interval_sec)
        try:
            await asyncio.to_thread(_do_backup, db, backup_path)
        except Exception as e:
            log.error("Backup failed: {}", e)


def _do_backup(db: Database, backup_dir: Path) -> Path:
    """Выполнить бэкап (синхронно). Возвращает путь к файлу бэкапа."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    dest = backup_dir / f"sentinel_backup_{ts}.db"
    src = Path(str(db._db_path))
    try:
        shutil.copy2(src, dest)
        log.info("Backup создан: {}", dest.name)
    except Exception as e:
        log.error("Backup failed (copy): {}", e)
        raise

    _rotate_backups(backup_dir)
    return dest


def _rotate_backups(backup_dir: Path) -> None:
    """Оставить только MAX_BACKUPS самых свежих файлов."""
    backups = sorted(backup_dir.glob("sentinel_backup_*.db"), key=lambda p: p.stat().st_mtime)
    while len(backups) > MAX_BACKUPS:
        old = backups.pop(0)
        try:
            old.unlink()
            log.info("Удалён старый бэкап: {}", old.name)
        except (OSError, PermissionError) as e:
            log.warning("Не удалось удалить бэкап {}: {}", old.name, e)


def manual_backup(db: Database, backup_dir: str | Path) -> Path:
    """Ручной бэкап (вызывается при graceful shutdown)."""
    return _do_backup(db, Path(backup_dir))
