"""
Structured JSONL event log for post-mortem analysis.

Loguru text logs are great for humans reading errors at 3am, but useless
for machine analytics: ``"Multi-TF mismatch: 4h_alignment, 1d_alignment
(TA=0.45, 4h_EMA50=58000, 1d_EMA50=55000)"`` is one regexp away from
unparseable.

This module writes one JSON object per line to ``logs/events.jsonl`` —
every event carries a timestamp, an event-type, and an arbitrary payload.
Pandas can ingest the whole file with one ``pd.read_json(..., lines=True)``
call and pivot/groupby/correlate without parsing.

Event-type vocabulary (extend as needed):
- ``signal_generated``       — strategy emitted a Signal
- ``signal_approved``        — passed all risk gates
- ``signal_rejected``        — blocked by a specific gate
- ``order_filled``           — exchange acknowledged fill
- ``position_opened``        — position state created
- ``position_closed``        — exit with realised PnL
- ``guard_tripped``          — DD breaker / circuit breaker / kill switch
- ``regime_change``          — market regime transition
- ``news_critical``          — critical bearish news fired
- ``ml_prediction``          — ML predictor verdict logged

The writer is async-safe via a single ``threading.Lock`` because pickle/
asyncio file ops can interleave on Windows. Throughput target: 1000 events/s
on a laptop SSD, well below typical Sentinel rate (~5-50/min).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _json_default(obj: Any) -> Any:
    """Best-effort serialiser for non-JSON-native objects."""
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "value"):  # Enum
        return obj.value
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    return repr(obj)


class EventLog:
    """Append-only JSONL writer with size-based rotation.

    Each call to ``emit`` writes one line: ``{"ts": ..., "type": ..., ...}``.
    The active file is ``logs/events.jsonl``; on rotation the current file
    is renamed to ``events.jsonl.1`` / ``.2`` etc. up to ``backup_count``.

    The writer is enabled only when ``path`` is set; tests construct
    ``EventLog(path=None)`` to use the in-memory ``recent_events`` buffer.
    """

    def __init__(
        self,
        path: Optional[Path] = None,
        max_bytes: int = 50 * 1024 * 1024,   # 50 MB per file
        backup_count: int = 5,
        in_memory_buffer: int = 1000,         # always keep last N in RAM
    ) -> None:
        self._path = path
        self._max_bytes = max_bytes
        self._backup_count = backup_count
        self._lock = threading.Lock()
        self._recent: list[dict] = []
        self._buffer_max = in_memory_buffer
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────
    # Emission
    # ──────────────────────────────────────────────

    def emit(self, event_type: str, **payload: Any) -> dict:
        """Write one event. Returns the assembled record (useful in tests)."""
        record = {"ts": int(time.time() * 1000), "type": event_type, **payload}
        with self._lock:
            self._recent.append(record)
            if len(self._recent) > self._buffer_max:
                self._recent = self._recent[-self._buffer_max:]
            if self._path is not None:
                try:
                    self._rotate_if_needed()
                    with self._path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(record, default=_json_default, ensure_ascii=False))
                        f.write("\n")
                except Exception as exc:
                    logger.error("EventLog write failed: %s", exc)
        return record

    @contextmanager
    def time_event(self, event_type: str, **payload: Any):
        """Context manager that emits an event with measured duration_ms.

            with event_log.time_event("risk_check", symbol="BTC"):
                run_check()
        """
        start = time.time()
        try:
            yield
        finally:
            self.emit(event_type, duration_ms=int((time.time() - start) * 1000), **payload)

    # ──────────────────────────────────────────────
    # Recent buffer (for live dashboards / tests)
    # ──────────────────────────────────────────────

    def recent_events(self, type_filter: Optional[str] = None, limit: int = 100) -> list[dict]:
        with self._lock:
            evs = list(self._recent)
        if type_filter:
            evs = [e for e in evs if e.get("type") == type_filter]
        return evs[-limit:]

    def clear_buffer(self) -> None:
        with self._lock:
            self._recent.clear()

    # ──────────────────────────────────────────────
    # Rotation
    # ──────────────────────────────────────────────

    def _rotate_if_needed(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            size = self._path.stat().st_size
        except OSError:
            return
        if size < self._max_bytes:
            return
        # Rotate: events.jsonl → events.jsonl.1, .1 → .2, ..., drop the oldest.
        for i in range(self._backup_count - 1, 0, -1):
            src = self._path.with_suffix(self._path.suffix + f".{i}")
            dst = self._path.with_suffix(self._path.suffix + f".{i + 1}")
            if src.exists():
                if dst.exists():
                    dst.unlink()
                src.rename(dst)
        backup = self._path.with_suffix(self._path.suffix + ".1")
        if backup.exists():
            backup.unlink()
        self._path.rename(backup)


# ──────────────────────────────────────────────
# Module-level singleton accessor
# ──────────────────────────────────────────────

_default_log: Optional[EventLog] = None


def get_event_log() -> EventLog:
    """Return the process-wide EventLog. Creates an in-memory one if not set."""
    global _default_log
    if _default_log is None:
        _default_log = EventLog(path=None)
    return _default_log


def set_event_log(log: EventLog) -> None:
    """Install a process-wide EventLog (called by main.py at startup)."""
    global _default_log
    _default_log = log
