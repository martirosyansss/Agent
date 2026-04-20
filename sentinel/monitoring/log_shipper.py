"""
Log-shipping uploader for ``events.jsonl``.

Subscribes to the in-process ``EventLog`` and POSTs batches of events
to a configurable HTTP endpoint (S3-compatible bucket, Loki, generic
webhook collector, etc). Without this, the audit trail dies if the
local SSD dies.

Design constraints:
- Default OFF — only activates when ``SENTINEL_LOG_SHIP_URL`` is set.
- Batched: events are buffered and POSTed every ``flush_interval`` or
  when ``batch_max`` is reached. POSTing per-event would burn HTTP cost
  and saturate the receiver.
- Non-blocking: subscriber callback only enqueues. A background asyncio
  task drains the queue and does the network I/O.
- Best-effort retry: a failed POST puts the batch back at the head of
  the queue once; second failure drops it (and increments
  ``self._dropped_batches``) so a permanently broken receiver doesn't
  back-pressure the producer.
- The shipper does NOT re-read from disk — it ships only events emitted
  while it was running. For historical replay use a separate tool.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Optional

from monitoring.event_log import EventLog, get_event_log

logger = logging.getLogger(__name__)


class LogShipper:
    """Background HTTP uploader for EventLog events."""

    def __init__(
        self,
        url: Optional[str] = None,
        *,
        event_log: Optional[EventLog] = None,
        flush_interval_sec: float = 10.0,
        batch_max: int = 100,
        http_timeout_sec: float = 5.0,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        self._url = url or os.environ.get("SENTINEL_LOG_SHIP_URL", "").strip()
        self._event_log = event_log or get_event_log()
        self._flush_interval = flush_interval_sec
        self._batch_max = batch_max
        self._http_timeout = http_timeout_sec
        self._headers = {"Content-Type": "application/x-ndjson", **(headers or {})}

        self._queue: list[dict] = []
        self._lock: Optional[asyncio.Lock] = None
        self._task: Optional[asyncio.Task] = None
        self._unsubscribe = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Observability of the shipper itself — exposed to /metrics in
        # production so operators can see the receiver is up.
        self.shipped: int = 0
        self.failed_batches: int = 0
        self.dropped_batches: int = 0

    @property
    def enabled(self) -> bool:
        return bool(self._url)

    async def start(self) -> None:
        if not self.enabled:
            logger.info("LogShipper disabled (SENTINEL_LOG_SHIP_URL unset)")
            return
        if self._task is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._lock = asyncio.Lock()
        self._unsubscribe = self._event_log.subscribe(self._on_event)
        self._task = asyncio.create_task(self._flush_loop(), name="log-shipper")
        logger.info("LogShipper started (target=%s, batch_max=%d, flush_interval=%.1fs)",
                    self._url, self._batch_max, self._flush_interval)

    async def stop(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        # Drain remaining queue on shutdown — don't lose the last batch.
        if self._queue:
            await self._flush_once()

    def _on_event(self, record: dict) -> None:
        """Subscriber callback — runs under EventLog lock, MUST be quick.
        Just append to the in-memory queue; the background task does I/O."""
        if self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(self._enqueue, record)
        except RuntimeError:
            pass  # loop closed during shutdown

    def _enqueue(self, record: dict) -> None:
        self._queue.append(record)
        # Trigger immediate flush if we hit the batch ceiling — don't
        # wait for the timer.
        if len(self._queue) >= self._batch_max and self._task is not None:
            asyncio.create_task(self._flush_once())

    async def _flush_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._flush_interval)
                if self._queue:
                    await self._flush_once()
        except asyncio.CancelledError:
            return

    async def _flush_once(self) -> None:
        async with self._lock:  # type: ignore[union-attr]
            if not self._queue:
                return
            batch = self._queue[:self._batch_max]
            del self._queue[:len(batch)]

        body = "\n".join(json.dumps(r, ensure_ascii=False) for r in batch).encode("utf-8")
        ok = await self._post(body, attempt=1)
        if not ok:
            ok = await self._post(body, attempt=2)
        if ok:
            self.shipped += len(batch)
        else:
            self.failed_batches += 1
            self.dropped_batches += 1
            logger.warning("LogShipper dropped batch of %d (total dropped=%d)",
                           len(batch), self.dropped_batches)

    async def _post(self, body: bytes, attempt: int) -> bool:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=self._http_timeout) as client:
                resp = await client.post(self._url, content=body, headers=self._headers)
                if 200 <= resp.status_code < 300:
                    return True
                logger.warning("LogShipper POST attempt %d → HTTP %d", attempt, resp.status_code)
                return False
        except Exception as exc:
            logger.warning("LogShipper POST attempt %d failed: %s", attempt, exc)
            return False
