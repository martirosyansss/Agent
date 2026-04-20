"""
Telegram push-alerter for critical observability events.

Subscribes to the in-process ``EventLog`` and pushes a Telegram message
when a high-severity event (``critical`` / kill-switch / circuit-breaker
trip / multiple ``error``s in a window) lands. Without this, dashboard
badges are only seen by someone who happens to have the page open —
operators want a phone buzz when something is on fire.

Design constraints:
- ``EventLog.subscribe`` callbacks run under the EventLog lock and MUST
  be cheap. We therefore enqueue messages to an ``asyncio.Queue`` and
  let a background task drain them with the actual HTTP call.
- Per-(event_type, key) cooldown so a flapping condition doesn't
  produce 100 messages — the in-process dedup in ``event_log`` is at
  TTL=60s; here we layer a longer cooldown (default 5 min) tuned to
  human attention. Operators want to know it's still happening, not be
  notified every 60 seconds.
- Default cooldown / severity threshold are env-overridable so on-call
  can tune during incident review.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from monitoring.event_log import EventLog, EventType, get_event_log

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# Severity rank — events at or above the configured floor get pushed.
_SEVERITY_RANK = {"info": 0, "warning": 1, "error": 2, "critical": 3}


@dataclass
class AlertPolicy:
    """Per-event-type alert rules.

    ``severity_floor`` is the lowest severity that fires an alert. For
    ``component_error`` the default is ``"error"`` — warnings are too
    chatty. For ``guard_tripped`` and the kill-switch we always alert
    regardless of severity (these are inherently critical).

    ``cooldown_sec`` is the minimum interval between alerts keyed by
    ``(event_type, dedup_key)`` — defaults to 5 minutes so a persistent
    condition produces a periodic reminder, not a stream.
    """
    severity_floor: str = "error"
    cooldown_sec: float = field(
        default_factory=lambda: _env_float("SENTINEL_ALERT_COOLDOWN_SEC", 300.0)
    )
    always_alert: bool = False


_DEFAULT_POLICIES: dict[str, AlertPolicy] = {
    EventType.COMPONENT_ERROR: AlertPolicy(severity_floor="error"),
    EventType.GUARD_TRIPPED: AlertPolicy(always_alert=True),
}


class TelegramAlerter:
    """Background pump from ``EventLog`` → Telegram message.

    Lifecycle::

        alerter = TelegramAlerter(send_callback=tg_bot.send_message)
        await alerter.start()        # spawns drain task, subscribes
        ...
        await alerter.stop()          # unsubscribes, drains pending

    The send callback is injected so this module doesn't depend on
    ``telegram_bot`` directly (and so tests can pass a stub).
    """

    def __init__(
        self,
        send_callback: Callable[[str], "asyncio.Future[Any] | Any"],
        *,
        event_log: Optional[EventLog] = None,
        policies: Optional[dict[str, AlertPolicy]] = None,
        max_queue: int = 1000,
    ) -> None:
        self._send = send_callback
        self._event_log = event_log or get_event_log()
        self._policies = dict(_DEFAULT_POLICIES)
        if policies:
            self._policies.update(policies)
        self._queue: "asyncio.Queue[str]" = asyncio.Queue(maxsize=max_queue)
        self._cooldowns: dict[tuple[str, str], float] = {}
        self._task: Optional[asyncio.Task] = None
        self._unsubscribe: Optional[Callable[[], None]] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._dropped: int = 0  # queue-full drop counter (visible in /metrics later)

    # ──────────────────────────────────────────────
    # Public lifecycle
    # ──────────────────────────────────────────────

    async def start(self) -> None:
        if self._task is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._unsubscribe = self._event_log.subscribe(self._on_event)
        self._task = asyncio.create_task(self._drain(), name="tg-alerter-drain")
        logger.info("TelegramAlerter started (cooldown=%.0fs)",
                    self._policies[EventType.COMPONENT_ERROR].cooldown_sec)

    async def stop(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        if self._task is not None:
            await self._queue.put(_SHUTDOWN_SENTINEL)
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except asyncio.TimeoutError:
                self._task.cancel()
            self._task = None

    # ──────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────

    def _on_event(self, record: dict) -> None:
        """Subscriber callback — runs under EventLog lock, MUST be quick.
        We do filtering + cooldown check + enqueue (no I/O)."""
        event_type = str(record.get("type") or "")
        policy = self._policies.get(event_type)
        if policy is None:
            return
        severity = str(record.get("severity") or "info").lower()
        if not policy.always_alert:
            sev_rank = _SEVERITY_RANK.get(severity, 0)
            floor_rank = _SEVERITY_RANK.get(policy.severity_floor, 0)
            if sev_rank < floor_rank:
                return

        # Cooldown key — group by what an operator considers "same alert":
        # for component_error → component; for guard_tripped → guard+name.
        if event_type == EventType.COMPONENT_ERROR:
            dedup_key = str(record.get("component") or "unknown")
        elif event_type == EventType.GUARD_TRIPPED:
            dedup_key = f"{record.get('guard')}:{record.get('name', '')}"
        else:
            dedup_key = ""

        now = time.time()
        key = (event_type, dedup_key)
        last = self._cooldowns.get(key, 0.0)
        if (now - last) < policy.cooldown_sec:
            return
        self._cooldowns[key] = now

        # Bound the cooldown dict so a runaway cardinality doesn't leak.
        if len(self._cooldowns) > 1024:
            cutoff = now - 3600
            for k in [k for k, ts in self._cooldowns.items() if ts < cutoff]:
                self._cooldowns.pop(k, None)

        text = self._format(event_type, record, severity)
        # Cross-thread enqueue: subscribe() may be called from a non-loop
        # thread (e.g. risk-pipeline thread). Use call_soon_threadsafe to
        # hop into the loop before touching the queue.
        if self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(self._enqueue, text)
        except RuntimeError:
            pass  # loop closed during shutdown

    def _enqueue(self, text: str) -> None:
        try:
            self._queue.put_nowait(text)
        except asyncio.QueueFull:
            self._dropped += 1
            logger.warning("TelegramAlerter queue full — dropped (total dropped=%d)", self._dropped)

    @staticmethod
    def _format(event_type: str, record: dict, severity: str) -> str:
        emoji = {"critical": "🚨", "error": "⚠️", "warning": "⚡"}.get(severity, "ℹ️")
        if event_type == EventType.COMPONENT_ERROR:
            return (
                f"{emoji} *Component error* (`{severity}`)\n"
                f"`{record.get('component', '?')}`\n"
                f"{record.get('reason', '')}\n"
                f"_exc:_ `{record.get('exc_type', '-')}`"
            )
        if event_type == EventType.GUARD_TRIPPED:
            extra = ""
            if record.get("name"):
                extra = f" / `{record['name']}`"
            return (
                f"{emoji} *Guard tripped*: `{record.get('guard', '?')}`{extra}\n"
                f"{record.get('reason', '')}"
            )
        return f"{emoji} {event_type}: {record}"

    async def _drain(self) -> None:
        while True:
            text = await self._queue.get()
            if text is _SHUTDOWN_SENTINEL:
                return
            try:
                result = self._send(text)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                # Telegram failures must not crash the alerter. Log it
                # (operator will see the dashboard for the full picture).
                logger.warning("TelegramAlerter send failed: %s", exc)

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def dropped(self) -> int:
        return self._dropped


_SHUTDOWN_SENTINEL = object()  # type: ignore[assignment]
