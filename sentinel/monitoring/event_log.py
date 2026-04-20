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
- ``strategy_toggled``       — UI flag flipped a strategy on/off (transition)

The writer is async-safe via a single ``threading.Lock`` because pickle/
asyncio file ops can interleave on Windows. Throughput target: 1000 events/s
on a laptop SSD, well below typical Sentinel rate (~5-50/min).
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import json
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, TypeVar


# Schema version for events written to events.jsonl. Bump when the
# top-level shape changes in a way that breaks downstream consumers
# (dashboards, replayers). Field additions don't require a bump;
# renames or semantic changes do.
EVENT_SCHEMA_VERSION = 1


# ContextVar carrying the current trace_id for the in-flight request /
# signal evaluation. Set by callers at the entry point of a logical
# operation (e.g. when a Signal is created) and inherited by every
# event emitted on that task — so signal → risk → fill → close-out
# can be reconstructed by ``GROUP BY trace_id`` in the dashboard.
_trace_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "sentinel_trace_id", default=None,
)


def new_trace_id() -> str:
    """Generate a fresh trace id (URL-safe, 12 hex chars — short enough
    to read in logs, large enough for collision-free trade volume)."""
    return uuid.uuid4().hex[:12]


def get_trace_id() -> Optional[str]:
    """Return the trace_id active in the current context, or None."""
    return _trace_id_var.get()


def set_trace_id(trace_id: Optional[str]) -> contextvars.Token:
    """Install a trace_id for the current context. Returns the token so
    callers can ``reset(token)`` to restore the prior value (or use the
    ``trace_context`` helper which does this for you)."""
    return _trace_id_var.set(trace_id)


@contextmanager
def trace_context(trace_id: Optional[str] = None):
    """Context manager that sets a trace_id for the duration of a block.

    ::

        with trace_context() as tid:
            ...   # tid is auto-generated and visible to every emit() call

        with trace_context("abc123def456") as tid:
            ...   # explicit propagation across system boundaries
    """
    token = _trace_id_var.set(trace_id or new_trace_id())
    try:
        yield _trace_id_var.get()
    finally:
        _trace_id_var.reset(token)

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    """Canonical severity vocabulary for component_error / guard_tripped.

    Subclassing ``str`` so existing call sites that pass plain strings
    (``severity="warning"``) keep working — Severity.WARNING == "warning".
    This means we can validate without breaking back-compat.
    """
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


_KNOWN_SEVERITIES: frozenset[str] = frozenset(s.value for s in Severity)


def _normalise_severity(value: Any) -> str:
    """Coerce a severity argument to a canonical string.

    Unknown values fall back to ``"error"`` and emit a one-time
    loguru warning so a typo (``"warninng"``) is loudly visible in
    development without breaking production telemetry.
    """
    if isinstance(value, Severity):
        return value.value
    s = str(value).lower().strip()
    if s in _KNOWN_SEVERITIES:
        return s
    logger.warning(
        "Unknown severity %r — falling back to 'error'. Use Severity enum or one of %s.",
        value, sorted(_KNOWN_SEVERITIES),
    )
    return Severity.ERROR.value


class EventType:
    """Canonical event-type identifiers — use these instead of raw strings.

    Keeping the vocabulary in one place prevents typos like
    ``"siganl_rejected"`` silently producing orphan events that no dashboard
    query picks up. Extend when new kinds of observability data show up;
    *do not* rename existing values without a migration plan — the JSONL
    archive on disk uses these strings as keys.
    """
    # Strategy → risk → execution lifecycle
    SIGNAL_GENERATED = "signal_generated"
    # Full per-signal decision audit: risk pipeline verdict + every gate's
    # verdict + feature snapshot. Emitted from main.py after
    # evaluate_with_trace(). Dashboard's "Strategy Decisions" tab consumes this.
    SIGNAL_DECISION = "signal_decision"
    # Emitted by BaseStrategy right after a Signal is produced — carries the
    # strategy's own reasoning + feature snapshot, BEFORE risk-gate filtering.
    # Lets the dashboard show "strategy wanted X, but gate Y blocked it".
    STRATEGY_DECISION = "strategy_decision"
    SIGNAL_APPROVED = "signal_approved"
    SIGNAL_REJECTED = "signal_rejected"
    ORDER_FILLED = "order_filled"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    # Risk guards
    GUARD_TRIPPED = "guard_tripped"
    # Regime & context
    REGIME_CHANGE = "regime_change"
    NEWS_CRITICAL = "news_critical"
    # ML
    ML_PREDICTION = "ml_prediction"
    # Component health — failure & degradation visibility
    COMPONENT_ERROR = "component_error"
    COMPONENT_DEGRADED = "component_degraded"
    # Configuration transitions — emitted only on actual value change
    # (transition events, not per-tick) so the JSONL stays signal-rich.
    STRATEGY_TOGGLED = "strategy_toggled"


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
        multi_process_safe: bool = False,
    ) -> None:
        self._path = path
        self._max_bytes = max_bytes
        self._backup_count = backup_count
        self._lock = threading.Lock()
        self._recent: list[dict] = []
        self._buffer_max = in_memory_buffer
        # Subscribers fired after each emit. Used by alerters / shippers
        # that want push semantics (Telegram push on critical, HTTP ship
        # batches). Subscribers run under the EventLog lock — they MUST
        # be quick (enqueue to a queue, not block on I/O).
        self._subscribers: list[Callable[[dict], None]] = []
        # Multi-process safety: when bot + backtester (or replicas) share
        # the same logs/events.jsonl, two processes appending concurrently
        # can interleave bytes mid-line and corrupt JSONL. ``filelock``
        # provides a cross-platform exclusive lock keyed on a sidecar file
        # so writes serialise globally. Opt-in to keep single-process
        # callers (most tests) free of the small overhead.
        self._mp_lock = None
        if path is not None and multi_process_safe:
            from filelock import FileLock
            lock_path = path.with_suffix(path.suffix + ".lock")
            self._mp_lock = FileLock(str(lock_path), timeout=5.0)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────
    # Emission
    # ──────────────────────────────────────────────

    def subscribe(self, callback: Callable[[dict], None]) -> Callable[[], None]:
        """Register a callback fired (best-effort, sync) after each emit.

        Subscribers MUST be cheap — they run inside the EventLog lock.
        For anything slow (HTTP push, Telegram), the subscriber should
        enqueue to a background worker rather than block here. Exceptions
        from a subscriber are caught and logged so one bad subscriber
        can't break the rest of the pipeline.

        Returns an unsubscribe function so callers (especially tests)
        can clean up reliably.
        """
        with self._lock:
            self._subscribers.append(callback)

        def _unsubscribe() -> None:
            with self._lock:
                try:
                    self._subscribers.remove(callback)
                except ValueError:
                    pass

        return _unsubscribe

    def emit(self, event_type: str, **payload: Any) -> dict:
        """Write one event. Returns the assembled record (useful in tests).

        Every record carries:
          - ``ts``  — UTC milliseconds
          - ``type`` — one of EventType.*
          - ``schema_version`` — bumped only on breaking shape changes
          - ``trace_id`` — auto-derived from the ContextVar if a caller
            installed one via ``trace_context``; explicit ``trace_id`` in
            ``payload`` always wins so cross-system handoffs work.
        """
        # Auto-fill trace_id from context if caller didn't provide one.
        if "trace_id" not in payload:
            ctx_tid = _trace_id_var.get()
            if ctx_tid is not None:
                payload["trace_id"] = ctx_tid
        record = {
            "ts": int(time.time() * 1000),
            "type": event_type,
            "schema_version": EVENT_SCHEMA_VERSION,
            **payload,
        }
        with self._lock:
            self._recent.append(record)
            if len(self._recent) > self._buffer_max:
                self._recent = self._recent[-self._buffer_max:]
            if self._path is not None:
                try:
                    self._rotate_if_needed()
                    if self._mp_lock is not None:
                        # Cross-process serialisation. Acquired inside the
                        # in-process lock so a stuck flock can't deadlock
                        # the singleton — timeout=5s configured at __init__.
                        with self._mp_lock:
                            with self._path.open("a", encoding="utf-8") as f:
                                f.write(json.dumps(record, default=_json_default, ensure_ascii=False))
                                f.write("\n")
                    else:
                        with self._path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(record, default=_json_default, ensure_ascii=False))
                            f.write("\n")
                except Exception as exc:
                    logger.error("EventLog write failed: %s", exc)
            # Subscriber fan-out: snapshot the list before iterating so a
            # subscriber that calls subscribe()/unsubscribe inside its
            # callback doesn't mutate-during-iteration.
            subs = list(self._subscribers)
        for cb in subs:
            try:
                cb(record)
            except Exception as exc:
                logger.warning("EventLog subscriber raised: %s", exc)
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


# ──────────────────────────────────────────────
# Observability-policy helpers
#
# These exist so every "something blocked a signal" and every "a component
# failed" path uses one canonical shape — otherwise the dashboard has to
# guess field names per call site.
# ──────────────────────────────────────────────


def emit_rejection(
    gate: str,
    reason: str,
    *,
    symbol: Optional[str] = None,
    direction: Optional[str] = None,
    **context: Any,
) -> dict:
    """Record a risk-gate blocking a candidate signal.

    Canonical payload: ``{"gate": ..., "reason": ..., "symbol": ..., ...}``.
    Pick ``gate`` values that match the class name (``"liquidity_gate"``,
    ``"drawdown_breaker"``) so dashboards can ``GROUP BY gate`` without
    string munging.
    """
    payload: dict[str, Any] = {"gate": gate, "reason": reason}
    if symbol is not None:
        payload["symbol"] = symbol
    if direction is not None:
        payload["direction"] = direction
    payload.update(context)
    return get_event_log().emit(EventType.SIGNAL_REJECTED, **payload)


# TTL-based dedup state for component_error emissions.
# Keyed by (component, exc_type) so a recurring failure doesn't flood
# events.jsonl (e.g. ml_predictor failing on every tick). Within the window,
# repeat emissions are dropped BUT a suppressed counter is attached to the
# next emission after the window expires so the dashboard can still see that
# a storm happened — we don't lose signal, we just compress it.
#
# Pruning: stale entries (last_ts older than ``_PRUNE_FACTOR * TTL``) are
# evicted opportunistically — once the dict exceeds ``_PRUNE_TRIGGER_SIZE``
# entries we sweep on the next emit. Without this the dict would grow
# without bound under high-cardinality error churn (each unique
# ``(component, exc_type)`` pair lives forever) — a slow leak.
#
# All four knobs are env-overridable so operators can tune without code
# changes (e.g. raise dedup TTL during a known incident burst).


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r — using default %s", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r — using default %s", name, raw, default)
        return default


_COMPONENT_ERROR_DEDUP_TTL_SEC = _env_float(
    "SENTINEL_OBS_COMPONENT_ERROR_TTL_SEC", 60.0,
)
_PRUNE_FACTOR = _env_float("SENTINEL_OBS_PRUNE_FACTOR", 5.0)
_PRUNE_TRIGGER_SIZE = _env_int("SENTINEL_OBS_PRUNE_TRIGGER_SIZE", 256)
_component_error_state: dict[tuple[str, str], dict[str, Any]] = {}
_component_error_lock = threading.Lock()


def _prune_component_error_state_locked(now: float) -> None:
    """Drop dedup entries whose ``last_ts`` is older than ``_PRUNE_FACTOR * TTL``.
    Caller must hold ``_component_error_lock``.
    """
    cutoff = now - _COMPONENT_ERROR_DEDUP_TTL_SEC * _PRUNE_FACTOR
    stale = [k for k, v in _component_error_state.items() if v["last_ts"] < cutoff]
    for k in stale:
        del _component_error_state[k]


def emit_component_error(
    component: str,
    reason: str,
    *,
    exc: Optional[BaseException] = None,
    severity: str = "error",
    **context: Any,
) -> Optional[dict]:
    """Record a component failure (exception, degraded state, timeout, etc.).

    Use at any ``except Exception`` site in a critical module where a silent
    loguru trace would lose analytical value. ``severity`` follows the
    standard levels (``"warning"``, ``"error"``, ``"critical"``) so
    dashboards can threshold alerts.

    Dedup: identical ``(component, exc_type)`` failures within
    ``_COMPONENT_ERROR_DEDUP_TTL_SEC`` are suppressed — the next emission
    after the window carries a ``suppressed_count`` field so the dashboard
    sees the storm without 1000 raw lines. Returns ``None`` when the event
    was suppressed.
    """
    exc_type = type(exc).__name__ if exc is not None else "none"
    key = (component, exc_type)
    now = time.time()
    with _component_error_lock:
        # Opportunistic prune: only when the dict has actually grown enough
        # to matter, to keep the hot path branch-cheap.
        if len(_component_error_state) > _PRUNE_TRIGGER_SIZE:
            _prune_component_error_state_locked(now)
        state = _component_error_state.get(key)
        if state is not None and (now - state["last_ts"]) < _COMPONENT_ERROR_DEDUP_TTL_SEC:
            state["suppressed"] += 1
            return None
        suppressed = state["suppressed"] if state is not None else 0
        _component_error_state[key] = {"last_ts": now, "suppressed": 0}
    payload: dict[str, Any] = {
        "component": component,
        "reason": reason,
        "severity": _normalise_severity(severity),
    }
    if exc is not None:
        payload["exc_type"] = exc_type
    if suppressed > 0:
        payload["suppressed_count"] = suppressed
    payload.update(context)
    return get_event_log().emit(EventType.COMPONENT_ERROR, **payload)


def _reset_component_error_dedup() -> None:
    """Test helper — wipes the dedup state so each test starts fresh."""
    with _component_error_lock:
        _component_error_state.clear()
    with _guard_tripped_lock:
        _guard_tripped_state.clear()


# Dedup state for ``guard_tripped`` events. Same shape as the
# component_error dedup but a separate dict so guard storms (e.g. a
# flapping CB) and component-error storms don't compete for the same
# trigger threshold. TTL is shorter (15s) because guard trips are
# higher-signal and operators want to see chatter.
_GUARD_TRIPPED_DEDUP_TTL_SEC = _env_float(
    "SENTINEL_OBS_GUARD_TRIPPED_TTL_SEC", 15.0,
)
_guard_tripped_state: dict[tuple[str, str], dict[str, Any]] = {}
_guard_tripped_lock = threading.Lock()


def emit_guard_tripped(
    guard: str,
    *,
    name: str = "",
    reason: str = "",
    severity: str = "warning",
    **context: Any,
) -> Optional[dict]:
    """Record a guard activation (CB trip, DD breach, kill-switch fire).

    Canonical shape ``{"guard": ..., "name": ..., "reason": ..., ...}`` so
    the dashboard can ``GROUP BY guard`` cleanly. Dedup follows the same
    pattern as ``emit_component_error`` — repeated trips of the same
    ``(guard, name)`` within the TTL are suppressed and counted, with the
    suppressed count attached to the next emission.
    """
    key = (guard, name)
    now = time.time()
    with _guard_tripped_lock:
        if len(_guard_tripped_state) > _PRUNE_TRIGGER_SIZE:
            cutoff = now - _GUARD_TRIPPED_DEDUP_TTL_SEC * _PRUNE_FACTOR
            stale = [k for k, v in _guard_tripped_state.items() if v["last_ts"] < cutoff]
            for k in stale:
                del _guard_tripped_state[k]
        state = _guard_tripped_state.get(key)
        if state is not None and (now - state["last_ts"]) < _GUARD_TRIPPED_DEDUP_TTL_SEC:
            state["suppressed"] += 1
            return None
        suppressed = state["suppressed"] if state is not None else 0
        _guard_tripped_state[key] = {"last_ts": now, "suppressed": 0}
    payload: dict[str, Any] = {
        "guard": guard,
        "severity": _normalise_severity(severity),
    }
    if name:
        payload["name"] = name
    if reason:
        payload["reason"] = reason
    if suppressed > 0:
        payload["suppressed_count"] = suppressed
    payload.update(context)
    return get_event_log().emit(EventType.GUARD_TRIPPED, **payload)


def emit_position_opened(
    symbol: str,
    *,
    entry_price: float,
    quantity: float,
    strategy: str = "",
    signal_id: str = "",
    stop_loss: float = 0.0,
    take_profit: float = 0.0,
    **context: Any,
) -> dict:
    """Record a position becoming active in memory + DB.

    Pairs with ``emit_position_closed`` — together they let the dashboard
    reconstruct the full open→close trajectory from events.jsonl without
    joining against the SQLite snapshot.
    """
    payload: dict[str, Any] = {
        "symbol": symbol,
        "entry_price": entry_price,
        "quantity": quantity,
    }
    if strategy:
        payload["strategy"] = strategy
    if signal_id:
        payload["signal_id"] = signal_id
    if stop_loss:
        payload["stop_loss"] = stop_loss
    if take_profit:
        payload["take_profit"] = take_profit
    payload.update(context)
    return get_event_log().emit(EventType.POSITION_OPENED, **payload)


def emit_position_closed(
    symbol: str,
    *,
    entry_price: float,
    exit_price: float,
    quantity: float,
    realized_pnl: float,
    strategy: str = "",
    signal_id: str = "",
    hold_ms: Optional[int] = None,
    exit_reason: str = "",
    **context: Any,
) -> dict:
    """Record a position closing with the realised PnL.

    ``exit_reason`` carries the human label used by the UI
    (``"sl_hit"``, ``"tp1"``, ``"manual_close"``, ``"regime_flip"``, …).
    """
    payload: dict[str, Any] = {
        "symbol": symbol,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "quantity": quantity,
        "realized_pnl": realized_pnl,
    }
    if strategy:
        payload["strategy"] = strategy
    if signal_id:
        payload["signal_id"] = signal_id
    if hold_ms is not None:
        payload["hold_ms"] = hold_ms
    if exit_reason:
        payload["exit_reason"] = exit_reason
    payload.update(context)
    return get_event_log().emit(EventType.POSITION_CLOSED, **payload)


F = TypeVar("F", bound=Callable[..., Any])


def traced_component(
    component: str,
    *,
    severity: str = "error",
    reraise: bool = True,
) -> Callable[[F], F]:
    """Decorator: emit ``component_error`` when the wrapped function raises.

    Works on both sync and async functions — we detect coroutine functions
    via ``asyncio.iscoroutinefunction`` and return a matching wrapper so the
    caller's ``await`` behaviour is preserved.

    The original exception still propagates by default (``reraise=True``) —
    this only *adds* structured visibility, it does not swallow errors.
    Set ``reraise=False`` for best-effort paths (background refreshers, opt-in
    telemetry) where failures should be recorded but not break the caller.
    """
    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    emit_component_error(
                        component,
                        str(exc) or type(exc).__name__,
                        exc=exc,
                        severity=severity,
                        function=func.__qualname__,
                    )
                    if reraise:
                        raise
                    return None
            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                emit_component_error(
                    component,
                    str(exc) or type(exc).__name__,
                    exc=exc,
                    severity=severity,
                    function=func.__qualname__,
                )
                if reraise:
                    raise
                return None
        return sync_wrapper  # type: ignore[return-value]

    return decorator
