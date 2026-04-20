"""
Tests for the observability-policy helpers in ``monitoring.event_log``.

Policy (see memory/feedback_observability_policy.md): every component
failure and every risk-gate rejection must show up in ``events.jsonl`` as
a structured event, not only as a loguru text line. These tests lock in
the canonical shape of those events so dashboards and analytics can rely
on it.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from monitoring.event_log import (
    EventLog,
    EventType,
    _reset_component_error_dedup,
    emit_component_error,
    emit_rejection,
    get_event_log,
    set_event_log,
    traced_component,
)


@pytest.fixture(autouse=True)
def _fresh_event_log():
    """Install a clean in-memory EventLog for each test."""
    _reset_component_error_dedup()
    log = EventLog(path=None)
    set_event_log(log)
    yield log
    # Leave a fresh default for whatever runs next.
    _reset_component_error_dedup()
    set_event_log(EventLog(path=None))


# ──────────────────────────────────────────────
# emit_rejection
# ──────────────────────────────────────────────


def test_emit_rejection_canonical_shape(_fresh_event_log: EventLog) -> None:
    record = emit_rejection(
        gate="liquidity_gate",
        reason="volume_ratio 0.20 < 0.40",
        symbol="BTCUSDT",
        direction="BUY",
        volume_ratio=0.20,
    )
    assert record["type"] == EventType.SIGNAL_REJECTED
    assert record["gate"] == "liquidity_gate"
    assert record["symbol"] == "BTCUSDT"
    assert record["direction"] == "BUY"
    assert record["volume_ratio"] == 0.20
    # Every event carries the schema version — downstream consumers can
    # branch on this when the format changes in a non-additive way.
    assert record["schema_version"] >= 1
    # Record is appended to the recent-buffer as well.
    recent = _fresh_event_log.recent_events(type_filter=EventType.SIGNAL_REJECTED)
    assert len(recent) == 1
    assert recent[0]["gate"] == "liquidity_gate"


def test_trace_id_propagates_via_context_var(_fresh_event_log: EventLog) -> None:
    """A trace_context block stamps every event emitted inside it with
    the same trace_id — so signal → risk → fill → close-out can be
    reconstructed by ``GROUP BY trace_id`` in the dashboard."""
    from monitoring.event_log import get_trace_id, trace_context

    assert get_trace_id() is None  # outside context: no implicit id
    with trace_context() as tid:
        assert tid == get_trace_id()
        emit_rejection(gate="g", reason="r")
        emit_component_error("c", "r", exc=ValueError())
    assert get_trace_id() is None  # context restored on exit

    events = _fresh_event_log.recent_events()
    # Both events emitted inside the context carry the same trace_id.
    assert {e.get("trace_id") for e in events} == {tid}


def test_explicit_trace_id_overrides_context(_fresh_event_log: EventLog) -> None:
    from monitoring.event_log import trace_context
    with trace_context("ctx-id"):
        rec = emit_rejection(gate="g", reason="r", trace_id="explicit-id")
    assert rec["trace_id"] == "explicit-id"


def test_subscriber_receives_emits(_fresh_event_log: EventLog) -> None:
    """The subscribe() API delivers each emitted record to registered
    callbacks. Backbone for Telegram alerter, log shipper, /metrics counters."""
    received: list[dict] = []
    unsubscribe = _fresh_event_log.subscribe(lambda r: received.append(r))
    emit_rejection(gate="g1", reason="r1")
    emit_component_error("c1", "r2", exc=ValueError())
    assert len(received) == 2
    types = {r["type"] for r in received}
    assert EventType.SIGNAL_REJECTED in types
    assert EventType.COMPONENT_ERROR in types

    # Unsubscribe stops further deliveries.
    unsubscribe()
    emit_rejection(gate="g2", reason="r3")
    assert len(received) == 2  # unchanged


def test_multi_process_safe_writes_serialise(tmp_path) -> None:
    """With ``multi_process_safe=True`` two EventLog instances writing to
    the same file produce well-formed JSONL — no interleaved bytes mid-line.
    Simulates two processes (we use threads here for portability — the
    file-lock is the same code path)."""
    import threading as _threading
    log_path = tmp_path / "events.jsonl"
    log_a = EventLog(path=log_path, multi_process_safe=True)
    log_b = EventLog(path=log_path, multi_process_safe=True)

    def writer(log, tag, n):
        for i in range(n):
            log.emit("test_event", tag=tag, idx=i, payload="x" * 200)

    t1 = _threading.Thread(target=writer, args=(log_a, "A", 50))
    t2 = _threading.Thread(target=writer, args=(log_b, "B", 50))
    t1.start(); t2.start()
    t1.join(); t2.join()

    # Every line in events.jsonl must be valid JSON. If file-locking
    # failed, we'd see truncated/interleaved lines that fail to parse.
    import json as _json
    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 100
    for line in lines:
        rec = _json.loads(line)
        assert rec["type"] == "test_event"


def test_subscriber_exception_does_not_break_emit(_fresh_event_log: EventLog) -> None:
    """A misbehaving subscriber must not block the EventLog. The
    emit() call still returns the record and other subscribers fire."""
    good_calls: list[dict] = []

    def bad(_record: dict) -> None:
        raise RuntimeError("subscriber broken")

    def good(record: dict) -> None:
        good_calls.append(record)

    _fresh_event_log.subscribe(bad)
    _fresh_event_log.subscribe(good)
    rec = emit_rejection(gate="g", reason="r")
    assert rec["type"] == EventType.SIGNAL_REJECTED
    assert len(good_calls) == 1


def test_emit_rejection_omits_none_optional_fields(_fresh_event_log: EventLog) -> None:
    record = emit_rejection(gate="regime_gate", reason="regime=volatile")
    assert "symbol" not in record
    assert "direction" not in record


# ──────────────────────────────────────────────
# emit_component_error
# ──────────────────────────────────────────────


def test_emit_component_error_shape(_fresh_event_log: EventLog) -> None:
    exc = RuntimeError("calibrator dropped to plateau")
    record = emit_component_error(
        "ml_predictor",
        "plateau detected",
        exc=exc,
        severity="warning",
        n_samples=500,
    )
    assert record is not None
    assert record["type"] == EventType.COMPONENT_ERROR
    assert record["component"] == "ml_predictor"
    assert record["severity"] == "warning"
    assert record["exc_type"] == "RuntimeError"
    assert record["n_samples"] == 500


def test_emit_component_error_dedup_within_ttl(_fresh_event_log: EventLog) -> None:
    """Repeated identical failures within the TTL window are suppressed.
    Without this, a tight-loop failure would flood events.jsonl with 1000s
    of identical lines and drown out every other signal."""
    exc = ValueError("same error")
    first = emit_component_error("flaky_component", "fail 1", exc=exc)
    second = emit_component_error("flaky_component", "fail 2", exc=exc)
    third = emit_component_error("flaky_component", "fail 3", exc=exc)
    assert first is not None
    assert second is None
    assert third is None
    events = _fresh_event_log.recent_events(type_filter=EventType.COMPONENT_ERROR)
    assert len(events) == 1


def test_emit_component_error_different_exc_types_not_deduped(
    _fresh_event_log: EventLog,
) -> None:
    """Different exception types are distinct keys — both should emit."""
    emit_component_error("x", "r", exc=ValueError("a"))
    emit_component_error("x", "r", exc=TypeError("b"))
    events = _fresh_event_log.recent_events(type_filter=EventType.COMPONENT_ERROR)
    assert len(events) == 2


def test_severity_enum_accepted(_fresh_event_log: EventLog) -> None:
    """Severity enum members and equivalent strings produce identical output."""
    from monitoring.event_log import Severity
    record = emit_component_error("x", "r", exc=ValueError(), severity=Severity.CRITICAL)
    assert record is not None
    assert record["severity"] == "critical"


def test_unknown_severity_falls_back_to_error(_fresh_event_log: EventLog) -> None:
    """Typo-ed severity strings don't silently create new buckets — they
    coerce to ``error`` so the dashboard's severity histogram stays clean."""
    record = emit_component_error("y", "r", exc=ValueError(), severity="warninng")
    assert record is not None
    assert record["severity"] == "error"


def test_emit_guard_tripped_dedup_within_ttl(_fresh_event_log: EventLog) -> None:
    """A flapping guard (e.g. CB tripping repeatedly) must not flood
    events.jsonl. Identical (guard, name) within 15s collapses with a
    suppressed counter on the next emission."""
    from monitoring.event_log import emit_guard_tripped

    first = emit_guard_tripped(guard="circuit_breaker", name="CB-1", reason="spike")
    second = emit_guard_tripped(guard="circuit_breaker", name="CB-1", reason="spike again")
    assert first is not None
    assert second is None
    events = _fresh_event_log.recent_events(type_filter=EventType.GUARD_TRIPPED)
    assert len(events) == 1
    assert events[0]["guard"] == "circuit_breaker"
    assert events[0]["name"] == "CB-1"


def test_emit_guard_tripped_distinct_names_not_deduped(_fresh_event_log: EventLog) -> None:
    """Different ``name`` values are distinct keys — both should emit
    (CB-1 trip and CB-2 trip are independent events)."""
    from monitoring.event_log import emit_guard_tripped
    emit_guard_tripped(guard="circuit_breaker", name="CB-1")
    emit_guard_tripped(guard="circuit_breaker", name="CB-2")
    events = _fresh_event_log.recent_events(type_filter=EventType.GUARD_TRIPPED)
    assert len(events) == 2


def test_emit_component_error_state_pruned_on_growth(_fresh_event_log: EventLog) -> None:
    """Stale dedup entries are evicted once the table exceeds the trigger size.
    Without pruning, a long-lived bot accumulating diverse (component, exc_type)
    pairs would leak memory in the singleton dict.
    """
    from monitoring import event_log as el

    el._reset_component_error_dedup()
    # Synthesize many stale entries (last_ts well outside the prune cutoff).
    stale_ts = el.time.time() - el._COMPONENT_ERROR_DEDUP_TTL_SEC * el._PRUNE_FACTOR - 100
    with el._component_error_lock:
        for i in range(el._PRUNE_TRIGGER_SIZE + 50):
            el._component_error_state[(f"comp_{i}", "Stale")] = {
                "last_ts": stale_ts, "suppressed": 0,
            }
    pre = len(el._component_error_state)
    assert pre > el._PRUNE_TRIGGER_SIZE

    # Any emit triggers the opportunistic prune sweep.
    el.emit_component_error("trigger", "wakeup", exc=RuntimeError("x"))
    post = len(el._component_error_state)
    # All synthetic stale entries should be gone; only the one we just emitted remains.
    assert post == 1, f"expected 1 entry after prune, got {post}"


# ──────────────────────────────────────────────
# @traced_component — sync
# ──────────────────────────────────────────────


def test_traced_component_emits_on_sync_exception(_fresh_event_log: EventLog) -> None:
    @traced_component("fake_component")
    def boom() -> None:
        raise ValueError("kaboom")

    with pytest.raises(ValueError):
        boom()

    events = _fresh_event_log.recent_events(type_filter=EventType.COMPONENT_ERROR)
    assert len(events) == 1
    ev = events[0]
    assert ev["component"] == "fake_component"
    assert ev["exc_type"] == "ValueError"
    assert ev["function"].endswith("boom")


def test_traced_component_passes_through_success(_fresh_event_log: EventLog) -> None:
    @traced_component("fake_component")
    def fine(x: int) -> int:
        return x * 2

    assert fine(21) == 42
    assert _fresh_event_log.recent_events(type_filter=EventType.COMPONENT_ERROR) == []


def test_traced_component_can_swallow_with_reraise_false(_fresh_event_log: EventLog) -> None:
    @traced_component("fake_component", reraise=False)
    def boom() -> str:
        raise RuntimeError("soft-fail")

    assert boom() is None
    events = _fresh_event_log.recent_events(type_filter=EventType.COMPONENT_ERROR)
    assert len(events) == 1
    assert events[0]["exc_type"] == "RuntimeError"


# ──────────────────────────────────────────────
# @traced_component — async
# ──────────────────────────────────────────────


def test_traced_component_emits_on_async_exception(_fresh_event_log: EventLog) -> None:
    @traced_component("async_fake")
    async def boom() -> None:
        raise TimeoutError("no response")

    with pytest.raises(TimeoutError):
        asyncio.run(boom())

    events = _fresh_event_log.recent_events(type_filter=EventType.COMPONENT_ERROR)
    assert len(events) == 1
    assert events[0]["component"] == "async_fake"
    assert events[0]["exc_type"] == "TimeoutError"


def test_traced_component_async_passes_through_success(_fresh_event_log: EventLog) -> None:
    @traced_component("async_fake")
    async def ok() -> str:
        await asyncio.sleep(0)
        return "done"

    assert asyncio.run(ok()) == "done"
    assert _fresh_event_log.recent_events(type_filter=EventType.COMPONENT_ERROR) == []


# ──────────────────────────────────────────────
# Singleton wiring sanity
# ──────────────────────────────────────────────


def test_get_event_log_returns_installed_instance(_fresh_event_log: EventLog) -> None:
    assert get_event_log() is _fresh_event_log


# ──────────────────────────────────────────────
# Stage B — RiskSentinel canonical emits
#
# These lock in that every decision through ``evaluate_with_trace`` produces
# a canonical ``signal_rejected`` or ``signal_approved`` event regardless of
# which gate triggered the block. The dashboard can then group by ``gate``
# without having to unpack the full decision-audit blob.
# ──────────────────────────────────────────────


def _make_signal(direction: str = "BUY", *, stop_loss_price: float = 59000.0) -> "Signal":  # type: ignore[name-defined]
    """Construct a minimal Signal for risk-pipeline tests."""
    from core.models import Direction, Signal
    return Signal(
        timestamp=0,
        symbol="BTCUSDT",
        direction=Direction.BUY if direction == "BUY" else Direction.SELL,
        confidence=0.85,
        strategy_name="ema_crossover_rsi",
        reason="test",
        suggested_quantity=0.001,
        stop_loss_price=stop_loss_price,
        take_profit_price=62000.0,
        signal_id="sig_test_1",
    )


def _make_sentinel():
    from core.events import EventBus
    from risk.sentinel import RiskLimits, RiskSentinel
    from risk.state_machine import RiskStateMachine
    sm = RiskStateMachine(event_bus=EventBus())
    return RiskSentinel(limits=RiskLimits(), state_machine=sm)


def test_approval_emits_canonical_signal_approved(_fresh_event_log: EventLog) -> None:
    sentinel = _make_sentinel()
    signal = _make_signal()
    result, trace = sentinel.evaluate_with_trace(
        signal=signal,
        daily_pnl=0.0,
        open_positions_count=0,
        total_exposure_pct=0.0,
        balance=500.0,
        current_market_price=60000.0,
    )
    assert result.approved, f"expected approval, got: {result.reason}"
    events = _fresh_event_log.recent_events(type_filter=EventType.SIGNAL_APPROVED)
    assert len(events) == 1
    assert events[0]["symbol"] == "BTCUSDT"
    assert events[0]["strategy"] == "ema_crossover_rsi"


def test_rejection_emits_canonical_signal_rejected_with_gate(_fresh_event_log: EventLog) -> None:
    # Missing stop-loss triggers the legacy [6] check, producing a rejection
    # surfaced via the "legacy_checks" gate in the decision trace.
    sentinel = _make_sentinel()
    signal = _make_signal(stop_loss_price=0.0)
    result, trace = sentinel.evaluate_with_trace(
        signal=signal,
        daily_pnl=0.0,
        open_positions_count=0,
        total_exposure_pct=0.0,
        balance=500.0,
        current_market_price=60000.0,
    )
    assert not result.approved
    events = _fresh_event_log.recent_events(type_filter=EventType.SIGNAL_REJECTED)
    assert len(events) == 1
    ev = events[0]
    assert ev["symbol"] == "BTCUSDT"
    assert ev["direction"] == "BUY"
    assert ev["gate"]  # some gate id recorded
    assert "stop" in ev["reason"].lower()


def test_gate_exception_emits_component_error(_fresh_event_log: EventLog) -> None:
    """A gate that raises must surface a component_error event — otherwise
    the pipeline fails open silently and the dashboard never sees it."""
    from risk.decision_tracer import DecisionTrace, GateTimer

    trace = DecisionTrace(symbol="BTCUSDT", strategy="ema_crossover_rsi")
    with GateTimer(trace, "broken_gate") as _t:
        raise RuntimeError("synthetic gate failure")

    events = _fresh_event_log.recent_events(type_filter=EventType.COMPONENT_ERROR)
    assert len(events) == 1
    ev = events[0]
    assert ev["component"] == "risk_gate.broken_gate"
    assert ev["exc_type"] == "RuntimeError"
    assert "synthetic gate failure" in ev["reason"]
    assert ev["symbol"] == "BTCUSDT"


# ──────────────────────────────────────────────
# CI guard — observability-policy regression test
#
# Critical modules MUST keep their structured-emit coverage above a
# minimum baseline. If someone deletes an ``emit_component_error`` /
# ``emit_rejection`` / ``GUARD_TRIPPED`` call, or adds new
# ``logger.error/critical`` paths without a matching emit, this test
# fails — forcing the regression to be addressed in the PR rather than
# silently drifting the observability surface.
# ──────────────────────────────────────────────


# (relative_path_under_sentinel, minimum_required_emit_count)
# Counts CALL sites only — imports don't count. When you ADD coverage,
# raise the threshold; when you legitimately remove a site, lower it AND
# explain in the PR. The whole point is that drops are visible.
_CRITICAL_MODULES_MIN_EMITS: list[tuple[str, int]] = [
    # risk/sentinel.py emits both branches via the canonical-decision helper:
    # the approve branch references EventType.SIGNAL_APPROVED, the reject
    # branch goes through emit_rejection(...). Both patterns are tracked so
    # silent removal of either branch fails this test.
    ("risk/sentinel.py", 2),                       # SIGNAL_APPROVED + emit_rejection
    ("risk/decision_tracer.py", 1),                # GateTimer.__exit__ on gate exception
    ("risk/circuit_breakers.py", 1),               # CircuitBreakerState.trip → GUARD_TRIPPED
    ("risk/drawdown_breaker.py", 1),               # update() trip path
    ("risk/kill_switch.py", 3),                    # activate header + close failure + final
    ("position/manager.py", 4),                    # 3 open + 1 close validation sites
    ("execution/live_executor.py", 2),             # execute_order + emergency_sell
    ("collector/news_collector.py", 1),            # @traced_component on _fetch_all
    # Round-10 refactor: ``predict()``'s silent-fallback emit moved with
    # the rest of the prediction path into ``analyzer/ml/prediction/
    # predictor.py``. ``ml_predictor.py`` is now a façade — its emit
    # coverage is accounted for in the new module.
    ("analyzer/ml_predictor.py", 0),               # façade — emits live in ml/ subpackage
    # predict_from_features emits one component_error on silent fallback.
    # The regex also counts the ``def emit_component_error(*args, **kwargs)``
    # optional-import fallback stub as a second match — hence baseline 2
    # rather than 1. A future cleanup that replaces the stub with a
    # conditional import would drop this back to 1.
    ("analyzer/ml/prediction/predictor.py", 2),    # real emit + optional-import stub
    ("analyzer/ml_ensemble.py", 3),                # 3 silent-fallback sites
    ("analyzer/ml_stacking.py", 2),                # fit + predict
    ("analyzer/ml_regime_router.py", 2),           # train + predict
    ("analyzer/ml_walk_forward.py", 1),            # fold-trainer failure
]

# Patterns intentionally require a trailing ``(`` so that a ``from … import``
# line never inflates the count. ``EventType.{GUARD_TRIPPED, COMPONENT_ERROR,
# SIGNAL_APPROVED, SIGNAL_REJECTED}`` count as call-context references —
# they only appear inside ``emit(...)`` arguments in production code.
# The decorator pattern is anchored to start-of-line whitespace
# (``re.MULTILINE``) so a docstring mention of ``@traced_component(...)``
# does NOT inflate the count.
_EMIT_CALL_PATTERNS = (
    r"\bemit_component_error\s*\(",
    r"\bemit_rejection\s*\(",
    r"\bemit_guard_tripped\s*\(",
    r"EventType\.GUARD_TRIPPED",
    r"EventType\.COMPONENT_ERROR",
    r"EventType\.SIGNAL_APPROVED",
    r"EventType\.SIGNAL_REJECTED",
    r"^\s*@traced_component\s*\(",
)


def _count_emits(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    return sum(
        len(re.findall(p, text, re.MULTILINE)) for p in _EMIT_CALL_PATTERNS
    )


def test_observability_emit_baseline_holds() -> None:
    """Critical modules must keep structured-emit coverage at or above the
    documented baseline. Failing this test means observability regressed —
    fix by restoring the emit, or (if the deletion was intentional) lower
    the baseline in this file with a justification in the PR description.
    """
    sentinel_root = Path(__file__).resolve().parent.parent
    failures: list[str] = []
    for rel_path, expected_min in _CRITICAL_MODULES_MIN_EMITS:
        full = sentinel_root / rel_path
        assert full.exists(), f"missing critical module: {rel_path}"
        actual = _count_emits(full)
        if actual < expected_min:
            failures.append(
                f"  {rel_path}: {actual} emit-call(s), expected ≥ {expected_min}"
            )
    assert not failures, (
        "Observability-policy regression — emit coverage dropped:\n"
        + "\n".join(failures)
        + "\n\nFix: restore the missing emit (preferred) or, if intentional,"
        " lower the baseline in test_observability_policy.py with rationale."
    )
