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
    # Record is appended to the recent-buffer as well.
    recent = _fresh_event_log.recent_events(type_filter=EventType.SIGNAL_REJECTED)
    assert len(recent) == 1
    assert recent[0]["gate"] == "liquidity_gate"


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
