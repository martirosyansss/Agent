"""
Tests for ``monitoring.telegram_alerter`` — push-alert pump.

These exercise the full subscribe → format → enqueue → drain path with
a stub send callback so we don't need a real Telegram bot. The dedup
logic gets its own coverage because operator inbox spam is the primary
failure mode of an alerter.
"""
from __future__ import annotations

import asyncio

import pytest

from monitoring.event_log import (
    EventLog,
    EventType,
    _reset_component_error_dedup,
    emit_component_error,
    emit_guard_tripped,
    set_event_log,
)
from monitoring.telegram_alerter import AlertPolicy, TelegramAlerter


@pytest.fixture(autouse=True)
def _fresh_log():
    _reset_component_error_dedup()
    log = EventLog(path=None)
    set_event_log(log)
    yield log
    _reset_component_error_dedup()
    set_event_log(EventLog(path=None))


class _StubSender:
    def __init__(self):
        self.sent: list[str] = []

    async def send(self, text: str) -> None:
        self.sent.append(text)


@pytest.mark.asyncio
async def test_alerter_pushes_critical_component_error(_fresh_log):
    sender = _StubSender()
    alerter = TelegramAlerter(send_callback=sender.send, event_log=_fresh_log)
    await alerter.start()
    try:
        emit_component_error("ml_predictor", "boom", exc=RuntimeError("x"), severity="critical")
        # Give the drain task one event-loop tick to process the queue.
        await asyncio.sleep(0.05)
    finally:
        await alerter.stop()
    assert len(sender.sent) == 1
    assert "Component error" in sender.sent[0]
    assert "ml_predictor" in sender.sent[0]


@pytest.mark.asyncio
async def test_alerter_skips_below_severity_floor(_fresh_log):
    """warning-level component errors must NOT page the operator —
    only error/critical warrant a phone buzz."""
    sender = _StubSender()
    alerter = TelegramAlerter(send_callback=sender.send, event_log=_fresh_log)
    await alerter.start()
    try:
        emit_component_error("c", "minor", exc=ValueError(), severity="warning")
        await asyncio.sleep(0.05)
    finally:
        await alerter.stop()
    assert sender.sent == []


@pytest.mark.asyncio
async def test_alerter_dedup_within_cooldown(_fresh_log):
    """Two errors from the same component in the cooldown window
    must produce ONE Telegram message, not two."""
    sender = _StubSender()
    alerter = TelegramAlerter(
        send_callback=sender.send,
        event_log=_fresh_log,
        # short cooldown for the test, but >> the EventLog dedup TTL
        policies={
            EventType.COMPONENT_ERROR: AlertPolicy(severity_floor="error", cooldown_sec=60.0),
        },
    )
    await alerter.start()
    try:
        emit_component_error("c", "fail1", exc=RuntimeError("x"), severity="error")
        # Different exc_type bypasses event_log dedup but should hit alerter cooldown
        emit_component_error("c", "fail2", exc=ValueError("x"), severity="error")
        await asyncio.sleep(0.05)
    finally:
        await alerter.stop()
    assert len(sender.sent) == 1


@pytest.mark.asyncio
async def test_alerter_pushes_guard_tripped_regardless_of_severity(_fresh_log):
    """Guard-tripped events are inherently critical — alert always."""
    sender = _StubSender()
    alerter = TelegramAlerter(send_callback=sender.send, event_log=_fresh_log)
    await alerter.start()
    try:
        emit_guard_tripped(guard="kill_switch", reason="manual stop", severity="critical")
        await asyncio.sleep(0.05)
    finally:
        await alerter.stop()
    assert len(sender.sent) == 1
    assert "Guard tripped" in sender.sent[0]
    assert "kill_switch" in sender.sent[0]


@pytest.mark.asyncio
async def test_alerter_send_failure_does_not_crash_drain(_fresh_log):
    """If Telegram is unreachable, the alerter logs and continues.
    A second emit must still get processed."""
    call_count = {"n": 0}

    async def flaky(text: str) -> None:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("telegram down")

    alerter = TelegramAlerter(send_callback=flaky, event_log=_fresh_log)
    await alerter.start()
    try:
        emit_component_error("a", "first", exc=RuntimeError(), severity="critical")
        emit_component_error("b", "second", exc=ValueError(), severity="critical")
        await asyncio.sleep(0.05)
    finally:
        await alerter.stop()
    assert call_count["n"] == 2  # both attempts made
