"""
Tests for ``monitoring.log_shipper`` — HTTP-batch uploader.

Uses a stub HTTPX transport so we don't need a real server. Validates:
  - shipper does nothing when URL is unset (default-off behaviour)
  - batches assemble + POST in NDJSON format
  - failed POST retries once then drops the batch
  - shutdown drains the pending queue (don't lose last events)
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

from monitoring.event_log import (
    EventLog,
    _reset_component_error_dedup,
    emit_component_error,
    set_event_log,
)
from monitoring.log_shipper import LogShipper


@pytest.fixture(autouse=True)
def _fresh_log():
    _reset_component_error_dedup()
    log = EventLog(path=None)
    set_event_log(log)
    yield log
    _reset_component_error_dedup()
    set_event_log(EventLog(path=None))


@pytest.mark.asyncio
async def test_shipper_disabled_when_url_unset(monkeypatch, _fresh_log):
    monkeypatch.delenv("SENTINEL_LOG_SHIP_URL", raising=False)
    shipper = LogShipper(event_log=_fresh_log)
    assert shipper.enabled is False
    await shipper.start()  # no-op
    await shipper.stop()


@pytest.mark.asyncio
async def test_shipper_posts_batch_in_ndjson(_fresh_log, monkeypatch):
    posted_bodies: list[bytes] = []

    async def stub_post(self, url, *, content, headers, **kw):
        posted_bodies.append(content)
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", stub_post)
    shipper = LogShipper(
        url="https://collector.example/ingest",
        event_log=_fresh_log,
        flush_interval_sec=0.05,
        batch_max=100,
    )
    await shipper.start()
    try:
        emit_component_error("c1", "fail", exc=RuntimeError())
        emit_component_error("c2", "fail", exc=ValueError())
        await asyncio.sleep(0.15)
    finally:
        await shipper.stop()

    assert len(posted_bodies) >= 1
    # NDJSON: each event on its own line, valid JSON per line.
    import json as _json
    lines = posted_bodies[0].decode("utf-8").strip().split("\n")
    assert len(lines) == 2
    parsed = [_json.loads(ln) for ln in lines]
    assert {p["component"] for p in parsed} == {"c1", "c2"}
    assert shipper.shipped == 2


@pytest.mark.asyncio
async def test_shipper_drops_batch_after_two_failed_attempts(_fresh_log, monkeypatch):
    attempts = {"n": 0}

    async def stub_post(self, url, *, content, headers, **kw):
        attempts["n"] += 1
        return httpx.Response(500, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", stub_post)
    shipper = LogShipper(
        url="https://collector.example/ingest",
        event_log=_fresh_log,
        flush_interval_sec=0.05,
    )
    await shipper.start()
    try:
        emit_component_error("c", "fail", exc=RuntimeError())
        await asyncio.sleep(0.15)
    finally:
        await shipper.stop()
    # Two attempts (initial + retry), then drop.
    assert attempts["n"] == 2
    assert shipper.dropped_batches == 1
    assert shipper.shipped == 0


@pytest.mark.asyncio
async def test_shipper_drains_on_shutdown(_fresh_log, monkeypatch):
    """Pending events at stop() must be flushed — otherwise restarting
    the bot loses the events queued during the previous shutdown."""
    posted_bodies: list[bytes] = []

    async def stub_post(self, url, *, content, headers, **kw):
        posted_bodies.append(content)
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", stub_post)
    shipper = LogShipper(
        url="https://collector.example/ingest",
        event_log=_fresh_log,
        flush_interval_sec=60.0,  # never tick during the test
        batch_max=1000,
    )
    await shipper.start()
    emit_component_error("c", "during-shutdown", exc=RuntimeError())
    # Tiny delay so the cross-thread enqueue completes.
    await asyncio.sleep(0.05)
    await shipper.stop()
    assert len(posted_bodies) == 1
    assert b"during-shutdown" in posted_bodies[0]
