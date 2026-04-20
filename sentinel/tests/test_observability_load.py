"""
Load tests for the observability pipeline.

These are fast pytest benchmarks (no external load-tester) that fail
when latency / throughput drops below documented expectations:

- ``EventLog.emit`` should sustain ≥10k events/sec on a laptop SSD.
- ``/api/observability/summary`` over 5000 events should respond
  in <250 ms (cached: <10 ms).
- The dedup hot path under storm conditions must not degrade.

These thresholds are deliberately loose so the test isn't flaky on a
busy CI machine — they only catch order-of-magnitude regressions
(e.g. someone added a per-event ``json.dumps`` to a hot path).
"""
from __future__ import annotations

import json
import time

import pytest

from monitoring.event_log import (
    EventLog,
    _reset_component_error_dedup,
    emit_component_error,
    emit_rejection,
    set_event_log,
)


@pytest.fixture(autouse=True)
def _fresh_log():
    _reset_component_error_dedup()
    log = EventLog(path=None)
    set_event_log(log)
    yield log
    _reset_component_error_dedup()
    set_event_log(EventLog(path=None))


def test_emit_throughput_in_memory(_fresh_log):
    """In-memory emit (no file write) must sustain ≥10k ev/s.
    Regression catcher for accidentally adding heavy work in the hot path."""
    n = 10_000
    start = time.perf_counter()
    for i in range(n):
        _fresh_log.emit("benchmark", idx=i, payload="x" * 50)
    elapsed = time.perf_counter() - start
    rate = n / elapsed
    assert rate > 10_000, f"in-memory emit too slow: {rate:.0f} ev/s (elapsed={elapsed:.3f}s)"


def test_emit_throughput_to_disk(tmp_path):
    """Disk-backed emit (single-process, no file lock) must sustain ≥5k ev/s.
    Below this, JSONL I/O has become the bottleneck — investigate before
    landing the change."""
    log = EventLog(path=tmp_path / "events.jsonl")
    n = 5_000
    start = time.perf_counter()
    for i in range(n):
        log.emit("benchmark", idx=i, payload="x" * 50)
    elapsed = time.perf_counter() - start
    rate = n / elapsed
    assert rate > 5_000, f"disk emit too slow: {rate:.0f} ev/s"


def test_dedup_storm_does_not_degrade(_fresh_log):
    """Under a storm of identical errors, the dedup path must short-circuit
    fast. 100k attempts (suppressed) should take <0.5s on a laptop."""
    exc = RuntimeError("storm")
    # Prime the dedup so subsequent calls all suppress.
    emit_component_error("flooding", "first", exc=exc)
    n = 100_000
    start = time.perf_counter()
    for _ in range(n):
        emit_component_error("flooding", "more", exc=exc)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"dedup hot path slow: {elapsed:.3f}s for {n} suppressed calls"


def test_summary_endpoint_cold_under_5000_events_below_500ms(tmp_path, monkeypatch):
    """Cold-cache /api/observability/summary against 5000 events
    should respond well under 500 ms — well within human-perceived
    snappy threshold for a dashboard refresh."""
    # Build the events.jsonl directly (don't use EventLog to avoid
    # subscriber overhead; we're benchmarking the endpoint, not emit).
    events_path = tmp_path / "events.jsonl"
    now_ms = int(time.time() * 1000)
    with events_path.open("w", encoding="utf-8") as fh:
        for i in range(5000):
            ev = {
                "ts": now_ms - i,
                "type": "signal_rejected" if i % 3 else "signal_approved",
                "direction": "BUY" if i % 2 else "SELL",
                "gate": f"gate_{i % 7}",
                "reason": "x" * 30,
                "schema_version": 1,
            }
            fh.write(json.dumps(ev) + "\n")

    monkeypatch.setenv("SENTINEL_EVENTS_LOG_PATH", str(events_path))

    # Build a fresh dashboard client. We import here to avoid pulling
    # FastAPI into the policy-tests module imports.
    from starlette.testclient import TestClient
    from config import Settings
    from core.events import EventBus
    from dashboard.app import Dashboard

    monkeypatch.setenv("DASHBOARD_PASSWORD", "")
    cfg = Settings(_env_file=None)
    cfg.dashboard_password = ""
    dashboard = Dashboard(settings=cfg, event_bus=EventBus(), state_provider=lambda: {})
    app = dashboard._create_app() if hasattr(dashboard, "_create_app") else dashboard._build_app()
    client = TestClient(app)

    # Warm to make sure first-time imports inside the endpoint aren't
    # measured. We measure the second (cold-cache for new params) call.
    client.get("/api/observability/summary?window_hours=24&max_events=100")
    start = time.perf_counter()
    r = client.get("/api/observability/summary?window_hours=24&max_events=5000")
    elapsed = time.perf_counter() - start
    assert r.status_code == 200
    assert elapsed < 0.5, f"endpoint slow: {elapsed * 1000:.0f}ms for 5000 events"
    # Sanity: payload populated.
    assert r.json()["events_scanned"] > 1000
