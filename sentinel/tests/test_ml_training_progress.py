"""
Smoke test for the dashboard ML training progress flow:

  POST /api/ml/retrain  → pre-arms progress (active=True) synchronously
  GET  /api/ml/training-progress  → visible without race
  second POST while running  → 409 busy
  after task finishes → active=False, finished_at set

This mirrors what the Settings-page card does at runtime.
"""
from __future__ import annotations

import asyncio
import time

import pytest
from starlette.testclient import TestClient


@pytest.fixture
def retrain_dashboard(monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSWORD", "")
    from dashboard.app import Dashboard
    from config import Settings
    from core.events import EventBus

    progress: dict = {
        "active": False, "phase": "idle", "message": "", "symbols_total": 0,
        "symbols_done": 0, "current_symbol": None, "percent": 0,
        "started_at": None, "finished_at": None, "ok": None, "metrics": None,
    }

    def progress_set(**kw):
        progress.update(kw)
        t = progress.get("symbols_total") or 0
        d = progress.get("symbols_done") or 0
        if t > 0:
            progress["percent"] = min(100, int(round(d / t * 100)))

    def get_progress():
        return dict(progress)

    async def fake_retrain():
        # Simulate 3 steps: BTCUSDT -> ETHUSDT -> unified
        progress_set(phase="collecting_data", message="data", symbols_total=3, symbols_done=0)
        await asyncio.sleep(0.02)
        for i, sym in enumerate(["BTCUSDT", "ETHUSDT", "unified"], start=1):
            progress_set(phase="training", current_symbol=sym,
                         message=f"Обучение {sym}…", symbols_done=i)
            await asyncio.sleep(0.02)
        progress_set(phase="done", message="ok", ok=True, active=False,
                     finished_at=time.time())
        return True

    state = {
        "ml_retrain_fn": fake_retrain,
        "ml_training_progress_fn": get_progress,
        "ml_training_progress_set_fn": progress_set,
    }

    cfg = Settings(_env_file=None)
    cfg.dashboard_password = ""
    dashboard = Dashboard(
        settings=cfg,
        event_bus=EventBus(),
        state_provider=lambda: state,
    )
    app = dashboard._create_app() if hasattr(dashboard, "_create_app") else dashboard._build_app()
    client = TestClient(app)
    # Seed the sentinel_csrf cookie the middleware expects on mutating calls.
    client.get("/")
    return client, progress


def _post_retrain(client):
    token = client.cookies.get("sentinel_csrf", "")
    return client.post("/api/ml/retrain", headers={"X-CSRF-Token": token})


def test_retrain_pre_arms_progress_synchronously(retrain_dashboard):
    """Regression guard: POST must mark the job active before returning."""
    client, progress = retrain_dashboard
    r = _post_retrain(client)
    assert r.status_code == 200, r.text
    # Even without awaiting the coroutine, progress is already 'active=True'
    snap = client.get("/api/ml/training-progress").json()
    assert snap["active"] is True
    assert snap["phase"] in ("queued", "collecting_data", "training", "starting")


def test_second_retrain_returns_409_busy(retrain_dashboard):
    client, _ = retrain_dashboard
    r1 = _post_retrain(client)
    assert r1.status_code == 200
    r2 = _post_retrain(client)
    assert r2.status_code == 409
    assert r2.json()["status"] == "busy"


def test_progress_endpoint_reports_snapshot_shape(retrain_dashboard):
    """The endpoint must return the full snapshot shape even when idle."""
    client, _ = retrain_dashboard
    snap = client.get("/api/ml/training-progress").json()
    for key in ("active", "phase", "message", "symbols_total", "symbols_done",
                "current_symbol", "percent", "started_at", "finished_at",
                "ok", "metrics"):
        assert key in snap, f"snapshot missing '{key}'"
