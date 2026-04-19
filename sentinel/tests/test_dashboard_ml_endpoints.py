"""
Tests for Phase-7 ML Robustness dashboard endpoints.

These are the first tests that exercise the FastAPI app directly. We mount
a minimal Dashboard() instance with a fake state_provider returning a
stub predictor so we can drive every response path — ready / not ready /
feature-flag-off — without pulling in the full main.py runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from starlette.testclient import TestClient


# ─────────────────────────────────────────────
# Stub predictor — just enough surface area
# for the four new /api/ml/* routes to read.
# ─────────────────────────────────────────────

@dataclass
class _FakeFoldResult:
    fold_idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    precision: float = 0.7
    recall: float = 0.6
    roc_auc: float = 0.75
    skill_score: float = 0.65
    train_precision: float = 0.72
    n_train: int = 400
    n_test: int = 50
    calibrated_threshold: float = 0.55


class _FakeWFReport:
    def __init__(self, n_folds: int = 3):
        self.fold_results = [
            _FakeFoldResult(i, i * 100, i * 100 + 400, i * 100 + 400, i * 100 + 450)
            for i in range(n_folds)
        ]
        self.mean_auc = 0.75
        self.std_auc = 0.04
        self.mean_precision = 0.7
        self.min_auc = 0.71
        self.mean_recall = 0.6
        self.mean_skill = 0.65
        self.degradation = 0.95
        self.n_folds_completed = n_folds
        self.mode = "rolling"

    def summary(self):
        return {
            "mean_precision": self.mean_precision, "mean_recall": self.mean_recall,
            "mean_auc": self.mean_auc, "std_auc": self.std_auc, "min_auc": self.min_auc,
            "mean_skill": self.mean_skill, "degradation": self.degradation,
            "n_folds_completed": self.n_folds_completed, "mode": self.mode,
        }


class _FakeRegimeRouter:
    def __init__(self, empty: bool = False):
        self.is_ready = not empty
        self.trained_regimes = [] if empty else ["trending_up", "sideways"]

    def get_regime_stats(self):
        if not self.is_ready:
            return {}
        return {
            "trending_up": {"regime": "trending_up", "n_trades": 200, "trained": True,
                            "skill_score": 0.72, "mean_precision": 0.78, "mean_auc": 0.82,
                            "fallback_reason": ""},
            "sideways": {"regime": "sideways", "n_trades": 180, "trained": True,
                         "skill_score": 0.68, "mean_precision": 0.71, "mean_auc": 0.76,
                         "fallback_reason": ""},
        }


class _FakePredictor:
    def __init__(
        self,
        has_wf: bool = True,
        has_router: bool = False,
        has_bootstrap: bool = False,
        has_correlation: bool = False,
    ):
        from analyzer.ml_predictor import MLMetrics, MLConfig
        self._metrics = MLMetrics(precision=0.7, recall=0.6, roc_auc=0.75, skill_score=0.65,
                                  train_samples=600, test_samples=120)
        self._cfg = MLConfig()
        self._model_version_str = "stub_v1"
        self._calibrated_threshold_val = 0.55
        self.rollout_mode = "shadow"
        self.is_ready = True
        self._wf = _FakeWFReport() if has_wf else None
        self._router = _FakeRegimeRouter() if has_router else None
        self._boot = (
            {"precision": {"metric": "precision", "mean": 0.7, "p5": 0.62, "p50": 0.7,
                           "p95": 0.78, "std": 0.05, "n_samples": 100, "n_simulations": 1000},
             "roc_auc": {"metric": "roc_auc", "mean": 0.75, "p5": 0.68, "p50": 0.75,
                         "p95": 0.82, "std": 0.04, "n_samples": 100, "n_simulations": 1000},
             "probability_above_random": 0.97}
            if has_bootstrap else {}
        )
        self._corr = ({"rf__lgbm": 0.62, "rf__xgb": 0.71, "lgbm__xgb": 0.58}
                      if has_correlation else {})

    @property
    def metrics(self):
        return self._metrics

    @property
    def walk_forward_report(self):
        return self._wf

    @property
    def regime_router(self):
        return self._router

    @property
    def bootstrap_ci(self):
        return dict(self._boot)

    @property
    def member_error_correlation(self):
        return dict(self._corr)

    # C4-M2/M3: mirror the new @property surface that /api/ml/status now reads
    # through. The real MLPredictor exposes these; the fake must too, otherwise
    # every test hitting /api/ml/status AttributeError'd instead of serialising.
    @property
    def model_version(self):
        return self._model_version_str

    @property
    def block_threshold(self):
        return float(self._cfg.block_threshold)

    @property
    def reduce_threshold(self):
        return float(getattr(self._cfg, "reduce_threshold", 0.65))

    @property
    def calibrated_threshold(self):
        return self._calibrated_threshold_val

    def needs_retrain(self):
        return False


# ─────────────────────────────────────────────
# Pytest setup
# ─────────────────────────────────────────────

@pytest.fixture
def dashboard_client(tmp_path, monkeypatch):
    # Isolate the dashboard from whatever .env is on disk by forcing
    # an empty password (auth middleware short-circuits when password is "").
    monkeypatch.setenv("DASHBOARD_PASSWORD", "")
    from dashboard.app import Dashboard
    from config import Settings
    from core.events import EventBus

    state = {"predictor": None}

    def state_provider():
        return {"ml_predictor": state["predictor"]}

    cfg = Settings(_env_file=None)
    cfg.dashboard_password = ""
    dashboard = Dashboard(
        settings=cfg,
        event_bus=EventBus(),
        state_provider=state_provider,
    )
    # Dashboard exposes _create_app internally — call it the same way start() does
    app = dashboard._create_app() if hasattr(dashboard, "_create_app") else dashboard._build_app()

    def set_predictor(p):
        state["predictor"] = p

    client = TestClient(app)
    client._set_predictor = set_predictor  # type: ignore[attr-defined]
    return client


# ─────────────────────────────────────────────
# /api/ml/status — C4-M2/M3 guard: endpoint reads through the new
# @property surface on MLPredictor; the fake must match or it breaks.
# ─────────────────────────────────────────────

class TestStatusEndpoint:
    def test_status_returns_enabled_false_when_no_predictor(self, dashboard_client):
        dashboard_client._set_predictor(None)
        r = dashboard_client.get("/api/ml/status")
        assert r.status_code == 200
        d = r.json()
        assert d["enabled"] is False

    def test_status_returns_metrics_through_public_properties(self, dashboard_client):
        """Regression guard: verify the endpoint wiring. Metrics come out
        correctly, and the threshold properties forward the dataclass values.
        Numeric values are hardcoded (m6-1 fix): this test actually fails if
        somebody changes MLConfig's block_threshold / reduce_threshold
        defaults silently — previously both sides read from MLConfig() so
        drift was invisible."""
        # Expected values frozen from MLConfig defaults at the time this
        # test was written. If you intentionally change the defaults, bump
        # them here too — deliberate, not silent.
        EXPECTED_BLOCK_THRESHOLD = 0.55
        EXPECTED_REDUCE_THRESHOLD = 0.65

        dashboard_client._set_predictor(_FakePredictor())
        r = dashboard_client.get("/api/ml/status")
        assert r.status_code == 200
        d = r.json()
        assert d["enabled"] is True
        assert d["ready"] is True
        assert d["version"] == "stub_v1"
        assert d["mode"] == "shadow"
        # Metrics serialize correctly
        m = d["metrics"]
        assert m["precision"] == 0.7
        assert m["roc_auc"] == 0.75
        # Threshold surface pinned to frozen numerics, not re-read from the
        # same dataclass — catches silent default drift.
        assert d["block_threshold"] == pytest.approx(EXPECTED_BLOCK_THRESHOLD)
        assert d["reduce_threshold"] == pytest.approx(EXPECTED_REDUCE_THRESHOLD)
        # Phase-ML1/4 fields exposed by /api/ml/status. All five keys must
        # be present even when the underlying MLMetrics instance pre-dates
        # PSR/DSR (a stale pickle on disk) — the endpoint fills ``None`` /
        # ``False`` / ``{}`` rather than omitting the fields, so the
        # dashboard's render path never has to feature-detect.
        for k in ("psr", "dsr", "psr_gate_passed", "psr_n_trials",
                  "feature_importance_stability"):
            assert k in m, f"/api/ml/status metrics missing {k}"
        # Old pickles without the fields should serialise without error.
        assert m["psr_gate_passed"] is False or m["psr_gate_passed"] is True
        assert isinstance(m["feature_importance_stability"], dict)


# ─────────────────────────────────────────────
# /api/ml/walk-forward
# ─────────────────────────────────────────────

class TestWalkForwardEndpoint:
    def test_returns_not_initialized_when_no_predictor(self, dashboard_client):
        dashboard_client._set_predictor(None)
        r = dashboard_client.get("/api/ml/walk-forward")
        assert r.status_code == 200
        d = r.json()
        assert d["enabled"] is False
        assert "not initialized" in d["reason"]

    def test_returns_not_available_when_wf_never_ran(self, dashboard_client):
        dashboard_client._set_predictor(_FakePredictor(has_wf=False))
        r = dashboard_client.get("/api/ml/walk-forward")
        d = r.json()
        assert d["enabled"] is True
        assert d["available"] is False

    def test_returns_folds_when_available(self, dashboard_client):
        dashboard_client._set_predictor(_FakePredictor(has_wf=True))
        r = dashboard_client.get("/api/ml/walk-forward")
        d = r.json()
        assert d["available"] is True
        assert len(d["folds"]) == 3
        assert d["summary"]["mean_auc"] == 0.75
        assert d["summary"]["mode"] == "rolling"


# ─────────────────────────────────────────────
# /api/ml/regime-performance
# ─────────────────────────────────────────────

class TestRegimePerformanceEndpoint:
    def test_not_active_without_router(self, dashboard_client):
        dashboard_client._set_predictor(_FakePredictor(has_router=False))
        r = dashboard_client.get("/api/ml/regime-performance")
        d = r.json()
        assert d["available"] is False

    def test_returns_regimes_when_router_ready(self, dashboard_client):
        dashboard_client._set_predictor(_FakePredictor(has_router=True))
        r = dashboard_client.get("/api/ml/regime-performance")
        d = r.json()
        assert d["available"] is True
        assert set(d["trained_regimes"]) == {"trending_up", "sideways"}
        assert "trending_up" in d["regimes"]


# ─────────────────────────────────────────────
# /api/ml/bootstrap-ci
# ─────────────────────────────────────────────

class TestBootstrapCiEndpoint:
    def test_returns_available_false_when_flag_off(self, dashboard_client):
        dashboard_client._set_predictor(_FakePredictor(has_bootstrap=False))
        r = dashboard_client.get("/api/ml/bootstrap-ci")
        d = r.json()
        assert d["available"] is False

    def test_returns_intervals_when_populated(self, dashboard_client):
        dashboard_client._set_predictor(_FakePredictor(has_bootstrap=True))
        r = dashboard_client.get("/api/ml/bootstrap-ci")
        d = r.json()
        assert d["available"] is True
        assert "roc_auc" in d["intervals"]
        assert d["intervals"]["probability_above_random"] == 0.97


# ─────────────────────────────────────────────
# /api/ml/member-correlation
# ─────────────────────────────────────────────

class TestMemberCorrelationEndpoint:
    def test_returns_available_false_when_empty(self, dashboard_client):
        dashboard_client._set_predictor(_FakePredictor(has_correlation=False))
        r = dashboard_client.get("/api/ml/member-correlation")
        d = r.json()
        assert d["available"] is False

    def test_returns_pairs_and_threshold(self, dashboard_client):
        dashboard_client._set_predictor(_FakePredictor(has_correlation=True))
        r = dashboard_client.get("/api/ml/member-correlation")
        d = r.json()
        assert d["available"] is True
        assert d["warn_threshold"] == 0.85
        assert "rf__lgbm" in d["correlations"]


# ─────────────────────────────────────────────
# HTML page
# ─────────────────────────────────────────────

class TestMLRobustnessPage:
    def test_page_loads(self, dashboard_client):
        r = dashboard_client.get("/ml-robustness")
        assert r.status_code == 200
        assert "ML Robustness" in r.text

    def test_page_references_all_four_endpoints(self, dashboard_client):
        """Round-7 #5: the /ml-robustness HTML fetches four ML endpoints.
        If any of those URLs get renamed/removed the page silently turns
        into an empty grid. Lock in the contract at the markup level so
        refactoring an endpoint name trips this test."""
        r = dashboard_client.get("/ml-robustness")
        html = r.text
        assert "/api/ml/walk-forward" in html
        assert "/api/ml/regime-performance" in html
        assert "/api/ml/bootstrap-ci" in html
        assert "/api/ml/member-correlation" in html

    def test_all_four_endpoints_serve_together_when_data_present(self, dashboard_client):
        """Full frontend-path smoke: one fake predictor populates ALL four
        sources, every endpoint returns available=true with the right shape,
        and the HTML loads. Catches cases where one endpoint's response
        regresses in a way an isolated unit test wouldn't notice."""
        dashboard_client._set_predictor(
            _FakePredictor(has_wf=True, has_router=True,
                           has_bootstrap=True, has_correlation=True),
        )
        # HTML still reachable
        r_page = dashboard_client.get("/ml-robustness")
        assert r_page.status_code == 200
        # Every /api/ml/* endpoint that the page consumes returns
        # available=True when the predictor has data.
        for path in (
            "/api/ml/walk-forward",
            "/api/ml/regime-performance",
            "/api/ml/bootstrap-ci",
            "/api/ml/member-correlation",
        ):
            r = dashboard_client.get(path)
            assert r.status_code == 200, f"{path} returned {r.status_code}"
            body = r.json()
            assert body.get("enabled") is True, f"{path}: enabled != True"
            assert body.get("available") is True, (
                f"{path}: available != True when fake has data; body={body}"
            )


# ─────────────────────────────────────────────
# /api/observability/summary — observability dashboard endpoint
# ─────────────────────────────────────────────

class TestObservabilityEndpoint:
    """Exercise the summary endpoint end-to-end.

    The endpoint reads from ``sentinel/logs/events.jsonl`` at a path hard-
    derived from the module's ``__file__``, so we write real events into
    that file and snapshot its prior contents to restore after the test.
    This is less tidy than a monkeypatch but exercises the real code path
    including the tail-reader.
    """

    @pytest.fixture
    def events_file(self, tmp_path, monkeypatch):
        """Point the endpoint at a tmp events.jsonl via env override.
        Production logs/events.jsonl is never touched — earlier versions
        of this fixture snapshotted/restored the real file, which was
        unsafe if a live bot was writing to it concurrently."""
        import json as _json
        import time as _time
        events_path = tmp_path / "events.jsonl"
        monkeypatch.setenv("SENTINEL_EVENTS_LOG_PATH", str(events_path))
        now_ms = int(_time.time() * 1000)

        def _write(records: list[dict]) -> None:
            with events_path.open("w", encoding="utf-8") as fh:
                for r in records:
                    fh.write(_json.dumps(r) + "\n")

        yield _write, now_ms

    def test_summary_returns_empty_sections_when_file_missing_or_empty(
        self, dashboard_client, events_file
    ):
        write, _ = events_file
        write([])  # empty file
        r = dashboard_client.get("/api/observability/summary?window_hours=24")
        assert r.status_code == 200
        d = r.json()
        assert d["signals"]["approved"] == 0
        assert d["signals"]["rejected"] == 0
        assert d["signals"]["approval_rate"] is None
        assert d["signals"]["buy"]["approval_rate"] is None
        assert d["signals"]["sell"]["approval_rate"] is None
        assert d["top_blocking_gates"] == []
        assert d["errors_by_component"] == []
        assert d["guards_tripped"] == []
        assert d["events_scanned"] == 0

    def test_summary_aggregates_signals_and_gates(self, dashboard_client, events_file):
        write, now_ms = events_file
        write([
            {"ts": now_ms - 1000, "type": "signal_approved", "symbol": "BTCUSDT"},
            {"ts": now_ms - 2000, "type": "signal_rejected", "gate": "liquidity_gate",
             "reason": "thin volume"},
            {"ts": now_ms - 3000, "type": "signal_rejected", "gate": "liquidity_gate",
             "reason": "thin volume"},
            {"ts": now_ms - 4000, "type": "signal_rejected", "gate": "drawdown_breaker",
             "reason": "dd tripped"},
        ])
        r = dashboard_client.get("/api/observability/summary?window_hours=24")
        assert r.status_code == 200
        d = r.json()
        assert d["signals"]["approved"] == 1
        assert d["signals"]["rejected"] == 3
        assert d["signals"]["approval_rate"] == 0.25
        # Top blocker must be liquidity_gate (2) then drawdown_breaker (1)
        gates = d["top_blocking_gates"]
        assert gates[0]["gate"] == "liquidity_gate"
        assert gates[0]["count"] == 2
        assert gates[1]["gate"] == "drawdown_breaker"

    def test_summary_surfaces_component_errors_and_guards(self, dashboard_client, events_file):
        write, now_ms = events_file
        # Chronological order (oldest first) — matches real events.jsonl semantics.
        write([
            {"ts": now_ms - 4000, "type": "guard_tripped", "guard": "circuit_breaker",
             "name": "CB-1", "reason": "price spike"},
            {"ts": now_ms - 3000, "type": "guard_tripped", "guard": "drawdown_breaker",
             "window": "daily", "reason": "dd 7%"},
            {"ts": now_ms - 2000, "type": "component_error", "component": "ml_ensemble.calibrator",
             "severity": "warning", "exc_type": "RuntimeError", "reason": "calibrator broken"},
            {"ts": now_ms - 1000, "type": "component_error", "component": "ml_predictor.predict",
             "severity": "warning", "exc_type": "ValueError", "reason": "feature mismatch"},
        ])
        r = dashboard_client.get("/api/observability/summary?window_hours=24")
        assert r.status_code == 200
        d = r.json()
        err_components = {e["component"] for e in d["errors_by_component"]}
        assert "ml_predictor.predict" in err_components
        assert "ml_ensemble.calibrator" in err_components
        assert d["errors_by_severity"].get("warning") == 2
        assert any(g["guard"] == "drawdown_breaker" for g in d["guards_tripped"])
        assert any(g["guard"] == "circuit_breaker" for g in d["guards_tripped"])
        # Recent errors are returned newest-first
        assert len(d["recent_component_errors"]) == 2
        assert d["recent_component_errors"][0]["component"] == "ml_predictor.predict"

    def test_summary_respects_window_hours(self, dashboard_client, events_file):
        write, now_ms = events_file
        ancient = now_ms - int(10 * 3600 * 1000)   # 10 hours ago
        recent = now_ms - int(0.25 * 3600 * 1000)  # 15 minutes ago
        write([
            {"ts": ancient, "type": "signal_rejected", "gate": "old_gate", "reason": "old"},
            {"ts": recent,  "type": "signal_rejected", "gate": "new_gate", "reason": "new"},
        ])
        # 1-hour window should drop the ancient event entirely.
        r = dashboard_client.get("/api/observability/summary?window_hours=1")
        assert r.status_code == 200
        d = r.json()
        gates = {g["gate"] for g in d["top_blocking_gates"]}
        assert "new_gate" in gates
        assert "old_gate" not in gates

    def test_summary_page_reachable(self, dashboard_client):
        r = dashboard_client.get("/observability")
        assert r.status_code == 200
        assert "SENTINEL" in r.text
        assert "Observability" in r.text

    def test_summary_reads_rotated_backups_for_long_windows(
        self, dashboard_client, events_file, monkeypatch, tmp_path
    ):
        """``window_hours > 24`` must include rotated backup files
        (``events.jsonl.1`` etc.); otherwise multi-day views are silently
        truncated at the rotation boundary."""
        import json as _json
        import time as _time
        events_path = tmp_path / "events.jsonl"
        backup_path = tmp_path / "events.jsonl.1"
        now_ms = int(_time.time() * 1000)
        # In-window event lives in the rotated backup, NOT the active file.
        backup_path.write_text(
            _json.dumps({
                "ts": now_ms - int(36 * 3600 * 1000),  # 36h ago — within 7-day window
                "type": "signal_rejected",
                "gate": "from_backup",
                "reason": "old",
                "direction": "BUY",
            }) + "\n",
            encoding="utf-8",
        )
        events_path.write_text("", encoding="utf-8")  # empty active file

        # 24h window → backup ignored (window <= 24h threshold) → 0 events
        r = dashboard_client.get("/api/observability/summary?window_hours=24")
        assert r.status_code == 200
        assert r.json()["events_scanned"] == 0
        # 72h window → backup IS scanned → backup gate visible
        r = dashboard_client.get("/api/observability/summary?window_hours=72")
        assert r.status_code == 200
        d = r.json()
        assert d["events_scanned"] == 1
        assert d["files_read"] == 2
        gates = {g["gate"] for g in d["top_blocking_gates"]}
        assert "from_backup" in gates

    def test_summary_caches_within_ttl(self, dashboard_client, events_file):
        """Identical calls within the 10s TTL must serve from cache —
        otherwise N open tabs × 30s refresh × file IO compounds."""
        write, now_ms = events_file
        write([
            {"ts": now_ms - 1000, "type": "signal_approved", "direction": "BUY"},
        ])
        r1 = dashboard_client.get("/api/observability/summary?window_hours=24")
        d1 = r1.json()
        assert d1["events_scanned"] == 1

        # Mutate the file out from under the endpoint — within TTL the
        # cached payload should still be returned.
        write([
            {"ts": now_ms - 500, "type": "signal_approved", "direction": "BUY"},
            {"ts": now_ms - 600, "type": "signal_approved", "direction": "BUY"},
            {"ts": now_ms - 700, "type": "signal_approved", "direction": "BUY"},
        ])
        r2 = dashboard_client.get("/api/observability/summary?window_hours=24")
        d2 = r2.json()
        assert d2["events_scanned"] == d1["events_scanned"], (
            "expected cache hit (same payload) within TTL window"
        )

    def test_slo_endpoint_reports_breach_when_buy_rate_below_target(
        self, dashboard_client, events_file, monkeypatch
    ):
        """When BUY approval rate falls below the configured SLO target,
        the endpoint must report a breach so on-call sees it immediately."""
        monkeypatch.setenv("SENTINEL_SLO_BUY_APPROVAL_RATE", "0.80")
        write, now_ms = events_file
        # 1 BUY approved, 4 BUY rejected → 20% — well below 80% target.
        events = [{"ts": now_ms - 100, "type": "signal_approved", "direction": "BUY"}]
        for i in range(4):
            events.append({"ts": now_ms - 200 - i, "type": "signal_rejected",
                           "direction": "BUY", "gate": "any", "reason": "x"})
        write(events)
        r = dashboard_client.get("/api/observability/slo?window_hours=24")
        assert r.status_code == 200
        d = r.json()
        slo_buy = next(s for s in d["slos"] if s["name"] == "buy_approval_rate")
        assert slo_buy["status"] == "breach", f"expected breach, got {slo_buy}"
        assert d["any_breach"] is True

    def test_slo_endpoint_ok_when_within_targets(self, dashboard_client, events_file):
        write, now_ms = events_file
        # 8 approved, 2 rejected → 80% — well above default 40% target.
        events = []
        for i in range(8):
            events.append({"ts": now_ms - 100 - i, "type": "signal_approved", "direction": "BUY"})
        for i in range(2):
            events.append({"ts": now_ms - 200 - i, "type": "signal_rejected",
                           "direction": "BUY", "gate": "g", "reason": "x"})
        write(events)
        r = dashboard_client.get("/api/observability/slo?window_hours=24")
        d = r.json()
        slo_buy = next(s for s in d["slos"] if s["name"] == "buy_approval_rate")
        assert slo_buy["status"] == "ok"
        assert d["any_breach"] is False

    def test_metrics_endpoint_returns_prometheus_text(
        self, dashboard_client, events_file
    ):
        """The /api/observability/metrics endpoint must return parseable
        Prometheus text format with the expected metric families."""
        write, now_ms = events_file
        write([
            {"ts": now_ms - 100, "type": "signal_approved", "direction": "BUY"},
            {"ts": now_ms - 200, "type": "signal_rejected", "direction": "BUY",
             "gate": "drawdown_breaker", "reason": "dd"},
            {"ts": now_ms - 300, "type": "component_error", "component": "ml_predictor",
             "severity": "warning", "exc_type": "ValueError", "reason": "x"},
            {"ts": now_ms - 400, "type": "guard_tripped", "guard": "circuit_breaker", "name": "CB-1"},
        ])
        r = dashboard_client.get("/api/observability/metrics")
        assert r.status_code == 200
        body = r.text
        assert "sentinel_signals_total" in body
        assert 'direction="BUY"' in body
        assert "sentinel_gate_rejections_total" in body
        assert 'gate="drawdown_breaker"' in body
        assert "sentinel_component_errors_total" in body
        assert 'component="ml_predictor"' in body
        assert "sentinel_guard_trips_total" in body
        # Format sanity — every metric line must start with a metric name
        # or be a HELP/TYPE comment.
        for line in body.strip().split("\n"):
            assert line.startswith("#") or line.startswith("sentinel_"), (
                f"unexpected line: {line!r}"
            )

    def test_summary_splits_buy_and_sell_approval_rate(
        self, dashboard_client, events_file
    ):
        """SELL bypasses entry gates and would inflate the overall approval
        rate; the BUY-only rate must be tracked separately so a BUY-rejection
        storm isn't masked by routine SELL approvals."""
        write, now_ms = events_file
        # 1 BUY approved, 3 BUY rejected → BUY rate = 25%
        # 5 SELL approved, 0 SELL rejected → SELL rate = 100%
        # Overall: 6 approved, 3 rejected → 67% (would mislead)
        events = []
        for i in range(1):
            events.append({"ts": now_ms - 100 - i, "type": "signal_approved", "direction": "BUY"})
        for i in range(3):
            events.append({"ts": now_ms - 200 - i, "type": "signal_rejected",
                           "direction": "BUY", "gate": "liquidity_gate", "reason": "x"})
        for i in range(5):
            events.append({"ts": now_ms - 300 - i, "type": "signal_approved", "direction": "SELL"})
        write(events)
        r = dashboard_client.get("/api/observability/summary?window_hours=24")
        assert r.status_code == 200
        d = r.json()
        s = d["signals"]
        assert s["buy"]["approved"] == 1
        assert s["buy"]["rejected"] == 3
        assert s["buy"]["approval_rate"] == 0.25
        assert s["sell"]["approved"] == 5
        assert s["sell"]["rejected"] == 0
        assert s["sell"]["approval_rate"] == 1.0
        # Overall rate kept for back-compat but should equal 6/9
        assert s["approval_rate"] == round(6 / 9, 4)
