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
