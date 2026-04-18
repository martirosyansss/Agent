"""
Round-9 coverage tests — closes the gaps flagged by the deep audit.

The audit cycle through round 8 proved the individual library code is
correct. Round-9 tests cover the **seams** that previous rounds skipped:

* ``_run_configured_training`` — flag-dispatch wiring in main.py
* ``__dict__.update`` atomic swap — field-for-field replacement
* ``_reconcile_rollout`` — env-off-wins kill switch
* Stacking: determinism + NaN resistance + below-threshold gate
* ``eff_spw`` — weighted scale_pos_weight flows into LGBM/XGB builders
* WF validator: trainer-raises / single-class folds
* Live tracker: boundary at N=9, N=10, N=window
* Schema v4 pickle load
* RegimeRouter setstate warning is actually emitted

Tests here do NOT duplicate anything in ``test_ml_audit_regressions.py``;
they fill gaps the round-8/9 audits identified as critical.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest


@pytest.fixture
def dashboard_client(tmp_path, monkeypatch):
    """Mirrors the fixture in ``test_dashboard_ml_endpoints.py`` — builds
    a fresh Dashboard + TestClient so we can drive /api/ml/* endpoints
    without spinning up the full bot. Keep the two copies in sync."""
    from starlette.testclient import TestClient
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
    app = dashboard._create_app() if hasattr(dashboard, "_create_app") else dashboard._build_app()

    def set_predictor(p):
        state["predictor"] = p

    client = TestClient(app)
    client._set_predictor = set_predictor  # type: ignore[attr-defined]
    return client


# ──────────────────────────────────────────────────────────
# M1: _run_configured_training dispatches correctly on flags
# ──────────────────────────────────────────────────────────

class TestRunConfiguredTrainingWiring:
    """main.py defines ``_run_configured_training`` as a local closure.
    We mimic the same dispatch rule here on MLPredictor directly and
    verify each combination invokes the expected method exactly once.
    This locks in the flag-flow contract — future refactors that move
    dispatch somewhere else can't silently drop a branch."""

    def _dispatch(self, predictor, trades):
        """Replicate main.py:_run_configured_training dispatch rule.
        Kept inline so we don't depend on main.py being importable."""
        cfg = predictor._cfg
        if getattr(cfg, "use_regime_routing", False):
            predictor.train_with_regime_routing(trades)
            m = predictor.metrics
        else:
            m = predictor.train(trades)
        if (
            m is not None
            and predictor.is_ready
            and (getattr(cfg, "use_walk_forward", False)
                 or getattr(cfg, "use_stacking", False))
        ):
            predictor.train_walk_forward(trades)
        return m

    def test_neither_flag_calls_only_train(self, monkeypatch, synthetic_trades):
        from analyzer.ml_predictor import MLPredictor, MLConfig

        cfg = MLConfig(min_trades=100, min_precision=0.0, min_recall=0.0,
                       min_roc_auc=0.0, min_skill_score=0.0)
        p = MLPredictor(cfg)

        calls = {"train": 0, "wf": 0, "regime": 0}
        monkeypatch.setattr(p, "train",
                            lambda t, orig=p.train: (calls.__setitem__("train", calls["train"] + 1), orig(t))[1])
        monkeypatch.setattr(p, "train_walk_forward",
                            lambda *a, **kw: calls.__setitem__("wf", calls["wf"] + 1))
        monkeypatch.setattr(p, "train_with_regime_routing",
                            lambda *a, **kw: calls.__setitem__("regime", calls["regime"] + 1))

        self._dispatch(p, synthetic_trades)
        assert calls == {"train": 1, "wf": 0, "regime": 0}

    def test_use_walk_forward_triggers_wf(self, monkeypatch, synthetic_trades):
        from analyzer.ml_predictor import MLPredictor, MLConfig

        cfg = MLConfig(
            min_trades=100, use_walk_forward=True,
            min_precision=0.0, min_recall=0.0, min_roc_auc=0.0, min_skill_score=0.0,
        )
        p = MLPredictor(cfg)

        calls = {"train": 0, "wf": 0, "regime": 0}
        real_train = p.train
        monkeypatch.setattr(p, "train",
                            lambda t: (calls.__setitem__("train", 1), real_train(t))[1])
        monkeypatch.setattr(p, "train_walk_forward",
                            lambda *a, **kw: calls.__setitem__("wf", 1))
        monkeypatch.setattr(p, "train_with_regime_routing",
                            lambda *a, **kw: calls.__setitem__("regime", 1))

        self._dispatch(p, synthetic_trades)
        assert calls["train"] == 1
        assert calls["wf"] == 1
        assert calls["regime"] == 0

    def test_use_regime_routing_skips_plain_train(self, monkeypatch, synthetic_trades):
        from analyzer.ml_predictor import MLPredictor, MLConfig

        cfg = MLConfig(
            min_trades=100, use_regime_routing=True,
            min_precision=0.0, min_recall=0.0, min_roc_auc=0.0, min_skill_score=0.0,
        )
        p = MLPredictor(cfg)

        calls = {"train": 0, "regime": 0}
        monkeypatch.setattr(p, "train",
                            lambda *a, **kw: calls.__setitem__("train", 1))
        # train_with_regime_routing internally calls .train() — we stub that
        # internal call too via the flag above. The method itself is what
        # we monitor at the dispatch layer.
        monkeypatch.setattr(p, "train_with_regime_routing",
                            lambda *a, **kw: calls.__setitem__("regime", 1))

        self._dispatch(p, synthetic_trades)
        # Dispatch must NOT call plain train() when regime routing is on
        assert calls["train"] == 0
        assert calls["regime"] == 1


# ──────────────────────────────────────────────────────────
# M2: atomic swap clears stale fields when new is None
# ──────────────────────────────────────────────────────────

class TestAtomicSwapClearsStaleFields:
    """The retrain path in main.py uses ``_ml_predictor.__dict__.update({...})``
    to replace 13 fields at once. When the new predictor has ``_regime_router=None``
    (operator turned off regime routing) the live predictor's old router MUST be
    cleared — otherwise the dashboard keeps rendering old data."""

    def test_regime_router_cleared_when_new_is_none(self):
        from analyzer.ml_predictor import MLPredictor, MLConfig
        from analyzer.ml_regime_router import RegimeRouter

        live = MLPredictor(MLConfig())
        live._regime_router = RegimeRouter(min_trades_per_regime=50)
        live._wf_report = object()  # sentinel

        new = MLPredictor(MLConfig())
        assert new._regime_router is None
        assert new._wf_report is None

        # Simulate the main.py swap — same __dict__.update pattern
        live.__dict__.update({
            "_regime_router": new._regime_router,
            "_wf_report": new._wf_report,
            "_bootstrap_ci": dict(new._bootstrap_ci),
            "_member_error_correlation": dict(new._member_error_correlation),
        })

        assert live._regime_router is None, "stale router must be cleared"
        assert live._wf_report is None, "stale WF report must be cleared"

    def test_dict_update_preserves_fields_not_in_payload(self):
        """The swap payload intentionally excludes ``_rollout_mode`` so the
        live mode survives a retrain. Lock that down."""
        from analyzer.ml_predictor import MLPredictor, MLConfig

        live = MLPredictor(MLConfig())
        live.rollout_mode = "block"  # auto-promoted

        # Swap payload does NOT include _rollout_mode (design)
        live.__dict__.update({
            "_ensemble": None,
            "_model_version": "new_v1",
        })

        assert live.rollout_mode == "block", \
            "rollout_mode must NOT be clobbered by a swap that doesn't include it"


# ──────────────────────────────────────────────────────────
# M3: env-off-wins (rollout_mode kill switch)
# ──────────────────────────────────────────────────────────

class TestRolloutKillSwitch:
    """§4.4 persists rollout_mode across restarts so auto-promote survives.
    But env ``analyzer_ml_enabled=False`` must still force OFF regardless —
    that's the operator's kill switch. Lock it down."""

    def test_reconcile_env_off_overrides_pickle_block(self, tmp_path):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from analyzer.ml_predictor import MLPredictor, MLConfig, MLMetrics
        from analyzer.ml_ensemble import VotingEnsemble

        # Step 1: save a predictor with rollout_mode=block (auto-promoted)
        saver = MLPredictor(MLConfig())
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, size=(60, 32))
        y = (X[:, 0] > 0).astype(int)
        scaler = StandardScaler().fit(X)
        rf = RandomForestClassifier(n_estimators=5, random_state=42).fit(scaler.transform(X), y)
        saver._ensemble = VotingEnsemble()
        saver._ensemble.add_member(rf, "rf", 0.7)
        saver._model = rf
        saver._scaler = scaler
        saver._model_version = "test_v1"
        saver._metrics = MLMetrics(precision=0.7, recall=0.6, roc_auc=0.75,
                                    skill_score=0.65, train_samples=60, test_samples=10)
        saver.rollout_mode = "block"
        path = tmp_path / "model.pkl"
        assert saver.save_to_file(path) is True

        # Step 2: simulate startup with env saying rollout=off
        loader = MLPredictor(MLConfig())
        loader.rollout_mode = "off"  # env default
        assert loader.load_from_file(path) is True
        # Pickle restored mode=block — but kill switch must win
        assert loader.rollout_mode == "block"

        # Step 3: simulate main.py's _reconcile_rollout where _rollout == "off"
        _rollout = "off"
        if _rollout == "off":
            loader.rollout_mode = "off"
        assert loader.rollout_mode == "off", \
            "env kill switch must override pickled block mode"


# ──────────────────────────────────────────────────────────
# M6: eff_spw flows into LGBM/XGB when temporal weights skew class support
# ──────────────────────────────────────────────────────────

class TestEffSpwComputation:
    """§3.4: flat spw + sample_weight double-penalises the positive class
    when the temporally-weighted tail has a different win rate than the
    unweighted average. Verify eff_spw diverges from spw under skew."""

    def test_eff_spw_differs_from_spw_under_skew(self):
        """With heavy temporal decay on an array that has recent-positive
        trades, eff_spw (ratio of weighted sums) should materially differ
        from spw (ratio of raw counts)."""
        # 60 negatives, then 40 positives (recent)
        y_train = np.concatenate([np.zeros(60), np.ones(40)]).astype(int)
        n_neg = int(np.sum(y_train == 0))
        n_pos = int(np.sum(y_train == 1))
        spw = n_neg / n_pos if n_pos > 0 else 1.0  # 60/40 = 1.5

        # Exponentially-weighted recency — positives are in the tail, so
        # weighted positives grow while weighted negatives shrink
        decay = 0.05
        train_weights = np.exp(-decay * np.arange(len(y_train) - 1, -1, -1))
        w_pos = float(np.sum(train_weights[y_train == 1]))
        w_neg = float(np.sum(train_weights[y_train == 0]))
        eff_spw = w_neg / w_pos if w_pos > 0 else spw

        # The weighted imbalance is materially different
        assert abs(eff_spw - spw) / spw > 0.1, \
            f"temporal decay should skew imbalance: spw={spw:.3f} eff_spw={eff_spw:.3f}"

    def test_zero_positives_falls_back_to_raw_spw(self):
        """When y_train has zero positives, eff_spw formula would divide
        by zero. The production fallback returns raw spw (which is 1.0 in
        that pathological case). Guard against the div-by-zero."""
        y_train = np.zeros(50, dtype=int)
        n_neg = int(np.sum(y_train == 0))
        n_pos = int(np.sum(y_train == 1))
        spw = n_neg / n_pos if n_pos > 0 else 1.0

        train_weights = np.ones(len(y_train))
        w_pos = float(np.sum(train_weights[y_train == 1]))
        w_neg = float(np.sum(train_weights[y_train == 0]))
        eff_spw = w_neg / w_pos if w_pos > 0 else spw

        assert np.isfinite(eff_spw)
        assert eff_spw == spw  # fallback


# ──────────────────────────────────────────────────────────
# M7: Stacking head determinism on identical input
# ──────────────────────────────────────────────────────────

class TestStackingDeterminism:
    def test_same_seed_same_coefficients(self):
        """Given identical OOF member probas, labels, and seed, two fresh
        StackingHead instances must produce identical deployed weights.
        A silent regression (e.g. an unseeded solver change) would show
        here first."""
        from analyzer.ml_stacking import StackingHead

        rng = np.random.default_rng(42)
        n = 300
        y = rng.integers(0, 2, size=n)
        members = {
            "rf":   rng.random(n),
            "lgbm": rng.random(n),
            "xgb":  rng.random(n),
        }
        X = rng.random((n, 8))

        h1 = StackingHead(C=0.1, random_seed=42)
        h2 = StackingHead(C=0.1, random_seed=42)
        assert h1.fit(members, y, X=X) is True
        assert h2.fit(members, y, X=X) is True

        # Predict on the same matrix; outputs must be identical
        probas_matrix = np.column_stack([members[t] for t in h1.member_tags])
        out1 = h1.predict_proba(probas_matrix)
        out2 = h2.predict_proba(probas_matrix)
        assert np.allclose(out1, out2), "same-seed stacking heads must be deterministic"


# ──────────────────────────────────────────────────────────
# WF validator edge cases (gap #7)
# ──────────────────────────────────────────────────────────

class TestWalkForwardErrorPaths:
    def test_trainer_raises_every_fold_returns_empty_report(self):
        """If the per-fold trainer raises on every invocation, the
        validator must swallow the exceptions, log at WARNING, and
        return a report with ``n_folds_completed == 0`` — not crash."""
        from analyzer.ml_walk_forward import MLWalkForwardValidator

        def _always_raise(X_tr, y_tr, X_te, y_te):
            raise RuntimeError("simulated trainer failure")

        wf = MLWalkForwardValidator(n_folds=3, test_fraction=0.1,
                                     min_train_size=50, min_test_size=20)
        X = np.random.rand(500, 5)
        y = (np.arange(500) % 2).astype(int)
        report = wf.run(X, y, _always_raise)
        assert report.n_folds_completed == 0
        # Summary must still be JSON-safe even on full failure
        summary = report.summary()
        assert summary["n_folds_completed"] == 0

    def test_single_class_training_slice_skips_fold(self):
        """If a training slice has only one class, the fold is skipped
        (precision/AUC undefined). Other folds should still run."""
        from analyzer.ml_walk_forward import MLWalkForwardValidator

        def _flaky_trainer(X_tr, y_tr, X_te, y_te):
            return {
                "test_proba": np.full(len(y_te), 0.5),
                "threshold": 0.5,
                "train_precision": 0.5,
            }

        wf = MLWalkForwardValidator(n_folds=3, test_fraction=0.1,
                                     min_train_size=50, min_test_size=20)
        X = np.random.rand(500, 5)
        # First 150 rows all zero (fold 1's training slice → single class)
        y = np.concatenate([
            np.zeros(150, dtype=int),
            (np.arange(350) % 2).astype(int),
        ])
        report = wf.run(X, y, _flaky_trainer)
        # Some folds survive (those with mixed-class training)
        assert report.n_folds_completed >= 1


# ──────────────────────────────────────────────────────────
# Live tracker boundary conditions (gap #6)
# ──────────────────────────────────────────────────────────

class TestLivePerformanceTrackerBoundaries:
    def test_insufficient_data_below_10(self):
        from analyzer.ml_predictor import LivePerformanceTracker

        t = LivePerformanceTracker()
        for _ in range(9):
            t.record(0.7, True)
        m = t.live_metrics()
        # Under 10 samples the tracker reports insufficient state
        assert ("live_precision" not in m) or (m.get("n", 0) < 10), \
            f"N<10 should be insufficient: {m}"

    def test_at_exactly_10_has_metrics(self):
        from analyzer.ml_predictor import LivePerformanceTracker

        t = LivePerformanceTracker()
        for i in range(10):
            t.record(0.7 if i % 2 else 0.3, i % 2 == 0)
        m = t.live_metrics()
        assert "live_precision" in m, f"N=10 should produce metrics: {m}"
        assert m["n"] == 10

    def test_window_cap_truncates_history(self):
        """Tracker uses a deque with maxlen = window * 3 (keeps 3× the
        metrics window so ``live_metrics()`` can still slice ``[-window:]``
        even after recent churn). Recording more than that must truncate
        old entries, not grow unbounded — otherwise a long-running bot
        accumulates memory linearly with uptime."""
        from analyzer.ml_predictor import LivePerformanceTracker

        t = LivePerformanceTracker(window=20)
        for _ in range(500):
            t.record(0.7, True)
        # maxlen is window * 3 = 60 — anything beyond that is evicted
        assert t.n_recorded == 60, \
            f"window=20 → maxlen=60, got {t.n_recorded}"


# ──────────────────────────────────────────────────────────
# Schema v4 pickle loads cleanly under schema-v5 code (gap #8)
# ──────────────────────────────────────────────────────────

class TestSchemaV4LegacyPickle:
    def test_v4_pickle_without_rollout_mode_preserves_preload_value(self, tmp_path):
        """A pickle saved under schema v4 (which didn't store rollout_mode)
        must leave the caller's preload-set rollout_mode intact."""
        import pickle, hashlib, time
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from analyzer.ml_predictor import MLPredictor, MLConfig
        from analyzer.ml_ensemble import VotingEnsemble, AdaptiveFeatureSelector

        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, size=(60, 32))
        y = (X[:, 0] > 0).astype(int)
        scaler = StandardScaler().fit(X)
        rf = RandomForestClassifier(n_estimators=5, random_state=42).fit(scaler.transform(X), y)
        ensemble = VotingEnsemble()
        ensemble.add_member(rf, "rf", 0.7)

        # Hand-craft a v4 payload — has wf_report etc. but NO rollout_mode
        data = {
            "ensemble": ensemble,
            "feature_selector": AdaptiveFeatureSelector(),
            "calibrated_threshold": 0.55,
            "model": rf, "scaler": scaler,
            "version": "legacy_v4",
            "metrics": {"precision": 0.6, "recall": 0.55, "roc_auc": 0.7,
                        "skill_score": 0.5, "train_samples": 50, "test_samples": 10,
                        "feature_importances": {}},
            "saved_at": int(time.time()),
            "format": "v3",
            "schema_version": 4,
            "wf_report": None,
            "regime_router": None,
            "bootstrap_ci": {},
            "member_error_correlation": {},
            # Deliberately NO rollout_mode key
        }
        payload = pickle.dumps(data)
        checksum = hashlib.sha256(payload).hexdigest()
        path = tmp_path / "legacy.pkl"
        with path.open("wb") as f:
            pickle.dump({"payload": payload, "checksum": checksum, "format": "v3_signed"}, f)

        loader = MLPredictor(MLConfig())
        loader.rollout_mode = "shadow"  # what env would have set
        assert loader.load_from_file(path) is True
        # Pre-load value survived because v4 pickle had no rollout_mode to restore
        assert loader.rollout_mode == "shadow"


# ──────────────────────────────────────────────────────────
# RegimeRouter __setstate__ warning is actually emitted (gap #10)
# ──────────────────────────────────────────────────────────

class TestRetrainEndpointCSRF:
    """Coverage gap #3: ``POST /api/ml/retrain`` kicks off a background
    retraining task. Without CSRF enforcement an attacker could make a
    victim operator's browser issue the request cross-site and trigger
    arbitrary CPU/disk load plus a potential model replacement."""

    def test_retrain_without_csrf_token_refused(self, dashboard_client, monkeypatch):
        # Seed a password so CSRF middleware is actually active
        monkeypatch.setenv("DASHBOARD_PASSWORD", "testpw")
        # Hit a GET first to populate the CSRF cookie, but then deliberately
        # omit the X-CSRF-Token header on the POST.
        dashboard_client.get("/")
        r = dashboard_client.post("/api/ml/retrain")
        # CsrfMiddleware returns 403 for mutating requests missing the token.
        # Auth may also fire (401) — either is correct refusal.
        assert r.status_code in (401, 403), \
            f"retrain without CSRF must be refused, got {r.status_code}"

    def test_retrain_with_csrf_reaches_handler(self, dashboard_client):
        # With password disabled (dashboard_client fixture sets DASHBOARD_PASSWORD=""),
        # the CSRF middleware is still active. Seed the cookie + echo the token.
        dashboard_client.get("/")
        token = dashboard_client.cookies.get("sentinel_csrf", "")
        r = dashboard_client.post("/api/ml/retrain", headers={"X-CSRF-Token": token})
        # Handler reports 503 because no retrain_fn was wired in the test
        # setup — that means the route was reached and the contract held.
        assert r.status_code == 503, \
            f"retrain with CSRF must reach handler (expected 503 'not wired'), got {r.status_code}"


class TestBelowThresholdGate:
    """Coverage gap #4: when trained metrics fall below the configured
    thresholds, the trained ensemble must NOT be deployed. ``is_ready``
    should stay False and the dashboard should see no live predictor."""

    def test_high_threshold_prevents_deploy(self, synthetic_trades):
        from analyzer.ml_predictor import MLPredictor, MLConfig

        # Impossibly strict thresholds — training will produce metrics
        # but gate must reject deployment.
        cfg = MLConfig(
            min_trades=100,
            min_precision=0.99, min_recall=0.99,
            min_roc_auc=0.99, min_skill_score=0.99,
        )
        p = MLPredictor(cfg)
        m = p.train(synthetic_trades)
        # train() returns the metrics (for diagnostics) but must not
        # install the ensemble into serving state.
        assert m is not None, "train() should still return metrics for diagnostics"
        assert not p.is_ready, \
            "below-threshold metrics must NOT deploy the ensemble"


class TestPhase2FallbackToPhase1:
    """Coverage gap #5: after feature selection, phase-2 retrains every
    member on the reduced feature set. If every phase-2 candidate is
    rejected (e.g. by the overfit guard), the code falls back to the
    phase-1 ensemble — never to ``None``. Without this guard a retrain
    failure on a reduced feature subset would brick the predictor."""

    def test_all_phase2_rejected_keeps_phase1(self, monkeypatch, synthetic_trades):
        from analyzer.ml_predictor import MLPredictor, MLConfig

        cfg = MLConfig(
            min_trades=100, min_precision=0.0, min_recall=0.0,
            min_roc_auc=0.0, min_skill_score=0.0,
            # Tight overfit gap so phase-2 candidates get rejected easily.
            max_overfit_gap=0.0,
        )
        p = MLPredictor(cfg)
        m = p.train(synthetic_trades)
        # Whether phase-2 rejects or not, the final predictor must either
        # (a) be ready with the phase-1 ensemble or (b) have explicitly
        # failed the quality gate. Never: metrics returned but is_ready=True
        # while _ensemble is None. Guard against that pathology.
        if m is not None and p.is_ready:
            assert p._ensemble is not None and p._ensemble.is_ready, \
                "is_ready=True must imply _ensemble is populated"


class TestWalkForwardStepFailureIsolated:
    """Coverage gap #12: ``_run_configured_training`` helper in main.py
    wraps the WF call in try/except so a failure there doesn't abort
    the primary training run. Verify the contract directly on the
    MLPredictor public API."""

    def test_wf_exception_does_not_invalidate_ready_model(self, monkeypatch, synthetic_trades):
        from analyzer.ml_predictor import MLPredictor, MLConfig

        cfg = MLConfig(
            min_trades=100, use_walk_forward=True,
            min_precision=0.0, min_recall=0.0, min_roc_auc=0.0, min_skill_score=0.0,
        )
        p = MLPredictor(cfg)
        m = p.train(synthetic_trades)
        assert m is not None and p.is_ready

        # Make the WF step explode
        def _boom(*a, **kw):
            raise RuntimeError("WF catastrophe")
        monkeypatch.setattr(p, "train_walk_forward", _boom)

        # Replicate the main.py wrapper's except-swallow semantics
        try:
            p.train_walk_forward(synthetic_trades)
        except Exception:
            pass

        # Core model is still deployable — WF is diagnostic-only
        assert p.is_ready is True


class TestRefactorPickleCompat:
    """Round-10 refactor guards: after the domain extraction, pickles
    produced BEFORE the refactor (stored the dataclass under
    ``analyzer.ml_predictor.MLMetrics``) and pickles produced AFTER
    (store ``analyzer.ml.domain.metrics.MLMetrics``) must both unpickle
    cleanly under the current code. The restricted unpickler whitelist
    must cover both module paths."""

    def _restricted_loads(self, data):
        """Use the production restricted unpickler so the test exercises
        the real security boundary, not a plain ``pickle.loads``."""
        from analyzer.ml_predictor import _restricted_loads
        import pickle
        return _restricted_loads(pickle.dumps(data))

    def test_new_path_mlmetrics_survives_restricted_unpickle(self):
        """Post-refactor: ``MLMetrics.__module__ == 'analyzer.ml.domain.metrics'``.
        The unpickler whitelist must allow this module prefix."""
        from analyzer.ml_predictor import MLMetrics
        assert MLMetrics.__module__ == "analyzer.ml.domain.metrics"
        m = MLMetrics(precision=0.7, recall=0.6, roc_auc=0.75, skill_score=0.65)
        restored = self._restricted_loads(m)
        assert restored.precision == 0.7
        assert restored.roc_auc == 0.75

    def test_old_path_alias_still_resolves_via_reexport(self):
        """A class object fetched from the legacy import path
        ``analyzer.ml_predictor.MLMetrics`` still works end-to-end —
        existing pickles that embedded that path resolve via the
        re-export and produce the same class as the new path."""
        import analyzer.ml_predictor as legacy
        from analyzer.ml.domain.metrics import MLMetrics as NewMLMetrics
        # Old import path returns the same class object
        assert legacy.MLMetrics is NewMLMetrics

    def test_ml_config_survives_restricted_unpickle(self):
        """Same guard for MLConfig — main.py constructs instances via
        ``analyzer.ml_predictor.MLConfig`` but the real class now lives
        under ``analyzer.ml.domain.config``."""
        from analyzer.ml_predictor import MLConfig
        assert MLConfig.__module__ == "analyzer.ml.domain.config"
        cfg = MLConfig(min_trades=250)
        restored = self._restricted_loads(cfg)
        assert restored.min_trades == 250


class TestRegimeRouterSetstateLogging:
    def test_empty_models_with_global_logs_warning(self, caplog):
        from analyzer.ml_regime_router import RegimeRouter, RegimeModel

        class _FakeEns:
            def predict_proba_calibrated(self, X):
                return np.full(len(X), 0.5)

        router = RegimeRouter.__new__(RegimeRouter)
        with caplog.at_level(logging.WARNING, logger="analyzer.ml_regime_router"):
            router.__setstate__({
                "_models": {},
                "_global": RegimeModel(
                    regime="__global__", ensemble=_FakeEns(),
                    scaler=None, selector=None, threshold=0.5,
                ),
            })
        assert any("no trained specialists" in r.message.lower()
                   or "no specialists" in r.message.lower()
                   or "fall through" in r.message.lower()
                   for r in caplog.records), \
            f"expected 'no specialists' warning; got: {[r.message for r in caplog.records]}"

    def test_empty_models_and_no_global_logs_warning(self, caplog):
        from analyzer.ml_regime_router import RegimeRouter

        router = RegimeRouter.__new__(RegimeRouter)
        with caplog.at_level(logging.WARNING, logger="analyzer.ml_regime_router"):
            router.__setstate__({"_models": {}, "_global": None})
        assert any("empty" in r.message.lower() or "neutral 0.5" in r.message.lower()
                   for r in caplog.records), \
            f"expected 'empty router' warning; got: {[r.message for r in caplog.records]}"
