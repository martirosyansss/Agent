"""
Regression guards for the issues found in the two-round post-fix audit.

Each test here nails down an exact behavioural contract so the fix can't
silently re-regress. The audit called out that the original patches had
no regression guards; this file is that safety net. Tests are grouped
by the audit finding ID they protect.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


class _PicklableRegimeEnsemble:
    """Module-level stand-in ensemble for RegimeModel pickle tests. Must be
    importable by pickle.loads — nested test-local classes fail."""
    def __init__(self, p: float = 0.7) -> None:
        self._p = p

    def predict_proba_calibrated(self, X):
        return np.full(len(X), self._p)


class _PicklableOkMember:
    def __init__(self, p: float) -> None:
        self._p = p

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([1 - np.full(n, self._p), np.full(n, self._p)])


class _PicklableBrokenMember:
    def predict_proba(self, X):
        raise RuntimeError("simulated failure")


class _PicklableConstantEnsemble:
    """Ensemble stub — rendered module-level so regime-router tests that
    exercise predict() through a RegimeModel can survive pickle."""
    def __init__(self, p: float) -> None:
        self._p = p

    @property
    def is_ready(self) -> bool:
        return True

    def predict_proba_calibrated(self, X):
        return np.full(len(X), self._p)


# ──────────────────────────────────────────────────────────
# N-1: single WF run (not two) + per-member OOF populated
# Protects the C-2/M-7 fix.
# ──────────────────────────────────────────────────────────

class TestWalkForwardSingleRun:
    def test_fold_trainer_invoked_once_per_fold(self):
        """train_walk_forward must call the underlying trainer exactly once
        per completed fold — the old code ran it twice (once for the report,
        once for stacking OOF), doubling CPU."""
        from analyzer.ml_walk_forward import MLWalkForwardValidator

        call_count = {"n": 0}

        def trainer(X_tr, y_tr, X_te, y_te):
            call_count["n"] += 1
            rng = np.random.default_rng(1)
            return {
                "test_proba": rng.random(len(y_te)),
                "threshold": 0.5,
                "train_precision": 0.5,
                "member_probas": {"rf": rng.random(len(y_te))},
            }

        wf = MLWalkForwardValidator(n_folds=5, test_fraction=0.1,
                                    min_train_size=50, min_test_size=20)
        X = np.random.rand(500, 5)
        y = (np.arange(500) % 2).astype(int)
        report = wf.run(X, y, trainer)
        assert call_count["n"] == report.n_folds_completed
        assert call_count["n"] >= 3

    def test_wf_fold_result_stores_member_probas(self):
        """WFFoldResult.member_probas must round-trip whatever the trainer
        returned, keyed by tag, sized to the fold's test window."""
        from analyzer.ml_walk_forward import MLWalkForwardValidator

        def trainer(X_tr, y_tr, X_te, y_te):
            n = len(y_te)
            return {
                "test_proba": np.full(n, 0.5),
                "threshold": 0.5,
                "train_precision": 0.5,
                "member_probas": {
                    "rf": np.linspace(0, 1, n),
                    "lgbm": np.linspace(1, 0, n),
                },
            }

        wf = MLWalkForwardValidator(n_folds=3, test_fraction=0.1,
                                    min_train_size=50, min_test_size=20)
        X = np.random.rand(500, 5)
        y = (np.arange(500) % 2).astype(int)
        report = wf.run(X, y, trainer)
        for fr in report.fold_results:
            assert set(fr.member_probas.keys()) == {"rf", "lgbm"}
            assert len(fr.member_probas["rf"]) == fr.n_test
            assert len(fr.member_probas["lgbm"]) == fr.n_test

    def test_missing_member_probas_results_in_empty_dict(self):
        """Backwards compat: a trainer that doesn't return member_probas
        still produces a valid (empty) dict on each fold — downstream
        stacking just sees no tags and skips."""
        from analyzer.ml_walk_forward import MLWalkForwardValidator

        def trainer(X_tr, y_tr, X_te, y_te):
            return {"test_proba": np.full(len(y_te), 0.5), "threshold": 0.5,
                    "train_precision": 0.5}

        wf = MLWalkForwardValidator(n_folds=3, test_fraction=0.1,
                                    min_train_size=50, min_test_size=20)
        X = np.random.rand(500, 5)
        y = (np.arange(500) % 2).astype(int)
        report = wf.run(X, y, trainer)
        for fr in report.fold_results:
            assert fr.member_probas == {}


# ──────────────────────────────────────────────────────────
# N-2: save/load persists schema-v4 artifacts
# Protects the C-4 fix.
# ──────────────────────────────────────────────────────────

class TestSchemaV4Persistence:
    def _mk_loaded_predictor(self, tmp_path: Path):
        """Build a trained predictor using real sklearn models so save/load
        survives the restricted unpickler (which only whitelists sklearn /
        numpy / analyzer.* — not test-local stubs)."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from analyzer.ml_predictor import MLPredictor, MLConfig
        from analyzer.ml_ensemble import VotingEnsemble

        p = MLPredictor(MLConfig())
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, size=(60, 32))
        y = (X[:, 0] > 0).astype(int)
        scaler = StandardScaler().fit(X)
        rf = RandomForestClassifier(n_estimators=5, random_state=42).fit(scaler.transform(X), y)

        p._ensemble = VotingEnsemble()
        p._ensemble.add_member(rf, "rf", 0.7)
        p._model = rf
        p._scaler = scaler
        p._calibrated_threshold = 0.55
        p._model_version = "test_v1"

        from analyzer.ml_predictor import MLMetrics
        from analyzer.ml_walk_forward import WFReport
        p._metrics = MLMetrics(
            precision=0.7, recall=0.6, roc_auc=0.75, skill_score=0.65,
            train_samples=600, test_samples=120,
        )
        # Empty WFReport — whitelisted, pickleable, survives round-trip
        p._wf_report = WFReport(fold_results=[], mean_auc=0.7, std_auc=0.03,
                                 n_folds_completed=3, mode="rolling")
        p._bootstrap_ci = {"precision": {"metric": "precision", "mean": 0.7,
                                         "p5": 0.62, "p50": 0.7, "p95": 0.78,
                                         "std": 0.05, "n_samples": 100,
                                         "n_simulations": 1000}}
        p._member_error_correlation = {"rf__lgbm": 0.62}

        path = tmp_path / "model.pkl"
        assert p.save_to_file(path) is True

        p2 = MLPredictor(MLConfig())
        assert p2.load_from_file(path) is True
        return p2

    def test_wf_report_persists(self, tmp_path):
        p2 = self._mk_loaded_predictor(tmp_path)
        assert p2.walk_forward_report is not None

    def test_bootstrap_ci_persists(self, tmp_path):
        p2 = self._mk_loaded_predictor(tmp_path)
        assert "precision" in p2.bootstrap_ci

    def test_member_correlation_persists(self, tmp_path):
        p2 = self._mk_loaded_predictor(tmp_path)
        assert "rf__lgbm" in p2.member_error_correlation

    def test_old_schema_loads_without_new_fields(self, tmp_path):
        """A save produced by code *without* the schema-v4 fields must still
        load cleanly, *with the ensemble actually ready*. Previous test used
        ensemble=None which short-circuits load and never actually exercises
        the missing-v4-key code path. This version builds a real v3 payload
        with a trained ensemble so we exercise the .get() defaults for the
        new fields directly."""
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

        data = {
            "ensemble": ensemble,
            "feature_selector": AdaptiveFeatureSelector(),
            "calibrated_threshold": 0.5,
            "model": rf,
            "scaler": scaler,
            "version": "legacy_v3",
            "metrics": {"precision": 0.6, "recall": 0.55, "roc_auc": 0.7,
                        "skill_score": 0.5, "train_samples": 50, "test_samples": 10,
                        "feature_importances": {}},
            "saved_at": int(time.time()),
            "format": "v3",
            "schema_version": 3,
            # deliberately NO wf_report / regime_router / bootstrap_ci / member_error_correlation
        }
        payload = pickle.dumps(data)
        checksum = hashlib.sha256(payload).hexdigest()
        path = tmp_path / "legacy.pkl"
        with path.open("wb") as f:
            pickle.dump({"payload": payload, "checksum": checksum, "format": "v3_signed"}, f)

        p = MLPredictor(MLConfig())
        ok = p.load_from_file(path)
        assert ok is True
        assert p.is_ready is True
        # New v4 fields must default cleanly when missing
        assert p.walk_forward_report is None
        assert p.regime_router is None
        assert p.bootstrap_ci == {}
        assert p.member_error_correlation == {}


# ──────────────────────────────────────────────────────────
# N-3: per-regime threshold actually reaches the decision boundary
# Protects the M-4 fix at predict() level, not just router level.
# ──────────────────────────────────────────────────────────

class TestRegimeThresholdReachesDecision:
    def test_router_threshold_overrides_global_in_predict(self):
        """End-to-end: a specialist with a strict threshold (e.g. 0.80)
        must cause predict() to return 'block' for a prob below that bar,
        even when the global threshold would have said 'allow'."""
        from analyzer.ml_predictor import MLPredictor, MLConfig, N_FEATURES
        from analyzer.ml_regime_router import RegimeRouter, RegimeModel

        cfg = MLConfig(use_regime_routing=True, block_threshold=0.0)
        p = MLPredictor(cfg)
        p._rollout_mode = "block"
        p._calibrated_threshold = 0.30  # GLOBAL: permissive
        # Fake a global model so is_ready returns True even without router
        p._ensemble = _PicklableConstantEnsemble(0.7)
        p._scaler = None

        # Router with a strict specialist (threshold 0.80)
        router = RegimeRouter(min_trades_per_regime=20)
        router._models["trending_up"] = RegimeModel(
            regime="trending_up",
            ensemble=_PicklableConstantEnsemble(0.55),  # specialist returns 0.55
            scaler=None, selector=None,
            threshold=0.80,  # STRICT
        )
        router._global = RegimeModel(
            regime="__global__", ensemble=_PicklableConstantEnsemble(0.5),
            scaler=None, selector=None, threshold=0.30,
        )
        router._is_ready = True
        p._regime_router = router

        # Build a feature vector with regime_idx = 0 (trending_up)
        features = [0.0] * N_FEATURES
        features[11] = 0.0  # trending_up

        result = p.predict(features)
        # specialist probability = 0.55, specialist threshold = 0.80
        # 0.55 < 0.80 * 0.85 = 0.68  → block
        assert result.decision == "block", f"expected block, got {result.decision}"

    def test_no_router_falls_back_to_global_threshold(self):
        """Inverse: when router isn't active, predict() uses
        _calibrated_threshold as before (no regression on the default path)."""
        from analyzer.ml_predictor import MLPredictor, MLConfig, N_FEATURES

        cfg = MLConfig(use_regime_routing=False, block_threshold=0.0)
        p = MLPredictor(cfg)
        p._rollout_mode = "block"
        p._calibrated_threshold = 0.40
        p._ensemble = _PicklableConstantEnsemble(0.85)
        p._scaler = None

        features = [0.0] * N_FEATURES
        result = p.predict(features)
        # 0.85 > 0.40 → allow
        assert result.decision == "allow"


# ──────────────────────────────────────────────────────────
# J-4: StackingHead tolerates member failures at predict time
# ──────────────────────────────────────────────────────────

class TestStackingFallbackGuards:
    """C4-1 / MA5-1: the two fallback paths in _fit_stacking_head_from_report
    must stay on soft-voting when we can't safely recalibrate. Attaching the
    deploy head without a stacking-fit calibrator would silently route
    stacked output through the voting-distribution calibrator (J-1 repro).

    Both tests drive ``_fit_stacking_head_from_report`` directly with a
    hand-built WFReport so we can force each fallback without depending on
    stochastic walk-forward fold behaviour.
    """

    def _make_predictor_with_voting_calibrator(self):
        """Predictor with a pre-fit voting-distribution calibrator — the
        exact state ``train()`` leaves behind before stacking activates."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from analyzer.ml_predictor import MLPredictor, MLConfig
        from analyzer.ml_ensemble import VotingEnsemble

        cfg = MLConfig(
            min_trades=100, use_walk_forward=True, use_stacking=True,
            min_precision=0.0, min_recall=0.0, min_roc_auc=0.0, min_skill_score=0.0,
        )
        p = MLPredictor(cfg)
        rng = np.random.default_rng(42)
        Xs = rng.normal(0, 1, size=(80, 32))
        ys = (Xs[:, 0] > 0).astype(int)
        scaler = StandardScaler().fit(Xs)
        rf = RandomForestClassifier(n_estimators=5, random_state=42).fit(
            scaler.transform(Xs), ys
        )
        ens = VotingEnsemble()
        ens.add_member(rf, "rf", 0.7)
        # Pre-fit a voting-style calibrator so we can prove it's NOT
        # overwritten by the fallback branches.
        ens._fit_calibrator_on_probas(ys, rng.random(len(ys)), source="voting")
        p._ensemble = ens
        p._scaler = scaler
        p._calibrated_threshold = 0.42  # sentinel value to detect overwrites
        return p, Xs, ys

    def test_insufficient_oof_keeps_soft_voting_and_threshold(self):
        """When valid OOF rows < 60 we must: (a) NOT attach stacking head,
        (b) NOT overwrite the calibrator, (c) NOT overwrite the threshold."""
        from analyzer.ml_walk_forward import WFReport, WFFoldResult

        p, X, y = self._make_predictor_with_voting_calibrator()

        # Build a report whose member_probas cover < 60 rows total.
        fr = WFFoldResult(
            fold_idx=0, train_start=0, train_end=40,
            test_start=40, test_end=70,   # 30 rows < 60 threshold
            n_train=40, n_test=30,
            oof_probas=np.full(30, 0.5),
            y_test=y[40:70].copy(),
            member_probas={
                "rf": np.clip(y[40:70] * 0.8 + 0.1, 0, 1),
            },
        )
        report = WFReport(fold_results=[fr], mean_auc=0.7,
                          n_folds_completed=1, mode="rolling")

        pre_calibrator = p._ensemble._calibrator
        pre_threshold = p._calibrated_threshold
        p._fit_stacking_head_from_report(report, X=X, y=y, pnl=None)

        # Must fall back cleanly
        assert p._ensemble._stacking_head is None, \
            "fallback must NOT attach stacking head — otherwise J-1 repro"
        assert p._ensemble._calibrator is pre_calibrator, \
            "fallback must NOT replace voting-fit calibrator"
        assert p._calibrated_threshold == pre_threshold, \
            "fallback must NOT re-tune the decision threshold"

    def test_deploy_head_fit_failure_keeps_soft_voting(self, monkeypatch):
        """When the deploy head's StackingHead.fit fails outright, the earliest
        bail-out triggers (no CV attempted). All three invariants hold.

        Covers the `if not deploy_head.fit(...)` branch near the top of
        ``_fit_stacking_head_from_report``."""
        from analyzer.ml_stacking import StackingHead
        from analyzer.ml_walk_forward import WFReport, WFFoldResult

        p, X, y = self._make_predictor_with_voting_calibrator()

        n_oof = 80
        fr = WFFoldResult(
            fold_idx=0, train_start=0, train_end=10,
            test_start=10, test_end=10 + n_oof,
            n_train=10, n_test=n_oof,
            oof_probas=np.full(n_oof, 0.5),
            y_test=y[: n_oof].copy() if len(y) >= n_oof else y,
            member_probas={"rf": np.random.default_rng(1).random(n_oof)},
        )
        big_X = np.random.default_rng(2).normal(0, 1, size=(10 + n_oof, 32))
        big_y = np.concatenate([
            np.zeros(10 + n_oof // 2, dtype=np.int64),
            np.ones(n_oof - n_oof // 2, dtype=np.int64),
        ])
        report = WFReport(fold_results=[fr], mean_auc=0.7,
                          n_folds_completed=1, mode="rolling")

        # Every StackingHead.fit returns False → deploy head fails first.
        monkeypatch.setattr(StackingHead, "fit", lambda self, *a, **kw: False)
        pre_calibrator = p._ensemble._calibrator
        pre_threshold = p._calibrated_threshold
        p._fit_stacking_head_from_report(report, X=big_X, y=big_y, pnl=None)

        assert p._ensemble._stacking_head is None
        assert p._ensemble._calibrator is pre_calibrator
        assert p._calibrated_threshold == pre_threshold

    def test_subhead_cv_failure_keeps_soft_voting(self, monkeypatch):
        """When the DEPLOY head fits fine but one of the two 2-fold CV
        sub-heads fails, we must still fall back to soft-voting.

        Covers the `clean_on_A is None or clean_on_B is None` branch which
        the previous m6-2 test didn't actually reach."""
        from analyzer.ml_stacking import StackingHead
        from analyzer.ml_walk_forward import WFReport, WFFoldResult

        p, X, y = self._make_predictor_with_voting_calibrator()

        n_oof = 80
        rng = np.random.default_rng(3)
        fr = WFFoldResult(
            fold_idx=0, train_start=0, train_end=10,
            test_start=10, test_end=10 + n_oof,
            n_train=10, n_test=n_oof,
            oof_probas=rng.random(n_oof),
            y_test=(np.arange(n_oof) % 2).astype(np.int64),
            member_probas={
                "rf": rng.random(n_oof),
                "lgbm": rng.random(n_oof),
            },
        )
        big_X = rng.normal(0, 1, size=(10 + n_oof, 32))
        big_y = np.concatenate([
            np.zeros(10, dtype=np.int64),
            (np.arange(n_oof) % 2).astype(np.int64),
        ])
        report = WFReport(fold_results=[fr], mean_auc=0.7,
                          n_folds_completed=1, mode="rolling")

        # Counter that lets the first fit (deploy head) succeed but fails
        # every sub-head fit thereafter — this is the exact shape of the
        # sub-head-CV failure branch.
        call_count = {"n": 0}
        orig_fit = StackingHead.fit

        def _selective_fit(self, *args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return orig_fit(self, *args, **kwargs)
            return False

        monkeypatch.setattr(StackingHead, "fit", _selective_fit)
        pre_calibrator = p._ensemble._calibrator
        pre_threshold = p._calibrated_threshold
        p._fit_stacking_head_from_report(report, X=big_X, y=big_y, pnl=None)

        assert call_count["n"] >= 2, "sub-head fits must have been attempted"
        assert p._ensemble._stacking_head is None
        assert p._ensemble._calibrator is pre_calibrator
        assert p._calibrated_threshold == pre_threshold

    def test_calibrator_returns_false_on_single_class_labels(self):
        """Integration-style guard (round-7 #1): drive the real
        ``_fit_calibrator_on_probas`` into its single-class silent-skip
        branch with actual inputs — no monkeypatch. Verifies that the
        ``return False`` contract is honoured by the production code, not
        just by the test's own stub."""
        from analyzer.ml_ensemble import VotingEnsemble

        e = VotingEnsemble()
        # Install a known-good calibrator first so we can detect accidental
        # overwrites. The source tag doesn't matter for this test.
        ok = e._fit_calibrator_on_probas(
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
            np.array([0.1, 0.9, 0.2, 0.8, 0.15, 0.85, 0.1, 0.92, 0.22, 0.78, 0.3, 0.7]),
            source="voting",
        )
        assert ok is True
        pre_calibrator = e._calibrator
        assert pre_calibrator is not None

        # Now try to "refit" with single-class labels — should return False
        # and leave the existing calibrator untouched.
        single_class_y = np.zeros(50, dtype=np.int64)
        ret = e._fit_calibrator_on_probas(
            single_class_y,
            np.random.default_rng(1).random(50),
            source="stacking",
        )
        assert ret is False
        assert e._calibrator is pre_calibrator, \
            "voting-fit calibrator must survive a failed stacking refit"

    def test_calibrator_returns_false_on_too_few_samples(self):
        """Same integration guard for the n<10 branch."""
        from analyzer.ml_ensemble import VotingEnsemble

        e = VotingEnsemble()
        ret = e._fit_calibrator_on_probas(
            np.array([0, 1, 0]), np.array([0.3, 0.7, 0.5]), source="stacking",
        )
        assert ret is False
        assert e._calibrator is None

    def test_calibrator_refit_silent_skip_keeps_soft_voting(self, monkeypatch):
        """M6-1 guard: if _fit_calibrator_on_probas silently skips (e.g. OOF
        labels ended up single-class), we must NOT attach the deploy head.

        Without this guard the deploy head would be attached while the
        calibrator still held the voting-distribution fit from train() —
        which is exactly the J-1 bug these rounds have been killing."""
        from analyzer.ml_ensemble import VotingEnsemble
        from analyzer.ml_walk_forward import WFReport, WFFoldResult

        p, X, y = self._make_predictor_with_voting_calibrator()

        n_oof = 80
        rng = np.random.default_rng(4)
        fr = WFFoldResult(
            fold_idx=0, train_start=0, train_end=10,
            test_start=10, test_end=10 + n_oof,
            n_train=10, n_test=n_oof,
            oof_probas=rng.random(n_oof),
            y_test=(np.arange(n_oof) % 2).astype(np.int64),
            member_probas={"rf": rng.random(n_oof), "lgbm": rng.random(n_oof)},
        )
        big_X = rng.normal(0, 1, size=(10 + n_oof, 32))
        big_y = np.concatenate([
            np.zeros(10, dtype=np.int64),
            (np.arange(n_oof) % 2).astype(np.int64),
        ])
        report = WFReport(fold_results=[fr], mean_auc=0.7,
                          n_folds_completed=1, mode="rolling")

        # Force calibrator refit to silently return False
        monkeypatch.setattr(VotingEnsemble, "_fit_calibrator_on_probas",
                            lambda self, y, probas, source="voting": False)

        pre_calibrator = p._ensemble._calibrator
        pre_threshold = p._calibrated_threshold
        p._fit_stacking_head_from_report(report, X=big_X, y=big_y, pnl=None)

        assert p._ensemble._stacking_head is None, \
            "attaching head with stale calibrator would repro J-1"
        assert p._ensemble._calibrator is pre_calibrator
        assert p._calibrated_threshold == pre_threshold


class TestLegacyPickleCompat:
    """m6-3: verify `RegimeRouter.__setstate__` tolerates old pickles.

    Pickles saved before round-5's setdefault guards won't carry
    ``_stats`` / ``_is_ready`` / ``min_trades_per_regime``. The new
    ``__setstate__`` fills them with safe defaults; this test simulates
    an old save by serialising a dict missing those fields and confirms
    unpickle completes without AttributeError on subsequent attribute
    access.
    """

    def test_pre_setstate_pickle_unmarshalls_cleanly(self):
        import pickle
        from analyzer.ml_regime_router import RegimeRouter

        router = RegimeRouter.__new__(RegimeRouter)
        # Minimal pre-round-5 state — only the fields that existed at the
        # time the router was designed. Crucially missing: _stats,
        # _is_ready, min_trades_per_regime.
        router.__setstate__({
            "_models": {},
            "_global": None,
        })
        # All attributes that newer code reads must now be present
        assert router._stats == {}
        assert router._models == {}
        assert router._global is None
        assert router._is_ready is False
        assert router.min_trades_per_regime == 100
        # Public API should not raise
        _ = router.is_ready
        _ = router.trained_regimes
        _ = router.get_regime_stats()

    def test_pickle_round_trip_preserves_state(self):
        """Round-7 #3: actually run the full pickle.dumps/pickle.loads cycle
        (not just __setstate__ direct). Catches interactions between the
        setstate hook and pickle's __reduce__ / __getstate__ defaults that
        the direct-call test can't detect."""
        import pickle
        from analyzer.ml_regime_router import RegimeRouter, RegimeModel

        router = RegimeRouter(min_trades_per_regime=50)
        router._global = RegimeModel(
            regime="__global__", ensemble=_PicklableRegimeEnsemble(),
            scaler=None, selector=None, threshold=0.5,
        )
        router._is_ready = True

        # Full round-trip via Python pickle (not the restricted unpickler,
        # which would reject the _FakeEnsemble test stub — that's tested
        # separately for the real save/load path).
        blob = pickle.dumps(router)
        restored = pickle.loads(blob)

        assert restored.min_trades_per_regime == 50
        assert restored._models == {}
        assert restored._global is not None
        assert restored.is_ready is True
        assert restored.get_regime_stats() == {}

    def test_pickle_round_trip_on_new_format_router(self):
        """A router saved by current code must also round-trip cleanly —
        regression guard against the setstate guards accidentally dropping
        fields that do exist."""
        import pickle
        from analyzer.ml_regime_router import RegimeRouter, RegimeStats

        router = RegimeRouter(min_trades_per_regime=100)
        router._stats["trending_up"] = RegimeStats(
            regime="trending_up", n_trades=200, trained=True,
            skill_score=0.8, mean_precision=0.75, mean_auc=0.82,
        )

        blob = pickle.dumps(router)
        restored = pickle.loads(blob)

        assert "trending_up" in restored._stats
        assert restored._stats["trending_up"].trained is True
        assert restored._stats["trending_up"].skill_score == 0.8


class TestLengthDriftGuard:
    """m6-4: verify train_walk_forward raises ValueError when pnl_arr drifts
    from X/y length — replaces the (now-removed) assert which would have
    been stripped under `python -O`."""

    def test_mismatched_pnl_raises_value_error(self, monkeypatch, synthetic_trades):
        """Directly induce the length drift that the guard at
        ``train_walk_forward`` is meant to catch.

        Earlier attempts monkey-patched ``np.array`` globally; that was
        flaky because ``extract_features_batch`` and other helpers call
        ``np.array`` many times, and the first match consumed the tamper
        state before the real pnl construction ran. Instead, we let the
        training pipeline run normally and patch ``extract_features_batch``
        to return an array whose row count is one less than the trade
        list — which is exactly the invariant the guard checks.
        """
        from analyzer.ml_predictor import MLPredictor, MLConfig

        cfg = MLConfig(
            min_trades=100, use_walk_forward=True, use_stacking=True,
            min_precision=0.0, min_recall=0.0, min_roc_auc=0.0, min_skill_score=0.0,
        )
        p = MLPredictor(cfg)
        p.train(synthetic_trades)
        if not p.is_ready:
            pytest.skip("train could not produce a deployable ensemble on synthetic data")

        real_extract = p.extract_features_batch

        def _short_extract(trades):
            # Drop one row so len(X) != len(trades) = len(pnl_arr)
            X = real_extract(trades)
            return X[:-1]

        monkeypatch.setattr(p, "extract_features_batch", _short_extract)

        with pytest.raises(ValueError, match="length drift"):
            p.train_walk_forward(synthetic_trades, n_folds=3)


class TestRegimeRoutingE2E:
    """Round-7 #4: end-to-end coverage for train_with_regime_routing.

    The method was previously only covered by the unit tests on its
    pieces (RegimeRouter._models dict, RegimeModel dataclass, etc.)
    and on the downstream predict() path. Nothing drove the full
    train()-then-partition-then-sub-train flow with realistic data.

    These tests use small synthetic datasets so they run in a second
    each; the goal is path coverage, not metric quality.
    """

    def test_under_min_trades_warns_but_does_not_crash(self, caplog, synthetic_trades_factory):
        """J-3 warning path: when effective min_trades < 80, operators
        see an upfront log instead of a silent specialist-training failure."""
        import logging
        from analyzer.ml_predictor import MLPredictor, MLConfig

        cfg = MLConfig(
            min_trades=50,
            min_trades_per_regime=50,  # < 80 threshold
            use_regime_routing=True,
            min_precision=0.0, min_recall=0.0, min_roc_auc=0.0, min_skill_score=0.0,
        )
        trades = synthetic_trades_factory(n=150, seed=11)
        p = MLPredictor(cfg)
        with caplog.at_level(logging.WARNING, logger="analyzer.ml_predictor"):
            p.train_with_regime_routing(trades)
        # The upfront warning must fire — operators are meant to see this
        # before they wonder why regime routing isn't helping.
        assert any("effective min_trades=" in r.message for r in caplog.records), \
            "J-3 warning should fire when effective_min < 80"

    def test_returns_none_when_global_train_fails(self, synthetic_trades_factory):
        """When train() can't produce a deployable model, the method must
        return None instead of attaching a half-built router — the router's
        predict path assumes a global fallback exists."""
        from analyzer.ml_predictor import MLPredictor, MLConfig

        # Impossibly strict thresholds so train() returns metrics but never
        # deploys.
        cfg = MLConfig(
            min_trades=50,
            use_regime_routing=True,
            min_precision=0.99, min_recall=0.99, min_roc_auc=0.99, min_skill_score=0.99,
        )
        trades = synthetic_trades_factory(n=200, seed=12)
        p = MLPredictor(cfg)
        out = p.train_with_regime_routing(trades)
        assert out is None
        assert p.regime_router is None

    def test_flag_off_is_noop(self, synthetic_trades_factory):
        """With use_regime_routing=False the method must short-circuit
        without running train() at all — otherwise we're paying the cost
        of a full global training on the guaranteed-unused path."""
        from analyzer.ml_predictor import MLPredictor, MLConfig

        cfg = MLConfig(
            min_trades=50, use_regime_routing=False,
            min_precision=0.0, min_recall=0.0, min_roc_auc=0.0, min_skill_score=0.0,
        )
        trades = synthetic_trades_factory(n=150, seed=13)
        p = MLPredictor(cfg)
        called = {"train": 0}
        orig_train = p.train
        def _counting_train(ts):
            called["train"] += 1
            return orig_train(ts)
        p.train = _counting_train
        out = p.train_with_regime_routing(trades)
        assert out is None
        assert called["train"] == 0
        assert p.regime_router is None


class TestRound8Fixes:
    """Round-8 deep-audit regression guards. These check cross-cutting
    contracts that unit tests on individual modules can't see."""

    def test_save_to_file_rejects_path_traversal(self, tmp_path):
        """§6.1 security: `..` segments in model_path must be refused even
        for otherwise-ready predictors. Prevents arbitrary-file-write if a
        future interface propagates user input into save_to_file."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from analyzer.ml_predictor import MLPredictor, MLConfig, MLMetrics
        from analyzer.ml_ensemble import VotingEnsemble

        p = MLPredictor(MLConfig())
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, size=(60, 32))
        y = (X[:, 0] > 0).astype(int)
        scaler = StandardScaler().fit(X)
        rf = RandomForestClassifier(n_estimators=5, random_state=42).fit(scaler.transform(X), y)
        p._ensemble = VotingEnsemble()
        p._ensemble.add_member(rf, "rf", 0.7)
        p._model = rf
        p._scaler = scaler
        p._model_version = "test_v1"
        p._metrics = MLMetrics(precision=0.7, recall=0.6, roc_auc=0.75,
                               skill_score=0.65, train_samples=60, test_samples=10)

        # Traversal attempt via `..` segment must be refused
        bad_path = tmp_path / "sub" / ".." / "model.pkl"
        assert p.save_to_file(bad_path) is False

        # Clean absolute path stays allowed — tests using tmp_path must work
        good_path = tmp_path / "model.pkl"
        assert p.save_to_file(good_path) is True

    def test_mlbootstrap_reproducible_across_call_order(self):
        """§3.3: calling bootstrap_metrics then probability_above_baseline
        must produce the same numbers as the reverse order. A shared RNG
        would make the two orderings diverge."""
        from analyzer.ml_bootstrap import MLBootstrap

        rng = np.random.default_rng(7)
        y = rng.integers(0, 2, size=300)
        p = rng.random(300)

        # Two fresh bootstraps, same seed, different call orders
        bs1 = MLBootstrap(n_simulations=200, seed=7)
        cis_first = bs1.bootstrap_metrics(y, p)
        prob_first = bs1.probability_above_baseline(y, p, baseline_auc=0.5)

        bs2 = MLBootstrap(n_simulations=200, seed=7)
        prob_second = bs2.probability_above_baseline(y, p, baseline_auc=0.5)
        cis_second = bs2.bootstrap_metrics(y, p)

        assert prob_first == prob_second, \
            f"probability_above_baseline differs by call order: {prob_first} vs {prob_second}"
        for metric in ("precision", "recall", "roc_auc"):
            assert cis_first[metric].p50 == cis_second[metric].p50, \
                f"{metric} p50 differs by call order"

    def test_rollout_mode_survives_pickle_round_trip(self, tmp_path):
        """§4.4: an auto-promoted ``rollout_mode=block`` must survive save
        and reload. Previously the field wasn't persisted, so bot restart
        silently reverted to the env default and operators wondered why
        their promotion didn't stick."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from analyzer.ml_predictor import MLPredictor, MLConfig, MLMetrics
        from analyzer.ml_ensemble import VotingEnsemble

        p = MLPredictor(MLConfig())
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, size=(60, 32))
        y = (X[:, 0] > 0).astype(int)
        scaler = StandardScaler().fit(X)
        rf = RandomForestClassifier(n_estimators=5, random_state=42).fit(scaler.transform(X), y)
        p._ensemble = VotingEnsemble()
        p._ensemble.add_member(rf, "rf", 0.7)
        p._model = rf
        p._scaler = scaler
        p._model_version = "test_v1"
        p._metrics = MLMetrics(precision=0.7, recall=0.6, roc_auc=0.75,
                               skill_score=0.65, train_samples=60, test_samples=10)
        p.rollout_mode = "block"  # simulate auto-promote

        path = tmp_path / "model.pkl"
        assert p.save_to_file(path) is True

        p2 = MLPredictor(MLConfig())
        p2.rollout_mode = "shadow"  # what env would default to
        assert p2.load_from_file(path) is True
        # Pickled value wins — operator's auto-promoted mode survives
        assert p2.rollout_mode == "block"

    def test_stacking_rejects_worse_than_voting(self, monkeypatch):
        """§3.2: when the stacking head's OOF AUC is worse than the simple
        voting mean, we must NOT attach it. Otherwise the meta-model
        replaces the simpler, better performing combiner on every predict."""
        from analyzer.ml_predictor import MLPredictor, MLConfig
        from analyzer.ml_ensemble import VotingEnsemble
        from analyzer.ml_walk_forward import WFReport, WFFoldResult

        p = MLPredictor(MLConfig(use_walk_forward=True, use_stacking=True))
        p._ensemble = VotingEnsemble()
        # Add a no-op member so the voting path has something to predict
        class _Dummy:
            def predict_proba(self, X):
                return np.column_stack([np.zeros(len(X)), np.ones(len(X)) * 0.5])
        p._ensemble.add_member(_Dummy(), "rf", 1.0)

        pre_calibrator = p._ensemble._calibrator  # None
        pre_threshold = p._calibrated_threshold

        # Synthesise OOF where voting is perfect and stacking (random) loses
        n = 200
        rng = np.random.default_rng(1)
        y_labels = (rng.integers(0, 2, size=n)).astype(np.int64)
        # Voting mean = member's signal = near-perfect
        perfect_signal = y_labels * 0.85 + 0.075
        fr = WFFoldResult(
            fold_idx=0, train_start=0, train_end=10,
            test_start=10, test_end=10 + n,
            n_train=10, n_test=n,
            oof_probas=perfect_signal.copy(),
            y_test=y_labels.copy(),
            member_probas={
                "rf": perfect_signal.copy(),
                "lgbm": perfect_signal.copy(),
            },
        )
        X = rng.normal(0, 1, size=(10 + n, 32))
        y_full = np.concatenate([
            np.zeros(10, dtype=np.int64),
            y_labels,
        ])
        report = WFReport(fold_results=[fr], mean_auc=0.95,
                          n_folds_completed=1, mode="rolling")

        # Make the stacking head predict pure noise → stacked_auc ≈ 0.5,
        # voting_auc ≈ 1.0, so stacking should be rejected.
        from analyzer.ml_stacking import StackingHead
        def _noise_predict_proba(self, member_probas, X=None):
            return np.random.default_rng(999).random(len(member_probas))
        monkeypatch.setattr(StackingHead, "predict_proba", _noise_predict_proba)

        p._fit_stacking_head_from_report(report, X=X, y=y_full, pnl=None)

        # Regression guard fires — no head attached, calibrator unchanged
        assert p._ensemble._stacking_head is None
        assert p._ensemble._calibrator is pre_calibrator
        assert p._calibrated_threshold == pre_threshold


class TestMemberFailureFallback:
    def test_failed_member_replaced_by_consensus_not_naive_half(self):
        """When a member raises, _member_probas_matrix must fill its column
        with the row-wise mean of successful members — NOT a blanket 0.5,
        which collides with genuine uncertainty at predict time."""
        from analyzer.ml_ensemble import VotingEnsemble

        e = VotingEnsemble()
        e.add_member(_PicklableOkMember(0.8), "rf", 0.7)
        e.add_member(_PicklableOkMember(0.7), "lgbm", 0.7)
        e.add_member(_PicklableBrokenMember(), "xgb", 0.7)

        X = np.zeros((5, 3))
        matrix, tags = e._member_probas_matrix(X)
        assert tags == ["rf", "lgbm", "xgb"]
        # xgb column should equal the mean of rf/lgbm (0.75), not 0.5
        assert np.allclose(matrix[:, 2], 0.75), f"got {matrix[:, 2]}"
