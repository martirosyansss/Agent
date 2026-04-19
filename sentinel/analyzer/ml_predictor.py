"""
Trade Analyzer Level 3 — ML Predictor façade (Triple-Engine Ensemble, v3).

Round-10 refactor status: this module is a **façade** over the
``analyzer.ml`` subpackage. Historically it was a 2800-LOC monolith;
after the refactor, most concerns have been extracted and only the
``MLPredictor`` class remains here as the stateful orchestrator that
composes the ML pipeline:

    analyzer.ml/
    ├── domain/         — MLConfig, MLMetrics, MLPrediction, constants, scoring
    ├── features/       — StrategyTrade → feature matrix
    ├── models/         — RF / LGBM / XGB / ElasticNet builders
    ├── training/       — calibration, stacking fit, walk-forward, regime routing
    ├── prediction/     — feature vector → MLPrediction
    └── persistence/    — pickle codec + signed envelope + version registry

Every class and helper that used to be defined here is still importable
from this module's namespace (``from analyzer.ml_predictor import MLConfig``
and friends) — the re-exports below guarantee that, and the pickle-
unpickler whitelist covers both the legacy and new module paths so
saved models from before the refactor keep loading.

Round-10 Step 9: the central training pipeline (phase-1 build, phase-2
feature-selection retrain, phase-B precision-recovery retrain,
bootstrap CI, OOT validation, calibration, skill gate) was extracted
into ``analyzer/ml/training/trainer.py::run_training``. The method on
this class is now a 15-line wrapper that delegates to the free
function while still owning all state mutations on ``self``.

VotingEnsemble: RF + LightGBM + XGBoost + ElasticNet soft-voting
(round-3 added ElasticNet as the decorrelated 4th member). ML ONLY
filters (block/reduce), NEVER initiates trades.

Rollout modes: off → shadow → block
- shadow: logs predictions, never blocks
- block:  actively blocks signals with low probability

32 features (all pre-trade, no forward-looking bias) — canonical list
in ``analyzer.ml.domain.constants.FEATURE_NAMES``.

Skill score = 0.30*precision + 0.10*recall + 0.35*roc_auc + 0.25*profit_factor
(precision weighted 3x recall — filter-mode: false positives > false negatives).
"""

from __future__ import annotations

import logging
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from core.models import StrategyTrade

logger = logging.getLogger(__name__)

# Round-10 Step 1 refactor: pure-data types and constants now live in
# ``analyzer.ml.domain.*``. Re-exported here so every existing caller
# (``from analyzer.ml_predictor import MLConfig`` etc.) keeps working
# — and so pickles that reference ``analyzer.ml_predictor.MLMetrics``
# still resolve via this module's namespace.
from analyzer.ml.domain.config import MLConfig  # noqa: E402
from analyzer.ml.domain.metrics import (  # noqa: E402
    MLMetrics,
    MLPrediction,
    LivePerformanceTracker,
)
from analyzer.ml.domain.scoring import (  # noqa: E402
    compute_skill_score,
    wilson_lower_bound,
)
from analyzer.ml.domain.versions import capture_package_versions as _capture_package_versions  # noqa: E402
from analyzer.ml.domain.constants import (  # noqa: E402
    _TEMPORAL_DECAY,
    N_FEATURES,
    REGIME_ENCODING,
    STRATEGY_REGIME_FIT,
    FEATURE_NAMES,
    _SKILL_W_PRECISION,
    _SKILL_W_RECALL,
    _SKILL_W_ROC_AUC,
    _SKILL_W_PROFIT_FACTOR,
)


# Round-10 Step 6: pickle codec moved to ``analyzer.ml.persistence.codec``.
# These module-level names stay as aliases so the existing restricted-
# unpickler tests in ``test_ml_predictor.py`` (which reach into
# ``analyzer.ml_predictor._RestrictedUnpickler`` directly) continue
# working without edits.
# ``_RestrictedUnpickler`` / ``_PICKLE_ALLOWED_PREFIXES`` / ``_restricted_loads``
# are reached into by ``tests/test_ml_predictor.py::TestRestrictedUnpickler``
# (the tests import from this module, not from ``ml/persistence/codec``).
# They look unused here but the test contract requires them to stay.
from analyzer.ml.persistence.codec import (  # noqa: E402, F401
    RestrictedUnpickler as _RestrictedUnpickler,
    PICKLE_ALLOWED_PREFIXES as _PICKLE_ALLOWED_PREFIXES,
    restricted_loads as _restricted_loads,
)


# Why so many re-imports above? Round-10 extraction moved every pure-data
# type (MLConfig, MLMetrics, MLPrediction, LivePerformanceTracker) and
# every module-level constant (N_FEATURES, REGIME_ENCODING,
# STRATEGY_REGIME_FIT, FEATURE_NAMES, _SKILL_W_*, _TEMPORAL_DECAY) plus
# the scoring / versions helpers into ``analyzer.ml.domain.*``. Callers
# still ``from analyzer.ml_predictor import MLConfig`` and pickles still
# reference ``analyzer.ml_predictor.MLMetrics``; the re-exports keep both
# paths resolving to the same class objects.


class MLPredictor:
    """Level 3 Trade Analyzer — ML фильтрация сигналов (VotingEnsemble v3)."""

    def __init__(self, config: MLConfig | None = None) -> None:
        from analyzer.ml_ensemble import VotingEnsemble, AdaptiveFeatureSelector
        self._cfg = config or MLConfig()
        self._model: Any = None          # Legacy: kept for load_from_file compat
        self._ensemble: Optional[VotingEnsemble] = None  # v3: primary predictor
        self._scaler: Any = None
        self._feature_selector: AdaptiveFeatureSelector = AdaptiveFeatureSelector(min_importance=0.01)
        self._model_version: str = ""
        self._metrics: Optional[MLMetrics] = None
        self._rollout_mode: str = "off"  # off, shadow, block
        self._calibrated_threshold: float = 0.5  # W-5: proper init
        self._last_train_ts: int = 0
        self._live_tracker: LivePerformanceTracker = LivePerformanceTracker(
            drift_threshold=self._cfg.drift_threshold,
        )
        # Phase-1/2/4/5 artifacts — populated only when the relevant feature
        # flags are on. Kept as Optional[Any] to avoid importing modules that
        # aren't used in the default path (keeps cold-start time down).
        self._wf_report: Optional[Any] = None           # WFReport from walk-forward
        self._regime_router: Optional[Any] = None        # RegimeRouter
        self._bootstrap_ci: dict[str, Any] = {}          # {metric: BootstrapCI.summary()}
        self._member_error_correlation: dict[str, float] = {}  # {"rf__lgbm": 0.67, ...}
        # Feature-drift monitor — populated by trainer.run_training after
        # the reference is fit, then exercised on every predict() call to
        # accumulate live samples. Optional to keep cold-start cheap when
        # the predictor is constructed only to hold loaded artifacts.
        self._feature_drift_monitor: Optional[Any] = None

    @property
    def is_ready(self) -> bool:
        # v3: ensemble-first, fallback to legacy model
        return (self._ensemble is not None and self._ensemble.is_ready) or (self._model is not None)

    @property
    def rollout_mode(self) -> str:
        return self._rollout_mode

    @rollout_mode.setter
    def rollout_mode(self, mode: str) -> None:
        if mode in ("off", "shadow", "block"):
            self._rollout_mode = mode

    @property
    def metrics(self) -> Optional[MLMetrics]:
        return self._metrics

    @property
    def walk_forward_report(self) -> Optional[Any]:
        """Most recent walk-forward evaluation result, or None if never run.

        See ``analyzer.ml_walk_forward.WFReport``. Populated only by
        ``train_walk_forward()``; regular ``train()`` leaves this unset.
        """
        return self._wf_report

    @property
    def regime_router(self) -> Optional[Any]:
        """The per-regime router (RegimeRouter) when regime routing is enabled."""
        return self._regime_router

    @property
    def bootstrap_ci(self) -> dict[str, Any]:
        """Bootstrap CI summaries from the most recent training run.

        Empty dict unless ``MLConfig.use_bootstrap_ci`` was on during training.
        Keys: ``precision``, ``recall``, ``roc_auc`` each mapping to a
        BootstrapCI.summary() dict; plus ``probability_above_random``.
        """
        return dict(self._bootstrap_ci)

    @property
    def member_error_correlation(self) -> dict[str, float]:
        """Pairwise Pearson correlation of ensemble members' holdout errors.

        Keys formatted ``"tag_a__tag_b"``. Values near 1.0 mean the two
        members agree on their mistakes (redundant); near 0 means they
        diversify; negative means they disagree (rare, positive signal).
        """
        return dict(self._member_error_correlation)

    # ──────────────────────────────────────────────────────────
    # Dashboard-facing public surface — M-5 fix: these expose the
    # values the dashboard endpoints used to read off private attrs
    # directly. Keeping a stable public API here means internal
    # refactors (e.g. renaming _model_version, moving config fields)
    # don't silently break the dashboard.
    # ──────────────────────────────────────────────────────────

    @property
    def model_version(self) -> str:
        """Opaque identifier of the currently-deployed ensemble."""
        return self._model_version or ""

    @property
    def block_threshold(self) -> float:
        """Environment-level floor for the block decision (env-configurable
        via ``ANALYZER_ML_BLOCK_THRESHOLD``). Production predict takes
        ``max(calibrated_threshold, block_threshold)``."""
        return float(self._cfg.block_threshold)

    @property
    def reduce_threshold(self) -> float:
        """Threshold below which probabilistic 'reduce' (rather than 'block')
        is returned. Currently informational — the actual reduce zone is
        computed as ``calibrated_threshold * reduce_margin`` at predict time."""
        return float(getattr(self._cfg, "reduce_threshold", 0.65))

    @property
    def calibrated_threshold(self) -> float:
        """The threshold learned during training, as distinct from the env
        floor. Exposed so dashboards can visualise both axes."""
        return float(self._calibrated_threshold)

    @property
    def drift_detected(self) -> bool:
        """True when live precision has drifted >12% below training precision."""
        if self._metrics is None:
            return False
        return self._live_tracker.is_drifting(self._metrics.precision)

    @property
    def live_metrics(self) -> dict:
        """Rolling live performance metrics from actual trade outcomes."""
        return self._live_tracker.live_metrics()

    def record_outcome(self, predicted_prob: float, actual_win: bool) -> None:
        """Record a live/paper trade outcome for concept drift monitoring.

        Call this after each trade closes with the ML probability that was
        predicted at entry time and whether the trade was actually profitable.
        """
        self._live_tracker.record(predicted_prob, actual_win)
        m = self._live_tracker.live_metrics()
        if "live_precision" in m and m["n"] >= 20 and m["n"] % 10 == 0:
            logger.info(
                "Live ML tracker (n=%d): precision=%.3f win_rate=%.3f auc=%.3f calib_err=%.3f%s",
                m["n"], m["live_precision"], m["live_win_rate"], m["live_auc"],
                m["calibration_error"],
                " ⚠ DRIFT DETECTED" if self.drift_detected else "",
            )

    @staticmethod
    def _parse_trade_timestamp(value: str) -> Optional[datetime]:
        if not value:
            return None
        normalized = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            return None

    # Round-10 Step 2: feature extraction moved to
    # ``analyzer.ml.features.extractor``. Wrappers below keep the old
    # method surface on MLPredictor so callers (``train``, ``predict``,
    # ``main.py``'s hot path at line 1761) don't need to change.

    @staticmethod
    def _regime_bias(regime: str) -> float:
        from analyzer.ml.features.extractor import regime_bias
        return regime_bias(regime)

    @staticmethod
    def _strategy_regime_fit(strategy_name: str, regime: str) -> float:
        from analyzer.ml.features.extractor import strategy_regime_fit
        return strategy_regime_fit(strategy_name, regime)

    def extract_features(
        self,
        trade: StrategyTrade,
        previous_trades: Optional[list[StrategyTrade]] = None,
    ) -> list[float]:
        from analyzer.ml.features.extractor import extract_features
        return extract_features(trade, previous_trades)

    def _build_rf(self, conservative: bool = False):
        """Build a RandomForest classifier.

        Round-10 Step 3 refactor: real implementation now lives in
        ``analyzer.ml.models.factories.build_rf`` as a free function
        taking the config explicitly. This method stays as a thin
        compatibility wrapper so internal callers (``train``, phase-2
        retrain, precision recovery) keep working verbatim.
        """
        from analyzer.ml.models.factories import build_rf
        return build_rf(self._cfg, conservative=conservative)

    def _build_lgbm(self, scale_pos_weight: float = 1.0, conservative: bool = False):
        """Compatibility wrapper — delegates to ``build_lgbm(cfg, ...)``."""
        from analyzer.ml.models.factories import build_lgbm
        return build_lgbm(self._cfg, scale_pos_weight=scale_pos_weight, conservative=conservative)

    def _build_elastic_net(self, conservative: bool = False):
        """Compatibility wrapper — delegates to ``build_elastic_net(cfg, ...)``."""
        from analyzer.ml.models.factories import build_elastic_net
        return build_elastic_net(self._cfg, conservative=conservative)

    def _build_xgb(self, scale_pos_weight: float = 1.0, conservative: bool = False):
        """Compatibility wrapper — delegates to ``build_xgb(cfg, ...)``."""
        from analyzer.ml.models.factories import build_xgb
        return build_xgb(self._cfg, scale_pos_weight=scale_pos_weight, conservative=conservative)

    # Round-10 Step 4 refactor: five static methods below are now thin
    # wrappers around pure functions in ``analyzer.ml.training.calibration``.
    # Keeping the method names on MLPredictor lets the rest of this class
    # call them exactly as before. The real arithmetic lives in one file
    # and is independently unit-testable.

    @staticmethod
    def _calibrate_threshold(
        y_true, y_proba, min_precision: float = 0.55,
        pnl: np.ndarray = None, min_recall: float = 0.30,
    ) -> float:
        from analyzer.ml.training.calibration import calibrate_threshold
        return calibrate_threshold(
            y_true, y_proba,
            min_precision=min_precision, pnl=pnl, min_recall=min_recall,
        )

    @staticmethod
    def _compute_profit_factor_score(y_pred, pnl_values) -> float:
        from analyzer.ml.training.calibration import compute_profit_factor_score
        return compute_profit_factor_score(y_pred, pnl_values)

    @staticmethod
    def _overfit_noise_margin(
        p_train: float, p_val: float, n_train: int, n_val: int,
        z: float = 1.96, n_tests: int = 1,
    ) -> float:
        from analyzer.ml.training.calibration import overfit_noise_margin
        return overfit_noise_margin(p_train, p_val, n_train, n_val, z=z, n_tests=n_tests)

    @staticmethod
    def _expected_calibration_error(
        y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10,
    ) -> float:
        from analyzer.ml.training.calibration import expected_calibration_error
        return expected_calibration_error(y_true, y_proba, n_bins=n_bins)

    @staticmethod
    def _compute_temporal_weights(n: int, decay: float = _TEMPORAL_DECAY) -> np.ndarray:
        from analyzer.ml.training.calibration import compute_temporal_weights
        return compute_temporal_weights(n, decay=decay)

    def extract_features_batch(self, trades: list[StrategyTrade]) -> np.ndarray:
        """Compatibility wrapper — real implementation in
        :mod:`analyzer.ml.features.extractor`."""
        from analyzer.ml.features.extractor import extract_features_batch
        return extract_features_batch(trades)

    def train(self, trades: list[StrategyTrade]) -> Optional[MLMetrics]:
        """Round-10 Step 9 compatibility wrapper.

        Real implementation in
        :func:`analyzer.ml.training.trainer.run_training`. This façade
        method keeps the old public signature so every caller in
        main.py (per-symbol + unified retrain paths) keeps working
        without changes. All state reads / writes still happen on
        self because the trainer receives the predictor as its
        first argument.
        """
        from analyzer.ml.training.trainer import run_training
        return run_training(self, trades)

    def train_walk_forward(
        self,
        trades: list[StrategyTrade],
        n_folds: int = 5,
        anchored: bool = False,
    ) -> Optional[Any]:
        """Round-10 Step 8 compatibility wrapper.

        Real implementation in
        :func:`analyzer.ml.training.walk_forward_runner.run_walk_forward`.
        The façade supplies the builder / extractor callables and
        receives ``(report, stacking_attach_result)`` back; it then
        updates its own ``_wf_report`` and ``_calibrated_threshold``
        fields to match the result.
        """
        from analyzer.ml.training.walk_forward_runner import run_walk_forward

        report, stacking = run_walk_forward(
            cfg=self._cfg,
            trades=trades,
            extract_features_batch_fn=self.extract_features_batch,
            build_rf_fn=lambda: self._build_rf(),
            build_lgbm_fn=lambda spw: self._build_lgbm(scale_pos_weight=spw),
            build_xgb_fn=lambda spw: self._build_xgb(scale_pos_weight=spw),
            build_elastic_net_fn=lambda: self._build_elastic_net(),
            ensemble=self._ensemble,
            n_folds=n_folds,
            anchored=anchored,
        )
        if report is not None:
            self._wf_report = report
        if stacking.attached and stacking.new_threshold is not None:
            self._calibrated_threshold = stacking.new_threshold
        return report

    def _fit_stacking_head_from_report(
        self,
        report: Any,
        X: np.ndarray,
        y: np.ndarray,
        pnl: Optional[np.ndarray] = None,
    ) -> None:
        """Round-10 Step 7 compatibility wrapper.

        Real implementation lives in
        :func:`analyzer.ml.training.stacking_fitter.fit_stacking_head_from_report`.
        The free function receives explicit state (ensemble, cfg,
        report, X, y, pnl, threshold-tuner) and returns a
        ``StackingAttachResult`` describing what, if anything, it did
        to the ensemble. The wrapper mirrors the old side-effect
        contract — updates ``self._calibrated_threshold`` when a new
        threshold was produced — so every existing caller and the
        round-9 test monkeypatches keep working.
        """
        from analyzer.ml.training.stacking_fitter import fit_stacking_head_from_report
        result = fit_stacking_head_from_report(
            ensemble=self._ensemble,
            cfg=self._cfg,
            report=report,
            X=X,
            y=y,
            pnl=pnl,
            calibrate_threshold_fn=self._calibrate_threshold,
        )
        if result.attached and result.new_threshold is not None:
            self._calibrated_threshold = result.new_threshold
        return None

    # ──────────────────────────────────────────────────────────
    # Phase-4: regime routing
    # ──────────────────────────────────────────────────────────

    def train_with_regime_routing(
        self,
        trades: list[StrategyTrade],
    ) -> Optional[Any]:
        """Round-10 Step 8 compatibility wrapper.

        Real implementation in
        :func:`analyzer.ml.training.regime_trainer.train_regime_routing`.
        The façade supplies a ``train_global_fn`` that runs ``.train()``
        on itself, a ``get_global_snapshot_fn`` that packages its
        post-train state as a :class:`RegimeModel`, and a factory that
        builds fresh MLPredictor instances for the specialists. On
        success it stashes the router in ``self._regime_router`` so
        ``predict()`` can route through it.
        """
        from analyzer.ml.training.regime_trainer import train_regime_routing

        def _train_global(t):
            m = self.train(t)
            return m if (m is not None and self.is_ready) else None

        def _global_snapshot():
            from analyzer.ml_regime_router import RegimeModel
            return RegimeModel(
                regime="__global__",
                ensemble=self._ensemble,
                scaler=self._scaler,
                selector=self._feature_selector,
                threshold=self._calibrated_threshold,
                skill_score=self._metrics.skill_score if self._metrics else 0.0,
                n_train=self._metrics.train_samples if self._metrics else 0,
                metrics_summary={
                    "precision": self._metrics.precision if self._metrics else 0.0,
                    "roc_auc": self._metrics.roc_auc if self._metrics else 0.5,
                },
            )

        router = train_regime_routing(
            cfg=self._cfg,
            trades=trades,
            train_global_fn=_train_global,
            get_global_snapshot_fn=_global_snapshot,
            predictor_factory=MLPredictor,
        )
        if router is not None:
            self._regime_router = router
        return router

    def predict(
        self,
        trade_features: list[float],
    ) -> MLPrediction:
        """Round-10 Step 10 compatibility wrapper.

        Real implementation in
        :func:`analyzer.ml.prediction.predictor.predict_from_features`.
        Callers everywhere (main.py's hot path, test fixtures, the
        warmup ML check) keep calling ``predictor.predict(features)``
        unchanged.
        """
        from analyzer.ml.prediction.predictor import (
            PredictionState,
            predict_from_features,
        )
        state = PredictionState(
            cfg=self._cfg,
            ensemble=self._ensemble,
            model=self._model,
            scaler=self._scaler,
            feature_selector=self._feature_selector,
            calibrated_threshold=self._calibrated_threshold,
            model_version=self._model_version,
            rollout_mode=self._rollout_mode,
            regime_router=self._regime_router,
        )
        result = predict_from_features(state, trade_features)
        # Feed live feature vector into the PSI drift monitor so we can
        # detect feature-distribution shift between training time and
        # production. Wrapped in try/except so a broken monitor never
        # tears down the prediction call — drift detection is observability,
        # not a hard dependency of trading decisions.
        if self._feature_drift_monitor is not None:
            try:
                self._feature_drift_monitor.record(trade_features)
            except Exception as _drift_err:  # noqa: BLE001
                logger.debug("PSI monitor record failed: %s", _drift_err)
        return result

    def needs_retrain(self) -> bool:
        """Return True if retraining is needed: scheduled interval OR concept drift."""
        if self._last_train_ts == 0:
            return True
        days_since = (time.time() * 1000 - self._last_train_ts) / (86400 * 1000)
        if days_since >= self._cfg.retrain_days:
            return True
        # Concept drift: live performance has diverged from training metrics
        if self.drift_detected:
            logger.warning(
                "Concept drift detected — triggering early retrain (live: %s)",
                self._live_tracker.live_metrics(),
            )
            return True
        return False

    # Default directory for model checkpoints. Used as the allow-list root
    # when path-traversal defence kicks in (see save_to_file).
    _DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "data" / "ml_models"

    def save_to_file(self, model_path: str | Path) -> bool:
        """Round-10 Step 6 compatibility wrapper.

        Builds the payload dict from ``self``'s trained state, delegates
        the signed-envelope write to
        :func:`analyzer.ml.persistence.codec.save_signed_payload_with_checksum`,
        and on success appends to the registry via
        :func:`analyzer.ml.persistence.registry.append_registry_entry`.
        """
        if not self.is_ready:
            logger.warning("ML save skipped: model not ready")
            return False

        from analyzer.ml.persistence.codec import save_signed_payload_with_checksum
        from analyzer.ml.persistence.registry import append_registry_entry

        path = Path(model_path)
        metrics_dict: dict[str, Any] = {}
        if self._metrics:
            metrics_dict = {
                "precision": self._metrics.precision,
                "recall": self._metrics.recall,
                "roc_auc": self._metrics.roc_auc,
                "accuracy": self._metrics.accuracy,
                "skill_score": self._metrics.skill_score,
                "train_samples": self._metrics.train_samples,
                "test_samples": self._metrics.test_samples,
                "feature_importances": self._metrics.feature_importances,
                "precision_ci_95": list(self._metrics.precision_ci_95),
                "auc_ci_95": list(self._metrics.auc_ci_95),
                "baseline_win_rate": self._metrics.baseline_win_rate,
                "precision_lift": self._metrics.precision_lift,
                "auc_lift": self._metrics.auc_lift,
                "oot_auc": self._metrics.oot_auc,
                "brier_score": self._metrics.brier_score,
                "ece": self._metrics.ece,
                "mean_proba": self._metrics.mean_proba,
                "median_proba": self._metrics.median_proba,
                "proba_p10": self._metrics.proba_p10,
                "proba_p90": self._metrics.proba_p90,
                "calibration_method": self._metrics.calibration_method,
            }
        data = {
            # v3: full ensemble (primary predictor)
            "ensemble": self._ensemble,
            "feature_selector": self._feature_selector,
            "calibrated_threshold": self._calibrated_threshold,
            # Legacy single-model fallback
            "model": self._model,
            "scaler": self._scaler,
            "version": self._model_version,
            "metrics": metrics_dict,
            "saved_at": int(time.time()),
            "format": "v3",
            "package_versions": _capture_package_versions(),
            "schema_version": 5,  # bumped round 8 §4.4 (rollout_mode persisted)
            "rollout_mode": self._rollout_mode,
            # Phase-1/2/4/5 artifacts — dashboard reads these after restart
            "wf_report": self._wf_report,
            "regime_router": self._regime_router,
            "bootstrap_ci": self._bootstrap_ci,
            "member_error_correlation": self._member_error_correlation,
            # PSI feature-drift monitor — survives restarts so the live
            # window keeps growing across deploys instead of resetting.
            "feature_drift_monitor": self._feature_drift_monitor,
        }
        ok, checksum = save_signed_payload_with_checksum(path, data)
        if not ok:
            return False

        n_members = self._ensemble.member_count() if self._ensemble else 0
        logger.info(
            "ML model saved to %s (version=%s, ensemble_members=%d)",
            path, self._model_version, n_members,
        )
        append_registry_entry(
            model_path=path,
            model_version=self._model_version,
            checksum=checksum or "",
            n_members=n_members,
            calibrated_threshold=self._calibrated_threshold,
            metrics=self._metrics,
        )
        return True

    def _append_to_registry(self, model_path: Path, checksum: str, n_members: int) -> None:
        """Deprecated compatibility wrapper — delegates to the registry
        module. Kept so any residual external caller / test continues
        to resolve the method name."""
        from analyzer.ml.persistence.registry import append_registry_entry
        append_registry_entry(
            model_path=model_path,
            model_version=self._model_version,
            checksum=checksum,
            n_members=n_members,
            calibrated_threshold=self._calibrated_threshold,
            metrics=self._metrics,
        )

    def load_from_file(self, model_path: str | Path) -> bool:
        """Round-10 Step 6 compatibility wrapper.

        Uses :func:`analyzer.ml.persistence.codec.load_signed_payload` to
        verify the envelope + checksum and return the inner data dict,
        then restores MLPredictor state from it. Any read / checksum /
        unpickle failure → returns False and logs at ERROR (codec logs
        its own diagnostic).
        """
        from analyzer.ml.persistence.codec import load_signed_payload

        path = Path(model_path)
        data = load_signed_payload(path)
        if data is None:
            return False
        try:
            # Warn on package-version divergence so operators know a model
            # trained against older sklearn/lgbm/xgb could behave
            # differently under the current install (tree-split ordering,
            # isotonic impl).
            saved_versions = data.get("package_versions") if isinstance(data, dict) else None
            if saved_versions:
                current = _capture_package_versions()
                diffs = [
                    f"{k}: saved={saved_versions.get(k,'?')} current={current.get(k,'?')}"
                    for k in saved_versions
                    if saved_versions.get(k) not in (current.get(k), "missing")
                    and current.get(k) != "missing"
                ]
                if diffs:
                    logger.warning(
                        "ML load: package version drift — %s. Retrain recommended.",
                        "; ".join(diffs),
                    )

            fmt = data.get("format", "v1")

            # v3: restore ensemble (primary predictor)
            if fmt == "v3":
                self._ensemble = data.get("ensemble")
                self._feature_selector = data.get("feature_selector", self._feature_selector)
                self._calibrated_threshold = data.get("calibrated_threshold", 0.5)
                n_members = self._ensemble.member_count() if self._ensemble else 0
                logger.info(
                    "ML ensemble loaded from %s (version=%s, members=%d)",
                    path, data.get("version", ""), n_members,
                )
            else:
                logger.info("ML loading legacy v1 format from %s", path)

            # Always restore legacy components (fallback + compat)
            self._model = data.get("model")
            self._scaler = data.get("scaler")
            self._model_version = data.get("version", "")
            saved_at = data.get("saved_at", 0)
            self._last_train_ts = saved_at * 1000 if saved_at else 0

            # Phase-1/2/4/5 artifacts — older saves (schema < 4) don't have them,
            # so .get() with sensible defaults keeps the load backwards-compatible.
            # None / empty values just mean the dashboard will show "not yet run"
            # until the next training cycle populates them.
            self._wf_report = data.get("wf_report")
            self._regime_router = data.get("regime_router")
            self._bootstrap_ci = data.get("bootstrap_ci", {}) or {}
            self._member_error_correlation = data.get("member_error_correlation", {}) or {}
            # PSI feature-drift monitor — older pickles (schema < 6) don't
            # carry it; the trainer re-fits a fresh monitor on the next
            # training cycle, so missing here is operationally fine.
            self._feature_drift_monitor = data.get("feature_drift_monitor")

            # Round-8 §4.4: restore rollout_mode if persisted (schema ≥ 5).
            # Pre-schema-5 pickles don't carry the mode, so fall back to
            # whatever the caller set via the property before load (env
            # default). Callers are expected to reconcile with env on top —
            # e.g. main.py forces "off" when analyzer_ml_enabled=False.
            _saved_rollout = data.get("rollout_mode")
            if _saved_rollout in ("off", "shadow", "block"):
                self._rollout_mode = _saved_rollout

            metrics_dict = data.get("metrics", {})
            if metrics_dict:
                # .get() with dataclass defaults so older saves stay loadable.
                pci = metrics_dict.get("precision_ci_95")
                aci = metrics_dict.get("auc_ci_95")
                self._metrics = MLMetrics(
                    precision=metrics_dict.get("precision", 0.0),
                    recall=metrics_dict.get("recall", 0.0),
                    roc_auc=metrics_dict.get("roc_auc", 0.0),
                    accuracy=metrics_dict.get("accuracy", 0.0),
                    skill_score=metrics_dict.get("skill_score", 0.0),
                    train_samples=metrics_dict.get("train_samples", 0),
                    test_samples=metrics_dict.get("test_samples", 0),
                    feature_importances=metrics_dict.get("feature_importances", {}),
                    precision_ci_95=tuple(pci) if pci else (0.0, 0.0),
                    auc_ci_95=tuple(aci) if aci else (0.0, 0.0),
                    baseline_win_rate=metrics_dict.get("baseline_win_rate", 0.0),
                    precision_lift=metrics_dict.get("precision_lift", 0.0),
                    auc_lift=metrics_dict.get("auc_lift", 0.0),
                    oot_auc=metrics_dict.get("oot_auc"),
                    brier_score=metrics_dict.get("brier_score", 0.0),
                    ece=metrics_dict.get("ece", 0.0),
                    mean_proba=metrics_dict.get("mean_proba", 0.5),
                    median_proba=metrics_dict.get("median_proba", 0.5),
                    proba_p10=metrics_dict.get("proba_p10", 0.0),
                    proba_p90=metrics_dict.get("proba_p90", 1.0),
                    calibration_method=metrics_dict.get("calibration_method", "none"),
                )

            # Ready if ensemble present OR legacy model present
            ready = (
                (self._ensemble is not None and self._ensemble.is_ready)
                or self._model is not None
            )
            if ready:
                logger.info("ML predictor ready (version=%s)", self._model_version)
            return ready
        except Exception as exc:
            logger.error("ML model load failed from %s: %s", path, exc)
            return False
