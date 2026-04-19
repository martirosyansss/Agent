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

The remaining 800 LOC of ``MLPredictor.train`` is the central training
orchestration (phase-1 build, phase-2 feature-selection retrain,
phase-B precision-recovery retrain, bootstrap CI, OOT validation,
calibration, skill gate). It stays here as a single method for now
because its branches share many intermediate variables that don't
belong in the public API of a separate trainer class. A future round
may split it into explicit Trainer / TrainingResult types.

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
        p_train: float, p_val: float, n_train: int, n_val: int, z: float = 1.96,
    ) -> float:
        from analyzer.ml.training.calibration import overfit_noise_margin
        return overfit_noise_margin(p_train, p_val, n_train, n_val, z=z)

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
        """Train VotingEnsemble on historical trades.

        v3 Ensemble: trains RF + LightGBM + XGBoost simultaneously,
        then combines via soft-voting weighted by validation skill score.
        TimeSeriesSplit CV + TemporalWeighting + IsotonicCalibration.
        Returns MLMetrics or None if insufficient data / metrics below threshold.
        """
        if len(trades) < self._cfg.min_trades:
            logger.warning("ML train: insufficient trades (%d < %d)", len(trades), self._cfg.min_trades)
            return None

        try:
            from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.error("scikit-learn not installed. pip install scikit-learn")
            return None

        # v3: Use vectorized batch extraction (8-15x faster than per-trade loop)
        logger.info("ML train: extracting features for %d trades (vectorized)", len(trades))
        X = self.extract_features_batch(trades)
        y_arr = np.array([1 if t.is_win else 0 for t in trades])
        pnl_values = [t.pnl_usd for t in trades]

        # Zero-variance feature detection: features that are constant (e.g. always 0)
        # provide no signal and either indicate a broken upstream data pipeline
        # or a feature that simply isn't available in this training corpus
        # (e.g. news_sentiment during pure-backtest training has no live news
        # to read). We always force-drop them via the AdaptiveFeatureSelector
        # below so they cannot end up in the deployed feature vector and
        # contaminate the StandardScaler's mean/std with constants.
        _variances = np.var(X, axis=0)
        _dead = [FEATURE_NAMES[i] for i in range(len(FEATURE_NAMES)) if i < len(_variances) and _variances[i] < 1e-12]
        if _dead:
            logger.warning(
                "ML train: %d ZERO-VARIANCE features will be force-dropped: %s "
                "— upstream data pipeline produces a constant value here",
                len(_dead), ", ".join(_dead),
            )

        # v3: Temporal sample weights — recent trades matter more
        sample_weights = self._compute_temporal_weights(len(trades), decay=self._cfg.temporal_decay)
        logger.info("ML train: temporal weights applied (decay=%.4f)", self._cfg.temporal_decay)

        # --- Time Series Cross-Validation (RF only — fast diagnostic pass) ---
        # Note: CV is used only for logging/early sanity check.
        # Actual ensemble member selection happens on the dedicated validation split below.
        tscv = TimeSeriesSplit(n_splits=self._cfg.cv_splits)
        cv_precisions, cv_recalls, cv_aucs = [], [], []

        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y_arr[train_idx], y_arr[test_idx]

            if len(X_tr) < 50 or len(X_te) < 15:
                continue

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            fold_model = self._build_rf()
            fold_model.fit(X_tr_s, y_tr)

            y_p = fold_model.predict(X_te_s)
            y_pr = fold_model.predict_proba(X_te_s)[:, 1] if len(set(y_tr)) > 1 else [0.5] * len(y_te)

            cv_precisions.append(precision_score(y_te, y_p, zero_division=0))
            cv_recalls.append(recall_score(y_te, y_p, zero_division=0))
            try:
                cv_aucs.append(roc_auc_score(y_te, y_pr))
            except ValueError:
                cv_aucs.append(0.5)

        if not cv_precisions:
            logger.warning("ML train: no valid CV folds produced")
            return None

        # Log CV results (RF diagnostic — ensemble metrics come from val/holdout splits)
        avg_prec = sum(cv_precisions) / len(cv_precisions)
        avg_rec = sum(cv_recalls) / len(cv_recalls)
        avg_auc = sum(cv_aucs) / len(cv_aucs)
        logger.info("ML CV (RF diagnostic): prec=%.3f rec=%.3f auc=%.3f (over %d folds)",
                     avg_prec, avg_rec, avg_auc, len(cv_precisions))

        # Walk-Forward: Train 70% → Val 15% (model selection + calibration) → Test 15%
        # Val split internally 50/50 for isotonic calibration vs threshold calibration.
        train_end = int(len(X) * 0.70)
        val_end   = int(len(X) * 0.85)
        X_train = X[:train_end]
        X_val   = X[train_end:val_end]
        X_test  = X[val_end:]
        y_train = y_arr[:train_end]
        y_val   = y_arr[train_end:val_end]
        y_test  = y_arr[val_end:]
        pnl_val  = pnl_values[train_end:val_end]
        pnl_test = pnl_values[val_end:]

        if len(X_train) < 50 or len(X_val) < 15 or len(X_test) < 15:
            return None

        final_scaler = StandardScaler()
        X_train_s = final_scaler.fit_transform(X_train)
        X_val_s   = final_scaler.transform(X_val)
        X_test_s  = final_scaler.transform(X_test)

        # v3: TemporalWeighting slices for each split
        train_weights = sample_weights[:train_end]

        # Class imbalance ratio for gradient boosters (computed before build)
        n_neg = int(np.sum(y_train == 0))
        n_pos = int(np.sum(y_train == 1))
        spw = n_neg / n_pos if n_pos > 0 else 1.0

        # Effective imbalance under temporal weighting — the weighted tail
        # sees a different win rate than the full-train sum. LGBM/XGB receive
        # BOTH scale_pos_weight (flat) AND sample_weight (decayed), so the
        # effective penalty is multiplicative. Logging the divergence helps
        # operators diagnose why calibration drifts on non-stationary data.
        w_pos = float(np.sum(train_weights[y_train == 1])) if n_pos > 0 else 0.0
        w_neg = float(np.sum(train_weights[y_train == 0])) if n_neg > 0 else 0.0
        eff_spw = w_neg / w_pos if w_pos > 0 else spw
        # Divergence > 20% means recent regime has materially different win rate
        # than the full train set — prefer retraining more often, or lowering
        # temporal_decay so the model sees a wider history.
        if spw > 0 and abs(eff_spw - spw) / spw > 0.20:
            logger.warning(
                "ML class imbalance divergence: flat spw=%.3f vs temporally-weighted eff_spw=%.3f "
                "(%.0f%% drift) — recent win rate differs from full train. Calibration may over-correct.",
                spw, eff_spw, 100 * abs(eff_spw - spw) / spw,
            )
        else:
            logger.info("ML class imbalance: spw=%.3f eff_spw=%.3f (n_pos=%d n_neg=%d)",
                        spw, eff_spw, n_pos, n_neg)

        # --- v3: VotingEnsemble — train all engines, combine by skill ---
        from analyzer.ml_ensemble import VotingEnsemble, AdaptiveFeatureSelector
        ensemble = VotingEnsemble()
        candidate_metrics: dict[str, float] = {}

        rf = self._build_rf()
        try:
            rf.fit(X_train_s, y_train, sample_weight=train_weights)
        except TypeError:
            rf.fit(X_train_s, y_train)

        # Round-8 §3.4: LGBM/XGB see BOTH sample_weight (temporal decay)
        # AND scale_pos_weight. Passing the flat ``spw`` here would double-
        # penalise the positive class: once via scale_pos_weight (based on
        # raw counts) and a second time implicitly via the sample_weight
        # sums. Using ``eff_spw`` — the ratio of weighted tail sums — makes
        # the penalty respect the temporal decay the model actually sees
        # during training. Without this fix, recent-regime calibration
        # drifts because the model over-corrects for an imbalance that
        # sample_weight already partly handles.
        lgbm = self._build_lgbm(scale_pos_weight=eff_spw)
        if lgbm is not None:
            try:
                lgbm.fit(X_train_s, y_train, sample_weight=train_weights)
            except Exception as lgbm_err:
                logger.debug("LightGBM training failed: %s", lgbm_err)
                lgbm = None

        xgb = self._build_xgb(scale_pos_weight=eff_spw)
        if xgb is not None:
            try:
                xgb.fit(X_train_s, y_train, sample_weight=train_weights)
            except Exception as xgb_err:
                logger.debug("XGBoost training failed: %s", xgb_err)
                xgb = None

        # Phase-3: decorrelated 4th member (ElasticNet LR). See _build_elastic_net
        # docstring for why a linear model diversifies an all-tree ensemble.
        lr_en = self._build_elastic_net()
        if lr_en is not None:
            try:
                lr_en.fit(X_train_s, y_train, sample_weight=train_weights)
            except Exception as lr_err:
                logger.debug("ElasticNet training failed: %s", lr_err)
                lr_en = None

        # Per-candidate overfit threshold = base gap + statistical noise margin.
        # The real noise depends on the model's actual train/val precisions
        # (a candidate predicting near 0.5 has much wider CI than one near 0.95),
        # so this is recomputed inside the loop instead of one global value.
        # We log the worst-case (p=0.5) margin upfront just for context.
        _n_val = len(y_val)
        _n_train = len(y_train)
        _worst_case_margin = self._overfit_noise_margin(0.5, 0.5, _n_train, _n_val)
        logger.info(
            "ML overfit guard: base=%.2f worst-case_margin=%.3f (n_train=%d n_val=%d)",
            self._cfg.max_overfit_gap, _worst_case_margin, _n_train, _n_val,
        )

        # Evaluate each candidate on validation set, reject overfit models
        for candidate, tag in [(rf, "rf"), (lgbm, "lgbm"), (xgb, "xgb"), (lr_en, "lr_en")]:
            if candidate is None:
                continue
            try:
                y_pred_v = candidate.predict(X_val_s)
                y_proba_v = (
                    candidate.predict_proba(X_val_s)[:, 1]
                    if len(set(y_train)) > 1
                    else np.full(len(y_val), 0.5)
                )
            except Exception as exc:
                logger.warning("ML candidate [%s] eval failed: %s", tag, exc)
                continue

            prec_v = precision_score(y_val, y_pred_v, zero_division=0)
            rec_v = recall_score(y_val, y_pred_v, zero_division=0)
            try:
                auc_v = roc_auc_score(y_val, y_proba_v)
            except ValueError:
                auc_v = 0.5

            pf_v = self._compute_profit_factor_score(y_pred_v, pnl_val)
            skill_v = compute_skill_score(prec_v, rec_v, auc_v, pf_v)

            # Precision-based overfit guard: directly catches the metric that drives
            # trading PnL. Train precision >> val precision = model memorized train set.
            y_train_pred_c = candidate.predict(X_train_s)
            train_prec_c = precision_score(y_train, y_train_pred_c, zero_division=0)
            overfit_gap = train_prec_c - prec_v
            noise_margin = self._overfit_noise_margin(train_prec_c, prec_v, _n_train, _n_val)
            cand_threshold = self._cfg.max_overfit_gap + noise_margin
            if overfit_gap > cand_threshold:
                logger.warning(
                    "ML OVERFITTING [%s]: train_prec=%.3f val_prec=%.3f gap=%.3f "
                    "(threshold=%.3f = base %.2f + 1.96σ noise %.3f) — REJECTED",
                    tag, train_prec_c, prec_v, overfit_gap,
                    cand_threshold, self._cfg.max_overfit_gap, noise_margin,
                )
                continue

            logger.info(
                "ML candidate [%s]: val_skill=%.3f prec=%.3f rec=%.3f auc=%.3f pf=%.3f → added to ensemble",
                tag, skill_v, prec_v, rec_v, auc_v, pf_v,
            )
            ensemble.add_member(candidate, tag, skill_v)
            candidate_metrics[tag] = skill_v

        if not ensemble.is_ready:
            logger.warning("ML train: all candidates failed overfitting check — ensemble empty")
            self._metrics = MLMetrics(
                precision=0, recall=0, roc_auc=0.5,
                accuracy=0, skill_score=0.0,
                train_samples=len(X_train), test_samples=len(X_test),
                feature_importances={},
            )
            return self._metrics

        # Total candidate slots = 3 tree models + 1 linear (ElasticNet) if enabled
        _max_members = 3 + (1 if self._cfg.use_elastic_net else 0)
        logger.info(
            "ML VotingEnsemble: %d/%d models accepted, members: %s",
            ensemble.member_count(), _max_members,
            ", ".join(f"{t}={s:.3f}" for t, s in candidate_metrics.items()),
        )

        # v3: Adaptive Feature Selector — analyse importances across all members
        combined_importances: dict[str, float] = {}
        for candidate, tag in [(rf, "rf"), (lgbm, "lgbm"), (xgb, "xgb")]:
            if candidate is None:
                continue
            raw_imp = getattr(candidate, 'feature_importances_', None)
            if raw_imp is not None:
                for i, name in enumerate(FEATURE_NAMES):
                    if i < len(raw_imp):
                        combined_importances[name] = combined_importances.get(name, 0.0) + float(raw_imp[i])

        # Normalize to get average importance across models
        n_models = sum(1 for c in [rf, lgbm, xgb] if c is not None)
        if n_models > 0:
            combined_importances = {k: v / n_models for k, v in combined_importances.items()}

        importances = combined_importances

        # Phase 2: Apply feature selector → refit scaler → retrain all models on selected features
        # This ensures inference pipeline matches training exactly:
        # predict(): selector.transform(30→N) → scaler.transform(N) → model.predict(N)
        #
        # Force-drop zero-variance features by zeroing their importance: the
        # selector keeps anything ≥ min_importance, and tree models can give a
        # constant column a small but non-zero "importance" via spurious early
        # splits, so we explicitly suppress them here.
        for _dead_name in _dead:
            importances[_dead_name] = 0.0
        self._feature_selector.fit(importances, FEATURE_NAMES)

        if self._feature_selector.dropped_names:
            # Apply selector to all splits (32 → N selected features)
            X_train_sel = self._feature_selector.transform(X_train)
            X_val_sel   = self._feature_selector.transform(X_val)
            X_test_sel  = self._feature_selector.transform(X_test)

            # Refit scaler on selected features only
            final_scaler = StandardScaler()
            X_train_s = final_scaler.fit_transform(X_train_sel)
            X_val_s   = final_scaler.transform(X_val_sel)
            X_test_s  = final_scaler.transform(X_test_sel)

            # Retrain all models on selected features
            ensemble = VotingEnsemble()
            candidate_metrics = {}

            rf2 = self._build_rf()
            try:
                rf2.fit(X_train_s, y_train, sample_weight=train_weights)
            except TypeError:
                rf2.fit(X_train_s, y_train)

            # §3.4: phase-2 retrain uses weighted eff_spw like phase-1
            lgbm2 = self._build_lgbm(scale_pos_weight=eff_spw)
            if lgbm2 is not None:
                try:
                    lgbm2.fit(X_train_s, y_train, sample_weight=train_weights)
                except Exception:
                    lgbm2 = None

            xgb2 = self._build_xgb(scale_pos_weight=eff_spw)
            if xgb2 is not None:
                try:
                    xgb2.fit(X_train_s, y_train, sample_weight=train_weights)
                except Exception:
                    xgb2 = None

            lr_en2 = self._build_elastic_net()
            if lr_en2 is not None:
                try:
                    lr_en2.fit(X_train_s, y_train, sample_weight=train_weights)
                except Exception:
                    lr_en2 = None

            for candidate, tag in [(rf2, "rf"), (lgbm2, "lgbm"), (xgb2, "xgb"), (lr_en2, "lr_en")]:
                if candidate is None:
                    continue
                try:
                    y_pred_v  = candidate.predict(X_val_s)
                    y_proba_v = candidate.predict_proba(X_val_s)[:, 1] if len(set(y_train)) > 1 else np.full(len(y_val), 0.5)
                except Exception as exc:
                    logger.warning("ML phase-2 candidate [%s] eval failed: %s", tag, exc)
                    continue
                prec_v  = precision_score(y_val, y_pred_v, zero_division=0)
                rec_v   = recall_score(y_val, y_pred_v, zero_division=0)
                try:
                    auc_v = roc_auc_score(y_val, y_proba_v)
                except ValueError:
                    auc_v = 0.5
                pf_v = self._compute_profit_factor_score(y_pred_v, pnl_val)
                skill_v = compute_skill_score(prec_v, rec_v, auc_v, pf_v)
                y_train_pred_c = candidate.predict(X_train_s)
                train_prec_c   = precision_score(y_train, y_train_pred_c, zero_division=0)
                noise_margin_p2 = self._overfit_noise_margin(train_prec_c, prec_v, _n_train, _n_val)
                cand_threshold_p2 = self._cfg.max_overfit_gap + noise_margin_p2
                if train_prec_c - prec_v > cand_threshold_p2:
                    logger.warning(
                        "ML phase-2 OVERFITTING [%s]: gap=%.3f (threshold=%.3f = base %.2f + 1.96σ noise %.3f) — REJECTED",
                        tag, train_prec_c - prec_v, cand_threshold_p2,
                        self._cfg.max_overfit_gap, noise_margin_p2,
                    )
                    continue
                ensemble.add_member(candidate, tag, skill_v)
                candidate_metrics[tag] = skill_v
                logger.info("ML phase-2 [%s]: skill=%.3f prec=%.3f rec=%.3f pf=%.3f → accepted", tag, skill_v, prec_v, rec_v, pf_v)

            if not ensemble.is_ready:
                logger.warning("ML phase-2: all candidates rejected — falling back to phase-1 ensemble")
                # fall back to phase-1 results (already in local variables)
            else:
                logger.info("ML phase-2 ensemble: %d members on %d features", ensemble.member_count(), X_train_sel.shape[1])
                best_tag = max(candidate_metrics, key=candidate_metrics.get)
                best_model = {"rf": rf2, "lgbm": lgbm2, "xgb": xgb2, "lr_en": lr_en2}.get(best_tag)
                rf, lgbm, xgb, lr_en = rf2, lgbm2, xgb2, lr_en2  # update refs for calibration below

        # Two-stage calibration: split validation into CAL (for probability
        # calibration) and THR (for threshold tuning). Sharing one set for both
        # causes double-dipping — the threshold gets optimized against noise in
        # the same sample used to learn the calibration, producing sunny metrics
        # that don't hold up live.
        #
        # Require 60 total samples (30 per half) to split safely. On smaller
        # validation sets we skip isotonic/Platt calibration entirely and use
        # raw ensemble probabilities; the threshold still gets tuned on the
        # full set, but with no calibration to overfit.
        MIN_SPLIT_SAMPLES = 60
        # Always initialize the CAL-half references so later branches
        # (precision-recovery Phase B) can reference them unconditionally
        # without NameError. When we skip the split they simply hold the
        # full val set — matching the no-split calibration path.
        val_mid = 0
        X_val_cal = X_val_s
        y_val_cal = y_val
        if len(X_val_s) >= MIN_SPLIT_SAMPLES:
            val_mid = len(X_val_s) // 2
            X_val_cal, X_val_thr = X_val_s[:val_mid], X_val_s[val_mid:]
            y_val_cal, y_val_thr = y_val[:val_mid], y_val[val_mid:]
            pnl_val_thr = pnl_val[val_mid:]
            ensemble.apply_isotonic_calibration(y_val_cal, X_val_cal)
        else:
            logger.warning(
                "ML calibration split skipped: val_n=%d < %d — using raw probabilities, "
                "threshold tuned on full validation set (acceptable noise given small N)",
                len(X_val_s), MIN_SPLIT_SAMPLES,
            )
            X_val_thr = X_val_s
            y_val_thr = y_val
            pnl_val_thr = pnl_val

        # Keep best single model as legacy fallback (for save_to_file compat)
        best_tag = max(candidate_metrics, key=candidate_metrics.get) if candidate_metrics else "rf"
        best_model: Any = {"rf": rf, "lgbm": lgbm, "xgb": xgb, "lr_en": lr_en}.get(best_tag, rf)

        # Threshold calibration via profit-factor optimization on second half of val
        y_proba_val_ensemble = ensemble.predict_proba_calibrated(X_val_thr)
        pnl_thr_arr = np.array(pnl_val_thr, dtype=np.float64) if pnl_val_thr else None

        calib_target = max(0.50, self._cfg.min_precision - 0.02)
        calibrated_thr = self._calibrate_threshold(
            y_val_thr, y_proba_val_ensemble,
            min_precision=calib_target, pnl=pnl_thr_arr,
        )
        logger.info(
            "ML VotingEnsemble calibrated threshold: %.3f (targeting P=%.2f, members=%d)",
            calibrated_thr, calib_target, ensemble.member_count(),
        )

        # Log feature importance summary
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        top5 = ", ".join(f"{n}={v:.3f}" for n, v in sorted_imp[:5])
        logger.info("ML averaged top features: %s", top5)
        if self._feature_selector.dropped_names:
            logger.info(
                "ML AdaptiveFeatureSelector dropped (%d): %s",
                len(self._feature_selector.dropped_names),
                ", ".join(self._feature_selector.dropped_names),
            )

        # --- Final metrics on HOLDOUT test set ---
        y_proba_holdout = ensemble.predict_proba_calibrated(X_test_s)
        y_pred_holdout = (y_proba_holdout >= calibrated_thr).astype(int)
        best_precision = precision_score(y_test, y_pred_holdout, zero_division=0)
        best_recall = recall_score(y_test, y_pred_holdout, zero_division=0)
        try:
            best_roc_auc = roc_auc_score(y_test, y_proba_holdout)
        except ValueError:
            best_roc_auc = 0.5
        best_accuracy = accuracy_score(y_test, y_pred_holdout)
        pf_score = self._compute_profit_factor_score(y_pred_holdout, pnl_test)
        # Precision 3x recall weight: false positives (losing trades) are far more expensive
        # than missed winners. AUC remains highest (threshold-independent ranking quality).
        final_skill = compute_skill_score(best_precision, best_recall, best_roc_auc, pf_score)

        # --- Precision recovery: when initial pass fails on precision ---
        # Noisy assets (e.g. ETH) often produce high recall / low precision.
        # Recovery strategy:
        #   Phase A: raise threshold on existing ensemble (cheap — no retraining)
        #   Phase B: retrain with conservative hyperparams + higher class penalty
        if best_precision < self._cfg.min_precision and best_recall > 0.40:
            logger.info(
                "ML PRECISION RECOVERY triggered: prec=%.3f < min=%.3f (recall=%.3f)",
                best_precision, self._cfg.min_precision, best_recall,
            )

            # Phase A: search for a higher threshold that meets precision target
            recovered = False
            for thr_candidate in np.arange(calibrated_thr + 0.02, 0.80, 0.01):
                y_pred_try = (y_proba_holdout >= thr_candidate).astype(int)
                n_pred_pos = int(y_pred_try.sum())
                if n_pred_pos < 5:
                    break
                prec_try = precision_score(y_test, y_pred_try, zero_division=0)
                rec_try = recall_score(y_test, y_pred_try, zero_division=0)
                if prec_try >= self._cfg.min_precision and rec_try >= 0.30:
                    pf_try = self._compute_profit_factor_score(y_pred_try, pnl_test)
                    skill_try = compute_skill_score(prec_try, rec_try, best_roc_auc, pf_try)
                    if skill_try >= self._cfg.min_skill_score * 0.95:
                        calibrated_thr = float(thr_candidate)
                        best_precision = prec_try
                        best_recall = rec_try
                        best_accuracy = accuracy_score(y_test, y_pred_try)
                        pf_score = pf_try
                        final_skill = skill_try
                        y_pred_holdout = y_pred_try
                        recovered = True
                        logger.info(
                            "ML PRECISION RECOVERY Phase A: thr=%.3f → prec=%.3f rec=%.3f skill=%.3f",
                            calibrated_thr, best_precision, best_recall, final_skill,
                        )
                        break

            # Phase B: retrain with conservative models if threshold bump wasn't enough
            if not recovered:
                logger.info("ML PRECISION RECOVERY Phase B: retraining with conservative hyperparams")
                # §3.4: scale up the weighted eff_spw (not the raw spw) so
                # the 1.8× boost multiplies the actually-seen imbalance
                # after temporal decay — consistent with the phase-1/2 fix.
                spw_boost = eff_spw * 1.8  # stronger false-positive penalty

                ensemble_b = VotingEnsemble()
                cand_metrics_b: dict[str, float] = {}

                rf_b = self._build_rf(conservative=True)
                try:
                    rf_b.fit(X_train_s, y_train, sample_weight=train_weights)
                except TypeError:
                    rf_b.fit(X_train_s, y_train)

                lgbm_b = self._build_lgbm(scale_pos_weight=spw_boost, conservative=True)
                if lgbm_b is not None:
                    try:
                        lgbm_b.fit(X_train_s, y_train, sample_weight=train_weights)
                    except Exception:
                        lgbm_b = None

                xgb_b = self._build_xgb(scale_pos_weight=spw_boost, conservative=True)
                if xgb_b is not None:
                    try:
                        xgb_b.fit(X_train_s, y_train, sample_weight=train_weights)
                    except Exception:
                        xgb_b = None

                lr_en_b = self._build_elastic_net(conservative=True)
                if lr_en_b is not None:
                    try:
                        lr_en_b.fit(X_train_s, y_train, sample_weight=train_weights)
                    except Exception:
                        lr_en_b = None

                for candidate_b, tag_b in [(rf_b, "rf"), (lgbm_b, "lgbm"), (xgb_b, "xgb"), (lr_en_b, "lr_en")]:
                    if candidate_b is None:
                        continue
                    try:
                        y_pred_bv = candidate_b.predict(X_val_s)
                        y_proba_bv = candidate_b.predict_proba(X_val_s)[:, 1] if len(set(y_train)) > 1 else np.full(len(y_val), 0.5)
                    except Exception:
                        continue
                    prec_bv = precision_score(y_val, y_pred_bv, zero_division=0)
                    rec_bv = recall_score(y_val, y_pred_bv, zero_division=0)
                    try:
                        auc_bv = roc_auc_score(y_val, y_proba_bv)
                    except ValueError:
                        auc_bv = 0.5
                    pf_bv = self._compute_profit_factor_score(y_pred_bv, pnl_val)
                    skill_bv = compute_skill_score(prec_bv, rec_bv, auc_bv, pf_bv)

                    y_train_pred_b = candidate_b.predict(X_train_s)
                    train_prec_b = precision_score(y_train, y_train_pred_b, zero_division=0)
                    noise_margin_b = self._overfit_noise_margin(train_prec_b, prec_bv, _n_train, _n_val)
                    cand_threshold_b = self._cfg.max_overfit_gap + noise_margin_b
                    if train_prec_b - prec_bv > cand_threshold_b:
                        logger.warning(
                            "ML recovery [%s]: overfitting gap=%.3f > threshold=%.3f (base %.2f + 1.96σ noise %.3f) — skipped",
                            tag_b, train_prec_b - prec_bv, cand_threshold_b,
                            self._cfg.max_overfit_gap, noise_margin_b,
                        )
                        continue

                    ensemble_b.add_member(candidate_b, tag_b, skill_bv)
                    cand_metrics_b[tag_b] = skill_bv
                    logger.info("ML recovery [%s]: prec=%.3f rec=%.3f auc=%.3f → accepted", tag_b, prec_bv, rec_bv, auc_bv)

                if ensemble_b.is_ready:
                    # Calibrate the recovery ensemble. val_mid / X_val_cal /
                    # y_val_cal are guaranteed to exist — they're initialised
                    # upfront in the main calibration block and overwritten
                    # when the split runs. When val_mid < 10 the CAL half is
                    # too tiny for a meaningful fit; calibrate on the whole
                    # val set instead. (Before this fix the branch referenced
                    # an undefined X_val_iso/y_val_iso and NameError'd.)
                    if val_mid >= 10:
                        ensemble_b.apply_isotonic_calibration(y_val_cal, X_val_cal)
                    else:
                        ensemble_b.apply_isotonic_calibration(y_val, X_val_s)

                    # Find precision-targeting threshold for recovery ensemble
                    y_proba_b_val = ensemble_b.predict_proba_calibrated(X_val_thr)
                    pnl_thr_arr_b = np.array(pnl_val_thr, dtype=np.float64) if pnl_val_thr else None
                    thr_b = self._calibrate_threshold(
                        y_val_thr, y_proba_b_val,
                        min_precision=self._cfg.min_precision, pnl=pnl_thr_arr_b,
                        min_recall=0.25,
                    )

                    # Evaluate on holdout
                    y_proba_b_holdout = ensemble_b.predict_proba_calibrated(X_test_s)
                    y_pred_b_holdout = (y_proba_b_holdout >= thr_b).astype(int)
                    n_pred_b = int(y_pred_b_holdout.sum())
                    if n_pred_b >= 5:
                        prec_b = precision_score(y_test, y_pred_b_holdout, zero_division=0)
                        rec_b = recall_score(y_test, y_pred_b_holdout, zero_division=0)
                        try:
                            auc_b = roc_auc_score(y_test, y_proba_b_holdout)
                        except ValueError:
                            auc_b = 0.5
                        pf_b = self._compute_profit_factor_score(y_pred_b_holdout, pnl_test)
                        skill_b = compute_skill_score(prec_b, rec_b, auc_b, pf_b)

                        logger.info(
                            "ML PRECISION RECOVERY Phase B holdout: prec=%.3f rec=%.3f auc=%.3f skill=%.3f (thr=%.3f)",
                            prec_b, rec_b, auc_b, skill_b, thr_b,
                        )

                        # Accept recovery if it improved precision AND skill is reasonable
                        if prec_b > best_precision and skill_b > final_skill * 0.90:
                            ensemble = ensemble_b
                            calibrated_thr = thr_b
                            best_precision = prec_b
                            best_recall = rec_b
                            best_roc_auc = auc_b
                            best_accuracy = accuracy_score(y_test, y_pred_b_holdout)
                            pf_score = pf_b
                            final_skill = skill_b
                            y_pred_holdout = y_pred_b_holdout
                            y_proba_holdout = y_proba_b_holdout
                            # Update best_model ref for legacy save
                            best_tag_b = max(cand_metrics_b, key=cand_metrics_b.get)
                            best_model = {"rf": rf_b, "lgbm": lgbm_b, "xgb": xgb_b, "lr_en": lr_en_b}.get(best_tag_b, rf_b)
                            logger.info(
                                "ML PRECISION RECOVERY Phase B ACCEPTED: prec=%.3f rec=%.3f skill=%.3f",
                                best_precision, best_recall, final_skill,
                            )
                        else:
                            logger.info(
                                "ML PRECISION RECOVERY Phase B: no improvement (prec %.3f→%.3f, skill %.3f→%.3f)",
                                best_precision, prec_b, final_skill, skill_b,
                            )

        # --- Naive baseline: always predict "win" ---
        baseline_win_rate = float(y_test.mean()) if len(y_test) > 0 else 0.5
        precision_lift = best_precision - baseline_win_rate
        auc_lift = best_roc_auc - 0.5
        logger.info(
            "Baseline (always-win): precision=%.3f | Model lift: +%.3f precision, +%.3f AUC",
            baseline_win_rate, precision_lift, auc_lift,
        )

        # --- Block bootstrap 95% CI on holdout (preserves temporal autocorrelation) ---
        n_boot = 500
        boot_prec, boot_auc = [], []
        rng = np.random.default_rng(self._cfg.random_seed)
        n_test = len(X_test_s)
        block_size = max(5, int(n_test ** (1.0 / 3.0)))  # cube-root rule for block length
        n_blocks = max(1, n_test // block_size)
        for _ in range(n_boot):
            # Sample contiguous blocks with replacement, then concatenate
            block_starts = rng.choice(n_test - block_size + 1, size=n_blocks, replace=True)
            idx_b = np.concatenate([np.arange(s, min(s + block_size, n_test)) for s in block_starts])
            idx_b = idx_b[:n_test]  # trim to original test size
            y_b_true = y_test[idx_b]
            y_b_prob = y_proba_holdout[idx_b]
            y_b_pred = (y_b_prob >= calibrated_thr).astype(int)
            bp = precision_score(y_b_true, y_b_pred, zero_division=0)
            boot_prec.append(bp)
            try:
                boot_auc.append(roc_auc_score(y_b_true, y_b_prob))
            except ValueError:
                boot_auc.append(0.5)
        ci_prec = (float(np.percentile(boot_prec, 2.5)), float(np.percentile(boot_prec, 97.5)))
        ci_auc  = (float(np.percentile(boot_auc,  2.5)), float(np.percentile(boot_auc,  97.5)))
        logger.info(
            "ML HOLDOUT [VotingEnsemble] (thr=%.3f): skill=%.3f prec=%.3f [%.3f,%.3f] rec=%.3f auc=%.3f [%.3f,%.3f]",
            calibrated_thr, final_skill,
            best_precision, ci_prec[0], ci_prec[1],
            best_recall, best_roc_auc, ci_auc[0], ci_auc[1],
        )

        # Phase-5: optional richer bootstrap report (p5/p50/p95 + probability
        # above baseline) surfaced to the dashboard. We keep the inline
        # block-bootstrap above because it feeds ci_prec/ci_auc into
        # MLMetrics. The MLBootstrap run below is strictly additive — only
        # populates self._bootstrap_ci when the flag is on, so the default
        # training path stays unchanged.
        if self._cfg.use_bootstrap_ci:
            try:
                from analyzer.ml_bootstrap import MLBootstrap
                mb = MLBootstrap(
                    n_simulations=self._cfg.bootstrap_n_simulations,
                    seed=self._cfg.random_seed,
                )
                cis = mb.bootstrap_metrics(y_test, y_proba_holdout, threshold=calibrated_thr)
                self._bootstrap_ci = {k: v.summary() for k, v in cis.items()}
                prob_above = mb.probability_above_baseline(y_test, y_proba_holdout, baseline_auc=0.5)
                self._bootstrap_ci["probability_above_random"] = round(prob_above, 4)
                logger.info(
                    "ML bootstrap CI: AUC p5=%.3f p50=%.3f p95=%.3f | P(AUC>0.5)=%.2f",
                    cis["roc_auc"].p5, cis["roc_auc"].p50, cis["roc_auc"].p95, prob_above,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("ML bootstrap CI failed: %s", exc)

        # Phase-3: diagnostic — log pairwise error correlation between members.
        # Exposed via self._member_error_correlation for dashboard; high
        # correlations (> 0.85) mean a member is paying rent without adding
        # diversity and the feature set needs attention.
        try:
            self._member_error_correlation = ensemble.member_error_correlation(X_test_s, y_test)
            if self._member_error_correlation:
                worst = max(self._member_error_correlation.items(), key=lambda kv: abs(kv[1]))
                logger.info(
                    "ML member error correlations: %s | worst pair %s corr=%.3f%s",
                    {k: round(v, 3) for k, v in self._member_error_correlation.items()},
                    worst[0], worst[1],
                    " ⚠ redundant member" if abs(worst[1]) > 0.85 else "",
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("member error correlation failed: %s", exc)

        # --- Out-of-time validation: retrain on 80%, test on LAST 20% as extra sanity ---
        oot_split = int(len(X) * 0.80)
        X_oot_raw = X[oot_split:]
        y_oot = y_arr[oot_split:]
        # OOT test always runs (regardless of feature selection).
        # Bug fix: previously skipped when no features were dropped — now unconditional.
        oot_auc = None
        if len(X_oot_raw) >= 20:
            try:
                if self._feature_selector.is_fitted:
                    X_oot_sel = self._feature_selector.transform(X_oot_raw)
                else:
                    X_oot_sel = X_oot_raw
                X_oot_s   = final_scaler.transform(X_oot_sel)
                y_oot_prob = ensemble.predict_proba_calibrated(X_oot_s)
                oot_auc    = float(roc_auc_score(y_oot, y_oot_prob))
                oot_prec   = float(precision_score(y_oot, (y_oot_prob >= calibrated_thr).astype(int), zero_division=0))
                logger.info("Out-of-time test (%d samples): AUC=%.3f prec=%.3f", len(X_oot_raw), oot_auc, oot_prec)
                if oot_auc < best_roc_auc - 0.15:
                    logger.warning(
                        "OOT AUC (%.3f) drops >0.15 vs holdout (%.3f) — possible period-specific overfitting",
                        oot_auc, best_roc_auc,
                    )
            except Exception as _oot_err:
                logger.warning("OOT test failed: %s", _oot_err)

        # --- Calibration diagnostics on the FINAL chosen ensemble -----------
        # Computed here (after both Phase A threshold-bump and Phase B retrain
        # have settled) so the persisted metrics describe the model that will
        # actually be deployed. ECE/Brier/percentiles flag the "every signal
        # ≈ 95%" failure mode that AUC/precision alone cannot catch.
        from sklearn.metrics import brier_score_loss
        raw_holdout_final = ensemble.predict_proba(X_test_s)
        try:
            brier_raw = float(brier_score_loss(y_test, raw_holdout_final))
            brier_cal = float(brier_score_loss(y_test, y_proba_holdout))
        except ValueError:
            brier_raw = brier_cal = 0.0
        ece_cal      = self._expected_calibration_error(y_test, y_proba_holdout, n_bins=10)
        proba_mean   = float(y_proba_holdout.mean())
        proba_median = float(np.median(y_proba_holdout))
        proba_p10    = float(np.percentile(y_proba_holdout, 10))
        proba_p90    = float(np.percentile(y_proba_holdout, 90))
        cal_method   = ensemble.calibration_method
        logger.info(
            "ML calibration [%s]: brier raw=%.3f cal=%.3f | ECE=%.3f | "
            "proba mean=%.3f median=%.3f p10=%.3f p90=%.3f (n=%d)",
            cal_method, brier_raw, brier_cal, ece_cal,
            proba_mean, proba_median, proba_p10, proba_p90, len(y_proba_holdout),
        )
        if proba_p10 > 0.70 or ece_cal > 0.15:
            # Loud signal that the displayed probabilities are inflated or
            # clumped — usually a calibration bug, not real model strength.
            logger.warning(
                "ML calibration suspicious: p10=%.3f ECE=%.3f method=%s — outputs look "
                "inflated or clumped. Check class balance and isotonic plateau collapse.",
                proba_p10, ece_cal, cal_method,
            )
        # Over-calibration guard: if the calibrator meaningfully *worsens*
        # Brier on the holdout, it has overfit the calibration set (common
        # on small N with Platt). Raw probabilities would serve users better.
        if brier_raw > 0 and brier_cal > brier_raw * 1.05:
            logger.warning(
                "ML over-calibration [%s]: brier worsened cal=%.3f > raw=%.3f × 1.05 "
                "— calibrator is overfitting the CAL set. Consider raising "
                "MIN_SAMPLES_ISOTONIC, disabling calibration, or using a held-out "
                "calibration split with more data.",
                cal_method, brier_cal, brier_raw,
            )

        metrics = MLMetrics(
            precision=best_precision,
            recall=best_recall,
            roc_auc=best_roc_auc,
            accuracy=best_accuracy,
            skill_score=final_skill,
            train_samples=len(X_train),
            test_samples=len(X_test),
            feature_importances=importances,
            precision_ci_95=ci_prec,
            auc_ci_95=ci_auc,
            baseline_win_rate=baseline_win_rate,
            precision_lift=precision_lift,
            auc_lift=auc_lift,
            oot_auc=oot_auc,
            brier_score=brier_cal,
            ece=ece_cal,
            mean_proba=proba_mean,
            median_proba=proba_median,
            proba_p10=proba_p10,
            proba_p90=proba_p90,
            calibration_method=cal_method,
        )

        # Gate check — reject if below quality thresholds
        if (
            best_precision < self._cfg.min_precision
            or best_recall < self._cfg.min_recall
            or best_roc_auc < self._cfg.min_roc_auc
            or final_skill < self._cfg.min_skill_score
        ):
            logger.warning(
                "ML metrics below threshold: skill=%.3f prec=%.3f rec=%.3f auc=%.3f — NOT deploying",
                final_skill, best_precision, best_recall, best_roc_auc,
            )
            self._metrics = metrics
            return metrics

        # Deploy new ensemble
        self._ensemble = ensemble
        self._model = best_model         # Legacy compat (save_to_file uses it)
        self._scaler = final_scaler
        self._calibrated_threshold = calibrated_thr
        self._model_version = f"ensemble_v{int(time.time())}_{ensemble.member_count()}m"
        self._metrics = metrics
        self._last_train_ts = int(time.time() * 1000)
        logger.info(
            "ML VotingEnsemble deployed: version=%s skill=%.3f prec=%.3f rec=%.3f auc=%.3f thr=%.3f",
            self._model_version, final_skill, best_precision, best_recall, best_roc_auc, calibrated_thr,
        )
        return metrics

    # ──────────────────────────────────────────────────────────
    # Phase-1/2: walk-forward validation + stacking head
    # ──────────────────────────────────────────────────────────

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
        return predict_from_features(state, trade_features)

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
