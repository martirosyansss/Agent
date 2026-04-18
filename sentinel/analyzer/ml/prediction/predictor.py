"""Prediction use case — take a feature vector, return an :class:`MLPrediction`.

Extracted from ``MLPredictor.predict`` during the round-10 refactor.
The free function takes all state it needs as explicit arguments, so
it can be unit-tested without instantiating MLPredictor and so the
contracts (regime specialist threshold bypass, shadow-mode allow-only,
fail-open exception handling) are co-located in one file.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..domain.config import MLConfig
from ..domain.constants import N_FEATURES, REGIME_ENCODING
from ..domain.metrics import MLPrediction

logger = logging.getLogger(__name__)

try:  # observability hook — optional at import time so plain tests work
    from monitoring.event_log import emit_component_error
except ImportError:  # pragma: no cover
    def emit_component_error(*args, **kwargs) -> None:  # type: ignore[misc]
        pass


@dataclass
class PredictionState:
    """Bundle of MLPredictor state the prediction path reads.

    Passing this dataclass into :func:`predict_from_features` keeps the
    function's signature small while making every field it touches
    explicit. The façade assembles one of these per call — zero-copy,
    just a dataclass of references to existing attributes.
    """
    cfg: MLConfig
    ensemble: Any                 # VotingEnsemble (or None)
    model: Any                    # legacy single-model fallback (or None)
    scaler: Any                   # StandardScaler (or None)
    feature_selector: Any         # AdaptiveFeatureSelector
    calibrated_threshold: float
    model_version: str
    rollout_mode: str             # "off" | "shadow" | "block"
    regime_router: Any = None     # RegimeRouter (or None)


def predict_from_features(
    state: PredictionState,
    trade_features: list[float],
) -> MLPrediction:
    """Compute ``MLPrediction`` for one feature vector.

    Branching summary:
    1. Early-return neutral 0.5 / "allow" when the predictor isn't
       ready or rollout is off.
    2. Validate feature-vector length (``N_FEATURES``); mismatch
       returns neutral + logs an error.
    3. If ``use_regime_routing`` is on and a specialist fires, use the
       router's probability + its own threshold (the env floor
       ``block_threshold`` does NOT override a specialist — round-8 §2.3).
    4. Otherwise apply feature selector + scaler + ensemble's
       calibrated probability.
    5. Map probability → decision using
       ``proba < cal_thr * reduce_margin`` → block, else
       ``proba < cal_thr`` → reduce, else allow.
    6. In shadow mode, decision is forced to "allow" regardless of the
       computed one (logging only, no blocking).
    7. On any exception inside the scoring pipeline, fail OPEN: log a
       warning and return 0.5 / "allow" so a misbehaving model never
       locks out live trades.
    """
    if not _is_ready(state) or state.rollout_mode == "off":
        return MLPrediction(
            probability=0.5,
            decision="allow",
            model_version="",
            rollout_mode=state.rollout_mode,
        )

    if len(trade_features) != N_FEATURES:
        logger.error(
            "ML predict: feature count mismatch: got %d, expected %d",
            len(trade_features), N_FEATURES,
        )
        return MLPrediction(
            probability=0.5, decision="allow",
            model_version=state.model_version,
            rollout_mode=state.rollout_mode,
        )

    try:
        features_raw = np.array([trade_features], dtype=np.float64)
        features_raw = np.nan_to_num(features_raw, nan=0.0, posinf=0.0, neginf=0.0)

        router_used: Optional[str] = None
        router_threshold: Optional[float] = None
        regime_specialist_active = False

        if (
            state.cfg.use_regime_routing
            and state.regime_router is not None
            and state.regime_router.is_ready
        ):
            # Regime is encoded at feature index 11 — decode back to a name.
            regime_idx = int(features_raw[0][11]) if features_raw.shape[1] > 11 else 5
            inv = {v: k for k, v in REGIME_ENCODING.items()}
            regime_name = inv.get(regime_idx, "unknown")
            proba, router_used, router_threshold = state.regime_router.predict_proba(
                regime_name, features_raw[0]
            )
            if router_used == "none":
                # Router had no model — fall through to non-routed path.
                router_used = None
                router_threshold = None
            elif router_used not in ("global", None):
                # Specialist fired → bypass env floor (round-8 §2.3).
                regime_specialist_active = True

        if router_used is None:
            features_arr = features_raw
            if state.feature_selector.is_fitted and state.feature_selector.dropped_names:
                features_arr = state.feature_selector.transform(features_arr)
            if state.scaler is not None:
                features_arr = state.scaler.transform(features_arr)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names")
                if state.ensemble is not None and state.ensemble.is_ready:
                    proba = float(state.ensemble.predict_proba_calibrated(features_arr)[0])
                elif state.model is not None:
                    proba = float(state.model.predict_proba(features_arr)[0][1])
                else:
                    proba = 0.5
    except Exception as exc:  # noqa: BLE001
        logger.warning("ML predict failed (defaulting to allow): %s", exc, exc_info=True)
        emit_component_error(
            "ml_predictor.predict",
            f"predict pipeline failed: {exc}",
            exc=exc,
            severity="warning",
            degraded_to="allow@0.5",
            model_version=state.model_version,
            rollout_mode=state.rollout_mode,
        )
        proba = 0.5

    # Threshold selection. Specialist result → use its own threshold.
    if regime_specialist_active and router_threshold is not None:
        cal_thr = router_threshold
    else:
        effective_cal = router_threshold if router_threshold is not None else state.calibrated_threshold
        cal_thr = max(effective_cal or 0.0, state.cfg.block_threshold or 0.0)
    if cal_thr <= 0:
        cal_thr = 0.5

    if proba < cal_thr * state.cfg.reduce_margin:
        decision = "block"
    elif proba < cal_thr:
        decision = "reduce"
    else:
        decision = "allow"

    # Shadow mode: computed decision is logged-but-not-enforced.
    effective_decision = decision if state.rollout_mode == "block" else "allow"

    return MLPrediction(
        probability=proba,
        decision=effective_decision,
        model_version=state.model_version,
        rollout_mode=state.rollout_mode,
    )


def _is_ready(state: PredictionState) -> bool:
    """Mirror of ``MLPredictor.is_ready`` — a live ensemble OR legacy
    single-model fallback counts as ready."""
    ensemble_ready = state.ensemble is not None and state.ensemble.is_ready
    return ensemble_ready or (state.model is not None)
