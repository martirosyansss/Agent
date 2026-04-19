"""Core training pipeline — extracted from MLPredictor.train in Step 9.

The function receives the MLPredictor instance as its first argument and
mutates it via the same interface the old method used (predictor._cfg,
predictor._build_rf(), predictor._calibrate_threshold(...), etc).
This is a mechanical extraction — no behavioural changes; the goal is
getting 830 lines out of ml_predictor.py so the façade file stays under
1000 LOC.

Why pass predictor rather than its fields individually: train
touches ~20 attributes (both reads and writes) across 3 phases (initial
build, feature-selection retrain, phase-B precision recovery). Passing
them one at a time would produce a 25-arg signature that obscures the
flow. Taking the whole predictor keeps the call-site tiny and the
behaviour identical.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np

from core.models import StrategyTrade

from ..domain.constants import FEATURE_NAMES
from ..domain.metrics import MLMetrics
from ..domain.scoring import compute_skill_score

logger = logging.getLogger(__name__)

def run_training(predictor: Any, trades: list[StrategyTrade]) -> Optional[Any]:
    """Train VotingEnsemble on historical trades.

    v3 Ensemble: trains RF + LightGBM + XGBoost simultaneously,
    then combines via soft-voting weighted by validation skill score.
    TimeSeriesSplit CV + TemporalWeighting + IsotonicCalibration.
    Returns MLMetrics or None if insufficient data / metrics below threshold.
    """
    if len(trades) < predictor._cfg.min_trades:
        logger.warning("ML train: insufficient trades (%d < %d)", len(trades), predictor._cfg.min_trades)
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
    X = predictor.extract_features_batch(trades)
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
    sample_weights = predictor._compute_temporal_weights(len(trades), decay=predictor._cfg.temporal_decay)
    logger.info("ML train: temporal weights applied (decay=%.4f)", predictor._cfg.temporal_decay)

    # --- Time Series Cross-Validation (RF only — fast diagnostic pass) ---
    # Note: CV is used only for logging/early sanity check.
    # Actual ensemble member selection happens on the dedicated validation split below.
    tscv = TimeSeriesSplit(n_splits=predictor._cfg.cv_splits)
    cv_precisions, cv_recalls, cv_aucs = [], [], []

    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]

        if len(X_tr) < 50 or len(X_te) < 15:
            continue

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        fold_model = predictor._build_rf()
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

    rf = predictor._build_rf()
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
    lgbm = predictor._build_lgbm(scale_pos_weight=eff_spw)
    if lgbm is not None:
        try:
            lgbm.fit(X_train_s, y_train, sample_weight=train_weights)
        except Exception as lgbm_err:
            logger.debug("LightGBM training failed: %s", lgbm_err)
            lgbm = None

    xgb = predictor._build_xgb(scale_pos_weight=eff_spw)
    if xgb is not None:
        try:
            xgb.fit(X_train_s, y_train, sample_weight=train_weights)
        except Exception as xgb_err:
            logger.debug("XGBoost training failed: %s", xgb_err)
            xgb = None

    # Phase-3: decorrelated 4th member (ElasticNet LR). See _build_elastic_net
    # docstring for why a linear model diversifies an all-tree ensemble.
    lr_en = predictor._build_elastic_net()
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
    _worst_case_margin = predictor._overfit_noise_margin(0.5, 0.5, _n_train, _n_val)
    logger.info(
        "ML overfit guard: base=%.2f worst-case_margin=%.3f (n_train=%d n_val=%d)",
        predictor._cfg.max_overfit_gap, _worst_case_margin, _n_train, _n_val,
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

        pf_v = predictor._compute_profit_factor_score(y_pred_v, pnl_val)
        skill_v = compute_skill_score(prec_v, rec_v, auc_v, pf_v)

        # Precision-based overfit guard: directly catches the metric that drives
        # trading PnL. Train precision >> val precision = model memorized train set.
        y_train_pred_c = candidate.predict(X_train_s)
        train_prec_c = precision_score(y_train, y_train_pred_c, zero_division=0)
        overfit_gap = train_prec_c - prec_v
        noise_margin = predictor._overfit_noise_margin(train_prec_c, prec_v, _n_train, _n_val)
        cand_threshold = predictor._cfg.max_overfit_gap + noise_margin
        if overfit_gap > cand_threshold:
            logger.warning(
                "ML OVERFITTING [%s]: train_prec=%.3f val_prec=%.3f gap=%.3f "
                "(threshold=%.3f = base %.2f + 1.96σ noise %.3f) — REJECTED",
                tag, train_prec_c, prec_v, overfit_gap,
                cand_threshold, predictor._cfg.max_overfit_gap, noise_margin,
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
        predictor._metrics = MLMetrics(
            precision=0, recall=0, roc_auc=0.5,
            accuracy=0, skill_score=0.0,
            train_samples=len(X_train), test_samples=len(X_test),
            feature_importances={},
        )
        return predictor._metrics

    # Total candidate slots = 3 tree models + 1 linear (ElasticNet) if enabled
    _max_members = 3 + (1 if predictor._cfg.use_elastic_net else 0)
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
    predictor._feature_selector.fit(importances, FEATURE_NAMES)

    if predictor._feature_selector.dropped_names:
        # Apply selector to all splits (32 → N selected features)
        X_train_sel = predictor._feature_selector.transform(X_train)
        X_val_sel   = predictor._feature_selector.transform(X_val)
        X_test_sel  = predictor._feature_selector.transform(X_test)

        # Refit scaler on selected features only
        final_scaler = StandardScaler()
        X_train_s = final_scaler.fit_transform(X_train_sel)
        X_val_s   = final_scaler.transform(X_val_sel)
        X_test_s  = final_scaler.transform(X_test_sel)

        # Retrain all models on selected features
        ensemble = VotingEnsemble()
        candidate_metrics = {}

        rf2 = predictor._build_rf()
        try:
            rf2.fit(X_train_s, y_train, sample_weight=train_weights)
        except TypeError:
            rf2.fit(X_train_s, y_train)

        # §3.4: phase-2 retrain uses weighted eff_spw like phase-1
        lgbm2 = predictor._build_lgbm(scale_pos_weight=eff_spw)
        if lgbm2 is not None:
            try:
                lgbm2.fit(X_train_s, y_train, sample_weight=train_weights)
            except Exception:
                lgbm2 = None

        xgb2 = predictor._build_xgb(scale_pos_weight=eff_spw)
        if xgb2 is not None:
            try:
                xgb2.fit(X_train_s, y_train, sample_weight=train_weights)
            except Exception:
                xgb2 = None

        lr_en2 = predictor._build_elastic_net()
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
            pf_v = predictor._compute_profit_factor_score(y_pred_v, pnl_val)
            skill_v = compute_skill_score(prec_v, rec_v, auc_v, pf_v)
            y_train_pred_c = candidate.predict(X_train_s)
            train_prec_c   = precision_score(y_train, y_train_pred_c, zero_division=0)
            noise_margin_p2 = predictor._overfit_noise_margin(train_prec_c, prec_v, _n_train, _n_val)
            cand_threshold_p2 = predictor._cfg.max_overfit_gap + noise_margin_p2
            if train_prec_c - prec_v > cand_threshold_p2:
                logger.warning(
                    "ML phase-2 OVERFITTING [%s]: gap=%.3f (threshold=%.3f = base %.2f + 1.96σ noise %.3f) — REJECTED",
                    tag, train_prec_c - prec_v, cand_threshold_p2,
                    predictor._cfg.max_overfit_gap, noise_margin_p2,
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

    calib_target = max(0.50, predictor._cfg.min_precision - 0.02)
    calibrated_thr = predictor._calibrate_threshold(
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
    if predictor._feature_selector.dropped_names:
        logger.info(
            "ML AdaptiveFeatureSelector dropped (%d): %s",
            len(predictor._feature_selector.dropped_names),
            ", ".join(predictor._feature_selector.dropped_names),
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
    pf_score = predictor._compute_profit_factor_score(y_pred_holdout, pnl_test)
    # Precision 3x recall weight: false positives (losing trades) are far more expensive
    # than missed winners. AUC remains highest (threshold-independent ranking quality).
    final_skill = compute_skill_score(best_precision, best_recall, best_roc_auc, pf_score)

    # --- Precision recovery: when initial pass fails on precision ---
    # Noisy assets (e.g. ETH) often produce high recall / low precision.
    # Recovery strategy:
    #   Phase A: raise threshold on existing ensemble (cheap — no retraining)
    #   Phase B: retrain with conservative hyperparams + higher class penalty
    if best_precision < predictor._cfg.min_precision and best_recall > 0.40:
        logger.info(
            "ML PRECISION RECOVERY triggered: prec=%.3f < min=%.3f (recall=%.3f)",
            best_precision, predictor._cfg.min_precision, best_recall,
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
            if prec_try >= predictor._cfg.min_precision and rec_try >= 0.30:
                pf_try = predictor._compute_profit_factor_score(y_pred_try, pnl_test)
                skill_try = compute_skill_score(prec_try, rec_try, best_roc_auc, pf_try)
                if skill_try >= predictor._cfg.min_skill_score * 0.95:
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

            rf_b = predictor._build_rf(conservative=True)
            try:
                rf_b.fit(X_train_s, y_train, sample_weight=train_weights)
            except TypeError:
                rf_b.fit(X_train_s, y_train)

            lgbm_b = predictor._build_lgbm(scale_pos_weight=spw_boost, conservative=True)
            if lgbm_b is not None:
                try:
                    lgbm_b.fit(X_train_s, y_train, sample_weight=train_weights)
                except Exception:
                    lgbm_b = None

            xgb_b = predictor._build_xgb(scale_pos_weight=spw_boost, conservative=True)
            if xgb_b is not None:
                try:
                    xgb_b.fit(X_train_s, y_train, sample_weight=train_weights)
                except Exception:
                    xgb_b = None

            lr_en_b = predictor._build_elastic_net(conservative=True)
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
                pf_bv = predictor._compute_profit_factor_score(y_pred_bv, pnl_val)
                skill_bv = compute_skill_score(prec_bv, rec_bv, auc_bv, pf_bv)

                y_train_pred_b = candidate_b.predict(X_train_s)
                train_prec_b = precision_score(y_train, y_train_pred_b, zero_division=0)
                noise_margin_b = predictor._overfit_noise_margin(train_prec_b, prec_bv, _n_train, _n_val)
                cand_threshold_b = predictor._cfg.max_overfit_gap + noise_margin_b
                if train_prec_b - prec_bv > cand_threshold_b:
                    logger.warning(
                        "ML recovery [%s]: overfitting gap=%.3f > threshold=%.3f (base %.2f + 1.96σ noise %.3f) — skipped",
                        tag_b, train_prec_b - prec_bv, cand_threshold_b,
                        predictor._cfg.max_overfit_gap, noise_margin_b,
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
                thr_b = predictor._calibrate_threshold(
                    y_val_thr, y_proba_b_val,
                    min_precision=predictor._cfg.min_precision, pnl=pnl_thr_arr_b,
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
                    pf_b = predictor._compute_profit_factor_score(y_pred_b_holdout, pnl_test)
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
    rng = np.random.default_rng(predictor._cfg.random_seed)
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
    # populates predictor._bootstrap_ci when the flag is on, so the default
    # training path stays unchanged.
    if predictor._cfg.use_bootstrap_ci:
        try:
            from analyzer.ml_bootstrap import MLBootstrap
            mb = MLBootstrap(
                n_simulations=predictor._cfg.bootstrap_n_simulations,
                seed=predictor._cfg.random_seed,
            )
            cis = mb.bootstrap_metrics(y_test, y_proba_holdout, threshold=calibrated_thr)
            predictor._bootstrap_ci = {k: v.summary() for k, v in cis.items()}
            prob_above = mb.probability_above_baseline(y_test, y_proba_holdout, baseline_auc=0.5)
            predictor._bootstrap_ci["probability_above_random"] = round(prob_above, 4)
            logger.info(
                "ML bootstrap CI: AUC p5=%.3f p50=%.3f p95=%.3f | P(AUC>0.5)=%.2f",
                cis["roc_auc"].p5, cis["roc_auc"].p50, cis["roc_auc"].p95, prob_above,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("ML bootstrap CI failed: %s", exc)

    # Phase-3: diagnostic — log pairwise error correlation between members.
    # Exposed via predictor._member_error_correlation for dashboard; high
    # correlations (> 0.85) mean a member is paying rent without adding
    # diversity and the feature set needs attention.
    try:
        predictor._member_error_correlation = ensemble.member_error_correlation(X_test_s, y_test)
        if predictor._member_error_correlation:
            worst = max(predictor._member_error_correlation.items(), key=lambda kv: abs(kv[1]))
            logger.info(
                "ML member error correlations: %s | worst pair %s corr=%.3f%s",
                {k: round(v, 3) for k, v in predictor._member_error_correlation.items()},
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
            if predictor._feature_selector.is_fitted:
                X_oot_sel = predictor._feature_selector.transform(X_oot_raw)
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
    ece_cal      = predictor._expected_calibration_error(y_test, y_proba_holdout, n_bins=10)
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
        best_precision < predictor._cfg.min_precision
        or best_recall < predictor._cfg.min_recall
        or best_roc_auc < predictor._cfg.min_roc_auc
        or final_skill < predictor._cfg.min_skill_score
    ):
        logger.warning(
            "ML metrics below threshold: skill=%.3f prec=%.3f rec=%.3f auc=%.3f — NOT deploying",
            final_skill, best_precision, best_recall, best_roc_auc,
        )
        predictor._metrics = metrics
        return metrics

    # Deploy new ensemble
    predictor._ensemble = ensemble
    predictor._model = best_model         # Legacy compat (save_to_file uses it)
    predictor._scaler = final_scaler
    predictor._calibrated_threshold = calibrated_thr
    predictor._model_version = f"ensemble_v{int(time.time())}_{ensemble.member_count()}m"
    predictor._metrics = metrics
    predictor._last_train_ts = int(time.time() * 1000)
    logger.info(
        "ML VotingEnsemble deployed: version=%s skill=%.3f prec=%.3f rec=%.3f auc=%.3f thr=%.3f",
        predictor._model_version, final_skill, best_precision, best_recall, best_roc_auc, calibrated_thr,
    )
    return metrics

# ──────────────────────────────────────────────────────────
# Phase-1/2: walk-forward validation + stacking head
# ──────────────────────────────────────────────────────────
