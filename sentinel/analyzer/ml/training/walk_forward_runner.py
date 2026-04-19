"""Walk-forward validation driver.

Extracted from ``MLPredictor.train_walk_forward`` during the round-10
refactor. Keeps the same contract:

* Runs :class:`MLWalkForwardValidator` against a per-fold trainer that
  fits the four ensemble members (RF / LGBM / XGB / ElasticNet) and
  captures per-member OOF predictions on the fold's test window.
* Single pass — per-member OOF is collected on the stability run itself
  so the stacking fitter can read it directly, no second pass.

The runner mutates the caller's ``_wf_report`` attribute (via the
returned report) and, when ``use_stacking`` is on, delegates to
``fit_stacking_head_from_report`` which updates the ensemble +
``_calibrated_threshold`` via the caller's state.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np

from ..domain.config import MLConfig
from .calibration import calibrate_threshold as _calibrate_threshold_fn
from .stacking_fitter import StackingAttachResult, fit_stacking_head_from_report

logger = logging.getLogger(__name__)


def run_walk_forward(
    cfg: MLConfig,
    trades: list[Any],               # list[StrategyTrade]
    extract_features_batch_fn: Callable[[list[Any]], np.ndarray],
    build_rf_fn: Callable[[], Any],
    build_lgbm_fn: Callable[[float], Any],
    build_xgb_fn: Callable[[float], Any],
    build_elastic_net_fn: Callable[[], Any],
    ensemble: Any,                   # VotingEnsemble | None — receives stacking head
    n_folds: int = 5,
    anchored: bool = False,
) -> tuple[Optional[Any], StackingAttachResult]:
    """Run walk-forward validation + (optionally) attach a stacking head.

    Returns ``(WFReport | None, StackingAttachResult)``:

    * The first item is the WF report the caller stashes in
      ``_wf_report`` for the dashboard.
    * The second item describes whether stacking was attached and
      provides a new threshold the caller should install in its
      ``_calibrated_threshold``. Callers ignore the result when
      ``use_stacking`` is off — attached will always be False.
    """
    if not cfg.use_walk_forward and not cfg.use_stacking:
        logger.debug("train_walk_forward: both flags off — noop")
        return None, StackingAttachResult()
    if len(trades) < cfg.min_trades:
        logger.warning(
            "train_walk_forward: need ≥ %d trades, have %d — skipping WF",
            cfg.min_trades, len(trades),
        )
        return None, StackingAttachResult()

    try:
        from analyzer.ml_walk_forward import MLWalkForwardValidator
    except ImportError as exc:
        logger.warning("train_walk_forward: cannot import validator: %s", exc)
        return None, StackingAttachResult()

    X = extract_features_batch_fn(trades)
    y = np.array([1 if t.is_win else 0 for t in trades], dtype=np.int64)

    # Per-fold trainer: fits each ensemble member on the fold's train
    # slice and returns the ensemble's test-set probability. Per-member
    # OOF is collected via the ``member_probas`` key on the same pass so
    # stacking can fit off it directly — no second run, no zip-alignment
    # hazards between fold indices.
    def _fold_trainer(X_tr, y_tr, X_te, y_te):
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import precision_score
        from analyzer.ml_ensemble import VotingEnsemble

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        fold_ens = VotingEnsemble()

        n_neg = int(np.sum(y_tr == 0))
        n_pos = int(np.sum(y_tr == 1))
        spw = n_neg / n_pos if n_pos > 0 else 1.0

        member_probas: dict[str, np.ndarray] = {}
        for builder, tag in (
            (lambda: build_rf_fn(), "rf"),
            (lambda: build_lgbm_fn(spw), "lgbm"),
            (lambda: build_xgb_fn(spw), "xgb"),
            (lambda: build_elastic_net_fn(), "lr_en"),
        ):
            mdl = builder()
            if mdl is None:
                continue
            try:
                mdl.fit(X_tr_s, y_tr)
            except Exception:
                continue
            try:
                member_probas[tag] = mdl.predict_proba(X_te_s)[:, 1]
            except Exception:
                continue
            try:
                v_pred = mdl.predict(X_te_s)
                w = max(0.01, precision_score(y_te, v_pred, zero_division=0))
            except Exception:
                w = 0.1
            fold_ens.add_member(mdl, tag, w)

        if not fold_ens.is_ready:
            return {
                "test_proba": np.full(len(y_te), 0.5),
                "threshold": 0.5,
                "train_precision": 0.0,
                "member_probas": {},
            }

        test_proba = fold_ens.predict_proba(X_te_s)
        train_pred = (fold_ens.predict_proba(X_tr_s) >= 0.5).astype(int)
        try:
            train_prec = precision_score(y_tr, train_pred, zero_division=0)
        except Exception:
            train_prec = 0.0

        return {
            "test_proba": test_proba,
            "threshold": 0.5,
            "train_precision": float(train_prec),
            "member_probas": member_probas,
        }

    # Phase-11 wiring: ``wfv_purge`` and ``wfv_embargo`` live on MLConfig.
    # Both default to 0 so pre-Phase-11 deployments see identical splits.
    # Operators flip them on when rolling-window features (rolling_win_rate_10,
    # recent pnl averages, ...) are in play — those create implicit
    # autocorrelation between train and test that can't be seen in a simple
    # chronological split.
    wf = MLWalkForwardValidator(
        n_folds=n_folds,
        test_fraction=0.15,
        anchored=anchored,
        min_train_size=100,
        min_test_size=30,
        purge=int(getattr(cfg, "wfv_purge", 0) or 0),
        embargo=int(getattr(cfg, "wfv_embargo", 0) or 0),
    )
    if wf.purge or wf.embargo:
        logger.info(
            "walk-forward: purge=%d embargo=%d (López de Prado)",
            wf.purge, wf.embargo,
        )

    report = wf.run(X, y, _fold_trainer)

    stacking_result = StackingAttachResult()
    if cfg.use_stacking and ensemble is not None and getattr(ensemble, "is_ready", False):
        # Round-8 M-1: pass per-sample PnL so the post-stacking threshold
        # re-tune uses the same profit-factor objective as the training-
        # time calibration. Without it the two thresholds optimise
        # different metrics and stacking-on vs stacking-off produce
        # non-comparable decision boundaries.
        pnl_arr = np.array([t.pnl_usd for t in trades], dtype=np.float64)
        # C4-m4 / MA5-2 guardrail — see original docstring.
        if not (len(pnl_arr) == len(X) == len(y)):
            raise ValueError(
                f"pnl_arr / X / y length drift: {len(pnl_arr)} vs {len(X)} vs {len(y)}"
            )
        stacking_result = fit_stacking_head_from_report(
            ensemble=ensemble,
            cfg=cfg,
            report=report,
            X=X,
            y=y,
            pnl=pnl_arr,
            calibrate_threshold_fn=_calibrate_threshold_fn,
        )

    return report, stacking_result
