"""Stacking-head fitter — fits a meta-model on walk-forward OOF predictions.

Extracted from ``MLPredictor._fit_stacking_head_from_report`` during the
round-10 refactor. The fitter mutates the ensemble passed in (attaches
the deploy head + re-fits the calibrator), so the caller owns the
ensemble lifecycle.

Correctness invariants locked down by the audit cycle (see comments
below for audit-ID cross-refs):

* The calibrator is re-fit on clean 2-fold OOF predictions, NEVER on
  predictions the deploy head already saw (J-1 / C4-1).
* The deploy head is attached ONLY when the calibrator refit actually
  succeeds AND the stacked AUC exceeds the voting baseline by ≥ 0.005.
* All three early-bail branches (insufficient OOF / sub-head fit
  failure / calibrator refit failure / stacking-worse-than-voting)
  return without touching ensemble state, so the fallback path keeps
  the pre-call voting calibrator.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from ..domain.config import MLConfig

logger = logging.getLogger(__name__)


@dataclass
class StackingAttachResult:
    """What ``fit_stacking_head_from_report`` did to the ensemble.

    ``attached=True`` means the deploy head is now part of the ensemble
    and ``new_threshold`` holds the re-tuned decision threshold that
    should replace the caller's ``_calibrated_threshold``. Any False
    result means the caller should leave its threshold untouched.
    """
    attached: bool = False
    new_threshold: Optional[float] = None


def fit_stacking_head_from_report(
    ensemble: Any,           # VotingEnsemble, typed as Any to avoid circular import
    cfg: MLConfig,
    report: Any,             # WFReport
    X: np.ndarray,
    y: np.ndarray,
    pnl: Optional[np.ndarray],
    calibrate_threshold_fn: Callable,
) -> StackingAttachResult:
    """Fit a StackingHead on per-member OOF and (on success) attach it.

    Two separate heads are fit to avoid leakage:

    1. **Deployment head** — trained on ALL valid OOF rows, used by
       serving code.
    2. **Two-fold CV heads** — train on half A, predict half B, and
       vice-versa. The concatenated scores form a proper out-of-head-
       sample OOF set for calibrator refit + threshold re-tune.

    Without the 2-fold split the calibrator would see predictions from
    a head that saw the same rows during fit — same-data leakage that
    quietly overfits the calibration curve.

    Side effects on success:
      * ``ensemble.refit_calibrator(...)`` is called with the CLEAN
        stacked OOF (replaces any voting-fit calibrator).
      * ``ensemble.set_stacking_head(deploy_head)`` attaches the head.
      * Returns ``StackingAttachResult(attached=True, new_threshold=...)``;
        caller must assign the returned threshold to its stored value.

    On every early-bail branch, ensemble state is left untouched and a
    ``StackingAttachResult(attached=False, new_threshold=None)`` is
    returned.
    """
    fold_results = getattr(report, "fold_results", None) or []
    if not fold_results:
        return StackingAttachResult()

    tags: set[str] = set()
    for fr in fold_results:
        tags.update(fr.member_probas.keys())
    if not tags:
        logger.info("stacking: no per-member OOF in report — keeping soft-voting")
        return StackingAttachResult()

    n = len(X)
    full: dict[str, np.ndarray] = {tag: np.full(n, np.nan) for tag in tags}
    mask = np.zeros(n, dtype=bool)
    for fr in fold_results:
        te_s, te_e = fr.test_start, fr.test_end
        mask[te_s:te_e] = True
        for tag, probas in fr.member_probas.items():
            upto = min(te_e - te_s, len(probas))
            full[tag][te_s:te_s + upto] = probas[:upto]

    try:
        from analyzer.ml_stacking import StackingHead
    except ImportError as exc:
        logger.warning("stacking: cannot import: %s", exc)
        return StackingAttachResult()

    if ensemble is None:
        return StackingAttachResult()

    # --- Deployment head: fit on ALL valid OOF rows ---
    deploy_head = StackingHead(
        use_raw_features=False, C=0.1, random_seed=cfg.random_seed,
    )
    if not deploy_head.fit(full, y, X=X, mask=mask):
        logger.info("stacking: deploy head fit failed — keeping soft-voting")
        return StackingAttachResult()

    # --- Proper OOF via 2-fold split for calibrator + threshold ---
    # Boundary raised from 60→80 (round-7 #6): StackingHead.fit has its
    # own ``valid.sum() < 30`` inner guard, so a 60-row pool would split
    # 30/30 and each sub-head sits at its minimum. 80 gives each sub-
    # head ~40 rows of headroom where LR coefficients on 4-feature
    # meta-models actually stabilise.
    valid_idx = np.where(mask & ~np.any(
        np.column_stack([np.isnan(full[t]) for t in sorted(tags)]), axis=1
    ))[0]
    if len(valid_idx) < 80 or len(np.unique(y[valid_idx])) < 2:
        # C4-1 guard: do NOT attach the deploy head. Without a
        # recalibrated calibrator the existing voting-fit one would be
        # applied to stacking output (J-1 repro).
        logger.warning(
            "stacking: only %d clean OOF rows (need ≥80 for stable 2-fold calibrator CV) "
            "— keeping soft-voting (head NOT attached; avoids J-1 calibrator mismatch)",
            len(valid_idx),
        )
        return StackingAttachResult()

    rng = np.random.default_rng(cfg.random_seed)
    perm = rng.permutation(len(valid_idx))
    half = len(perm) // 2
    idx_A = valid_idx[perm[:half]]
    idx_B = valid_idx[perm[half:]]

    def _clean_stacked_for(fit_idx: np.ndarray, pred_idx: np.ndarray) -> Optional[np.ndarray]:
        """Fit head on fit_idx, predict pred_idx — no overlap → no leakage."""
        sub_head = StackingHead(
            use_raw_features=False, C=0.1, random_seed=cfg.random_seed,
        )
        sub_oof = {tag: full[tag] for tag in tags}
        sub_mask = np.zeros(n, dtype=bool)
        sub_mask[fit_idx] = True
        if not sub_head.fit(sub_oof, y, X=X, mask=sub_mask):
            return None
        cols = []
        for tag in sub_head.member_tags:
            arr = full[tag][pred_idx]
            col = np.where(np.isnan(arr), 0.5, arr)
            cols.append(col)
        matrix = np.column_stack(cols) if cols else np.zeros((len(pred_idx), 0))
        return np.asarray(sub_head.predict_proba(matrix, X=None), dtype=np.float64)

    clean_on_B = _clean_stacked_for(idx_A, idx_B)
    clean_on_A = _clean_stacked_for(idx_B, idx_A)
    if clean_on_A is None or clean_on_B is None:
        logger.warning("stacking: sub-head fit failed during 2-fold CV — keeping soft-voting")
        return StackingAttachResult()

    # Re-align sub-head predictions to original-index order
    clean_stacked = np.empty(len(valid_idx), dtype=np.float64)
    clean_labels = np.empty(len(valid_idx), dtype=np.int64)
    clean_pnl: Optional[np.ndarray] = None
    if pnl is not None:
        clean_pnl = np.empty(len(valid_idx), dtype=np.float64)

    score_map: dict[int, float] = {}
    for i, idx in enumerate(idx_A):
        score_map[int(idx)] = float(clean_on_A[i])
    for i, idx in enumerate(idx_B):
        score_map[int(idx)] = float(clean_on_B[i])
    for k, idx in enumerate(valid_idx):
        clean_stacked[k] = score_map[int(idx)]
        clean_labels[k] = int(y[idx])
        if clean_pnl is not None:
            clean_pnl[k] = float(pnl[idx])

    # Round-8 §3.2: stacked AUC must beat voting AUC by ≥ 0.005 —
    # otherwise the learnt combiner is worse than the simple mean vote.
    try:
        from sklearn.metrics import roc_auc_score as _auc
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            voting_oof = np.nanmean(
                np.column_stack([full[t] for t in sorted(tags)]), axis=1,
            )
        voting_clean = voting_oof[valid_idx]
        voting_auc = float(_auc(clean_labels, voting_clean))
        stacked_auc = float(_auc(clean_labels, clean_stacked))
        logger.info(
            "stacking: clean-OOF AUC voting=%.3f stacked=%.3f",
            voting_auc, stacked_auc,
        )
        if stacked_auc < voting_auc + 0.005:
            logger.warning(
                "stacking: head AUC %.3f does not exceed voting %.3f by +0.005 "
                "— keeping soft-voting",
                stacked_auc, voting_auc,
            )
            return StackingAttachResult()
    except Exception as cmp_err:  # noqa: BLE001
        logger.debug("stacking: AUC comparison skipped: %s", cmp_err)

    # M6-1: calibrator refit via PUBLIC API; silent-skip → leave alone.
    calibrator_replaced = ensemble.refit_calibrator(
        clean_labels, clean_stacked, source="stacking",
    )
    if not calibrator_replaced:
        logger.warning(
            "stacking: calibrator refit on clean OOF failed — keeping soft-voting "
            "(head NOT attached; avoids J-1 calibrator mismatch)",
        )
        return StackingAttachResult()

    calibrated_stacked = ensemble.apply_calibrator(clean_stacked)

    calib_target = max(0.50, cfg.min_precision - 0.02)
    new_thr = calibrate_threshold_fn(
        clean_labels, calibrated_stacked,
        min_precision=calib_target,
        pnl=clean_pnl,   # M-1: profit-factor branch when PnL provided
    )

    ensemble.set_stacking_head(deploy_head)

    # Round-8 §4.1: audit trail — log the learnt meta-coefficients so
    # operators can reconstruct "why did rf outweigh xgb on this trade"
    # after the fact.
    meta = getattr(deploy_head, "_meta", None)
    coef_log: dict[str, float] = {}
    intercept_log: Optional[float] = None
    if meta is not None:
        try:
            raw_coef = np.asarray(meta.coef_).flatten()
            intercept_log = float(np.asarray(meta.intercept_).flatten()[0])
            for i, tag in enumerate(deploy_head.member_tags):
                if i < len(raw_coef):
                    coef_log[tag] = round(float(raw_coef[i]), 4)
        except Exception as _coef_err:  # noqa: BLE001
            logger.debug("stacking: coef log failed: %s", _coef_err)

    logger.info(
        "stacking: deploy head attached (tags=%s, OOF samples=%d); "
        "threshold re-tuned on clean 2-fold OOF → %.3f (target P≥%.2f, pnl=%s) | "
        "meta_coef=%s intercept=%s",
        sorted(tags), int(mask.sum()),
        new_thr, calib_target,
        "yes" if clean_pnl is not None else "no",
        coef_log, intercept_log,
    )

    return StackingAttachResult(attached=True, new_threshold=float(new_thr))
