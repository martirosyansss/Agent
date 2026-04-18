"""
ML Walk-Forward Validation — per-fold training with stability diagnostics.

The current `MLPredictor.train()` uses a single static 70/15/15 chronological
split. That gives one point estimate of skill metrics, but tells you nothing
about how the model behaves across different market epochs. A model that
scores AUC=0.72 on one fold but 0.55 on another has no real edge — it just
got lucky on the holdout window.

This module wraps the existing train routine in a walk-forward loop:

- Split the full trade history into N sequential folds.
- Train on each fold's in-sample window, test on its out-of-sample window.
- Collect per-fold precision/recall/AUC and aggregate (mean, std, min).
- Optionally generate out-of-fold (OOF) probability predictions for every
  trade — these are the correct input for a stacking meta-model
  (in-sample ensemble predictions leak labels and overfit the meta).

Two window modes:
- ``rolling``: train window slides forward (older data drops off)
- ``anchored``: train window always starts at t=0 (accumulates history)

The divergence between anchored and rolling Sharpe is a classic overfitting
signal — see López de Prado, "Advances in Financial Machine Learning", §11.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from monitoring.event_log import emit_component_error

logger = logging.getLogger(__name__)


@dataclass
class WFFoldResult:
    """Result of a single walk-forward fold.

    ``oof_probas`` holds the model's predictions on THIS fold's test slice.
    Concatenating the oof_probas across all folds yields a set of predictions
    where every sample was forecast by a model that did NOT see it during
    training — i.e. true out-of-fold predictions suitable for stacking.
    """
    fold_idx: int
    train_start: int
    train_end: int          # exclusive
    test_start: int
    test_end: int           # exclusive
    precision: float = 0.0
    recall: float = 0.0
    roc_auc: float = 0.5
    skill_score: float = 0.0
    train_precision: float = 0.0       # in-sample precision (for degradation)
    n_train: int = 0
    n_test: int = 0
    calibrated_threshold: float = 0.5
    # Array of P(win) predictions for test_start:test_end; length == n_test
    oof_probas: np.ndarray = field(default_factory=lambda: np.array([]))
    # Raw test labels preserved for downstream stacking / bootstrap
    y_test: np.ndarray = field(default_factory=lambda: np.array([]))
    # Optional per-member OOF probability arrays indexed by ensemble tag.
    # Populated when the trainer surfaces them via the `member_probas` key;
    # used by the stacking head to fit a meta-model on out-of-fold predictions
    # without having to re-run the walk-forward loop a second time.
    member_probas: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class WFReport:
    """Aggregated walk-forward metrics + stability diagnostics.

    ``std_auc`` is the key stability metric: a model with mean_auc=0.72 and
    std_auc=0.03 is reliable; mean_auc=0.72 with std_auc=0.15 means half the
    folds were near-random and half looked great, which is noise, not skill.

    ``degradation`` = mean OOS precision / mean IS precision. Close to 1.0
    means the model generalises; below 0.7 is a red flag (in-sample
    optimisation is bleeding through).

    ``oof_probas`` is the full OOF prediction array (size N) across all folds,
    with NaN for samples never in any test set (first fold's train region
    when ``anchored=False``). Suitable as stacking meta-features.
    """
    fold_results: list[WFFoldResult]
    mean_precision: float = 0.0
    mean_recall: float = 0.0
    mean_auc: float = 0.5
    std_auc: float = 0.0
    min_auc: float = 0.5
    mean_skill: float = 0.0
    degradation: float = 1.0
    n_folds_completed: int = 0
    oof_probas: np.ndarray = field(default_factory=lambda: np.array([]))
    oof_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    mode: str = "rolling"

    def summary(self) -> dict[str, float]:
        """Flat dict for dashboard / JSON serialization."""
        return {
            "mean_precision": round(self.mean_precision, 4),
            "mean_recall": round(self.mean_recall, 4),
            "mean_auc": round(self.mean_auc, 4),
            "std_auc": round(self.std_auc, 4),
            "min_auc": round(self.min_auc, 4),
            "mean_skill": round(self.mean_skill, 4),
            "degradation": round(self.degradation, 4),
            "n_folds_completed": self.n_folds_completed,
            "mode": self.mode,
        }


# Signature a trainer callable must satisfy.
# Given train/test slices it must return a dict with at minimum:
#   {"test_proba": np.ndarray, "train_precision": float, "threshold": float}
TrainerFn = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], dict[str, Any]]


class MLWalkForwardValidator:
    """Walk-forward validator that wraps an arbitrary trainer callable.

    Minimum data requirement: ``n_folds * (min_train + min_test)``. For the
    defaults (5 folds × 100 train × 30 test = 650) that means the validator
    should not be invoked on symbols with fewer than ~650 trades — the caller
    is expected to enforce this via ``MLConfig.min_trades``.

    Two decoupled APIs:
    - ``run()``: full walk-forward with metrics aggregation (most callers)
    - ``generate_oof_predictions()``: just the OOF probability array,
      intended for feeding a stacking meta-model
    """

    def __init__(
        self,
        n_folds: int = 5,
        test_fraction: float = 0.15,
        anchored: bool = False,
        min_train_size: int = 100,
        min_test_size: int = 30,
    ) -> None:
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {n_folds}")
        if not 0.0 < test_fraction < 0.5:
            raise ValueError(f"test_fraction must be in (0, 0.5), got {test_fraction}")
        self.n_folds = n_folds
        self.test_fraction = test_fraction
        self.anchored = anchored
        self.min_train_size = min_train_size
        self.min_test_size = min_test_size

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def generate_splits(self, n_samples: int) -> list[tuple[int, int, int, int]]:
        """Return list of (train_start, train_end, test_start, test_end) tuples.

        Rolling mode (default):
            fold i: train=[i*step : i*step + train_size]
                    test =[i*step + train_size : i*step + train_size + test_size]

        Anchored mode: train always starts at 0, so later folds get progressively
        more training data (closer to how production retraining accumulates).

        Splits are dropped silently if either side is smaller than
        ``min_train_size`` / ``min_test_size`` — caller should check
        ``len(splits)`` to detect underspecified data.
        """
        test_size = max(self.min_test_size, int(n_samples * self.test_fraction))
        # Allocate the tail to tests, the remaining head to training across folds.
        # Reserve one full test_size per fold at the far end, step the window
        # forward in test_size increments.
        total_test_budget = test_size * self.n_folds
        if total_test_budget >= n_samples:
            return []

        train_pool = n_samples - total_test_budget
        # Base training size per fold — rolling window; anchored ignores this
        # and uses an expanding window instead.
        rolling_train_size = max(self.min_train_size, train_pool)

        splits: list[tuple[int, int, int, int]] = []
        for fold_idx in range(self.n_folds):
            test_start = train_pool + fold_idx * test_size
            test_end = test_start + test_size
            if test_end > n_samples:
                break
            if self.anchored:
                train_start = 0
                train_end = test_start
            else:
                train_start = max(0, test_start - rolling_train_size)
                train_end = test_start
            if train_end - train_start < self.min_train_size:
                continue
            if test_end - test_start < self.min_test_size:
                continue
            splits.append((train_start, train_end, test_start, test_end))
        return splits

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        trainer: TrainerFn,
        pnl: Optional[np.ndarray] = None,
    ) -> WFReport:
        """Run walk-forward validation.

        Args:
            X: Feature matrix (n_samples, n_features), chronologically ordered.
            y: Binary labels in {0, 1}, shape (n_samples,).
            trainer: Callable trained per fold — see ``TrainerFn`` signature.
                     Must return a dict with keys ``test_proba``,
                     ``train_precision``, ``threshold``.
            pnl: Optional per-sample PnL used by skill score (ignored if None).

        Returns:
            WFReport with per-fold metrics and aggregated OOF predictions.
        """
        n = len(X)
        splits = self.generate_splits(n)
        mode = "anchored" if self.anchored else "rolling"

        if not splits:
            logger.warning(
                "WF: no valid splits from n=%d samples (need ≥ %d folds × (%d+%d) = %d)",
                n, self.n_folds, self.min_train_size, self.min_test_size,
                self.n_folds * (self.min_train_size + self.min_test_size),
            )
            return WFReport(fold_results=[], mode=mode,
                            oof_probas=np.full(n, np.nan), oof_mask=np.zeros(n, dtype=bool))

        fold_results: list[WFFoldResult] = []
        oof_probas = np.full(n, np.nan)
        oof_mask = np.zeros(n, dtype=bool)

        for fold_idx, (tr_s, tr_e, te_s, te_e) in enumerate(splits):
            X_tr, X_te = X[tr_s:tr_e], X[te_s:te_e]
            y_tr, y_te = y[tr_s:tr_e], y[te_s:te_e]

            # Skip folds with only one class — precision/AUC undefined.
            if len(np.unique(y_tr)) < 2:
                logger.warning("WF fold %d: single-class training set, skipping", fold_idx)
                continue

            try:
                out = trainer(X_tr, y_tr, X_te, y_te)
            except Exception as exc:  # noqa: BLE001
                logger.warning("WF fold %d: trainer raised %s — skipping", fold_idx, exc)
                emit_component_error(
                    "ml_walk_forward.fold",
                    f"trainer failed on fold {fold_idx}: {exc}",
                    exc=exc,
                    severity="warning",
                    fold_idx=fold_idx,
                )
                continue

            probas = np.asarray(out.get("test_proba", np.full(len(y_te), 0.5)), dtype=np.float64)
            threshold = float(out.get("threshold", 0.5))
            train_prec = float(out.get("train_precision", 0.0))
            y_pred = (probas >= threshold).astype(int)

            prec, rec, auc = _metrics_safe(y_te, y_pred, probas)
            skill = _skill_proxy(prec, rec, auc)

            # Trainer may optionally surface per-member OOF probabilities so
            # downstream stacking can fit on them without a second WF run.
            raw_members = out.get("member_probas") or {}
            member_probas: dict[str, np.ndarray] = {
                str(tag): np.asarray(arr, dtype=np.float64)
                for tag, arr in raw_members.items()
                if arr is not None and len(arr) == len(y_te)
            }

            res = WFFoldResult(
                fold_idx=fold_idx,
                train_start=tr_s, train_end=tr_e,
                test_start=te_s, test_end=te_e,
                precision=prec, recall=rec, roc_auc=auc,
                skill_score=skill,
                train_precision=train_prec,
                n_train=tr_e - tr_s, n_test=te_e - te_s,
                calibrated_threshold=threshold,
                oof_probas=probas,
                y_test=y_te.copy(),
                member_probas=member_probas,
            )
            fold_results.append(res)
            oof_probas[te_s:te_e] = probas
            oof_mask[te_s:te_e] = True

            logger.info(
                "WF fold %d/%d [%s]: train=[%d:%d] test=[%d:%d] "
                "prec=%.3f rec=%.3f auc=%.3f skill=%.3f thr=%.3f",
                fold_idx + 1, len(splits), mode,
                tr_s, tr_e, te_s, te_e,
                prec, rec, auc, skill, threshold,
            )

        report = self._aggregate(fold_results, oof_probas, oof_mask, mode)
        logger.info(
            "WF summary [%s]: %d/%d folds | mean_auc=%.3f±%.3f min=%.3f "
            "mean_prec=%.3f degradation=%.3f",
            mode, report.n_folds_completed, len(splits),
            report.mean_auc, report.std_auc, report.min_auc,
            report.mean_precision, report.degradation,
        )
        return report

    def generate_oof_predictions(
        self,
        X: np.ndarray,
        y: np.ndarray,
        trainer: TrainerFn,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute OOF predictions and return (probas, mask).

        ``mask`` is True where OOF predictions exist — earlier samples (before
        the first fold's test window) are NaN and masked False, so the caller
        can filter them out when fitting a meta-model.
        """
        rep = self.run(X, y, trainer)
        return rep.oof_probas, rep.oof_mask

    # ──────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _aggregate(
        fold_results: list[WFFoldResult],
        oof_probas: np.ndarray,
        oof_mask: np.ndarray,
        mode: str,
    ) -> WFReport:
        if not fold_results:
            return WFReport(fold_results=[], mode=mode,
                            oof_probas=oof_probas, oof_mask=oof_mask)

        precs = np.array([r.precision for r in fold_results])
        recs = np.array([r.recall for r in fold_results])
        aucs = np.array([r.roc_auc for r in fold_results])
        skills = np.array([r.skill_score for r in fold_results])
        train_precs = np.array([r.train_precision for r in fold_results])

        mean_oos_prec = float(precs.mean())
        mean_is_prec = float(train_precs.mean()) if train_precs.sum() > 0 else mean_oos_prec
        degradation = mean_oos_prec / mean_is_prec if mean_is_prec > 1e-6 else 1.0

        return WFReport(
            fold_results=fold_results,
            mean_precision=mean_oos_prec,
            mean_recall=float(recs.mean()),
            mean_auc=float(aucs.mean()),
            std_auc=float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0,
            min_auc=float(aucs.min()),
            mean_skill=float(skills.mean()),
            degradation=float(degradation),
            n_folds_completed=len(fold_results),
            oof_probas=oof_probas,
            oof_mask=oof_mask,
            mode=mode,
        )


# ──────────────────────────────────────────────────────────
# Shared metric helpers (duplicated-light so this module stays standalone)
# ──────────────────────────────────────────────────────────

def _metrics_safe(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> tuple[float, float, float]:
    """Compute (precision, recall, auc) without raising on degenerate inputs."""
    try:
        from sklearn.metrics import precision_score, recall_score, roc_auc_score
    except ImportError:
        return 0.0, 0.0, 0.5

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        auc = 0.5
    return prec, rec, auc


def _skill_proxy(precision: float, recall: float, auc: float) -> float:
    """Lightweight skill proxy when no PnL is available.

    Matches the production ``compute_skill_score()`` weights for the three
    non-PnL components. PnL is neutralised (0.5 midpoint) so the skill score
    remains comparable across folds that do/don't pass PnL through.
    """
    return 0.30 * precision + 0.10 * recall + 0.35 * auc + 0.25 * 0.5
