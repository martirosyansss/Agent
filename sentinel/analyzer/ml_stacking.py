"""
ML Stacking — meta-model over OOF ensemble member predictions.

Soft voting (the default in VotingEnsemble) averages base-model probabilities
with fixed skill-derived weights. That works well when every member is about
equally informative in every region of feature space. It fails when a
member has a region-specific edge — e.g. the ElasticNet LR is the one you'd
trust in strong-trend regimes but the tree models dominate in sideways
regimes. A fixed weight blends the advice together and loses the local
edge.

Stacking replaces the fixed blend with a learnt one. A small LogisticRegression
sits on top of the members' probability outputs and learns when to trust
each. In return for the added capacity we pay a standard stacking tax:

* The meta-model MUST be fitted on out-of-fold (OOF) predictions, NOT on
  in-sample predictions. In-sample probas contain label-leakage — the base
  models saw those labels during training — and a meta fitted on them
  always looks brilliant in CV and never generalises.
* The validator that generated the OOF probas must cover the same samples
  that the caller will feed to ``fit``. Samples outside the validator's
  fold coverage have no OOF prediction and must be masked out (the
  validator exposes an explicit mask for this).

This module only defines the meta-model and its fit/predict API. The OOF
generation lives in ``ml_walk_forward.MLWalkForwardValidator``; the wiring
that calls both in sequence lives in ``MLPredictor.train_walk_forward``.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from monitoring.event_log import emit_component_error

logger = logging.getLogger(__name__)


class StackingHead:
    """LogisticRegression meta-model over per-member probability columns.

    Input shape at fit/predict: ``(n_samples, n_members)`` where columns are
    ordered by the caller's tag list. If ``use_raw_features=True`` the full
    raw feature matrix ``X`` is concatenated on the right, giving the meta
    context beyond just the members' votes (e.g. "trust LR more when ADX
    is high"). That's strictly stronger but also multiplies the number of
    meta-parameters by ~30, so on small OOF sets (< ~800 samples) it tends
    to overfit — start with ``use_raw_features=False`` and only turn it on
    after you've confirmed the simpler head already improves AUC.
    """

    def __init__(
        self,
        use_raw_features: bool = False,
        C: float = 0.1,
        random_seed: int = 42,
    ) -> None:
        self._use_raw = use_raw_features
        self._C = C
        self._seed = random_seed
        self._meta: Optional[Any] = None
        self._member_tags: list[str] = []
        self._n_features: int = 0
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def member_tags(self) -> list[str]:
        return list(self._member_tags)

    def fit(
        self,
        oof_probas: dict[str, np.ndarray],
        y: np.ndarray,
        X: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ) -> bool:
        """Fit the meta-model on OOF member predictions.

        Args:
            oof_probas: ``{tag: probas_array}`` mapping each member to its OOF
                        predictions. Every array must have length ``n_samples``
                        (use ``np.nan`` for samples without an OOF prediction
                        in that member).
            y:          Binary labels, length ``n_samples``.
            X:          Raw feature matrix ``(n_samples, n_features)`` —
                        required if ``use_raw_features=True``, ignored otherwise.
            mask:       Optional boolean mask selecting which samples have
                        full OOF coverage. If None, samples with any NaN
                        across members are dropped automatically.

        Returns:
            True if the meta-model fitted successfully, False otherwise
            (e.g. insufficient data, single-class labels, sklearn missing).
        """
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            logger.warning("StackingHead.fit: sklearn missing — stacking disabled")
            return False

        if not oof_probas:
            logger.warning("StackingHead.fit: empty oof_probas")
            return False

        tags = sorted(oof_probas.keys())
        stacked = np.column_stack([np.asarray(oof_probas[t], dtype=np.float64) for t in tags])
        y_arr = np.asarray(y, dtype=np.int64)

        if len(stacked) != len(y_arr):
            logger.warning(
                "StackingHead.fit: shape mismatch probas=%d labels=%d",
                len(stacked), len(y_arr),
            )
            return False

        if mask is None:
            valid = ~np.isnan(stacked).any(axis=1)
        else:
            valid = np.asarray(mask, dtype=bool)
            # Any NaN inside the masked region is still invalid
            valid = valid & ~np.isnan(stacked).any(axis=1)

        if valid.sum() < 30:
            logger.warning(
                "StackingHead.fit: only %d valid OOF samples, need ≥ 30 — skipping",
                int(valid.sum()),
            )
            return False

        y_fit = y_arr[valid]
        if len(np.unique(y_fit)) < 2:
            logger.warning("StackingHead.fit: single-class OOF labels — skipping")
            return False

        if self._use_raw:
            if X is None:
                logger.warning("StackingHead.fit: use_raw_features=True requires X")
                return False
            X_arr = np.asarray(X, dtype=np.float64)
            if len(X_arr) != len(stacked):
                logger.warning("StackingHead.fit: X length mismatch")
                return False
            features = np.hstack([stacked[valid], X_arr[valid]])
        else:
            features = stacked[valid]

        # m6-5: dropped explicit `penalty="l2"` — sklearn 1.8+ emits a
        # FutureWarning because l2 is the default and the argument is on
        # track for removal in 1.10.
        meta = LogisticRegression(
            C=self._C, solver="lbfgs",
            max_iter=500, class_weight="balanced",
            random_state=self._seed,
        )
        try:
            meta.fit(features, y_fit)
        except Exception as exc:  # noqa: BLE001
            logger.warning("StackingHead.fit: meta training failed: %s", exc)
            emit_component_error(
                "ml_stacking.fit",
                f"meta training failed: {exc}",
                exc=exc,
                severity="warning",
                degraded_to="unfitted_head_means_fallback",
            )
            return False

        self._meta = meta
        self._member_tags = tags
        self._n_features = features.shape[1]
        self._is_fitted = True
        logger.info(
            "StackingHead: fitted on %d samples, tags=%s, n_features=%d, use_raw=%s",
            int(valid.sum()), tags, self._n_features, self._use_raw,
        )
        return True

    def predict_proba(
        self,
        member_probas: np.ndarray,
        X: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply the meta-model.

        Args:
            member_probas: Matrix of per-member live probabilities, shape
                           ``(n_samples, n_members)``. Column order MUST
                           match ``member_tags`` (the caller is responsible;
                           VotingEnsemble._member_probas_matrix() does this).
            X:             Raw features if the head was fitted with
                           ``use_raw_features=True``; ignored otherwise.

        Returns:
            Array of P(win) of shape ``(n_samples,)``. Falls back to the
            column-wise mean if the meta-model is not fitted or prediction
            fails.
        """
        probas = np.asarray(member_probas, dtype=np.float64)
        if not self._is_fitted or self._meta is None:
            # Graceful fallback — mean of members. Caller's VotingEnsemble
            # also has its own fallback, but defence in depth is cheap here.
            return probas.mean(axis=1) if probas.size else np.array([])

        if self._use_raw:
            if X is None:
                logger.warning("StackingHead.predict: X required with use_raw_features")
                return probas.mean(axis=1)
            X_arr = np.asarray(X, dtype=np.float64)
            features = np.hstack([probas, X_arr])
        else:
            features = probas

        if features.shape[1] != self._n_features:
            logger.warning(
                "StackingHead.predict: feature dim mismatch got=%d expected=%d — fallback",
                features.shape[1], self._n_features,
            )
            return probas.mean(axis=1)

        try:
            return self._meta.predict_proba(features)[:, 1]
        except Exception as exc:  # noqa: BLE001
            logger.warning("StackingHead.predict: meta predict failed: %s", exc)
            emit_component_error(
                "ml_stacking.predict",
                f"meta predict failed: {exc}",
                exc=exc,
                severity="warning",
                degraded_to="members_mean",
            )
            return probas.mean(axis=1)
