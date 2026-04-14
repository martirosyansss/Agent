"""
ML Ensemble Module — Soft-Voting VotingEnsemble with skill-weighted probabilities.

Instead of winner-takes-all model selection, combines RF + LightGBM + XGBoost
into a weighted ensemble where each model's weight = its validation skill score.

Final probability = Σ(weight_i * prob_i) / Σ(weight_i)

Benefits:
- More robust predictions (diversification across model types)
- Reduces individual model variance
- Each model contributes proportionally to its quality
- Isotonic calibration ensures reliable probability estimates
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class VotingEnsemble:
    """Soft-voting ensemble combining multiple classifiers by skill weight.

    Implements weighted average of `predict_proba` outputs, where each
    model's weight equals its validation skill score.
    """

    def __init__(self) -> None:
        self._members: list[tuple[Any, str, float]] = []  # (model, tag, weight)
        self._calibrator: Optional[Any] = None
        self._is_calibrated: bool = False

    def add_member(self, model: Any, tag: str, skill_score: float) -> None:
        """Add a trained model to the ensemble."""
        if skill_score <= 0.0:
            logger.warning("VotingEnsemble: skipping %s — zero/negative skill", tag)
            return
        # W-9: Prevent duplicate tags (would double-count a model's vote)
        if any(t == tag for _, t, _ in self._members):
            logger.warning("VotingEnsemble: duplicate tag '%s' — skipping", tag)
            return
        self._members.append((model, tag, skill_score))
        logger.debug("VotingEnsemble: added member [%s] weight=%.3f", tag, skill_score)

    @property
    def is_ready(self) -> bool:
        return len(self._members) > 0

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute weighted average probability across all ensemble members.

        W-4 fix: Only divides by the weight of members that actually succeeded,
        preventing dilution toward zero when a member fails prediction.

        Args:
            X: Feature matrix, shape (n_samples, n_features)

        Returns:
            Probability array, shape (n_samples,) for the positive class
        """
        if not self.is_ready:
            return np.full(len(X), 0.5)

        weighted_proba = np.zeros(len(X))
        actual_weight = 0.0  # W-4: track only successful members' weight

        for model, tag, weight in self._members:
            try:
                proba = model.predict_proba(X)[:, 1]
                weighted_proba += weight * proba
                actual_weight += weight
            except Exception as exc:
                logger.warning("VotingEnsemble: member [%s] predict failed: %s", tag, exc)

        if actual_weight <= 0:
            return np.full(len(X), 0.5)

        return weighted_proba / actual_weight

    def apply_isotonic_calibration(
        self,
        y_val: np.ndarray,
        X_val_s: np.ndarray,
    ) -> None:
        """Calibrate the ensemble's probability output using Isotonic Regression.

        Isotonic calibration corrects systematic bias in predicted probabilities,
        making them correspond to the empirical win rate. This is critical for
        reliable threshold-based filtering.

        Args:
            y_val: True binary labels on validation set
            X_val_s: Scaled validation features
        """
        try:
            from sklearn.isotonic import IsotonicRegression

            raw_proba = self.predict_proba(X_val_s)

            # Fit isotonic regressor mapping raw_proba → empirical frequency
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(raw_proba, y_val)
            self._calibrator = ir
            self._is_calibrated = True

            logger.info(
                "VotingEnsemble: Isotonic calibration applied on %d validation samples",
                len(y_val),
            )
        except Exception as exc:
            logger.warning("VotingEnsemble: calibration failed: %s — using raw proba", exc)

    def predict_proba_calibrated(self, X: np.ndarray) -> np.ndarray:
        """Predict probability with isotonic calibration applied if available."""
        raw = self.predict_proba(X)
        if self._is_calibrated and self._calibrator is not None:
            try:
                return self._calibrator.predict(raw)
            except (ValueError, TypeError) as exc:
                logger.warning("VotingEnsemble: calibration predict failed: %s", exc)
        return raw

    def member_count(self) -> int:
        return len(self._members)

    def get_member_info(self) -> list[dict]:
        return [{"tag": tag, "weight": w} for _, tag, w in self._members]


class AdaptiveFeatureSelector:
    """Selects features with importance above threshold, removing low-signal noise.

    After training, low-importance features can add noise and slow predictions.
    This selector builds and applies a boolean mask to keep only significant features.

    Usage:
        selector = AdaptiveFeatureSelector(min_importance=0.01)
        selector.fit(feature_importances_dict, feature_names)
        X_filtered = selector.transform(X)
    """

    def __init__(self, min_importance: float = 0.01) -> None:
        self._min_importance = min_importance
        self._mask: Optional[np.ndarray] = None
        self._selected_names: list[str] = []
        self._dropped_names: list[str] = []

    @property
    def is_fitted(self) -> bool:
        return self._mask is not None

    def fit(
        self,
        importances: dict[str, float],
        feature_names: list[str],
    ) -> None:
        """Fit the selector using model feature importances.

        Args:
            importances: Dict mapping feature name → importance score
            feature_names: Ordered list of feature names (same order as columns in X)
        """
        mask = []
        selected = []
        dropped = []

        for name in feature_names:
            imp = importances.get(name, 0.0)
            keep = imp >= self._min_importance
            mask.append(keep)
            if keep:
                selected.append(name)
            else:
                dropped.append(name)

        self._mask = np.array(mask, dtype=bool)
        self._selected_names = selected
        self._dropped_names = dropped

        logger.info(
            "AdaptiveFeatureSelector: keeping %d/%d features (dropped: %s)",
            len(selected),
            len(feature_names),
            ", ".join(dropped) if dropped else "none",
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply feature mask to array (columns mode).

        Args:
            X: Shape (n_samples, n_features) or (1, n_features)

        Returns:
            X with low-importance columns removed
        """
        if self._mask is None:
            return X
        return X[:, self._mask]

    def transform_single(self, features: list[float]) -> list[float]:
        """Apply feature mask to a single flat feature list."""
        if self._mask is None:
            return features
        arr = np.array(features)
        return arr[self._mask].tolist()

    @property
    def selected_feature_count(self) -> int:
        return len(self._selected_names)

    @property
    def selected_names(self) -> list[str]:
        return list(self._selected_names)

    @property
    def dropped_names(self) -> list[str]:
        return list(self._dropped_names)
