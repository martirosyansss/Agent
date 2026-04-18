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

from monitoring.event_log import emit_component_error

logger = logging.getLogger(__name__)


class _PlattCalibrator:
    """Platt scaling calibrator with an IsotonicRegression-compatible predict() API.

    Wraps a sklearn LogisticRegression fitted on raw probabilities so callers
    can stay agnostic about which calibration method was used.
    """

    def __init__(self, logistic_reg: Any) -> None:
        self._lr = logistic_reg

    def predict(self, raw_proba: np.ndarray) -> np.ndarray:
        arr = np.asarray(raw_proba).reshape(-1, 1)
        return self._lr.predict_proba(arr)[:, 1]


class VotingEnsemble:
    """Soft-voting ensemble combining multiple classifiers by skill weight.

    Implements weighted average of `predict_proba` outputs, where each
    model's weight equals its validation skill score.
    """

    def __init__(self) -> None:
        self._members: list[tuple[Any, str, float]] = []  # (model, tag, weight)
        self._calibrator: Optional[Any] = None
        self._is_calibrated: bool = False
        self._calibration_method: str = "none"  # "none" | "platt" | "isotonic"
        # Phase-2: optional stacking meta-model. When set, overrides the soft-vote
        # weighted average in predict_proba_calibrated(). Must be fitted on OOF
        # predictions to avoid label leakage — see set_stacking_head() docstring.
        self._stacking_head: Optional[Any] = None

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

    # Minimum validation-set size for isotonic calibration. Below this we fall
    # back to Platt (sigmoid) scaling, which fits only two parameters and is
    # far more stable on small/medium samples.
    #
    # Empirically 50 was too low: at 50–150 samples isotonic still collapses
    # into 3–5 step plateaus, often with a high one (~0.85–0.96), so every raw
    # score gets mapped onto that plateau and the displayed "ML probability"
    # looks systematically inflated. Platt (1 free parameter, monotonic
    # sigmoid) keeps the ranking and produces smooth, well-spread outputs in
    # this regime. We only switch to isotonic once there are enough samples
    # (~20 per bin × 10 bins) for the staircase to actually resolve the
    # underlying P(win|score) curve.
    MIN_SAMPLES_ISOTONIC: int = 200

    # ──────────────────────────────────────────────────────────
    # Public calibrator surface (Round-10 Step 5)
    #
    # The training code previously reached into ``self._calibrator``
    # directly — see _fit_stacking_head_from_report. The four methods
    # below expose exactly what callers need without leaking the
    # private attribute:
    #
    #   has_calibrator()       → is a fitted calibrator in place?
    #   refit_calibrator(...)  → public alias for _fit_calibrator_on_probas
    #   apply_calibrator(...)  → run the calibrator on pre-computed probas
    #   calibration_method     → property (already existed) returning the tag
    #
    # After this change, ``_calibrator`` is a true implementation detail.
    # Future refactors that change its storage / shape (e.g. a registry of
    # calibrators for stacked vs voting modes) won't break any caller.
    # ──────────────────────────────────────────────────────────

    def has_calibrator(self) -> bool:
        """True when a calibrator has been fitted and can be applied."""
        return self._is_calibrated and self._calibrator is not None

    def refit_calibrator(
        self,
        y: np.ndarray,
        probas: np.ndarray,
        source: str = "voting",
    ) -> bool:
        """Fit (or refit) the ensemble's calibrator on pre-computed probas.

        Replaces direct access to ``_fit_calibrator_on_probas``. Returns
        True on success, False on silent-skip conditions (n<10, single
        class, sklearn exception) so callers can react without having to
        re-inspect the calibrator field to detect the skip.
        """
        return self._fit_calibrator_on_probas(y, probas, source=source)

    def apply_calibrator(self, probas: np.ndarray) -> np.ndarray:
        """Run the calibrator on pre-computed probabilities.

        Returns the input unchanged when no calibrator is fitted, so
        callers can write ``ens.apply_calibrator(raw)`` unconditionally
        without guarding with ``has_calibrator``. Errors (incompatible
        input shape, etc.) also fall back to the input.
        """
        if not self.has_calibrator():
            return probas
        try:
            return self._calibrator.predict(probas)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "VotingEnsemble.apply_calibrator: failed — returning input unchanged (%s)",
                exc,
            )
            emit_component_error(
                "ml_ensemble.calibrator",
                f"apply failed: {exc}",
                exc=exc,
                severity="warning",
                degraded_to="raw_proba",
            )
            return probas

    def apply_isotonic_calibration(
        self,
        y_val: np.ndarray,
        X_val_s: np.ndarray,
    ) -> None:
        """Calibrate ensemble probabilities using Isotonic (≥MIN_SAMPLES_ISOTONIC)
        or Platt (below) — see MIN_SAMPLES_ISOTONIC for the rationale.

        On small/medium validation sets, unconstrained isotonic regression
        collapses into 3-5 step plateaus — predictions become bimodal (e.g.
        0.6, 0.96) and the block threshold no longer separates signal from
        noise. Platt scaling (logistic regression, monotonic sigmoid) produces
        smooth, well-behaved probabilities and is the default until the
        validation set is large enough for isotonic bins to resolve.

        Args:
            y_val: True binary labels on validation set
            X_val_s: Scaled validation features
        """
        raw_proba = self.predict_proba(X_val_s)
        self._fit_calibrator_on_probas(y_val, raw_proba, source="voting")

    def _fit_calibrator_on_probas(
        self,
        y: np.ndarray,
        probas: np.ndarray,
        source: str = "voting",
    ) -> bool:
        """Fit Platt/Isotonic on pre-computed probabilities.

        Split out from ``apply_isotonic_calibration`` so both the voting
        path (which runs ``predict_proba(X)`` internally) AND the stacking
        path (which supplies stacked probabilities directly, from OOF
        predictions) use identical calibrator selection logic.

        Returns ``True`` only when a fresh calibrator was successfully
        installed. All three silent-skip branches below return ``False``
        so the caller can detect "old calibrator still in place" and take
        the appropriate action (e.g. the stacking path must NOT attach the
        deploy head if this returns False — doing so would route stacked
        output through the voting-fit calibrator, which is the original
        J-1 bug).
        """
        y = np.asarray(y)
        probas = np.asarray(probas, dtype=np.float64)
        n = len(y)

        if n < 10:
            logger.warning(
                "VotingEnsemble: calibration [%s] skipped — only %d samples",
                source, n,
            )
            return False
        if len(np.unique(y)) < 2:
            logger.warning(
                "VotingEnsemble: calibration [%s] skipped — validation set has only one class",
                source,
            )
            return False

        try:
            if n >= self.MIN_SAMPLES_ISOTONIC:
                from sklearn.isotonic import IsotonicRegression
                ir = IsotonicRegression(out_of_bounds="clip")
                ir.fit(probas, y)
                self._calibrator = ir
                self._is_calibrated = True
                self._calibration_method = "isotonic"
                logger.info(
                    "VotingEnsemble: Isotonic calibration [%s] on %d samples",
                    source, n,
                )
            else:
                from sklearn.linear_model import LogisticRegression
                lr = LogisticRegression(C=1.0, solver="lbfgs")
                lr.fit(probas.reshape(-1, 1), y)
                self._calibrator = _PlattCalibrator(lr)
                self._is_calibrated = True
                self._calibration_method = "platt"
                logger.info(
                    "VotingEnsemble: Platt scaling [%s] on %d samples (isotonic needs ≥%d)",
                    source, n, self.MIN_SAMPLES_ISOTONIC,
                )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "VotingEnsemble: calibration [%s] failed: %s — using raw proba",
                source, exc,
            )
            emit_component_error(
                "ml_ensemble.calibration_fit",
                f"fit failed ({source}): {exc}",
                exc=exc,
                severity="warning",
                source=source,
                degraded_to="raw_proba",
            )
            return False

    def predict_proba_calibrated(self, X: np.ndarray) -> np.ndarray:
        """Predict probability with isotonic calibration applied if available.

        Dispatch order (first match wins):
          1. Stacking head, if one has been attached via set_stacking_head().
             The head receives the per-member probability matrix and decides
             how to combine them (learnt meta-model, not a fixed weight rule).
             Its output is then passed through the same calibrator that the
             soft-vote path uses — downstream code treats the returned value
             as a calibrated probability, so the stacking branch must honour
             that contract rather than silently outputting raw LogisticRegression
             scores that could be mis-calibrated.
          2. Isotonic / Platt calibrator on top of soft-voting average.
          3. Raw soft-voting average.
        """
        if self._stacking_head is not None:
            try:
                member_probas, _ = self._member_probas_matrix(X)
                stacked = np.asarray(
                    self._stacking_head.predict_proba(member_probas, X),
                    dtype=np.float64,
                )
                # Apply the existing calibrator to the stacked output so the
                # mapping proba → P(win) matches what the rest of the pipeline
                # (threshold tuner, ECE, Brier) expects. The calibrator was
                # fit on soft-voting outputs, so it's a reasonable monotonic
                # remap; if it fails we return the stacked probability as-is.
                if self._is_calibrated and self._calibrator is not None:
                    try:
                        return self._calibrator.predict(stacked)
                    except (ValueError, TypeError) as exc:
                        logger.warning(
                            "VotingEnsemble: calibrator failed on stacked output (%s) — returning raw stacked",
                            exc,
                        )
                return stacked
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "VotingEnsemble: stacking head predict failed (%s) — falling back to voting",
                    exc,
                )
                emit_component_error(
                    "ml_ensemble.stacking_head",
                    f"stacking predict failed: {exc}",
                    exc=exc,
                    severity="warning",
                    degraded_to="soft_voting",
                )
        raw = self.predict_proba(X)
        if self._is_calibrated and self._calibrator is not None:
            try:
                return self._calibrator.predict(raw)
            except (ValueError, TypeError) as exc:
                logger.warning("VotingEnsemble: calibration predict failed: %s", exc)
        return raw

    def member_count(self) -> int:
        return len(self._members)

    @property
    def calibration_method(self) -> str:
        """Which calibrator is currently active: 'none' | 'platt' | 'isotonic'."""
        return self._calibration_method

    def get_member_info(self) -> list[dict]:
        return [{"tag": tag, "weight": w} for _, tag, w in self._members]

    # ──────────────────────────────────────────────────────────
    # Phase-3: diagnostic — pairwise error correlation between members
    # ──────────────────────────────────────────────────────────

    def member_error_correlation(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict[str, float]:
        """Pearson correlation of error vectors between every pair of members.

        A voting ensemble can only cancel *uncorrelated* errors. If two
        members agree on which samples they get wrong (corr > 0.85 is the
        rough red line), averaging their votes doesn't buy diversification
        — one of them is paying rent without contributing. Logging this
        number lets operators tell "we have 4 models" from "we have 1 model
        in 4 hats".

        Errors are signed residuals ``proba - y`` so pairs that overestimate
        the same samples show positive correlation and pairs that disagree
        in direction show negative correlation (rare but possible).

        Returns a dict keyed by ``"tag_a__tag_b"`` (sorted so each pair
        appears once). Empty dict if fewer than two members are ready.
        """
        if len(self._members) < 2:
            return {}
        errors: dict[str, np.ndarray] = {}
        y = np.asarray(y_val, dtype=np.float64)
        for model, tag, _ in self._members:
            try:
                proba = model.predict_proba(X_val)[:, 1]
            except Exception as exc:
                logger.debug("error_correlation: %s predict failed: %s", tag, exc)
                continue
            errors[tag] = np.asarray(proba, dtype=np.float64) - y
        tags = sorted(errors.keys())
        out: dict[str, float] = {}
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                a, b = tags[i], tags[j]
                ea, eb = errors[a], errors[b]
                if ea.std() < 1e-9 or eb.std() < 1e-9:
                    corr = 0.0
                else:
                    corr = float(np.corrcoef(ea, eb)[0, 1])
                key = f"{a}__{b}"
                out[key] = corr
        return out

    # ──────────────────────────────────────────────────────────
    # Phase-2: optional stacking head override
    # ──────────────────────────────────────────────────────────

    def set_stacking_head(self, head: Optional[Any]) -> None:
        """Attach a fitted StackingHead to override soft-voting at predict time.

        When set, ``predict_proba_calibrated()`` delegates to the head's
        ``predict_proba()`` using the current members' raw probabilities as
        inputs. Pass ``None`` to revert to weighted voting.

        The head must have been fitted on OUT-OF-FOLD member probabilities
        (see ``MLWalkForwardValidator.generate_oof_predictions``) — fitting
        it on in-sample predictions leaks training labels and produces a
        meta-model that appears to dominate the base members but fails on
        fresh data.
        """
        self._stacking_head = head

    def _member_probas_matrix(self, X: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Column-stacked member probas (shape: n_samples × n_members).

        When a member fails ``predict_proba`` we replace its column with the
        row-wise mean of the successful members, not a blind 0.5. Using 0.5
        would collide with "member is genuinely uncertain", and the stacking
        head — fit on OOF data where members did succeed — has no way to
        tell the two apart. Falling back to the consensus vote keeps the
        failed member's row vote close to what the other members say, so
        the meta-model sees a sample that looks like "everyone agrees"
        instead of "one member says 50/50". If all members fail, we default
        to 0.5 (no information left to recover).
        """
        cols: list[np.ndarray] = []
        tags: list[str] = []
        failures: list[int] = []
        for idx, (model, tag, _) in enumerate(self._members):
            try:
                cols.append(np.asarray(model.predict_proba(X)[:, 1], dtype=np.float64))
            except Exception:
                failures.append(idx)
                cols.append(np.full(len(X), np.nan))  # placeholder, fixed below
            tags.append(tag)

        if not cols:
            return np.empty((len(X), 0)), tags

        matrix = np.column_stack(cols)

        # Fill any column belonging to a crashed member with the row-wise
        # consensus of the successful members (explained above).
        if failures:
            mask = np.zeros(matrix.shape[1], dtype=bool)
            mask[failures] = True
            successful = matrix[:, ~mask]
            if successful.size:
                with np.errstate(invalid="ignore"):
                    row_mean = np.nanmean(successful, axis=1)
                row_mean = np.where(np.isnan(row_mean), 0.5, row_mean)
            else:
                row_mean = np.full(len(X), 0.5)
            for idx in failures:
                matrix[:, idx] = row_mean

        # C4-M1: clamp any lingering NaN in the final matrix regardless of
        # whether a member formally "failed". A member can succeed at
        # predict_proba but still emit NaN on specific rows — sklearn edge
        # cases, scaler-over-dead-feature outputs, XGBoost sparse-predict
        # degenerate cases all produce that. StackingHead.predict_proba
        # would then raise on NaN meta-features and the ensemble would
        # fall back to voting, hiding the real problem. Clamp to 0.5
        # (neutral probability) so the downstream path stays deterministic.
        if np.isnan(matrix).any():
            matrix = np.where(np.isnan(matrix), 0.5, matrix)

        return matrix, tags


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
