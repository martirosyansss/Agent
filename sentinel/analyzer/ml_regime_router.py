"""
ML Regime Router — per-regime VotingEnsembles with fallback to a global model.

A single ensemble trained on every trade is forced to find one decision
boundary that straddles every market regime it saw: the momentum
signatures of trending regimes and the mean-reversion signatures of
sideways regimes both get averaged into the same model. You end up with
a boundary that's mediocre in both — which is worse than two focussed
boundaries.

The router solves this by partitioning training trades on
``StrategyTrade.market_regime`` (produced upstream by
``strategy/market_regime.py``) and fitting a dedicated ensemble per
regime. At predict time the current regime is read from the incoming
features; the matching ensemble handles the decision.

Fallback rules:
- If a regime has fewer than ``min_trades_per_regime`` samples, we don't
  train a specialist — the ensemble would overfit. That bucket falls back
  to the global model.
- If the router is invoked with an unseen regime (e.g. ``unknown``),
  fallback to the global model.
- If no global model is available either, return a neutral 0.5 so the
  caller can decide what to do (typically: default to the non-ML signal).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from monitoring.event_log import emit_component_error

logger = logging.getLogger(__name__)


@dataclass
class RegimeStats:
    """Training statistics for a single regime bucket.

    ``trained`` tells the dashboard whether this regime has its own model
    (True) or falls back to the global one (False). ``skill_score`` is the
    aggregated validation skill of the regime-specific ensemble — useful
    when comparing which regimes benefit most from specialisation.
    """
    regime: str
    n_trades: int
    trained: bool
    skill_score: float = 0.0
    mean_precision: float = 0.0
    mean_auc: float = 0.5
    fallback_reason: str = ""

    def summary(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "n_trades": self.n_trades,
            "trained": self.trained,
            "skill_score": round(self.skill_score, 4),
            "mean_precision": round(self.mean_precision, 4),
            "mean_auc": round(self.mean_auc, 4),
            "fallback_reason": self.fallback_reason,
        }


@dataclass
class RegimeModel:
    """Trained artifacts for a single regime.

    Stored directly in the router so the unpickled router is ready to
    predict without any external model registry. Everything needed to go
    from raw features to a calibrated probability lives here.
    """
    regime: str
    ensemble: Any                    # VotingEnsemble — left as Any to avoid circular import
    scaler: Any                      # StandardScaler
    selector: Any                    # AdaptiveFeatureSelector (or None)
    threshold: float = 0.5
    skill_score: float = 0.0
    n_train: int = 0
    metrics_summary: dict[str, float] = field(default_factory=dict)


class RegimeRouter:
    """Routes predictions to the ensemble matching the trade's market regime.

    This class is intentionally trainer-agnostic: it doesn't know how to
    fit a model, only how to partition data and call a trainer callable.
    Caller (typically ``MLPredictor.train_with_regime_routing``) supplies
    a trainer that takes a list of trades and returns a ``RegimeModel``.
    That keeps all the sklearn / feature-engineering plumbing in one place
    instead of being duplicated here.
    """

    UNKNOWN_REGIMES: tuple[str, ...] = ("unknown", "transitioning")

    def __init__(
        self,
        min_trades_per_regime: int = 100,
    ) -> None:
        if min_trades_per_regime < 20:
            raise ValueError(f"min_trades_per_regime must be ≥ 20, got {min_trades_per_regime}")
        self.min_trades_per_regime = min_trades_per_regime
        self._models: dict[str, RegimeModel] = {}
        self._global: Optional[RegimeModel] = None
        self._stats: dict[str, RegimeStats] = {}
        self._is_ready: bool = False

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Unpickle hook: restore state and re-emit the 'no specialists'
        diagnostic if the reloaded router has no trained specialists.

        Rationale: ``train_with_regime_routing`` logs an upfront warning when
        ``min_trades_per_regime`` is too small for specialists to train. On
        pickle reload we skip training entirely, so that warning would never
        fire and operators could run for weeks with a router that silently
        always falls through to the global model. Emitting the same
        diagnostic on load surfaces the misconfiguration when the bot
        restarts.

        Two failure modes:
        * specialists empty, global present → silent all-fallback mode (WARNING)
        * specialists empty AND global absent → returns neutral 0.5 for
          every call (also WARNING, not ERROR: this is a DOCUMENTED supported
          state — see ``predict_proba`` neutral-signal branch — so alerting
          at ERROR would page operators for an intentional behaviour).

        Backwards compat: pickles from before this hook existed don't carry
        some of the newer attributes (e.g. ``_is_ready``, ``_stats``). We
        fall those back to safe defaults before applying ``__dict__.update``
        so ``is_ready`` / ``get_regime_stats`` don't AttributeError on a
        legacy save.
        """
        # MI5-3: older pickles may lack fields added later; set defaults so
        # attribute lookups post-load don't raise.
        state.setdefault("_stats", {})
        state.setdefault("_models", {})
        state.setdefault("_global", None)
        state.setdefault(
            "_is_ready",
            bool(state.get("_models")) or state.get("_global") is not None,
        )
        state.setdefault("min_trades_per_regime", 100)
        self.__dict__.update(state)

        if not self._models:
            if self._global is not None:
                logger.warning(
                    "RegimeRouter reloaded without any trained specialists — "
                    "every predict will fall through to the global model. "
                    "Raise min_trades_per_regime if this is unexpected, or "
                    "disable use_regime_routing to skip the indirection.",
                )
            else:
                # MI5-5: was logger.error — downgraded because the resulting
                # neutral-0.5 behaviour is a documented supported state, not
                # an impossible one. Keeping it at WARNING tells operators to
                # retrain without paging for something that isn't broken.
                logger.warning(
                    "RegimeRouter reloaded empty (no specialists, no global). "
                    "Every predict will return neutral 0.5 — expected after "
                    "the first start or a fresh install; retrain before "
                    "relying on ML signals in production.",
                )

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def trained_regimes(self) -> list[str]:
        return list(self._models.keys())

    # ──────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────

    def train(
        self,
        trades: list[Any],
        trainer: Callable[[list[Any], str], Optional[RegimeModel]],
        global_model: Optional[RegimeModel] = None,
    ) -> dict[str, RegimeStats]:
        """Partition trades by regime and train a specialist per bucket.

        Args:
            trades:       All StrategyTrade samples (chronologically ordered).
                          Each must have a ``market_regime`` attribute — the
                          router uses it verbatim for partitioning.
            trainer:      Callable ``(trades_subset, regime_name) ->
                          RegimeModel | None``. Return None when training
                          fails / metrics too weak to use; caller then falls
                          back to ``global_model`` for that regime.
            global_model: A pre-trained all-data model used for regimes that
                          fell back. Required — without it, fallback returns
                          neutral 0.5 probabilities. Supply the one from
                          ``MLPredictor.train()`` on the unsegmented data.

        Returns:
            ``{regime: RegimeStats}`` summarising per-regime outcomes.
        """
        if global_model is not None:
            self._global = global_model

        buckets: dict[str, list[Any]] = defaultdict(list)
        for t in trades:
            regime = getattr(t, "market_regime", None) or "unknown"
            buckets[str(regime)].append(t)

        stats: dict[str, RegimeStats] = {}

        for regime, subset in buckets.items():
            n = len(subset)
            if regime in self.UNKNOWN_REGIMES:
                stats[regime] = RegimeStats(
                    regime=regime, n_trades=n, trained=False,
                    fallback_reason="noise regime — always fall back",
                )
                continue
            if n < self.min_trades_per_regime:
                stats[regime] = RegimeStats(
                    regime=regime, n_trades=n, trained=False,
                    fallback_reason=f"only {n} trades (min {self.min_trades_per_regime})",
                )
                continue

            try:
                model = trainer(subset, regime)
            except Exception as exc:  # noqa: BLE001
                logger.warning("RegimeRouter [%s] trainer raised: %s", regime, exc)
                emit_component_error(
                    "ml_regime_router.train",
                    f"trainer failed for regime={regime}: {exc}",
                    exc=exc,
                    severity="warning",
                    regime=regime,
                )
                model = None

            if model is None:
                stats[regime] = RegimeStats(
                    regime=regime, n_trades=n, trained=False,
                    fallback_reason="trainer returned None",
                )
                continue

            self._models[regime] = model
            summary = model.metrics_summary or {}
            stats[regime] = RegimeStats(
                regime=regime,
                n_trades=n,
                trained=True,
                skill_score=float(model.skill_score),
                mean_precision=float(summary.get("precision", 0.0)),
                mean_auc=float(summary.get("roc_auc", 0.5)),
            )
            logger.info(
                "RegimeRouter trained [%s]: n=%d skill=%.3f prec=%.3f auc=%.3f",
                regime, n, model.skill_score,
                stats[regime].mean_precision, stats[regime].mean_auc,
            )

        self._stats = stats
        self._is_ready = bool(self._models) or self._global is not None
        logger.info(
            "RegimeRouter: %d specialists + %s global = %s",
            len(self._models),
            "yes" if self._global else "no",
            "ready" if self._is_ready else "not ready",
        )
        return stats

    # ──────────────────────────────────────────────────────────
    # Prediction
    # ──────────────────────────────────────────────────────────

    def predict_proba(
        self,
        regime: str,
        features_raw: np.ndarray,
    ) -> tuple[float, str, float]:
        """Return ``(probability, model_used, calibrated_threshold)``.

        The third item is the threshold the chosen model was calibrated
        against during training — callers should compare ``probability``
        to *this* value rather than the globally-configured threshold,
        otherwise per-regime calibration ends up being thrown away by a
        blanket cutoff. ``model_used`` is the regime name (specialist),
        ``"global"`` (fallback), or ``"none"`` (no model at all — caller
        should treat this as no ML signal and use 0.5 as a neutral threshold).
        """
        model = self._models.get(str(regime))
        if model is None:
            if self._global is None:
                return 0.5, "none", 0.5
            model = self._global
            tag = "global"
        else:
            tag = str(regime)

        proba = self._run_model(model, features_raw)
        return proba, tag, float(model.threshold)

    def get_regime_stats(self) -> dict[str, dict[str, Any]]:
        """Dashboard-serialisable snapshot of the router's state."""
        return {r: s.summary() for r, s in self._stats.items()}

    # ──────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _run_model(model: RegimeModel, features_raw: np.ndarray) -> float:
        """Apply selector → scaler → ensemble → threshold-neutral probability.

        Returns the raw calibrated probability (NOT thresholded); the caller
        decides how to interpret it (filter, reduce, allow).
        """
        x = np.asarray(features_raw, dtype=np.float64).reshape(1, -1)
        if model.selector is not None and getattr(model.selector, "is_fitted", False):
            x = model.selector.transform(x)
        if model.scaler is not None:
            try:
                x = model.scaler.transform(x)
            except Exception:  # noqa: BLE001
                pass
        try:
            proba = model.ensemble.predict_proba_calibrated(x)
            return float(np.asarray(proba).flatten()[0])
        except Exception as exc:  # noqa: BLE001
            logger.warning("RegimeModel predict failed: %s", exc)
            emit_component_error(
                "ml_regime_router.predict",
                f"regime predict failed: {exc}",
                exc=exc,
                severity="warning",
                degraded_to="0.5",
            )
            return 0.5
