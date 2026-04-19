"""Per-regime specialist training — wraps RegimeRouter.

Extracted from ``MLPredictor.train_with_regime_routing`` during the
round-10 refactor. The global model is built first (on all trades),
then each regime gets a fresh sub-predictor trained on just its
trades. Regimes with too few samples — or where sub-training fails —
fall back to the global model at predict time.

Why a free function with callbacks rather than a method:

* Keeps the circular ``MLPredictor ↔ RegimeRouter`` knowledge in one
  place. The caller owns the factory (so the sub-predictor is the
  same class as the outer one, guaranteed) and the router manages
  partitioning and fallback.
* Lets the unit tests inject a mock ``predictor_factory`` so the
  routing logic is exercised without sklearn fits.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from ..domain.config import MLConfig

logger = logging.getLogger(__name__)


def train_regime_routing(
    cfg: MLConfig,
    trades: list[Any],                        # list[StrategyTrade]
    train_global_fn: Callable[[list[Any]], Optional[Any]],
    get_global_snapshot_fn: Callable[[], Optional[Any]],  # → RegimeModel
    predictor_factory: Callable[[MLConfig], Any],          # → MLPredictor-like
) -> Optional[Any]:
    """Train the global model and then per-regime specialists.

    Args:
        cfg:               Outer-predictor config. The function short-
                           circuits when ``use_regime_routing`` is False.
        trades:            All trades (chronological).
        train_global_fn:   Runs ``train(trades)`` on the caller's
                           predictor. Returns its metrics (or None on
                           failure).
        get_global_snapshot_fn: Called AFTER a successful global train
                           to produce a ``RegimeModel`` wrapping the
                           caller's ensemble + scaler + selector +
                           calibrated threshold. This is the fallback
                           for under-represented regimes.
        predictor_factory: Callable that builds a fresh sub-predictor
                           from a ``MLConfig``. The factory is what
                           makes this function trainer-agnostic — tests
                           can inject a stub to avoid real sklearn fits.

    Returns:
        The assembled ``RegimeRouter`` on success, or None if the
        global train failed / the flag is off.
    """
    if not cfg.use_regime_routing:
        logger.debug("train_with_regime_routing: flag off — noop")
        return None

    # 1. Global model first — it's the fallback for under-represented
    #    regimes and for predict-time "unknown" states. Refuse to build
    #    a router when this fails, because a router without a fallback
    #    silently returns neutral 0.5 for every unseen regime.
    global_metrics = train_global_fn(trades)
    if global_metrics is None:
        logger.warning("regime routing: global model failed — aborting")
        return None

    try:
        from analyzer.ml_regime_router import RegimeRouter, RegimeModel
    except ImportError as exc:
        logger.warning("regime routing: cannot import router: %s", exc)
        return None

    global_model = get_global_snapshot_fn()
    if global_model is None:
        logger.warning("regime routing: global snapshot unavailable — aborting")
        return None

    # J-3 sanity check: per-regime training needs at least ~80 samples
    # (50 train + 15 val + 15 test). If the operator configured
    # ``min_trades_per_regime`` lower than that, every specialist will
    # silently fail to train and the whole router will collapse back
    # to the global model. Log it upfront so the "regime routing on
    # but no specialists" pattern is visible.
    _effective_min = min(cfg.min_trades, cfg.min_trades_per_regime)
    if _effective_min < 80:
        logger.warning(
            "regime routing: effective min_trades=%d is below the ~80 required by "
            "train()'s 70/15/15 split. Most specialists will fall back to global.",
            _effective_min,
        )

    def _regime_trainer(subset: list[Any], regime: str) -> Optional[RegimeModel]:
        # Build a fresh sub-predictor so its feature selector / scaler /
        # ensemble don't collide with the outer global one. Force
        # regime/stacking/WF flags OFF inside the sub-train to avoid
        # infinite recursion.
        sub_cfg = MLConfig(**{**cfg.__dict__})
        sub_cfg.use_regime_routing = False
        sub_cfg.use_stacking = False
        sub_cfg.use_walk_forward = False
        sub_cfg.min_trades = min(cfg.min_trades, cfg.min_trades_per_regime)
        sub_predictor = predictor_factory(sub_cfg)
        m = sub_predictor.train(subset)
        if m is None or not sub_predictor.is_ready:
            logger.info(
                "regime routing [%s]: specialist training returned %s — falling back to global",
                regime, "no metrics" if m is None else "not-ready predictor",
            )
            return None
        return RegimeModel(
            regime=regime,
            ensemble=sub_predictor._ensemble,
            scaler=sub_predictor._scaler,
            selector=sub_predictor._feature_selector,
            threshold=sub_predictor._calibrated_threshold,
            skill_score=m.skill_score,
            n_train=m.train_samples,
            metrics_summary={"precision": m.precision, "roc_auc": m.roc_auc},
        )

    router = RegimeRouter(min_trades_per_regime=cfg.min_trades_per_regime)
    router.train(trades, _regime_trainer, global_model=global_model)
    logger.info(
        "regime routing: router ready with %d specialists + global fallback",
        len(router.trained_regimes),
    )
    return router
