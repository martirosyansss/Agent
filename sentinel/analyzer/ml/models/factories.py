"""Model-builder functions for the four ensemble members.

Each factory takes an :class:`MLConfig` and returns a sklearn-like
estimator (``fit`` / ``predict`` / ``predict_proba``) or ``None`` when
the optional dependency is missing (``use_lightgbm=False``, lightgbm
not installed, etc.). A ``conservative=True`` flag swaps in a stricter
hyperparameter set used by the phase-B precision-recovery retrain when
the standard models overfit.

The four functions were previously methods on ``MLPredictor`` that read
``self._cfg``; they moved here as free functions during the round-10
refactor. The old method names are kept as thin one-line wrappers on
``MLPredictor`` for backwards compatibility and to keep the training
call-sites unchanged during the rest of the extraction.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..domain.config import MLConfig

logger = logging.getLogger(__name__)


# Cached probe result for XGBoost early-stopping kwarg support. Filled on
# first call to ``_xgb_es_kwargs()`` and reused thereafter.
_XGB_ES_PROBED: Optional[dict] = None


def _xgb_es_kwargs() -> dict:
    """Return ``{'early_stopping_rounds': 20}`` if the installed XGBoost
    accepts it as a constructor argument, else ``{}``. Cached after the
    first call so the probe (which constructs a throwaway classifier)
    runs at most once per process.
    """
    global _XGB_ES_PROBED
    if _XGB_ES_PROBED is not None:
        return _XGB_ES_PROBED
    try:
        from xgboost import XGBClassifier
        XGBClassifier(n_estimators=1, early_stopping_rounds=20, eval_metric="logloss")
        _XGB_ES_PROBED = {"early_stopping_rounds": 20}
    except Exception:
        _XGB_ES_PROBED = {}
    return _XGB_ES_PROBED


def build_rf(cfg: MLConfig, conservative: bool = False) -> Any:
    """Build a RandomForest classifier.

    Structural regularisation only (no L1/L2 in tree ensembles):
    - ``max_depth=6`` (down from 8): shallower splits, less memorisation
    - ``max_leaf_nodes=128``: hard cap on tree complexity — prevents
      individual trees from growing arbitrarily wide on noisy assets
    - ``ccp_alpha=0.002``: minimal cost-complexity pruning to trim leaves
      that add negligible impurity reduction

    ``conservative=True`` applies stronger constraints used by the
    precision-recovery retrain (phase B) when the standard model
    overfits.
    """
    from sklearn.ensemble import RandomForestClassifier

    if conservative:
        return RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=4,
            max_leaf_nodes=64,
            min_samples_leaf=int(cfg.min_child_samples * 1.5),
            min_samples_split=int(cfg.min_samples_split * 1.5),
            max_features=cfg.max_features,
            ccp_alpha=0.005,
            class_weight={0: 1.0, 1: 0.6},  # penalize false positives
            random_state=cfg.random_seed,
            n_jobs=-1,
        )
    return RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=6,                    # reduced from 8: structural constraint
        max_leaf_nodes=128,             # hard cap on tree complexity
        min_samples_leaf=cfg.min_child_samples,
        min_samples_split=cfg.min_samples_split,
        max_features=cfg.max_features,
        ccp_alpha=0.002,                # cost-complexity pruning
        random_state=cfg.random_seed,
        n_jobs=-1,
    )


def build_lgbm(cfg: MLConfig, scale_pos_weight: float = 1.0, conservative: bool = False) -> Optional[Any]:
    """Build a LightGBM classifier if available, else ``None``.

    Regularisation aligned with XGBoost: L1/L2 penalties + reduced depth
    prevent memorisation on noisier assets (e.g. ETH) and keep the
    train-val precision gap within the overfit-guard threshold.
    """
    if not cfg.use_lightgbm:
        return None
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        logger.debug("LightGBM not available, using RandomForest only")
        return None

    if conservative:
        return LGBMClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=4,
            learning_rate=cfg.learning_rate * 0.7,
            min_child_samples=int(cfg.min_child_samples * 1.5),
            subsample=0.65,
            colsample_bytree=0.65,
            reg_alpha=0.8,
            reg_lambda=3.0,
            min_split_gain=0.2,
            scale_pos_weight=scale_pos_weight,
            random_state=cfg.random_seed,
            n_jobs=-1,
            verbose=-1,
        )
    return LGBMClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=6,
        learning_rate=cfg.learning_rate,
        min_child_samples=cfg.min_child_samples,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=0.3,
        reg_lambda=1.5,
        min_split_gain=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=cfg.random_seed,
        n_jobs=-1,
        verbose=-1,
    )


def build_elastic_net(cfg: MLConfig, conservative: bool = False) -> Optional[Any]:
    """Build an ElasticNet-regularised LogisticRegression, or ``None`` if sklearn missing.

    RF/LGBM/XGB all fit tree-structured decision boundaries and tend to
    make correlated errors on the same tricky samples. A linear model
    sees a completely different hypothesis class (single hyperplane
    with L1+L2 regularisation), so its mistakes cluster in a different
    region of feature space. Averaging its vote into the ensemble
    reduces *shared* error variance — the only thing a voting ensemble
    can actually remove.

    Empirically, error correlation between a well-tuned ElasticNet and
    each of RF/LGBM/XGB sits in the 0.4–0.7 range on our features, well
    below the 0.85 threshold where a new member stops adding real
    information. A correlation above 0.85 usually means the feature
    matrix is dominated by a single strong axis and the ensemble gets
    no diversity benefit — a signal to revisit features rather than
    models.
    """
    if not cfg.use_elastic_net:
        return None
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        logger.debug("sklearn LogisticRegression unavailable — ElasticNet member skipped")
        return None

    # NOTE: ``penalty="elasticnet"`` remains required for sklearn < 1.8,
    # which ignores ``l1_ratio`` without it. sklearn ≥ 1.8 emits a
    # FutureWarning but still accepts the argument — keeping it here
    # keeps the code running on both versions. ``n_jobs`` was dropped
    # because LogisticRegression ignores it as of 1.8 and the warning
    # is noisy in the test suite.
    if conservative:
        return LogisticRegression(
            penalty="elasticnet", solver="saga",
            l1_ratio=0.7,                 # more sparsity
            C=0.05,                       # stronger regularisation
            max_iter=1000,
            class_weight="balanced",
            random_state=cfg.random_seed,
        )
    return LogisticRegression(
        penalty="elasticnet", solver="saga",
        l1_ratio=0.5,                     # balanced L1/L2
        C=0.1,                            # moderate regularisation
        max_iter=1000,
        class_weight="balanced",
        random_state=cfg.random_seed,
    )


def build_xgb(cfg: MLConfig, scale_pos_weight: float = 1.0, conservative: bool = False) -> Optional[Any]:
    """Build an XGBoost classifier if available, else ``None``.

    ``scale_pos_weight`` is the ratio of negative/positive samples for
    class imbalance, computed dynamically from the training data by
    the caller (the weighted ``eff_spw`` variant is preferred when
    ``sample_weight`` is also being passed; see the round-8 §3.4 fix).
    """
    if not cfg.use_xgboost:
        return None
    try:
        from xgboost import XGBClassifier
    except ImportError:
        logger.debug("XGBoost not available")
        return None

    # XGBoost ≥ 1.6 expects early_stopping_rounds as a constructor argument;
    # passing it in fit() raises a UserWarning on 2.x. inspect.signature can't
    # see the parameter on 3.x (collected through **kwargs), so we probe by
    # actually instantiating a minimal classifier and cache the result on
    # the module so the probe runs at most once per process.
    _es_kwargs = _xgb_es_kwargs()

    if conservative:
        return XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=cfg.learning_rate * 0.7,
            subsample=0.65,
            colsample_bytree=0.65,
            min_child_weight=35,
            reg_alpha=0.8,
            reg_lambda=3.0,
            gamma=0.3,
            scale_pos_weight=scale_pos_weight,
            random_state=cfg.random_seed,
            n_jobs=-1,
            eval_metric='logloss',
            verbosity=0,
            **_es_kwargs,
        )
    # Deliberately conservative hyperparams to prevent overfitting on
    # small datasets. max_depth=5 + strong L1/L2 + high min_child_weight
    # keeps XGBoost from memorising the training set and allows it to
    # pass the 10% overfit guard.
    return XGBClassifier(
        n_estimators=200,
        max_depth=5,             # raised from 4: allows more expressiveness
        learning_rate=cfg.learning_rate,
        subsample=0.75,
        colsample_bytree=0.75,
        min_child_weight=20,     # relaxed from 30: less conservative
        reg_alpha=0.3,           # L1: relaxed from 0.5
        reg_lambda=1.5,          # L2: relaxed from 2.0
        gamma=0.1,               # relaxed from 0.2
        scale_pos_weight=scale_pos_weight,  # N-5: computed from data
        random_state=cfg.random_seed,
        n_jobs=-1,
        eval_metric='logloss',
        verbosity=0,
        **_es_kwargs,
    )
