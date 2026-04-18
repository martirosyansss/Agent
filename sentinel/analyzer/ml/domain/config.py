"""``MLConfig`` — all tunable knobs for training + prediction.

Extracted from ``analyzer.ml_predictor`` during the round-10 refactor.
Re-exported from the old module path for backwards compatibility, so
existing imports (``from analyzer.ml_predictor import MLConfig``) and
pickled models constructed before the move continue to load.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class MLConfig:
    n_estimators: int = 250
    max_depth: int = 8
    learning_rate: float = 0.05
    min_child_samples: int = 20
    min_samples_split: int = 15
    max_features: str = "sqrt"
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    block_threshold: float = 0.55
    reduce_threshold: float = 0.65       # separate threshold for 'reduce' decision
    min_precision: float = 0.65
    min_recall: float = 0.58
    min_roc_auc: float = 0.65
    min_skill_score: float = 0.72
    retrain_days: int = 14
    min_trades: int = 500
    test_window_days: int = 60
    cv_splits: int = 5
    use_lightgbm: bool = True            # try LightGBM if available
    use_xgboost: bool = True             # try XGBoost if available
    # Phase-3 additions (2026-04-18): decorrelated 4th member.
    # ElasticNet LR gives the ensemble a linear voice that rarely makes the
    # same mistakes as the three tree models. On by default because its
    # downside is small (single extra fit) and it slots into the existing
    # overfit-guard / skill-score pipeline identically to the others.
    use_elastic_net: bool = True
    # Phase-1: walk-forward validation of ML skill. Off by default so the
    # existing single-split training path is unchanged; flip on only when
    # you have ≥ 500 trades to split into 5 folds. Dashboard consumes the
    # WFReport via /api/ml/walk-forward.
    use_walk_forward: bool = False
    # Phase-2: stacking meta-model (LogisticRegression on OOF member probas).
    # Requires use_walk_forward=True (meta is fit on OOF, not in-sample).
    use_stacking: bool = False
    # Phase-4: per-regime VotingEnsembles routed by StrategyTrade.market_regime.
    # Needs enough trades in each regime bucket; falls back to the global
    # ensemble for regimes with < min_trades_per_regime samples.
    use_regime_routing: bool = False
    min_trades_per_regime: int = 100
    # Phase-5: bootstrap CI for reported metrics. Off by default because
    # 1000×resample is ~5s extra per training run; enable for dashboards
    # that display confidence intervals.
    use_bootstrap_ci: bool = False
    bootstrap_n_simulations: int = 1000
    max_overfit_gap: float = 0.10        # max train-test precision gap
    # Temporal sample weighting: exp(decay·i) so most-recent trade has the
    # largest weight. 0.003 means trade 200 ago weighs ~0.55× the newest.
    # Previously hardcoded at module scope; exposed here so operators can
    # tune it for fast-moving vs stable market regimes.
    temporal_decay: float = 0.003
    # LivePerformanceTracker drift threshold. None = compute adaptively from
    # sample size via Wilson-width (recommended). Set a fixed float (e.g. 0.12)
    # to override — useful for stress testing or unit tests.
    drift_threshold: Optional[float] = None
    # Below ``calibrated_thr * reduce_margin`` the decision is "block" instead
    # of "reduce". Previously a magic 0.85 in predict(); made configurable so
    # the block-buffer can be tuned without editing source.
    reduce_margin: float = 0.85
    # Single random seed propagated to every estimator and bootstrap sampler —
    # changing it here (not in three builder methods) re-seeds the pipeline.
    random_seed: int = 42
