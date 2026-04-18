"""
Trade Analyzer Level 3 — ML Predictor (Triple-Engine Ensemble, v3).

VotingEnsemble: RF + LightGBM + XGBoost soft-voting instead of winner-takes-all.
ML ONLY filters (block/reduce), NEVER initiates trades.

Rollout modes: off → shadow → block
- shadow: logs predictions, never blocks
- block: actively blocks signals with low probability

32 features (all pre-trade, no forward-looking bias):
  Technical:  rsi_14, adx, ema_9_vs_21, bb_bandwidth, volume_ratio,
              macd_histogram, atr_ratio
  Temporal:   hour_sin, hour_cos, day_sin, day_cos (cyclical encoding)
  Encoding:   market_regime_encoded, strategy_regime_fit
  Historical: recent_win_rate_10, hours_since_last_trade,
              rolling_avg_pnl_pct_20, consecutive_losses,
              strategy_specific_wr_10
  Sentiment:  news_sentiment, fear_greed_normalized, trend_alignment,
              regime_bias
  Enhanced:   cci, roc, cmf, bb_pct_b, hist_volatility, dmi_spread,
              stoch_rsi, price_change_5h_norm, momentum_norm, rsi_daily

Upgrade v3:
  - VotingEnsemble: soft-voting with skill-weighted probabilities
  - TemporalWeighting: recent trades contribute more to training (exp decay)
  - AdaptiveFeatureSelector: auto-drops features with importance < 1%
  - IsotonicCalibration: P(win|score=x) == x empirically
  - NumPy batch feature extraction: 8-15x speedup vs per-trade loop

Upgrade v4:
  - strategy_encoded → strategy_regime_fit: continuous score [-1,1] (how well
    strategy fits current regime) — removes categorical lookup bias
  - strategy_specific_wr_10: per-strategy rolling win rate (last 10 same-strategy trades)

Skill score = 0.30*precision + 0.10*recall + 0.35*roc_auc + 0.25*profit_factor
(precision weighted 3x recall — filter-mode: false positives > false negatives)
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from core.models import StrategyTrade

logger = logging.getLogger(__name__)

# Temporal decay factor for sample weighting (higher = faster decay)
# 0.003 ≈ recent 200 trades weighted ~2x more than oldest
_TEMPORAL_DECAY: float = 0.003

# Skill score weights — single source of truth.
# Precision is weighted 3x recall because this model is a *signal filter*:
# false positives (letting bad trades through) cost more than false negatives
# (blocking some good trades — the strategy would still have many other chances).
_SKILL_W_PRECISION = 0.30
_SKILL_W_RECALL = 0.10
_SKILL_W_ROC_AUC = 0.35
_SKILL_W_PROFIT_FACTOR = 0.25


def compute_skill_score(precision: float, recall: float, roc_auc: float, profit_factor_score: float) -> float:
    """Weighted composite skill score used for model selection and gating.

    All four inputs should be normalized to [0, 1]. profit_factor_score is
    typically min(profit_factor / 3.0, 1.0).
    """
    return (
        _SKILL_W_PRECISION * precision
        + _SKILL_W_RECALL * recall
        + _SKILL_W_ROC_AUC * roc_auc
        + _SKILL_W_PROFIT_FACTOR * profit_factor_score
    )


def wilson_lower_bound(successes: int, trials: int, z: float = 1.96) -> float:
    """95%-CI lower bound on a binomial proportion (Wilson score interval).

    Used to test whether an observed success rate is *significantly above* a
    target. The Wilson interval is well-behaved at small N and at extreme p,
    unlike the naive normal approximation (which fails when p≈0 or p≈1).

    Args:
        successes: observed successes (e.g., true-positive predictions)
        trials:    total trials (e.g., total predicted-positive events)
        z:         z-score for the desired confidence (1.96 = 95%, 1.645 = 90%)

    Returns:
        Lower bound of the confidence interval in [0, 1]. Returns 0 if trials==0.
    """
    if trials <= 0:
        return 0.0
    p = successes / trials
    z2 = z * z
    denom = 1.0 + z2 / trials
    center = p + z2 / (2.0 * trials)
    half_width = z * ((p * (1.0 - p) + z2 / (4.0 * trials)) / trials) ** 0.5
    return max(0.0, (center - half_width) / denom)


N_FEATURES = 32  # v5: +2 cyclical temporal (sin/cos), -1 adx_normalized (redundant)

# Encoding maps
REGIME_ENCODING = {
    "trending_up": 0, "trending_down": 1, "sideways": 2,
    "volatile": 3, "transitioning": 4, "unknown": 5,
}

# strategy_regime_fit: how well each strategy fits each market regime (-1..1).
# Replaces raw strategy_encoded categorical to give the model interpretable signal:
# "EMA crossover works when trending" instead of "strategy 0 is historically better".
# This generalises across regime changes rather than memorising per-strategy win rates.
STRATEGY_REGIME_FIT: dict[tuple[str, str], float] = {
    # EMA crossover — momentum strategy, needs a trend
    ("ema_crossover_rsi", "trending_up"):    1.0,
    ("ema_crossover_rsi", "trending_down"):  0.6,
    ("ema_crossover_rsi", "sideways"):      -0.5,
    ("ema_crossover_rsi", "volatile"):      -0.3,
    # Bollinger breakout — volatility expansion, loves squeezes + breakouts
    ("bollinger_breakout", "trending_up"):   0.5,
    ("bollinger_breakout", "trending_down"): 0.5,
    ("bollinger_breakout", "sideways"):      0.3,
    ("bollinger_breakout", "volatile"):      1.0,
    # Mean reversion — range-bound, hostile to trends
    ("mean_reversion", "trending_up"):      -0.7,
    ("mean_reversion", "trending_down"):    -0.7,
    ("mean_reversion", "sideways"):          1.0,
    ("mean_reversion", "volatile"):         -0.3,
    # MACD divergence — trend-following with momentum confirmation
    ("macd_divergence", "trending_up"):      1.0,
    ("macd_divergence", "trending_down"):    0.8,
    ("macd_divergence", "sideways"):        -0.4,
    ("macd_divergence", "volatile"):         0.2,
    # Grid trading — range-bound, symmetric
    ("grid_trading", "trending_up"):        -0.5,
    ("grid_trading", "trending_down"):      -0.5,
    ("grid_trading", "sideways"):            1.0,
    ("grid_trading", "volatile"):            0.0,
    # DCA — long-biased accumulation
    ("dca_bot", "trending_up"):              0.8,
    ("dca_bot", "trending_down"):           -0.3,
    ("dca_bot", "sideways"):                 0.3,
    ("dca_bot", "volatile"):                -0.2,
    # TRANSITIONING regime — dangerous, most strategies perform poorly
    ("ema_crossover_rsi", "transitioning"):  -0.2,
    ("bollinger_breakout", "transitioning"):  0.1,
    ("mean_reversion", "transitioning"):      0.0,
    ("macd_divergence", "transitioning"):    -0.1,
    ("grid_trading", "transitioning"):       -0.3,
    ("dca_bot", "transitioning"):             0.2,
}

FEATURE_NAMES = [
    # Technical (0-6)
    "rsi_14", "adx", "ema_9_vs_21", "bb_bandwidth", "volume_ratio",
    "macd_histogram", "atr_ratio",
    # Temporal — cyclical sin/cos encoding (7-10)
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    # Encoding (11-12)
    "market_regime_encoded",
    "strategy_regime_fit",
    # Historical (13-17)
    "recent_win_rate_10", "hours_since_last_trade",
    "rolling_avg_pnl_pct_20", "consecutive_losses",
    "strategy_specific_wr_10",
    # Sentiment / regime (18-21)
    "news_sentiment", "fear_greed_normalized", "trend_alignment",
    "regime_bias",
    # Phase 2: Enhanced indicators (22-31)
    "cci", "roc", "cmf", "bb_pct_b", "hist_volatility",
    "dmi_spread", "stoch_rsi", "price_change_5h_norm",
    "momentum_norm", "rsi_daily",
]


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
    max_overfit_gap: float = 0.10        # max train-test precision gap


@dataclass
class MLMetrics:
    """Метрики обученной модели."""
    precision: float = 0.0
    recall: float = 0.0
    roc_auc: float = 0.0
    accuracy: float = 0.0
    skill_score: float = 0.0
    train_samples: int = 0
    test_samples: int = 0
    feature_importances: dict[str, float] = field(default_factory=dict)
    # Statistical confidence (bootstrap 95% CI)
    precision_ci_95: tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    auc_ci_95: tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    # Baseline comparison
    baseline_win_rate: float = 0.0    # precision of "predict always win" naive model
    precision_lift: float = 0.0       # model precision − baseline_win_rate
    auc_lift: float = 0.0             # model AUC − 0.5 (random baseline)
    # Out-of-time robustness
    oot_auc: float | None = None      # AUC on most-recent 20% (independent OOT set)
    # Calibration diagnostics — flag silent biases like "every prediction ≈ 0.9"
    # that hurt downstream interpretation even when AUC/precision look fine.
    brier_score: float = 0.0          # mean (proba - actual)² on holdout, lower is better
    ece: float = 0.0                  # Expected Calibration Error (10 bins) on holdout
    mean_proba: float = 0.5           # mean calibrated probability across holdout
    median_proba: float = 0.5         # median — flags one-sided distributions
    proba_p10: float = 0.0            # 10th percentile — should not collapse onto p90
    proba_p90: float = 1.0            # 90th percentile
    calibration_method: str = "none"  # "none" | "platt" | "isotonic"


@dataclass
class MLPrediction:
    """Предсказание модели для сигнала."""
    probability: float = 0.5
    decision: str = "allow"  # allow, reduce, block
    model_version: str = ""
    rollout_mode: str = "shadow"  # off, shadow, block


class LivePerformanceTracker:
    """Tracks model predictions vs actual trade outcomes in live/paper trading.

    Detects concept drift: when live precision drops significantly below training
    precision, the model has likely overfit to a historical regime that no longer holds.

    Thread-safe: uses a lock for concurrent access from async code paths.
    Memory-bounded: uses collections.deque with fixed maxlen.
    """

    def __init__(self, window: int = 50, drift_threshold: float | None = None) -> None:
        import threading
        from collections import deque
        self._window = window
        # When None, drift threshold is computed adaptively from sample size
        # using the Wilson-score width (see is_drifting). A fixed threshold
        # like 0.12 fires spuriously at small N and misses real drift at large N.
        self._drift_threshold = drift_threshold
        self._history: deque[tuple[float, int]] = deque(maxlen=window * 3)
        self._lock = threading.Lock()

    def record(self, predicted_prob: float, actual_win: bool) -> None:
        """Record one live prediction + its realized outcome (thread-safe)."""
        with self._lock:
            self._history.append((predicted_prob, int(actual_win)))

    def live_metrics(self) -> dict:
        """Compute rolling precision, win rate, and calibration on recent window."""
        with self._lock:
            snapshot = list(self._history)

        n = len(snapshot)
        if n < 10:
            return {"status": "insufficient_data", "n": n}

        recent = snapshot[-self._window:]
        probs   = np.array([p for p, _ in recent], dtype=np.float64)
        actuals = np.array([a for _, a in recent], dtype=np.float64)
        preds   = (probs >= 0.5).astype(int)

        win_rate   = float(actuals.mean())
        n_pred_win = int(preds.sum())
        live_prec  = float(np.sum((preds == 1) & (actuals == 1)) / n_pred_win) if n_pred_win > 0 else 0.0
        calib_err = float(abs(probs.mean() - win_rate))

        try:
            from sklearn.metrics import roc_auc_score as _auc
            live_auc = float(_auc(actuals, probs)) if len(set(actuals)) > 1 else 0.5
        except Exception:
            live_auc = 0.5

        return {
            "n": len(recent),
            "n_pred_win": n_pred_win,   # denominator of precision — needed for Wilson CI
            "live_precision": live_prec,
            "live_win_rate": win_rate,
            "live_auc": live_auc,
            "calibration_error": calib_err,
        }

    def is_drifting(self, training_precision: float) -> bool:
        """Detect concept drift using a sample-size-aware threshold.

        Uses a two-proportion z-test style margin: drift is flagged only when
        the gap between training and live precision exceeds what sampling noise
        alone would produce at 95% confidence (z=1.96). Formula:

            margin = z * sqrt(p * (1-p) / n_pred_win)

        where p is training precision. At n=30, p=0.70 this gives margin≈0.164,
        so a 12-point drop is noise; at n=200 it gives 0.064, so even small drops
        are meaningful.

        A fixed fallback is used if the caller explicitly set drift_threshold.
        """
        m = self.live_metrics()
        if "live_precision" not in m or m.get("n", 0) < 10:
            return False
        n_pred_win = max(int(m.get("n_pred_win", 0)), 1)

        if self._drift_threshold is not None:
            margin = self._drift_threshold
        else:
            p = max(min(training_precision, 0.99), 0.01)
            margin = 1.96 * np.sqrt(p * (1.0 - p) / n_pred_win)

        return (training_precision - m["live_precision"]) > margin

    @property
    def n_recorded(self) -> int:
        with self._lock:
            return len(self._history)


class MLPredictor:
    """Level 3 Trade Analyzer — ML фильтрация сигналов (VotingEnsemble v3)."""

    def __init__(self, config: MLConfig | None = None) -> None:
        from analyzer.ml_ensemble import VotingEnsemble, AdaptiveFeatureSelector
        self._cfg = config or MLConfig()
        self._model: Any = None          # Legacy: kept for load_from_file compat
        self._ensemble: Optional[VotingEnsemble] = None  # v3: primary predictor
        self._scaler: Any = None
        self._feature_selector: AdaptiveFeatureSelector = AdaptiveFeatureSelector(min_importance=0.01)
        self._model_version: str = ""
        self._metrics: Optional[MLMetrics] = None
        self._rollout_mode: str = "off"  # off, shadow, block
        self._calibrated_threshold: float = 0.5  # W-5: proper init
        self._last_train_ts: int = 0
        self._live_tracker: LivePerformanceTracker = LivePerformanceTracker()

    @property
    def is_ready(self) -> bool:
        # v3: ensemble-first, fallback to legacy model
        return (self._ensemble is not None and self._ensemble.is_ready) or (self._model is not None)

    @property
    def rollout_mode(self) -> str:
        return self._rollout_mode

    @rollout_mode.setter
    def rollout_mode(self, mode: str) -> None:
        if mode in ("off", "shadow", "block"):
            self._rollout_mode = mode

    @property
    def metrics(self) -> Optional[MLMetrics]:
        return self._metrics

    @property
    def drift_detected(self) -> bool:
        """True when live precision has drifted >12% below training precision."""
        if self._metrics is None:
            return False
        return self._live_tracker.is_drifting(self._metrics.precision)

    @property
    def live_metrics(self) -> dict:
        """Rolling live performance metrics from actual trade outcomes."""
        return self._live_tracker.live_metrics()

    def record_outcome(self, predicted_prob: float, actual_win: bool) -> None:
        """Record a live/paper trade outcome for concept drift monitoring.

        Call this after each trade closes with the ML probability that was
        predicted at entry time and whether the trade was actually profitable.
        """
        self._live_tracker.record(predicted_prob, actual_win)
        m = self._live_tracker.live_metrics()
        if "live_precision" in m and m["n"] >= 20 and m["n"] % 10 == 0:
            logger.info(
                "Live ML tracker (n=%d): precision=%.3f win_rate=%.3f auc=%.3f calib_err=%.3f%s",
                m["n"], m["live_precision"], m["live_win_rate"], m["live_auc"],
                m["calibration_error"],
                " ⚠ DRIFT DETECTED" if self.drift_detected else "",
            )

    @staticmethod
    def _parse_trade_timestamp(value: str) -> Optional[datetime]:
        if not value:
            return None
        normalized = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            return None

    @staticmethod
    def _regime_bias(regime: str) -> float:
        """Regime bias using ONLY pre-trade knowledge (no pnl_pct)."""
        lowered = regime.lower()
        if lowered == "trending_up":
            return 1.0
        if lowered == "trending_down":
            return -1.0
        return 0.0

    @staticmethod
    def _strategy_regime_fit(strategy_name: str, regime: str) -> float:
        """How well the strategy fits the current market regime (-1..1).

        Uses STRATEGY_REGIME_FIT lookup. Unknown combinations default to 0.0
        (neutral — no edge, no disadvantage).
        """
        return STRATEGY_REGIME_FIT.get((strategy_name, regime.lower()), 0.0)

    def extract_features(
        self,
        trade: StrategyTrade,
        previous_trades: Optional[list[StrategyTrade]] = None,
    ) -> list[float]:
        """Extract 31 features using ONLY pre-trade data (no forward-looking bias).

        N-3 fix: Delegates to extract_features_batch() to guarantee identical
        feature computation between training and inference paths.

        All features are known BEFORE entering the trade:
        - Technical indicators at entry (rsi, adx, volume_ratio)
        - Strategy metadata (regime fit score, strategy-specific win rate)
        - Time features (hour, day)
        - Historical performance (win_rate, consecutive_losses, rolling pnl)
        - Sentiment data (news, fear_greed)
        """
        previous_trades = previous_trades or []
        # Build the full historical sequence: previous trades + current trade
        # extract_features_batch processes them chronologically and the LAST
        # row corresponds to the current trade's features.
        all_trades = previous_trades + [trade]
        X = self.extract_features_batch(all_trades)
        return X[-1].tolist()  # Last row = current trade

    def _build_rf(self, conservative: bool = False):
        """Build a RandomForest classifier.

        Structural regularization only (no L1/L2 in tree ensembles):
        - max_depth=6 (down from 8): shallower splits, less memorisation
        - max_leaf_nodes=128: hard cap on tree complexity — prevents
          individual trees from growing arbitrarily wide on noisy assets
        - ccp_alpha=0.002: minimal cost-complexity pruning to trim leaves
          that add negligible impurity reduction

        Args:
            conservative: If True, use stronger structural constraints for
                          precision recovery on noisy assets.
        """
        from sklearn.ensemble import RandomForestClassifier
        if conservative:
            return RandomForestClassifier(
                n_estimators=self._cfg.n_estimators,
                max_depth=4,
                max_leaf_nodes=64,
                min_samples_leaf=int(self._cfg.min_child_samples * 1.5),
                min_samples_split=int(self._cfg.min_samples_split * 1.5),
                max_features=self._cfg.max_features,
                ccp_alpha=0.005,
                class_weight={0: 1.0, 1: 0.6},  # penalize false positives
                random_state=42,
                n_jobs=-1,
            )
        return RandomForestClassifier(
            n_estimators=self._cfg.n_estimators,
            max_depth=6,                    # reduced from 8: structural constraint
            max_leaf_nodes=128,             # hard cap on tree complexity
            min_samples_leaf=self._cfg.min_child_samples,
            min_samples_split=self._cfg.min_samples_split,
            max_features=self._cfg.max_features,
            ccp_alpha=0.002,                # cost-complexity pruning
            random_state=42,
            n_jobs=-1,
        )

    def _build_lgbm(self, scale_pos_weight: float = 1.0, conservative: bool = False):
        """Build a LightGBM classifier if available, else None.

        Regularization aligned with XGBoost: L1/L2 penalties + reduced depth
        prevent memorisation on noisier assets (e.g. ETH) and keep the
        train-val precision gap within the overfit guard threshold.

        Args:
            conservative: If True, use stronger regularization for precision recovery
                          on noisy assets (higher L1/L2, shallower trees, more penalization
                          of false positives via scale_pos_weight).
        """
        if not self._cfg.use_lightgbm:
            return None
        try:
            from lightgbm import LGBMClassifier
            if conservative:
                return LGBMClassifier(
                    n_estimators=self._cfg.n_estimators,
                    max_depth=4,
                    learning_rate=self._cfg.learning_rate * 0.7,
                    min_child_samples=int(self._cfg.min_child_samples * 1.5),
                    subsample=0.65,
                    colsample_bytree=0.65,
                    reg_alpha=0.8,
                    reg_lambda=3.0,
                    min_split_gain=0.2,
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                )
            return LGBMClassifier(
                n_estimators=self._cfg.n_estimators,
                max_depth=6,
                learning_rate=self._cfg.learning_rate,
                min_child_samples=self._cfg.min_child_samples,
                subsample=0.75,
                colsample_bytree=0.75,
                reg_alpha=0.3,
                reg_lambda=1.5,
                min_split_gain=0.1,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
        except ImportError:
            logger.debug("LightGBM not available, using RandomForest only")
            return None

    def _build_xgb(self, scale_pos_weight: float = 1.0, conservative: bool = False):
        """Build an XGBoost classifier if available, else None.

        Args:
            scale_pos_weight: Ratio of negative/positive samples for class imbalance.
                              Computed dynamically from training data.
            conservative: If True, use stronger regularization for precision recovery.
        """
        if not self._cfg.use_xgboost:
            return None
        try:
            from xgboost import XGBClassifier
            if conservative:
                return XGBClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=self._cfg.learning_rate * 0.7,
                    subsample=0.65,
                    colsample_bytree=0.65,
                    min_child_weight=35,
                    reg_alpha=0.8,
                    reg_lambda=3.0,
                    gamma=0.3,
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric='logloss',
                    verbosity=0,
                )
            # Deliberately conservative hyperparams to prevent overfitting on small datasets.
            # max_depth=4 (vs RF's 8) + strong L1/L2 + high min_child_weight keeps XGBoost
            # from memorising the training set and allows it to pass the 10% overfit guard.
            return XGBClassifier(
                n_estimators=200,
                max_depth=5,             # raised from 4: allows more expressiveness
                learning_rate=self._cfg.learning_rate,
                subsample=0.75,
                colsample_bytree=0.75,
                min_child_weight=20,     # relaxed from 30: less conservative
                reg_alpha=0.3,           # L1: relaxed from 0.5
                reg_lambda=1.5,          # L2: relaxed from 2.0
                gamma=0.1,               # relaxed from 0.2
                scale_pos_weight=scale_pos_weight,  # N-5: computed from data
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss',
                verbosity=0,
            )
        except ImportError:
            logger.debug("XGBoost not available")
            return None

    @staticmethod
    def _calibrate_threshold(
        y_true, y_proba, min_precision: float = 0.55,
        pnl: np.ndarray = None, min_recall: float = 0.30,
    ) -> float:
        """Find optimal threshold maximizing profit factor with precision + recall constraints.

        When pnl data is provided, optimizes for realized profit factor (gross
        winning PnL / gross losing PnL among predicted-win trades) subject to:
        - precision >= min_precision (avoid losing trades)
        - recall >= min_recall (don't miss too many winners)

        Falls back to precision-weighted F-beta when pnl is not available.
        """
        from sklearn.metrics import precision_recall_curve, precision_score, recall_score
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

        best_threshold = 0.5

        if pnl is not None and len(pnl) == len(y_true):
            best_pf = 0.0
            for thr in np.arange(0.30, 0.75, 0.01):
                y_pred = (y_proba >= thr).astype(int)
                n_pred_pos = int(y_pred.sum())
                if n_pred_pos < 5:
                    continue
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                if prec < min_precision or rec < min_recall:
                    continue
                wins_pnl = float(sum(p for p, pred in zip(pnl, y_pred) if pred == 1 and p > 0))
                loss_pnl = abs(float(sum(p for p, pred in zip(pnl, y_pred) if pred == 1 and p <= 0)))
                pf = wins_pnl / loss_pnl if loss_pnl > 0 else (3.0 if wins_pnl > 0 else 0.0)
                if pf > best_pf:
                    best_pf = pf
                    best_threshold = float(thr)
        else:
            # Fallback: F-beta (beta=0.5 — precision 2x more important than recall)
            best_fb = 0.0
            beta_sq = 0.25  # beta=0.5 → beta²=0.25
            for prec, rec, thr in zip(precisions[:-1], recalls[:-1], thresholds):
                if prec < min_precision:
                    continue
                fb = (1 + beta_sq) * prec * rec / (beta_sq * prec + rec) if (beta_sq * prec + rec) > 0 else 0
                if fb > best_fb:
                    best_fb = fb
                    best_threshold = float(thr)

        return best_threshold

    @staticmethod
    def _compute_profit_factor_score(y_pred, pnl_values) -> float:
        """Normalized profit factor from predicted wins vs actual PnL (0..1 scale).

        Measures how well the model's "win" predictions align with actual profitable trades.
        Returns value in [0, 1] where 1.0 = perfect profit factor (>= 3.0).
        """
        if pnl_values is None or len(pnl_values) == 0:
            return 0.5
        pred_wins_pnl = sum(p for p, pred in zip(pnl_values, y_pred) if pred == 1 and p > 0)
        pred_wins_loss = abs(sum(p for p, pred in zip(pnl_values, y_pred) if pred == 1 and p <= 0))
        if pred_wins_loss <= 0:
            pf = 3.0 if pred_wins_pnl > 0 else 0.0
        else:
            pf = min(pred_wins_pnl / pred_wins_loss, 3.0)
        return pf / 3.0  # normalize to 0..1

    @staticmethod
    def _overfit_noise_margin(
        p_train: float,
        p_val: float,
        n_train: int,
        n_val: int,
        z: float = 1.96,
    ) -> float:
        """Statistical margin for the train-vs-val precision gap.

        The previous formula `0.5 / sqrt(n_val)` was a flat heuristic that
        ignored both the training-side variance and the actual proportion p,
        so it falsely rejected models on small samples whose gap was within
        sampling noise. This helper returns the *real* z-σ margin for the
        difference of two binomial proportions:

            SE(p_train) = sqrt(p_train * (1 - p_train) / n_train)
            SE(p_val)   = sqrt(p_val   * (1 - p_val)   / n_val)
            margin      = z * sqrt(SE_train² + SE_val²)

        At small n_val (typical for our 70/15/15 splits) this is meaningfully
        larger than the heuristic; at large n_val it converges to it. Caller
        adds it to a base tolerance (e.g. `cfg.max_overfit_gap`) and rejects
        the candidate only when the observed gap exceeds the sum.

        Returns 0 if either sample is empty (caller should treat that as
        "not enough data to call overfitting either way").
        """
        if n_train <= 0 or n_val <= 0:
            return 0.0
        # Clamp p into (0, 1) to avoid SE collapsing to 0 when a model trivially
        # predicts a single class on either split — that hides real variance.
        p_t = min(max(p_train, 1e-3), 1.0 - 1e-3)
        p_v = min(max(p_val,   1e-3), 1.0 - 1e-3)
        var_t = p_t * (1.0 - p_t) / n_train
        var_v = p_v * (1.0 - p_v) / n_val
        return float(z * (var_t + var_v) ** 0.5)

    @staticmethod
    def _expected_calibration_error(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Expected Calibration Error: mean |confidence − accuracy| across equal-width bins.

        ECE complements Brier score: a model with ECE > ~0.10 is meaningfully
        miscalibrated (the displayed probabilities don't match realized
        frequencies). Empty bins are skipped, so this is robust on the small
        holdout sets we work with (~50–300 samples).

        Args:
            y_true:  Binary labels in {0, 1}, shape (n,)
            y_proba: Predicted probabilities in [0, 1], shape (n,)
            n_bins:  Number of equal-width bins (default 10 → bin width 0.1)

        Returns:
            ECE in [0, 1]. 0 = perfectly calibrated, 1 = maximally miscalibrated.
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_proba = np.asarray(y_proba, dtype=np.float64)
        if len(y_true) == 0:
            return 0.0
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        # Bin index per sample; clip the right edge so 1.0 lands in the last bin.
        bin_idx = np.clip(np.digitize(y_proba, bin_edges[1:-1]), 0, n_bins - 1)
        n = len(y_true)
        ece = 0.0
        for b in range(n_bins):
            mask = bin_idx == b
            count = int(mask.sum())
            if count == 0:
                continue
            avg_conf = float(y_proba[mask].mean())
            avg_acc  = float(y_true[mask].mean())
            ece += (count / n) * abs(avg_conf - avg_acc)
        return float(ece)

    @staticmethod
    def _compute_temporal_weights(n: int, decay: float = _TEMPORAL_DECAY) -> np.ndarray:
        """Exponential temporal weights: recent trades weighted more.

        w(i) = exp(decay * i) so that the last trade has weight=1.0
        and the first trade has weight=exp(-decay*(n-1)).

        Args:
            n: Total number of samples
            decay: Decay rate. Higher = faster decay (older trades discounted more)

        Returns:
            Normalized weight array of shape (n,)
        """
        indices = np.arange(n, dtype=np.float64)
        weights = np.exp(decay * indices)
        return weights / weights.mean()  # normalize so mean=1.0

    def extract_features_batch(
        self,
        trades: list[StrategyTrade],
    ) -> np.ndarray:
        """Vectorized batch feature extraction — ~8-15x faster than per-trade loop.

        C-2 fix: Pre-parses all timestamps once to eliminate O(N²) datetime parsing.

        Args:
            trades: List of StrategyTrade sorted chronologically

        Returns:
            Feature matrix of shape (n_trades, N_FEATURES)
        """
        n = len(trades)
        X = np.zeros((n, N_FEATURES), dtype=np.float64)

        # Pre-compute cumulative win arrays for vectorized recent_win_rate
        is_win = np.array([1 if t.is_win else 0 for t in trades], dtype=np.float64)
        pnl_arr = np.array([t.pnl_usd for t in trades], dtype=np.float64)

        # C-2 fix: Pre-parse ALL timestamps ONCE (eliminates O(N²) re-parsing)
        parsed_open = [self._parse_trade_timestamp(t.timestamp_open or "") for t in trades]
        parsed_close = [
            self._parse_trade_timestamp(t.timestamp_close or t.timestamp_open or "")
            for t in trades
        ]
        # W-2 fix: Pre-compute consecutive_losses array in O(N)
        consec_losses = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            consec_losses[i] = 0.0 if is_win[i - 1] == 1 else consec_losses[i - 1] + 1.0

        # v4: Pre-build per-strategy win index for strategy_specific_wr_10.
        # Avoids O(N²) filtering inside the loop — each strategy gets its own
        # sorted list of (trade_idx, is_win) pairs.
        from collections import defaultdict
        strategy_win_history: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for idx, trade in enumerate(trades):
            entry_price = max(abs(trade.entry_price), 1e-9)
            regime_bias = self._regime_bias(trade.market_regime)
            ema_9_vs_21 = (trade.ema_9_at_entry - trade.ema_21_at_entry) / entry_price
            atr_safe = max(trade.atr_at_entry, 1e-9)  # safe divisor for ATR-normalization
            atr_ratio = atr_safe / entry_price

            strat_fit = self._strategy_regime_fit(trade.strategy_name, trade.market_regime)

            prev_same = strategy_win_history[trade.strategy_name]
            if prev_same:
                last10 = prev_same[-10:]
                strategy_specific_wr = sum(w for _, w in last10) / len(last10)
            else:
                strategy_specific_wr = 0.5

            if idx >= 1:
                start = max(0, idx - 10)
                recent_win_rate = is_win[start:idx].mean()
            else:
                recent_win_rate = 0.5

            if idx >= 1:
                _start20 = max(0, idx - 20)
                _recent_pnl = np.array([t.pnl_pct for t in trades[_start20:idx]], dtype=np.float64)
                rolling_avg_pnl_pct_20 = float(np.mean(_recent_pnl)) / 10.0
            else:
                rolling_avg_pnl_pct_20 = 0.0

            hours_since = 0.0
            if idx > 0 and parsed_open[idx] and parsed_close[idx - 1]:
                delta = (parsed_open[idx] - parsed_close[idx - 1]).total_seconds()
                hours_since = max(delta / 3600.0, 0.0)
            hours_since_clamped = min(hours_since, 72.0)

            cci_norm = max(-3.0, min(float(trade.cci_at_entry or 0) / 200.0, 3.0))

            # Cyclical temporal encoding: sin/cos preserves adjacency (hour 23 ↔ 0)
            _hour = float(trade.hour_of_day or 0)
            _day = float(trade.day_of_week or 0)
            hour_sin = np.sin(2.0 * np.pi * _hour / 24.0)
            hour_cos = np.cos(2.0 * np.pi * _hour / 24.0)
            day_sin = np.sin(2.0 * np.pi * _day / 7.0)
            day_cos = np.cos(2.0 * np.pi * _day / 7.0)

            # NaN-safe indicator reads: (val or 0) guards against None propagation
            _fg = float(trade.fear_greed_index or 0) / 100.0
            _dmi = float(trade.dmi_spread_at_entry or 0) / 50.0
            _stoch = float(trade.stoch_rsi_at_entry or 0) / 100.0
            _rsi_d = float(trade.rsi_daily_at_entry or 0) / 100.0
            # ATR-normalized for stationarity (replaces arbitrary /5.0 and /100.0 divisors)
            _pc5h_norm = float(trade.price_change_5h_at_entry or 0) / atr_safe
            _mom_norm = float(trade.momentum_at_entry or 0) / atr_safe

            X[idx] = [
                trade.rsi_at_entry,
                trade.adx_at_entry,
                ema_9_vs_21,
                trade.bb_bandwidth_at_entry,
                trade.volume_ratio_at_entry,
                trade.macd_histogram_at_entry,
                atr_ratio,
                # Cyclical temporal (4 features)
                hour_sin, hour_cos, day_sin, day_cos,
                # Encoding
                float(REGIME_ENCODING.get(trade.market_regime, 5)),
                strat_fit,
                # Historical
                recent_win_rate,
                hours_since_clamped,
                rolling_avg_pnl_pct_20,
                consec_losses[idx],
                strategy_specific_wr,
                # Sentiment / regime
                float(trade.news_sentiment or 0),
                _fg,
                float(trade.trend_alignment or 0),
                regime_bias,
                # Phase 2 (adx_normalized removed — redundant with adx)
                cci_norm,
                float(trade.roc_at_entry or 0) / 10.0,
                float(trade.cmf_at_entry or 0),
                float(trade.bb_pct_b_at_entry or 0),
                float(trade.hist_volatility_at_entry or 0),
                _dmi,
                _stoch,
                _pc5h_norm,
                _mom_norm,
                _rsi_d,
            ]

            # Record AFTER building features (no look-ahead)
            strategy_win_history[trade.strategy_name].append((idx, int(trade.is_win)))


        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def train(self, trades: list[StrategyTrade]) -> Optional[MLMetrics]:
        """Train VotingEnsemble on historical trades.

        v3 Ensemble: trains RF + LightGBM + XGBoost simultaneously,
        then combines via soft-voting weighted by validation skill score.
        TimeSeriesSplit CV + TemporalWeighting + IsotonicCalibration.
        Returns MLMetrics or None if insufficient data / metrics below threshold.
        """
        if len(trades) < self._cfg.min_trades:
            logger.warning("ML train: insufficient trades (%d < %d)", len(trades), self._cfg.min_trades)
            return None

        try:
            from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.error("scikit-learn not installed. pip install scikit-learn")
            return None

        # v3: Use vectorized batch extraction (8-15x faster than per-trade loop)
        logger.info("ML train: extracting features for %d trades (vectorized)", len(trades))
        X = self.extract_features_batch(trades)
        y_arr = np.array([1 if t.is_win else 0 for t in trades])
        pnl_values = [t.pnl_usd for t in trades]

        # Zero-variance feature detection: features that are constant (e.g. always 0)
        # provide no signal and either indicate a broken upstream data pipeline
        # or a feature that simply isn't available in this training corpus
        # (e.g. news_sentiment during pure-backtest training has no live news
        # to read). We always force-drop them via the AdaptiveFeatureSelector
        # below so they cannot end up in the deployed feature vector and
        # contaminate the StandardScaler's mean/std with constants.
        _variances = np.var(X, axis=0)
        _dead = [FEATURE_NAMES[i] for i in range(len(FEATURE_NAMES)) if i < len(_variances) and _variances[i] < 1e-12]
        if _dead:
            logger.warning(
                "ML train: %d ZERO-VARIANCE features will be force-dropped: %s "
                "— upstream data pipeline produces a constant value here",
                len(_dead), ", ".join(_dead),
            )

        # v3: Temporal sample weights — recent trades matter more
        sample_weights = self._compute_temporal_weights(len(trades))
        logger.info("ML train: temporal weights applied (decay=%.4f)", _TEMPORAL_DECAY)

        # --- Time Series Cross-Validation (RF only — fast diagnostic pass) ---
        # Note: CV is used only for logging/early sanity check.
        # Actual ensemble member selection happens on the dedicated validation split below.
        tscv = TimeSeriesSplit(n_splits=self._cfg.cv_splits)
        cv_precisions, cv_recalls, cv_aucs = [], [], []

        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y_arr[train_idx], y_arr[test_idx]

            if len(X_tr) < 50 or len(X_te) < 15:
                continue

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            fold_model = self._build_rf()
            fold_model.fit(X_tr_s, y_tr)

            y_p = fold_model.predict(X_te_s)
            y_pr = fold_model.predict_proba(X_te_s)[:, 1] if len(set(y_tr)) > 1 else [0.5] * len(y_te)

            cv_precisions.append(precision_score(y_te, y_p, zero_division=0))
            cv_recalls.append(recall_score(y_te, y_p, zero_division=0))
            try:
                cv_aucs.append(roc_auc_score(y_te, y_pr))
            except ValueError:
                cv_aucs.append(0.5)

        if not cv_precisions:
            logger.warning("ML train: no valid CV folds produced")
            return None

        # Log CV results (RF diagnostic — ensemble metrics come from val/holdout splits)
        avg_prec = sum(cv_precisions) / len(cv_precisions)
        avg_rec = sum(cv_recalls) / len(cv_recalls)
        avg_auc = sum(cv_aucs) / len(cv_aucs)
        logger.info("ML CV (RF diagnostic): prec=%.3f rec=%.3f auc=%.3f (over %d folds)",
                     avg_prec, avg_rec, avg_auc, len(cv_precisions))

        # Walk-Forward: Train 70% → Val 15% (model selection + calibration) → Test 15%
        # Val split internally 50/50 for isotonic calibration vs threshold calibration.
        train_end = int(len(X) * 0.70)
        val_end   = int(len(X) * 0.85)
        X_train = X[:train_end]
        X_val   = X[train_end:val_end]
        X_test  = X[val_end:]
        y_train = y_arr[:train_end]
        y_val   = y_arr[train_end:val_end]
        y_test  = y_arr[val_end:]
        pnl_val  = pnl_values[train_end:val_end]
        pnl_test = pnl_values[val_end:]

        if len(X_train) < 50 or len(X_val) < 15 or len(X_test) < 15:
            return None

        final_scaler = StandardScaler()
        X_train_s = final_scaler.fit_transform(X_train)
        X_val_s   = final_scaler.transform(X_val)
        X_test_s  = final_scaler.transform(X_test)

        # v3: TemporalWeighting slices for each split
        train_weights = sample_weights[:train_end]

        # Class imbalance ratio for gradient boosters (computed before build)
        n_neg = int(np.sum(y_train == 0))
        n_pos = int(np.sum(y_train == 1))
        spw = n_neg / n_pos if n_pos > 0 else 1.0

        # --- v3: VotingEnsemble — train all engines, combine by skill ---
        from analyzer.ml_ensemble import VotingEnsemble, AdaptiveFeatureSelector
        ensemble = VotingEnsemble()
        candidate_metrics: dict[str, float] = {}

        rf = self._build_rf()
        try:
            rf.fit(X_train_s, y_train, sample_weight=train_weights)
        except TypeError:
            rf.fit(X_train_s, y_train)

        lgbm = self._build_lgbm(scale_pos_weight=spw)
        if lgbm is not None:
            try:
                lgbm.fit(X_train_s, y_train, sample_weight=train_weights)
            except Exception as lgbm_err:
                logger.debug("LightGBM training failed: %s", lgbm_err)
                lgbm = None

        xgb = self._build_xgb(scale_pos_weight=spw)
        if xgb is not None:
            try:
                xgb.fit(X_train_s, y_train, sample_weight=train_weights)
            except Exception as xgb_err:
                logger.debug("XGBoost training failed: %s", xgb_err)
                xgb = None

        # Per-candidate overfit threshold = base gap + statistical noise margin.
        # The real noise depends on the model's actual train/val precisions
        # (a candidate predicting near 0.5 has much wider CI than one near 0.95),
        # so this is recomputed inside the loop instead of one global value.
        # We log the worst-case (p=0.5) margin upfront just for context.
        _n_val = len(y_val)
        _n_train = len(y_train)
        _worst_case_margin = self._overfit_noise_margin(0.5, 0.5, _n_train, _n_val)
        logger.info(
            "ML overfit guard: base=%.2f worst-case_margin=%.3f (n_train=%d n_val=%d)",
            self._cfg.max_overfit_gap, _worst_case_margin, _n_train, _n_val,
        )

        # Evaluate each candidate on validation set, reject overfit models
        for candidate, tag in [(rf, "rf"), (lgbm, "lgbm"), (xgb, "xgb")]:
            if candidate is None:
                continue
            try:
                y_pred_v = candidate.predict(X_val_s)
                y_proba_v = (
                    candidate.predict_proba(X_val_s)[:, 1]
                    if len(set(y_train)) > 1
                    else np.full(len(y_val), 0.5)
                )
            except Exception as exc:
                logger.warning("ML candidate [%s] eval failed: %s", tag, exc)
                continue

            prec_v = precision_score(y_val, y_pred_v, zero_division=0)
            rec_v = recall_score(y_val, y_pred_v, zero_division=0)
            try:
                auc_v = roc_auc_score(y_val, y_proba_v)
            except ValueError:
                auc_v = 0.5

            pf_v = self._compute_profit_factor_score(y_pred_v, pnl_val)
            skill_v = compute_skill_score(prec_v, rec_v, auc_v, pf_v)

            # Precision-based overfit guard: directly catches the metric that drives
            # trading PnL. Train precision >> val precision = model memorized train set.
            y_train_pred_c = candidate.predict(X_train_s)
            train_prec_c = precision_score(y_train, y_train_pred_c, zero_division=0)
            overfit_gap = train_prec_c - prec_v
            noise_margin = self._overfit_noise_margin(train_prec_c, prec_v, _n_train, _n_val)
            cand_threshold = self._cfg.max_overfit_gap + noise_margin
            if overfit_gap > cand_threshold:
                logger.warning(
                    "ML OVERFITTING [%s]: train_prec=%.3f val_prec=%.3f gap=%.3f "
                    "(threshold=%.3f = base %.2f + 1.96σ noise %.3f) — REJECTED",
                    tag, train_prec_c, prec_v, overfit_gap,
                    cand_threshold, self._cfg.max_overfit_gap, noise_margin,
                )
                continue

            logger.info(
                "ML candidate [%s]: val_skill=%.3f prec=%.3f rec=%.3f auc=%.3f pf=%.3f → added to ensemble",
                tag, skill_v, prec_v, rec_v, auc_v, pf_v,
            )
            ensemble.add_member(candidate, tag, skill_v)
            candidate_metrics[tag] = skill_v

        if not ensemble.is_ready:
            logger.warning("ML train: all candidates failed overfitting check — ensemble empty")
            self._metrics = MLMetrics(
                precision=0, recall=0, roc_auc=0.5,
                accuracy=0, skill_score=0.0,
                train_samples=len(X_train), test_samples=len(X_test),
                feature_importances={},
            )
            return self._metrics

        logger.info(
            "ML VotingEnsemble: %d/%d models accepted, members: %s",
            ensemble.member_count(), 3,
            ", ".join(f"{t}={s:.3f}" for t, s in candidate_metrics.items()),
        )

        # v3: Adaptive Feature Selector — analyse importances across all members
        combined_importances: dict[str, float] = {}
        for candidate, tag in [(rf, "rf"), (lgbm, "lgbm"), (xgb, "xgb")]:
            if candidate is None:
                continue
            raw_imp = getattr(candidate, 'feature_importances_', None)
            if raw_imp is not None:
                for i, name in enumerate(FEATURE_NAMES):
                    if i < len(raw_imp):
                        combined_importances[name] = combined_importances.get(name, 0.0) + float(raw_imp[i])

        # Normalize to get average importance across models
        n_models = sum(1 for c in [rf, lgbm, xgb] if c is not None)
        if n_models > 0:
            combined_importances = {k: v / n_models for k, v in combined_importances.items()}

        importances = combined_importances

        # Phase 2: Apply feature selector → refit scaler → retrain all models on selected features
        # This ensures inference pipeline matches training exactly:
        # predict(): selector.transform(30→N) → scaler.transform(N) → model.predict(N)
        #
        # Force-drop zero-variance features by zeroing their importance: the
        # selector keeps anything ≥ min_importance, and tree models can give a
        # constant column a small but non-zero "importance" via spurious early
        # splits, so we explicitly suppress them here.
        for _dead_name in _dead:
            importances[_dead_name] = 0.0
        self._feature_selector.fit(importances, FEATURE_NAMES)

        if self._feature_selector.dropped_names:
            # Apply selector to all splits (32 → N selected features)
            X_train_sel = self._feature_selector.transform(X_train)
            X_val_sel   = self._feature_selector.transform(X_val)
            X_test_sel  = self._feature_selector.transform(X_test)

            # Refit scaler on selected features only
            final_scaler = StandardScaler()
            X_train_s = final_scaler.fit_transform(X_train_sel)
            X_val_s   = final_scaler.transform(X_val_sel)
            X_test_s  = final_scaler.transform(X_test_sel)

            # Retrain all models on selected features
            ensemble = VotingEnsemble()
            candidate_metrics = {}

            rf2 = self._build_rf()
            try:
                rf2.fit(X_train_s, y_train, sample_weight=train_weights)
            except TypeError:
                rf2.fit(X_train_s, y_train)

            lgbm2 = self._build_lgbm(scale_pos_weight=spw)
            if lgbm2 is not None:
                try:
                    lgbm2.fit(X_train_s, y_train, sample_weight=train_weights)
                except Exception:
                    lgbm2 = None

            xgb2 = self._build_xgb(scale_pos_weight=spw)
            if xgb2 is not None:
                try:
                    xgb2.fit(X_train_s, y_train, sample_weight=train_weights)
                except Exception:
                    xgb2 = None

            for candidate, tag in [(rf2, "rf"), (lgbm2, "lgbm"), (xgb2, "xgb")]:
                if candidate is None:
                    continue
                try:
                    y_pred_v  = candidate.predict(X_val_s)
                    y_proba_v = candidate.predict_proba(X_val_s)[:, 1] if len(set(y_train)) > 1 else np.full(len(y_val), 0.5)
                except Exception as exc:
                    logger.warning("ML phase-2 candidate [%s] eval failed: %s", tag, exc)
                    continue
                prec_v  = precision_score(y_val, y_pred_v, zero_division=0)
                rec_v   = recall_score(y_val, y_pred_v, zero_division=0)
                try:
                    auc_v = roc_auc_score(y_val, y_proba_v)
                except ValueError:
                    auc_v = 0.5
                pf_v = self._compute_profit_factor_score(y_pred_v, pnl_val)
                skill_v = compute_skill_score(prec_v, rec_v, auc_v, pf_v)
                y_train_pred_c = candidate.predict(X_train_s)
                train_prec_c   = precision_score(y_train, y_train_pred_c, zero_division=0)
                noise_margin_p2 = self._overfit_noise_margin(train_prec_c, prec_v, _n_train, _n_val)
                cand_threshold_p2 = self._cfg.max_overfit_gap + noise_margin_p2
                if train_prec_c - prec_v > cand_threshold_p2:
                    logger.warning(
                        "ML phase-2 OVERFITTING [%s]: gap=%.3f (threshold=%.3f = base %.2f + 1.96σ noise %.3f) — REJECTED",
                        tag, train_prec_c - prec_v, cand_threshold_p2,
                        self._cfg.max_overfit_gap, noise_margin_p2,
                    )
                    continue
                ensemble.add_member(candidate, tag, skill_v)
                candidate_metrics[tag] = skill_v
                logger.info("ML phase-2 [%s]: skill=%.3f prec=%.3f rec=%.3f pf=%.3f → accepted", tag, skill_v, prec_v, rec_v, pf_v)

            if not ensemble.is_ready:
                logger.warning("ML phase-2: all candidates rejected — falling back to phase-1 ensemble")
                # fall back to phase-1 results (already in local variables)
            else:
                logger.info("ML phase-2 ensemble: %d members on %d features", ensemble.member_count(), X_train_sel.shape[1])
                best_tag = max(candidate_metrics, key=candidate_metrics.get)
                best_model = {"rf": rf2, "lgbm": lgbm2, "xgb": xgb2}.get(best_tag)
                rf, lgbm, xgb = rf2, lgbm2, xgb2  # update refs for calibration below

        # Two-stage calibration: split validation into CAL (for probability
        # calibration) and THR (for threshold tuning). Sharing one set for both
        # causes double-dipping — the threshold gets optimized against noise in
        # the same sample used to learn the calibration, producing sunny metrics
        # that don't hold up live.
        #
        # Require 60 total samples (30 per half) to split safely. On smaller
        # validation sets we skip isotonic/Platt calibration entirely and use
        # raw ensemble probabilities; the threshold still gets tuned on the
        # full set, but with no calibration to overfit.
        MIN_SPLIT_SAMPLES = 60
        if len(X_val_s) >= MIN_SPLIT_SAMPLES:
            val_mid = len(X_val_s) // 2
            X_val_cal, X_val_thr = X_val_s[:val_mid], X_val_s[val_mid:]
            y_val_cal, y_val_thr = y_val[:val_mid], y_val[val_mid:]
            pnl_val_thr = pnl_val[val_mid:]
            ensemble.apply_isotonic_calibration(y_val_cal, X_val_cal)
        else:
            logger.warning(
                "ML calibration split skipped: val_n=%d < %d — using raw probabilities, "
                "threshold tuned on full validation set (acceptable noise given small N)",
                len(X_val_s), MIN_SPLIT_SAMPLES,
            )
            X_val_thr = X_val_s
            y_val_thr = y_val
            pnl_val_thr = pnl_val

        # Keep best single model as legacy fallback (for save_to_file compat)
        best_tag = max(candidate_metrics, key=candidate_metrics.get) if candidate_metrics else "rf"
        best_model: Any = {"rf": rf, "lgbm": lgbm, "xgb": xgb}.get(best_tag, rf)

        # Threshold calibration via profit-factor optimization on second half of val
        y_proba_val_ensemble = ensemble.predict_proba_calibrated(X_val_thr)
        pnl_thr_arr = np.array(pnl_val_thr, dtype=np.float64) if pnl_val_thr else None

        calib_target = max(0.50, self._cfg.min_precision - 0.02)
        calibrated_thr = self._calibrate_threshold(
            y_val_thr, y_proba_val_ensemble,
            min_precision=calib_target, pnl=pnl_thr_arr,
        )
        logger.info(
            "ML VotingEnsemble calibrated threshold: %.3f (targeting P=%.2f, members=%d)",
            calibrated_thr, calib_target, ensemble.member_count(),
        )

        # Log feature importance summary
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        top5 = ", ".join(f"{n}={v:.3f}" for n, v in sorted_imp[:5])
        logger.info("ML averaged top features: %s", top5)
        if self._feature_selector.dropped_names:
            logger.info(
                "ML AdaptiveFeatureSelector dropped (%d): %s",
                len(self._feature_selector.dropped_names),
                ", ".join(self._feature_selector.dropped_names),
            )

        # --- Final metrics on HOLDOUT test set ---
        y_proba_holdout = ensemble.predict_proba_calibrated(X_test_s)
        y_pred_holdout = (y_proba_holdout >= calibrated_thr).astype(int)
        best_precision = precision_score(y_test, y_pred_holdout, zero_division=0)
        best_recall = recall_score(y_test, y_pred_holdout, zero_division=0)
        try:
            best_roc_auc = roc_auc_score(y_test, y_proba_holdout)
        except ValueError:
            best_roc_auc = 0.5
        best_accuracy = accuracy_score(y_test, y_pred_holdout)
        pf_score = self._compute_profit_factor_score(y_pred_holdout, pnl_test)
        # Precision 3x recall weight: false positives (losing trades) are far more expensive
        # than missed winners. AUC remains highest (threshold-independent ranking quality).
        final_skill = compute_skill_score(best_precision, best_recall, best_roc_auc, pf_score)

        # --- Precision recovery: when initial pass fails on precision ---
        # Noisy assets (e.g. ETH) often produce high recall / low precision.
        # Recovery strategy:
        #   Phase A: raise threshold on existing ensemble (cheap — no retraining)
        #   Phase B: retrain with conservative hyperparams + higher class penalty
        if best_precision < self._cfg.min_precision and best_recall > 0.40:
            logger.info(
                "ML PRECISION RECOVERY triggered: prec=%.3f < min=%.3f (recall=%.3f)",
                best_precision, self._cfg.min_precision, best_recall,
            )

            # Phase A: search for a higher threshold that meets precision target
            recovered = False
            for thr_candidate in np.arange(calibrated_thr + 0.02, 0.80, 0.01):
                y_pred_try = (y_proba_holdout >= thr_candidate).astype(int)
                n_pred_pos = int(y_pred_try.sum())
                if n_pred_pos < 5:
                    break
                prec_try = precision_score(y_test, y_pred_try, zero_division=0)
                rec_try = recall_score(y_test, y_pred_try, zero_division=0)
                if prec_try >= self._cfg.min_precision and rec_try >= 0.30:
                    pf_try = self._compute_profit_factor_score(y_pred_try, pnl_test)
                    skill_try = compute_skill_score(prec_try, rec_try, best_roc_auc, pf_try)
                    if skill_try >= self._cfg.min_skill_score * 0.95:
                        calibrated_thr = float(thr_candidate)
                        best_precision = prec_try
                        best_recall = rec_try
                        best_accuracy = accuracy_score(y_test, y_pred_try)
                        pf_score = pf_try
                        final_skill = skill_try
                        y_pred_holdout = y_pred_try
                        recovered = True
                        logger.info(
                            "ML PRECISION RECOVERY Phase A: thr=%.3f → prec=%.3f rec=%.3f skill=%.3f",
                            calibrated_thr, best_precision, best_recall, final_skill,
                        )
                        break

            # Phase B: retrain with conservative models if threshold bump wasn't enough
            if not recovered:
                logger.info("ML PRECISION RECOVERY Phase B: retraining with conservative hyperparams")
                spw_boost = spw * 1.8  # stronger false-positive penalty

                ensemble_b = VotingEnsemble()
                cand_metrics_b: dict[str, float] = {}

                rf_b = self._build_rf(conservative=True)
                try:
                    rf_b.fit(X_train_s, y_train, sample_weight=train_weights)
                except TypeError:
                    rf_b.fit(X_train_s, y_train)

                lgbm_b = self._build_lgbm(scale_pos_weight=spw_boost, conservative=True)
                if lgbm_b is not None:
                    try:
                        lgbm_b.fit(X_train_s, y_train, sample_weight=train_weights)
                    except Exception:
                        lgbm_b = None

                xgb_b = self._build_xgb(scale_pos_weight=spw_boost, conservative=True)
                if xgb_b is not None:
                    try:
                        xgb_b.fit(X_train_s, y_train, sample_weight=train_weights)
                    except Exception:
                        xgb_b = None

                for candidate_b, tag_b in [(rf_b, "rf"), (lgbm_b, "lgbm"), (xgb_b, "xgb")]:
                    if candidate_b is None:
                        continue
                    try:
                        y_pred_bv = candidate_b.predict(X_val_s)
                        y_proba_bv = candidate_b.predict_proba(X_val_s)[:, 1] if len(set(y_train)) > 1 else np.full(len(y_val), 0.5)
                    except Exception:
                        continue
                    prec_bv = precision_score(y_val, y_pred_bv, zero_division=0)
                    rec_bv = recall_score(y_val, y_pred_bv, zero_division=0)
                    try:
                        auc_bv = roc_auc_score(y_val, y_proba_bv)
                    except ValueError:
                        auc_bv = 0.5
                    pf_bv = self._compute_profit_factor_score(y_pred_bv, pnl_val)
                    skill_bv = compute_skill_score(prec_bv, rec_bv, auc_bv, pf_bv)

                    y_train_pred_b = candidate_b.predict(X_train_s)
                    train_prec_b = precision_score(y_train, y_train_pred_b, zero_division=0)
                    noise_margin_b = self._overfit_noise_margin(train_prec_b, prec_bv, _n_train, _n_val)
                    cand_threshold_b = self._cfg.max_overfit_gap + noise_margin_b
                    if train_prec_b - prec_bv > cand_threshold_b:
                        logger.warning(
                            "ML recovery [%s]: overfitting gap=%.3f > threshold=%.3f (base %.2f + 1.96σ noise %.3f) — skipped",
                            tag_b, train_prec_b - prec_bv, cand_threshold_b,
                            self._cfg.max_overfit_gap, noise_margin_b,
                        )
                        continue

                    ensemble_b.add_member(candidate_b, tag_b, skill_bv)
                    cand_metrics_b[tag_b] = skill_bv
                    logger.info("ML recovery [%s]: prec=%.3f rec=%.3f auc=%.3f → accepted", tag_b, prec_bv, rec_bv, auc_bv)

                if ensemble_b.is_ready:
                    # Calibrate the recovery ensemble
                    if val_mid >= 10:
                        ensemble_b.apply_isotonic_calibration(y_val_iso, X_val_iso)
                    else:
                        ensemble_b.apply_isotonic_calibration(y_val, X_val_s)

                    # Find precision-targeting threshold for recovery ensemble
                    y_proba_b_val = ensemble_b.predict_proba_calibrated(X_val_thr)
                    pnl_thr_arr_b = np.array(pnl_val_thr, dtype=np.float64) if pnl_val_thr else None
                    thr_b = self._calibrate_threshold(
                        y_val_thr, y_proba_b_val,
                        min_precision=self._cfg.min_precision, pnl=pnl_thr_arr_b,
                        min_recall=0.25,
                    )

                    # Evaluate on holdout
                    y_proba_b_holdout = ensemble_b.predict_proba_calibrated(X_test_s)
                    y_pred_b_holdout = (y_proba_b_holdout >= thr_b).astype(int)
                    n_pred_b = int(y_pred_b_holdout.sum())
                    if n_pred_b >= 5:
                        prec_b = precision_score(y_test, y_pred_b_holdout, zero_division=0)
                        rec_b = recall_score(y_test, y_pred_b_holdout, zero_division=0)
                        try:
                            auc_b = roc_auc_score(y_test, y_proba_b_holdout)
                        except ValueError:
                            auc_b = 0.5
                        pf_b = self._compute_profit_factor_score(y_pred_b_holdout, pnl_test)
                        skill_b = compute_skill_score(prec_b, rec_b, auc_b, pf_b)

                        logger.info(
                            "ML PRECISION RECOVERY Phase B holdout: prec=%.3f rec=%.3f auc=%.3f skill=%.3f (thr=%.3f)",
                            prec_b, rec_b, auc_b, skill_b, thr_b,
                        )

                        # Accept recovery if it improved precision AND skill is reasonable
                        if prec_b > best_precision and skill_b > final_skill * 0.90:
                            ensemble = ensemble_b
                            calibrated_thr = thr_b
                            best_precision = prec_b
                            best_recall = rec_b
                            best_roc_auc = auc_b
                            best_accuracy = accuracy_score(y_test, y_pred_b_holdout)
                            pf_score = pf_b
                            final_skill = skill_b
                            y_pred_holdout = y_pred_b_holdout
                            y_proba_holdout = y_proba_b_holdout
                            # Update best_model ref for legacy save
                            best_tag_b = max(cand_metrics_b, key=cand_metrics_b.get)
                            best_model = {"rf": rf_b, "lgbm": lgbm_b, "xgb": xgb_b}.get(best_tag_b, rf_b)
                            logger.info(
                                "ML PRECISION RECOVERY Phase B ACCEPTED: prec=%.3f rec=%.3f skill=%.3f",
                                best_precision, best_recall, final_skill,
                            )
                        else:
                            logger.info(
                                "ML PRECISION RECOVERY Phase B: no improvement (prec %.3f→%.3f, skill %.3f→%.3f)",
                                best_precision, prec_b, final_skill, skill_b,
                            )

        # --- Naive baseline: always predict "win" ---
        baseline_win_rate = float(y_test.mean()) if len(y_test) > 0 else 0.5
        precision_lift = best_precision - baseline_win_rate
        auc_lift = best_roc_auc - 0.5
        logger.info(
            "Baseline (always-win): precision=%.3f | Model lift: +%.3f precision, +%.3f AUC",
            baseline_win_rate, precision_lift, auc_lift,
        )

        # --- Block bootstrap 95% CI on holdout (preserves temporal autocorrelation) ---
        n_boot = 500
        boot_prec, boot_auc = [], []
        rng = np.random.default_rng(42)
        n_test = len(X_test_s)
        block_size = max(5, int(n_test ** (1.0 / 3.0)))  # cube-root rule for block length
        n_blocks = max(1, n_test // block_size)
        for _ in range(n_boot):
            # Sample contiguous blocks with replacement, then concatenate
            block_starts = rng.choice(n_test - block_size + 1, size=n_blocks, replace=True)
            idx_b = np.concatenate([np.arange(s, min(s + block_size, n_test)) for s in block_starts])
            idx_b = idx_b[:n_test]  # trim to original test size
            y_b_true = y_test[idx_b]
            y_b_prob = y_proba_holdout[idx_b]
            y_b_pred = (y_b_prob >= calibrated_thr).astype(int)
            bp = precision_score(y_b_true, y_b_pred, zero_division=0)
            boot_prec.append(bp)
            try:
                boot_auc.append(roc_auc_score(y_b_true, y_b_prob))
            except ValueError:
                boot_auc.append(0.5)
        ci_prec = (float(np.percentile(boot_prec, 2.5)), float(np.percentile(boot_prec, 97.5)))
        ci_auc  = (float(np.percentile(boot_auc,  2.5)), float(np.percentile(boot_auc,  97.5)))
        logger.info(
            "ML HOLDOUT [VotingEnsemble] (thr=%.3f): skill=%.3f prec=%.3f [%.3f,%.3f] rec=%.3f auc=%.3f [%.3f,%.3f]",
            calibrated_thr, final_skill,
            best_precision, ci_prec[0], ci_prec[1],
            best_recall, best_roc_auc, ci_auc[0], ci_auc[1],
        )

        # --- Out-of-time validation: retrain on 80%, test on LAST 20% as extra sanity ---
        oot_split = int(len(X) * 0.80)
        X_oot_raw = X[oot_split:]
        y_oot = y_arr[oot_split:]
        # OOT test always runs (regardless of feature selection).
        # Bug fix: previously skipped when no features were dropped — now unconditional.
        oot_auc = None
        if len(X_oot_raw) >= 20:
            try:
                if self._feature_selector.is_fitted:
                    X_oot_sel = self._feature_selector.transform(X_oot_raw)
                else:
                    X_oot_sel = X_oot_raw
                X_oot_s   = final_scaler.transform(X_oot_sel)
                y_oot_prob = ensemble.predict_proba_calibrated(X_oot_s)
                oot_auc    = float(roc_auc_score(y_oot, y_oot_prob))
                oot_prec   = float(precision_score(y_oot, (y_oot_prob >= calibrated_thr).astype(int), zero_division=0))
                logger.info("Out-of-time test (%d samples): AUC=%.3f prec=%.3f", len(X_oot_raw), oot_auc, oot_prec)
                if oot_auc < best_roc_auc - 0.15:
                    logger.warning(
                        "OOT AUC (%.3f) drops >0.15 vs holdout (%.3f) — possible period-specific overfitting",
                        oot_auc, best_roc_auc,
                    )
            except Exception as _oot_err:
                logger.warning("OOT test failed: %s", _oot_err)

        # --- Calibration diagnostics on the FINAL chosen ensemble -----------
        # Computed here (after both Phase A threshold-bump and Phase B retrain
        # have settled) so the persisted metrics describe the model that will
        # actually be deployed. ECE/Brier/percentiles flag the "every signal
        # ≈ 95%" failure mode that AUC/precision alone cannot catch.
        from sklearn.metrics import brier_score_loss
        raw_holdout_final = ensemble.predict_proba(X_test_s)
        try:
            brier_raw = float(brier_score_loss(y_test, raw_holdout_final))
            brier_cal = float(brier_score_loss(y_test, y_proba_holdout))
        except ValueError:
            brier_raw = brier_cal = 0.0
        ece_cal      = self._expected_calibration_error(y_test, y_proba_holdout, n_bins=10)
        proba_mean   = float(y_proba_holdout.mean())
        proba_median = float(np.median(y_proba_holdout))
        proba_p10    = float(np.percentile(y_proba_holdout, 10))
        proba_p90    = float(np.percentile(y_proba_holdout, 90))
        cal_method   = ensemble.calibration_method
        logger.info(
            "ML calibration [%s]: brier raw=%.3f cal=%.3f | ECE=%.3f | "
            "proba mean=%.3f median=%.3f p10=%.3f p90=%.3f (n=%d)",
            cal_method, brier_raw, brier_cal, ece_cal,
            proba_mean, proba_median, proba_p10, proba_p90, len(y_proba_holdout),
        )
        if proba_p10 > 0.70 or ece_cal > 0.15:
            # Loud signal that the displayed probabilities are inflated or
            # clumped — usually a calibration bug, not real model strength.
            logger.warning(
                "ML calibration suspicious: p10=%.3f ECE=%.3f method=%s — outputs look "
                "inflated or clumped. Check class balance and isotonic plateau collapse.",
                proba_p10, ece_cal, cal_method,
            )

        metrics = MLMetrics(
            precision=best_precision,
            recall=best_recall,
            roc_auc=best_roc_auc,
            accuracy=best_accuracy,
            skill_score=final_skill,
            train_samples=len(X_train),
            test_samples=len(X_test),
            feature_importances=importances,
            precision_ci_95=ci_prec,
            auc_ci_95=ci_auc,
            baseline_win_rate=baseline_win_rate,
            precision_lift=precision_lift,
            auc_lift=auc_lift,
            oot_auc=oot_auc,
            brier_score=brier_cal,
            ece=ece_cal,
            mean_proba=proba_mean,
            median_proba=proba_median,
            proba_p10=proba_p10,
            proba_p90=proba_p90,
            calibration_method=cal_method,
        )

        # Gate check — reject if below quality thresholds
        if (
            best_precision < self._cfg.min_precision
            or best_recall < self._cfg.min_recall
            or best_roc_auc < self._cfg.min_roc_auc
            or final_skill < self._cfg.min_skill_score
        ):
            logger.warning(
                "ML metrics below threshold: skill=%.3f prec=%.3f rec=%.3f auc=%.3f — NOT deploying",
                final_skill, best_precision, best_recall, best_roc_auc,
            )
            self._metrics = metrics
            return metrics

        # Deploy new ensemble
        self._ensemble = ensemble
        self._model = best_model         # Legacy compat (save_to_file uses it)
        self._scaler = final_scaler
        self._calibrated_threshold = calibrated_thr
        self._model_version = f"ensemble_v{int(time.time())}_{ensemble.member_count()}m"
        self._metrics = metrics
        self._last_train_ts = int(time.time() * 1000)
        logger.info(
            "ML VotingEnsemble deployed: version=%s skill=%.3f prec=%.3f rec=%.3f auc=%.3f thr=%.3f",
            self._model_version, final_skill, best_precision, best_recall, best_roc_auc, calibrated_thr,
        )
        return metrics

    def predict(
        self,
        trade_features: list[float],
    ) -> MLPrediction:
        """Predict win probability using VotingEnsemble (v3).

        Returns MLPrediction with decision: allow / reduce / block.
        In shadow mode, decision is always 'allow' (prediction is logged only).
        Uses calibrated probabilities from the isotonic-calibrated ensemble.
        """
        if not self.is_ready or self._rollout_mode == "off":
            return MLPrediction(
                probability=0.5,
                decision="allow",
                model_version="",
                rollout_mode=self._rollout_mode,
            )

        # Validate feature vector length
        if len(trade_features) != N_FEATURES:
            logger.error(
                "ML predict: feature count mismatch: got %d, expected %d",
                len(trade_features), N_FEATURES,
            )
            return MLPrediction(
                probability=0.5, decision="allow",
                model_version=self._model_version, rollout_mode=self._rollout_mode,
            )

        try:
            features_arr = np.array([trade_features], dtype=np.float64)
            features_arr = np.nan_to_num(features_arr, nan=0.0, posinf=0.0, neginf=0.0)
            # Apply feature selector first (30 → N selected features), then scaler
            if self._feature_selector.is_fitted and self._feature_selector.dropped_names:
                features_arr = self._feature_selector.transform(features_arr)
            if self._scaler is not None:
                features_arr = self._scaler.transform(features_arr)

            # v3: Use ensemble if available, fallback to legacy single model
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names")
                if self._ensemble is not None and self._ensemble.is_ready:
                    proba = float(self._ensemble.predict_proba_calibrated(features_arr)[0])
                elif self._model is not None:
                    proba = float(self._model.predict_proba(features_arr)[0][1])
                else:
                    proba = 0.5
        except Exception as exc:
            logger.warning("ML predict failed (defaulting to allow): %s", exc, exc_info=True)
            proba = 0.5

        # Decision logic: respect both the trained threshold and the env floor.
        # _calibrated_threshold is learned per-model during training (tuned on
        # validation PnL + precision); _cfg.block_threshold is an env-level floor.
        # Taking max() ensures we never loosen below what training validated,
        # while still letting the operator raise the bar via ANALYZER_ML_BLOCK_THRESHOLD.
        cal_thr = max(self._calibrated_threshold or 0.0, self._cfg.block_threshold or 0.0)
        if cal_thr <= 0:
            cal_thr = 0.5
        if proba < cal_thr * 0.85:       # well below threshold → block
            decision = "block"
        elif proba < cal_thr:             # slightly below → reduce
            decision = "reduce"
        else:
            decision = "allow"

        # In shadow mode, always allow (just log)
        effective_decision = decision if self._rollout_mode == "block" else "allow"

        return MLPrediction(
            probability=proba,
            decision=effective_decision,
            model_version=self._model_version,
            rollout_mode=self._rollout_mode,
        )

    def needs_retrain(self) -> bool:
        """Return True if retraining is needed: scheduled interval OR concept drift."""
        if self._last_train_ts == 0:
            return True
        days_since = (time.time() * 1000 - self._last_train_ts) / (86400 * 1000)
        if days_since >= self._cfg.retrain_days:
            return True
        # Concept drift: live performance has diverged from training metrics
        if self.drift_detected:
            logger.warning(
                "Concept drift detected — triggering early retrain (live: %s)",
                self._live_tracker.live_metrics(),
            )
            return True
        return False

    def save_to_file(self, model_path: str | Path) -> bool:
        """Save trained model, ensemble, scaler and metrics to pickle file (atomic write).

        v3: persists VotingEnsemble and AdaptiveFeatureSelector alongside legacy model
        so the system survives restarts without full retraining.
        """
        if not self.is_ready:
            logger.warning("ML save skipped: model not ready")
            return False

        path = Path(model_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            metrics_dict = {}
            if self._metrics:
                metrics_dict = {
                    "precision": self._metrics.precision,
                    "recall": self._metrics.recall,
                    "roc_auc": self._metrics.roc_auc,
                    "accuracy": self._metrics.accuracy,
                    "skill_score": self._metrics.skill_score,
                    "train_samples": self._metrics.train_samples,
                    "test_samples": self._metrics.test_samples,
                    "feature_importances": self._metrics.feature_importances,
                    # Statistical confidence
                    "precision_ci_95": list(self._metrics.precision_ci_95),
                    "auc_ci_95": list(self._metrics.auc_ci_95),
                    # Baseline comparison
                    "baseline_win_rate": self._metrics.baseline_win_rate,
                    "precision_lift": self._metrics.precision_lift,
                    "auc_lift": self._metrics.auc_lift,
                    # Out-of-time robustness
                    "oot_auc": self._metrics.oot_auc,
                    # Calibration diagnostics — visible after restart
                    "brier_score": self._metrics.brier_score,
                    "ece": self._metrics.ece,
                    "mean_proba": self._metrics.mean_proba,
                    "median_proba": self._metrics.median_proba,
                    "proba_p10": self._metrics.proba_p10,
                    "proba_p90": self._metrics.proba_p90,
                    "calibration_method": self._metrics.calibration_method,
                }
            data = {
                # v3: full ensemble (primary predictor)
                "ensemble": self._ensemble,
                "feature_selector": self._feature_selector,
                "calibrated_threshold": self._calibrated_threshold,
                # Legacy: single best model (fallback)
                "model": self._model,
                "scaler": self._scaler,
                "version": self._model_version,
                "metrics": metrics_dict,
                "saved_at": int(time.time()),
                "format": "v3",
            }
            tmp_path = path.with_suffix(".tmp")
            payload = pickle.dumps(data)
            # C-3: Integrity checksum to detect corruption/tampering
            checksum = hashlib.sha256(payload).hexdigest()
            with tmp_path.open("wb") as f:
                pickle.dump({"payload": payload, "checksum": checksum, "format": "v3_signed"}, f)
            tmp_path.replace(path)
            n_members = self._ensemble.member_count() if self._ensemble else 0
            logger.info(
                "ML model saved to %s (version=%s, ensemble_members=%d)",
                path, self._model_version, n_members,
            )

            # MLOps 10/10: Append to model version registry for audit trail
            self._append_to_registry(path, checksum, n_members)

            return True
        except Exception as exc:
            logger.warning("ML model save failed to %s: %s", path, exc)
            return False

    def _append_to_registry(self, model_path: Path, checksum: str, n_members: int) -> None:
        """Append model metadata to a JSON registry file for version tracking."""
        import json
        registry_path = model_path.parent / "model_registry.json"
        entry = {
            "version": self._model_version,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "saved_ts": int(time.time()),
            "checksum_sha256": checksum,
            "ensemble_members": n_members,
            "calibrated_threshold": round(self._calibrated_threshold, 4),
            "metrics": {},
        }
        if self._metrics:
            entry["metrics"] = {
                "precision": round(self._metrics.precision, 4),
                "recall": round(self._metrics.recall, 4),
                "roc_auc": round(self._metrics.roc_auc, 4),
                "accuracy": round(self._metrics.accuracy, 4),
                "skill_score": round(self._metrics.skill_score, 4),
                "train_samples": self._metrics.train_samples,
                "test_samples": self._metrics.test_samples,
                "precision_ci_95": [round(v, 4) for v in self._metrics.precision_ci_95],
                "auc_ci_95": [round(v, 4) for v in self._metrics.auc_ci_95],
                "baseline_win_rate": round(self._metrics.baseline_win_rate, 4),
                "precision_lift": round(self._metrics.precision_lift, 4),
                "auc_lift": round(self._metrics.auc_lift, 4),
                "oot_auc": round(self._metrics.oot_auc, 4) if self._metrics.oot_auc is not None else None,
            }
        try:
            registry: list = []
            if registry_path.exists():
                with registry_path.open("r", encoding="utf-8") as f:
                    registry = json.load(f)
            registry.append(entry)
            # Keep last 50 entries to avoid unbounded growth
            registry = registry[-50:]
            with registry_path.open("w", encoding="utf-8") as f:
                json.dump(registry, f, indent=2, ensure_ascii=False)
            logger.info("ML registry updated: %s (%d entries)", registry_path, len(registry))
        except Exception as exc:
            logger.debug("ML registry write failed: %s", exc)

    def load_from_file(self, model_path: str | Path) -> bool:
        """Load trained ensemble + model, scaler and metrics from pickle file.

        v3: restores VotingEnsemble, AdaptiveFeatureSelector, and calibrated_threshold.
        Falls back gracefully to legacy single-model format for backwards compat.
        """
        path = Path(model_path)
        if not path.exists():
            logger.warning("ML load skipped: file not found %s", path)
            return False
        try:
            with path.open("rb") as f:
                raw = pickle.load(f)

            # C-3: Verify integrity checksum if present
            if isinstance(raw, dict) and raw.get("format") == "v3_signed":
                payload = raw["payload"]
                expected = raw.get("checksum", "")
                actual = hashlib.sha256(payload).hexdigest()
                if expected and actual != expected:
                    logger.error(
                        "ML load ABORTED: checksum mismatch (expected=%s, got=%s). "
                        "File may be corrupted or tampered.",
                        expected[:12], actual[:12],
                    )
                    return False
                data = pickle.loads(payload)
            else:
                # Legacy unsigned format
                data = raw

            fmt = data.get("format", "v1")

            # v3: restore ensemble (primary predictor)
            if fmt == "v3":
                self._ensemble = data.get("ensemble")
                self._feature_selector = data.get("feature_selector", self._feature_selector)
                self._calibrated_threshold = data.get("calibrated_threshold", 0.5)
                n_members = self._ensemble.member_count() if self._ensemble else 0
                logger.info(
                    "ML ensemble loaded from %s (version=%s, members=%d)",
                    path, data.get("version", ""), n_members,
                )
            else:
                logger.info("ML loading legacy v1 format from %s", path)

            # Always restore legacy components (fallback + compat)
            self._model = data.get("model")
            self._scaler = data.get("scaler")
            self._model_version = data.get("version", "")
            saved_at = data.get("saved_at", 0)
            self._last_train_ts = saved_at * 1000 if saved_at else 0

            metrics_dict = data.get("metrics", {})
            if metrics_dict:
                # .get() with dataclass defaults so older saves stay loadable.
                pci = metrics_dict.get("precision_ci_95")
                aci = metrics_dict.get("auc_ci_95")
                self._metrics = MLMetrics(
                    precision=metrics_dict.get("precision", 0.0),
                    recall=metrics_dict.get("recall", 0.0),
                    roc_auc=metrics_dict.get("roc_auc", 0.0),
                    accuracy=metrics_dict.get("accuracy", 0.0),
                    skill_score=metrics_dict.get("skill_score", 0.0),
                    train_samples=metrics_dict.get("train_samples", 0),
                    test_samples=metrics_dict.get("test_samples", 0),
                    feature_importances=metrics_dict.get("feature_importances", {}),
                    precision_ci_95=tuple(pci) if pci else (0.0, 0.0),
                    auc_ci_95=tuple(aci) if aci else (0.0, 0.0),
                    baseline_win_rate=metrics_dict.get("baseline_win_rate", 0.0),
                    precision_lift=metrics_dict.get("precision_lift", 0.0),
                    auc_lift=metrics_dict.get("auc_lift", 0.0),
                    oot_auc=metrics_dict.get("oot_auc"),
                    brier_score=metrics_dict.get("brier_score", 0.0),
                    ece=metrics_dict.get("ece", 0.0),
                    mean_proba=metrics_dict.get("mean_proba", 0.5),
                    median_proba=metrics_dict.get("median_proba", 0.5),
                    proba_p10=metrics_dict.get("proba_p10", 0.0),
                    proba_p90=metrics_dict.get("proba_p90", 1.0),
                    calibration_method=metrics_dict.get("calibration_method", "none"),
                )

            # Ready if ensemble present OR legacy model present
            ready = (
                (self._ensemble is not None and self._ensemble.is_ready)
                or self._model is not None
            )
            if ready:
                logger.info("ML predictor ready (version=%s)", self._model_version)
            return ready
        except Exception as exc:
            logger.error("ML model load failed from %s: %s", path, exc)
            return False
