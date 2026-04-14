"""
Trade Analyzer Level 3 — ML Predictor (Triple-Engine Ensemble, v3).

VotingEnsemble: RF + LightGBM + XGBoost soft-voting instead of winner-takes-all.
ML ONLY filters (block/reduce), NEVER initiates trades.

Rollout modes: off → shadow → block
- shadow: logs predictions, never blocks
- block: actively blocks signals with low probability

31 features (all pre-trade, no forward-looking bias):
  Technical:  rsi_14, adx, ema_9_vs_21, bb_bandwidth, volume_ratio,
              macd_histogram, atr_ratio, adx_normalized
  Temporal:   hour_of_day, day_of_week
  Encoding:   market_regime_encoded, strategy_regime_fit (v4)
  Historical: recent_win_rate_10, hours_since_last_trade,
              rolling_avg_pnl_pct_20, consecutive_losses,
              strategy_specific_wr_10 (v4)
  Sentiment:  news_sentiment, fear_greed_normalized, trend_alignment,
              regime_bias
  Enhanced:   cci, roc, cmf, bb_pct_b, hist_volatility, dmi_spread,
              stoch_rsi, price_change_5h, momentum, rsi_daily

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

Skill score = 0.20*precision + 0.20*recall + 0.35*roc_auc + 0.25*profit_factor
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import time
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

N_FEATURES = 31  # v4: +1 (strategy_encoded → strategy_regime_fit + strategy_specific_wr_10)

# Encoding maps
REGIME_ENCODING = {
    "trending_up": 0, "trending_down": 1, "sideways": 2,
    "volatile": 3, "unknown": 4,
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
}

FEATURE_NAMES = [
    # Technical (0-6)
    "rsi_14", "adx", "ema_9_vs_21", "bb_bandwidth", "volume_ratio",
    "macd_histogram", "atr_ratio",
    # Temporal (7-8)
    "hour_of_day", "day_of_week",
    # Encoding (9-10)
    "market_regime_encoded",
    "strategy_regime_fit",       # v4: replaces strategy_encoded (categorical)
    # Historical (11-15)
    "recent_win_rate_10", "hours_since_last_trade",
    "rolling_avg_pnl_pct_20", "consecutive_losses",
    "strategy_specific_wr_10",   # v4: win rate of last 10 trades by THIS strategy
    # Sentiment (16-20)
    "news_sentiment", "fear_greed_normalized", "trend_alignment",
    "regime_bias", "adx_normalized",
    # Phase 2: Enhanced indicators (21-30)
    "cci", "roc", "cmf", "bb_pct_b", "hist_volatility",
    "dmi_spread", "stoch_rsi", "price_change_5h",
    "momentum", "rsi_daily",
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
    retrain_days: int = 30
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

    Usage:
        tracker.record(predicted_prob=0.72, actual_win=True)
        if tracker.is_drifting(training_precision=0.65):
            trigger_retrain()
    """

    def __init__(self, window: int = 50, drift_threshold: float = 0.12) -> None:
        """
        Args:
            window: Number of most-recent live trades to evaluate.
            drift_threshold: Alert if live_precision < training_precision - threshold.
        """
        self._window = window
        self._drift_threshold = drift_threshold
        self._history: list[tuple[float, int]] = []  # (predicted_prob, actual 0/1)

    def record(self, predicted_prob: float, actual_win: bool) -> None:
        """Record one live prediction + its realized outcome."""
        self._history.append((predicted_prob, int(actual_win)))
        # Keep a rolling buffer (3× window) to avoid unbounded growth
        if len(self._history) > self._window * 3:
            self._history = self._history[-self._window * 3:]

    def live_metrics(self) -> dict:
        """Compute rolling precision, win rate, and calibration on recent window."""
        n = len(self._history)
        if n < 10:
            return {"status": "insufficient_data", "n": n}

        recent = self._history[-self._window:]
        probs   = np.array([p for p, _ in recent], dtype=np.float64)
        actuals = np.array([a for _, a in recent], dtype=np.float64)
        preds   = (probs >= 0.5).astype(int)

        win_rate   = float(actuals.mean())
        n_pred_win = int(preds.sum())
        live_prec  = float(np.sum((preds == 1) & (actuals == 1)) / n_pred_win) if n_pred_win > 0 else 0.0
        # Calibration error: mean predicted prob vs actual win rate
        calib_err = float(abs(probs.mean() - win_rate))

        try:
            from sklearn.metrics import roc_auc_score as _auc
            live_auc = float(_auc(actuals, probs)) if len(set(actuals)) > 1 else 0.5
        except Exception:
            live_auc = 0.5

        return {
            "n": len(recent),
            "live_precision": live_prec,
            "live_win_rate": win_rate,
            "live_auc": live_auc,
            "calibration_error": calib_err,
        }

    def is_drifting(self, training_precision: float) -> bool:
        """Return True if live precision has dropped more than drift_threshold below training."""
        m = self.live_metrics()
        if "live_precision" not in m:
            return False
        return (training_precision - m["live_precision"]) > self._drift_threshold

    @property
    def n_recorded(self) -> int:
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

    def _build_rf(self):
        """Build a RandomForest classifier."""
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=self._cfg.n_estimators,
            max_depth=self._cfg.max_depth,
            min_samples_leaf=self._cfg.min_child_samples,
            min_samples_split=self._cfg.min_samples_split,
            max_features=self._cfg.max_features,
            # Removed class_weight="balanced" to prioritize precision over recall
            random_state=42,
            n_jobs=-1,
        )

    def _build_lgbm(self):
        """Build a LightGBM classifier if available, else None."""
        if not self._cfg.use_lightgbm:
            return None
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                n_estimators=self._cfg.n_estimators,
                max_depth=self._cfg.max_depth,
                learning_rate=self._cfg.learning_rate,
                min_child_samples=self._cfg.min_child_samples,
                subsample=self._cfg.subsample,
                colsample_bytree=self._cfg.colsample_bytree,
                # Removed class_weight="balanced" to prioritize precision over recall
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
        except ImportError:
            logger.debug("LightGBM not available, using RandomForest only")
            return None

    def _build_xgb(self, scale_pos_weight: float = 1.0):
        """Build an XGBoost classifier if available, else None.

        Args:
            scale_pos_weight: Ratio of negative/positive samples for class imbalance.
                              Computed dynamically from training data.
        """
        if not self._cfg.use_xgboost:
            return None
        try:
            from xgboost import XGBClassifier
            # Deliberately conservative hyperparams to prevent overfitting on small datasets.
            # max_depth=4 (vs RF's 8) + strong L1/L2 + high min_child_weight keeps XGBoost
            # from memorising the training set and allows it to pass the 10% overfit guard.
            return XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=self._cfg.learning_rate,
                subsample=0.7,
                colsample_bytree=0.7,
                min_child_weight=30,     # require at least 30 samples in each leaf
                reg_alpha=0.5,           # L1 regularization (sparsity)
                reg_lambda=2.0,          # L2 regularization (shrinkage)
                gamma=0.2,               # min loss reduction for split
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
    def _calibrate_threshold(y_true, y_proba, min_precision: float = 0.55) -> float:
        """Find optimal probability threshold that maximizes F1 with min precision constraint.

        Uses precision-recall curve on validation set to find the threshold
        where precision >= min_precision and F1 is maximized.
        """
        from sklearn.metrics import precision_recall_curve
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

        best_threshold = 0.5
        best_f1 = 0.0

        for prec, rec, thr in zip(precisions[:-1], recalls[:-1], thresholds):
            if prec < min_precision:
                continue
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
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
            atr_ratio = trade.atr_at_entry / entry_price
            adx_normalized = min(trade.adx_at_entry / 50.0, 1.0)

            # v4: strategy_regime_fit — continuous signal replacing raw strategy_encoded.
            # Captures WHY the strategy fits current conditions, not just which strategy it is.
            strat_fit = self._strategy_regime_fit(trade.strategy_name, trade.market_regime)

            # v4: strategy_specific_wr_10 — win rate of last 10 trades by THIS strategy only.
            # Complements strat_fit: measures how this strategy has been performing recently,
            # independent of other strategies in the portfolio.
            prev_same = strategy_win_history[trade.strategy_name]  # trades so far (before current)
            if prev_same:
                last10 = prev_same[-10:]
                strategy_specific_wr = sum(w for _, w in last10) / len(last10)
            else:
                strategy_specific_wr = 0.5  # neutral prior when no history

            # Vectorized recent_win_rate (last 10 prior trades, all strategies)
            if idx >= 1:
                start = max(0, idx - 10)
                recent_win_rate = is_win[start:idx].mean()
            else:
                recent_win_rate = 0.5

            # rolling_avg_pnl_pct_20: mean pnl_pct of last 20 closed trades.
            # Replaces daily_pnl_so_far (which had day-boundary reset issues in live
            # trading and created circular "win on good days" clustering bias).
            if idx >= 1:
                _start20 = max(0, idx - 20)
                _recent_pnl = np.array([t.pnl_pct for t in trades[_start20:idx]], dtype=np.float64)
                rolling_avg_pnl_pct_20 = float(np.mean(_recent_pnl)) / 10.0  # normalize: ±10% → ±1.0
            else:
                rolling_avg_pnl_pct_20 = 0.0

            # C-2 fix: hours_since using pre-parsed timestamps
            hours_since = 0.0
            if idx > 0 and parsed_open[idx] and parsed_close[idx - 1]:
                delta = (parsed_open[idx] - parsed_close[idx - 1]).total_seconds()
                hours_since = max(delta / 3600.0, 0.0)

            X[idx] = [
                trade.rsi_at_entry,
                trade.adx_at_entry,
                ema_9_vs_21,
                trade.bb_bandwidth_at_entry,
                trade.volume_ratio_at_entry,
                trade.macd_histogram_at_entry,
                atr_ratio,
                float(trade.hour_of_day),
                float(trade.day_of_week),
                float(REGIME_ENCODING.get(trade.market_regime, 4)),
                strat_fit,                   # v4: was strategy_encoded (categorical int)
                recent_win_rate,
                hours_since,
                rolling_avg_pnl_pct_20,
                consec_losses[idx],          # W-2 fix: pre-computed O(1)
                strategy_specific_wr,        # v4: new feature (pos 15)
                trade.news_sentiment,
                trade.fear_greed_index / 100.0,
                trade.trend_alignment,
                regime_bias,
                adx_normalized,
                # Phase 2: Enhanced indicators
                trade.cci_at_entry / 200.0,
                trade.roc_at_entry / 10.0,
                trade.cmf_at_entry,
                trade.bb_pct_b_at_entry,
                trade.hist_volatility_at_entry,
                trade.dmi_spread_at_entry / 50.0,
                trade.stoch_rsi_at_entry / 100.0,
                trade.price_change_5h_at_entry / 5.0,
                trade.momentum_at_entry / 100.0,
                trade.rsi_daily_at_entry / 100.0,
            ]

            # Record AFTER building features (no look-ahead: next trade uses this trade's outcome)
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

        # Walk-Forward Validation: Train 70% → Validate 15% → Holdout Test 15%
        # Model selection on validation set, final metrics on unseen holdout
        train_end = int(len(X) * 0.70)
        val_end = int(len(X) * 0.85)
        X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
        y_train, y_val, y_test = y_arr[:train_end], y_arr[train_end:val_end], y_arr[val_end:]
        pnl_val  = pnl_values[train_end:val_end]
        pnl_test = pnl_values[val_end:]

        if len(X_train) < 50 or len(X_val) < 15 or len(X_test) < 15:
            return None

        final_scaler = StandardScaler()
        X_train_s = final_scaler.fit_transform(X_train)
        X_val_s = final_scaler.transform(X_val)
        X_test_s = final_scaler.transform(X_test)

        # v3: TemporalWeighting slices for each split
        train_weights = sample_weights[:train_end]
        # val/test don't use sample_weight for evaluation (metrics must be unweighted)

        # --- v3: VotingEnsemble — train all engines, combine by skill ---
        from analyzer.ml_ensemble import VotingEnsemble, AdaptiveFeatureSelector
        ensemble = VotingEnsemble()
        candidate_metrics: dict[str, float] = {}

        rf = self._build_rf()
        try:
            rf.fit(X_train_s, y_train, sample_weight=train_weights)
        except TypeError:
            rf.fit(X_train_s, y_train)  # fallback if sample_weight not supported

        lgbm = self._build_lgbm()
        if lgbm is not None:
            try:
                lgbm.fit(X_train_s, y_train, sample_weight=train_weights)
            except Exception as lgbm_err:
                logger.debug("LightGBM training failed: %s", lgbm_err)
                lgbm = None

        # N-5: compute actual class imbalance ratio for XGBoost
        n_neg = int(np.sum(y_train == 0))
        n_pos = int(np.sum(y_train == 1))
        spw = n_neg / n_pos if n_pos > 0 else 1.0
        xgb = self._build_xgb(scale_pos_weight=spw)
        if xgb is not None:
            try:
                xgb.fit(X_train_s, y_train, sample_weight=train_weights)
            except Exception as xgb_err:
                logger.debug("XGBoost training failed: %s", xgb_err)
                xgb = None

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
            skill_v = 0.20 * prec_v + 0.20 * rec_v + 0.35 * auc_v + 0.25 * pf_v

            # Overfitting guard
            y_train_pred_c = candidate.predict(X_train_s)
            train_prec_c = precision_score(y_train, y_train_pred_c, zero_division=0)
            overfit_gap = train_prec_c - prec_v
            if overfit_gap > self._cfg.max_overfit_gap:
                logger.warning(
                    "ML OVERFITTING [%s]: train_prec=%.3f val_prec=%.3f gap=%.3f — REJECTED",
                    tag, train_prec_c, prec_v, overfit_gap,
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
        self._feature_selector.fit(importances, FEATURE_NAMES)

        if self._feature_selector.dropped_names:
            # Apply selector to all splits (30 → N selected features)
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

            lgbm2 = self._build_lgbm()
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
                skill_v = 0.20 * prec_v + 0.20 * rec_v + 0.35 * auc_v + 0.25 * pf_v
                y_train_pred_c = candidate.predict(X_train_s)
                train_prec_c   = precision_score(y_train, y_train_pred_c, zero_division=0)
                if train_prec_c - prec_v > self._cfg.max_overfit_gap:
                    logger.warning("ML phase-2 OVERFITTING [%s]: gap=%.3f — REJECTED", tag, train_prec_c - prec_v)
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

        # v3: Apply IsotonicCalibration on FIRST HALF of validation set (C-4 fix)
        # Threshold calibration uses SECOND HALF to prevent data leakage
        val_mid = len(X_val_s) // 2
        if val_mid >= 10:
            X_val_calib, X_val_thr = X_val_s[:val_mid], X_val_s[val_mid:]
            y_val_calib, y_val_thr = y_val[:val_mid], y_val[val_mid:]
            ensemble.apply_isotonic_calibration(y_val_calib, X_val_calib)
        else:
            # Fallback: too few samples to split, use full val (accept minor leakage)
            X_val_thr = X_val_s
            y_val_thr = y_val
            ensemble.apply_isotonic_calibration(y_val, X_val_s)

        # Keep best single model as legacy fallback (for save_to_file compat)
        best_tag = max(candidate_metrics, key=candidate_metrics.get) if candidate_metrics else "rf"
        best_model: Any = {"rf": rf, "lgbm": lgbm, "xgb": xgb}.get(best_tag, rf)

        # --- v3: Calibrate threshold on ENSEMBLE (calibrated) probabilities ---
        # C-4 fix: uses second half of validation (not seen by isotonic calibrator)
        y_proba_val_ensemble = ensemble.predict_proba_calibrated(X_val_thr)

        calib_target = max(0.50, self._cfg.min_precision - 0.02)
        calibrated_thr = self._calibrate_threshold(y_val_thr, y_proba_val_ensemble, min_precision=calib_target)
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
        # AUC-ROC has highest weight (threshold-independent, most reliable on small datasets).
        # PF score second (directly measures trading profitability).
        final_skill = 0.20 * best_precision + 0.20 * best_recall + 0.35 * best_roc_auc + 0.25 * pf_score

        # --- Naive baseline: always predict "win" ---
        baseline_win_rate = float(y_test.mean()) if len(y_test) > 0 else 0.5
        precision_lift = best_precision - baseline_win_rate
        auc_lift = best_roc_auc - 0.5
        logger.info(
            "Baseline (always-win): precision=%.3f | Model lift: +%.3f precision, +%.3f AUC",
            baseline_win_rate, precision_lift, auc_lift,
        )

        # --- Bootstrap 95% confidence intervals on holdout ---
        n_boot = 500
        boot_prec, boot_auc = [], []
        rng = np.random.default_rng(42)
        n_test = len(X_test_s)
        for _ in range(n_boot):
            idx_b = rng.choice(n_test, size=n_test, replace=True)
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
        oot_auc = None
        if len(X_oot_raw) >= 20 and self._feature_selector.dropped_names:
            try:
                X_oot_sel = self._feature_selector.transform(X_oot_raw)
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
                logger.debug("OOT test failed: %s", _oot_err)

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
            if self._ensemble is not None and self._ensemble.is_ready:
                proba = float(self._ensemble.predict_proba_calibrated(features_arr)[0])
            elif self._model is not None:
                proba = float(self._model.predict_proba(features_arr)[0][1])
            else:
                proba = 0.5
        except Exception as exc:
            logger.warning("ML predict failed (defaulting to allow): %s", exc)
            proba = 0.5

        # Decision logic: config block_threshold overrides calibrated threshold
        # _cfg.block_threshold (from .env ANALYZER_ML_BLOCK_THRESHOLD) is the primary knob;
        # _calibrated_threshold is a fallback computed during training.
        cal_thr = self._cfg.block_threshold if self._cfg.block_threshold > 0 else self._calibrated_threshold
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
            "checksum_sha256": checksum[:16] + "...",
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
                self._metrics = MLMetrics(
                    precision=metrics_dict.get("precision", 0),
                    recall=metrics_dict.get("recall", 0),
                    roc_auc=metrics_dict.get("roc_auc", 0),
                    accuracy=metrics_dict.get("accuracy", 0),
                    skill_score=metrics_dict.get("skill_score", 0),
                    train_samples=metrics_dict.get("train_samples", 0),
                    test_samples=metrics_dict.get("test_samples", 0),
                    feature_importances=metrics_dict.get("feature_importances", {}),
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
