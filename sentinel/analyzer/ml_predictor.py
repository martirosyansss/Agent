"""
Trade Analyzer Level 3 — ML Predictor (Shadow Mode).

LightGBM / sklearn RandomForest для фильтрации торговых сигналов.
ML ТОЛЬКО фильтрует (block), НИКОГДА не инициирует сделки.

Rollout режимы: off → shadow → block
- shadow: логирует предсказания, не блокирует
- block: блокирует сигналы с low prediction

15 features: rsi_14, adx, ema_9_vs_21, bb_bandwidth, volume_ratio,
  macd_hist, atr_ratio, hour_of_day, day_of_week, market_regime_encoded,
  strategy_encoded, recent_win_rate_10, hours_since_last_trade,
  daily_pnl_so_far, consecutive_losses

Skill score = 0.40*precision + 0.25*recall + 0.25*roc_auc + 0.10*normalized_pnl
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from core.models import StrategyTrade

logger = logging.getLogger(__name__)

# Encoding maps
REGIME_ENCODING = {
    "trending_up": 0, "trending_down": 1, "sideways": 2,
    "volatile": 3, "unknown": 4,
}
STRATEGY_ENCODING = {
    "ema_crossover_rsi": 0, "grid_trading": 1, "mean_reversion": 2,
    "bollinger_breakout": 3, "dca_bot": 4, "macd_divergence": 5,
}

FEATURE_NAMES = [
    "rsi_14", "adx", "ema_9_vs_21", "bb_bandwidth", "volume_ratio",
    "macd_histogram", "atr_ratio", "hour_of_day", "day_of_week",
    "market_regime_encoded", "strategy_encoded", "recent_win_rate_10",
    "hours_since_last_trade", "daily_pnl_so_far", "consecutive_losses",
    "news_sentiment", "fear_greed_normalized", "confidence_at_entry",
    "regime_bias", "adx_normalized",
]


@dataclass
class MLConfig:
    n_estimators: int = 200
    max_depth: int = 8
    learning_rate: float = 0.05
    min_child_samples: int = 20
    min_samples_split: int = 15
    max_features: str = "sqrt"
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    block_threshold: float = 0.60
    min_precision: float = 0.65
    min_recall: float = 0.58
    min_roc_auc: float = 0.65
    min_skill_score: float = 0.72
    retrain_days: int = 30
    min_trades: int = 500
    test_window_days: int = 60
    cv_splits: int = 5


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


@dataclass
class MLPrediction:
    """Предсказание модели для сигнала."""
    probability: float = 0.5
    decision: str = "allow"  # allow, reduce, block
    model_version: str = ""
    rollout_mode: str = "shadow"  # off, shadow, block


class MLPredictor:
    """Level 3 Trade Analyzer — ML фильтрация сигналов."""

    def __init__(self, config: MLConfig | None = None) -> None:
        self._cfg = config or MLConfig()
        self._model: Any = None
        self._scaler: Any = None
        self._model_version: str = ""
        self._metrics: Optional[MLMetrics] = None
        self._rollout_mode: str = "off"  # off, shadow, block
        self._last_train_ts: int = 0

    @property
    def is_ready(self) -> bool:
        return self._model is not None

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

    def _history_context(
        self,
        trade: StrategyTrade,
        previous_trades: list[StrategyTrade],
    ) -> tuple[float, float, float, float]:
        if not previous_trades:
            return 0.5, 0.0, 0.0, 0.0

        # Ensure chronological order to prevent data leakage
        sorted_trades = sorted(
            previous_trades,
            key=lambda t: t.timestamp_close or t.timestamp_open or "",
        )

        # Filter: only trades that closed BEFORE current trade opened
        current_open = trade.timestamp_open or ""
        if current_open:
            sorted_trades = [t for t in sorted_trades if (t.timestamp_close or "") < current_open]

        if not sorted_trades:
            return 0.5, 0.0, 0.0, 0.0

        recent_trades = sorted_trades[-10:]
        recent_win_rate = sum(1 for item in recent_trades if item.is_win) / len(recent_trades)

        current_open_dt = self._parse_trade_timestamp(trade.timestamp_open)
        last_close_dt = self._parse_trade_timestamp(sorted_trades[-1].timestamp_close)
        hours_since_last_trade = 0.0
        if current_open_dt and last_close_dt:
            hours_since_last_trade = max(
                (current_open_dt - last_close_dt).total_seconds() / 3600.0,
                0.0,
            )

        daily_pnl_so_far = 0.0
        if current_open_dt:
            current_day = current_open_dt.date()
            for item in sorted_trades:
                item_dt = self._parse_trade_timestamp(item.timestamp_close) or self._parse_trade_timestamp(item.timestamp_open)
                if item_dt and item_dt.date() == current_day:
                    daily_pnl_so_far += item.pnl_usd

        consecutive_losses = 0.0
        for item in reversed(sorted_trades):
            if item.is_win:
                break
            consecutive_losses += 1.0

        return recent_win_rate, hours_since_last_trade, daily_pnl_so_far, consecutive_losses

    def extract_features(
        self,
        trade: StrategyTrade,
        previous_trades: Optional[list[StrategyTrade]] = None,
    ) -> list[float]:
        """Extract 20 features using ONLY pre-trade data (no forward-looking bias).

        All features are known BEFORE entering the trade:
        - Technical indicators at entry (rsi, adx, volume_ratio)
        - Strategy metadata (confidence, regime, strategy_name)
        - Time features (hour, day)
        - Historical performance (win_rate, consecutive_losses, daily_pnl)
        - Sentiment data (news, fear_greed)
        """
        previous_trades = previous_trades or []

        entry_price = max(abs(trade.entry_price), 1e-9)
        regime_bias = self._regime_bias(trade.market_regime)

        # Raw EMA difference as trend signal
        ema_9_vs_21 = (trade.ema_9_at_entry - trade.ema_21_at_entry) / entry_price if entry_price > 0 else 0.0

        # Raw BB bandwidth (from trade attributes)
        bb_bandwidth = trade.bb_bandwidth_at_entry

        # Raw MACD histogram
        macd_histogram = trade.macd_histogram_at_entry

        # ATR ratio (ATR / price)
        atr_ratio = (trade.atr_at_entry / entry_price) if entry_price > 0 else 0.0

        # ADX normalized to [0, 1]
        adx_normalized = min(trade.adx_at_entry / 50.0, 1.0)

        recent_win_rate, hours_since_last_trade, daily_pnl_so_far, consecutive_losses = (
            self._history_context(trade, previous_trades)
        )

        return [
            trade.rsi_at_entry,                                          # 0: RSI at entry
            trade.adx_at_entry,                                          # 1: ADX at entry
            ema_9_vs_21,                                                  # 2: EMA9-EMA21 / price
            bb_bandwidth,                                                 # 3: BB bandwidth (raw)
            trade.volume_ratio_at_entry,                                  # 4: volume confirmation
            macd_histogram,                                               # 5: MACD histogram (raw)
            atr_ratio,                                                    # 6: ATR / price
            float(trade.hour_of_day),                                     # 7: hour of day
            float(trade.day_of_week),                                     # 8: day of week
            float(REGIME_ENCODING.get(trade.market_regime, 4)),           # 9: regime encoded
            float(STRATEGY_ENCODING.get(trade.strategy_name, 0)),         # 10: strategy encoded
            recent_win_rate,                                              # 11: recent win rate
            hours_since_last_trade,                                       # 12: hours since last
            daily_pnl_so_far,                                             # 13: daily PnL before
            consecutive_losses,                                           # 14: loss streak
            getattr(trade, 'news_sentiment', 0.0),                        # 15: news sentiment
            getattr(trade, 'fear_greed_index', 50) / 100.0,              # 16: fear/greed norm
            trade.confidence,                                             # 17: strategy confidence
            regime_bias,                                                  # 18: regime bias (-1/0/1)
            adx_normalized,                                               # 19: ADX normalized
        ]

    def train(self, trades: list[StrategyTrade]) -> Optional[MLMetrics]:
        """Обучить модель на исторических сделках.

        TimeSeriesSplit CV + StandardScaler + regularized RF.
        Returns MLMetrics or None if insufficient data / metrics below threshold.
        """
        if len(trades) < self._cfg.min_trades:
            logger.warning("ML train: insufficient trades (%d < %d)", len(trades), self._cfg.min_trades)
            return None

        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            import numpy as np
        except ImportError:
            logger.error("scikit-learn not installed. pip install scikit-learn")
            return None

        # Prepare data
        X_raw = [
            self.extract_features(trade, previous_trades=trades[:idx])
            for idx, trade in enumerate(trades)
        ]
        y = [1 if t.is_win else 0 for t in trades]

        X = np.array(X_raw, dtype=np.float64)
        y_arr = np.array(y)

        # Replace NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Time Series Cross-Validation ---
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

            fold_model = RandomForestClassifier(
                n_estimators=self._cfg.n_estimators,
                max_depth=self._cfg.max_depth,
                min_samples_leaf=self._cfg.min_child_samples,
                min_samples_split=self._cfg.min_samples_split,
                max_features=self._cfg.max_features,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
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

        # Log CV results
        avg_prec = sum(cv_precisions) / len(cv_precisions)
        avg_rec = sum(cv_recalls) / len(cv_recalls)
        avg_auc = sum(cv_aucs) / len(cv_aucs)
        logger.info("ML CV results: prec=%.3f rec=%.3f auc=%.3f (over %d folds)",
                     avg_prec, avg_rec, avg_auc, len(cv_precisions))

        # Overfitting detection: train final model on 80%, evaluate on 20%
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y_arr[:split_idx], y_arr[split_idx:]

        if len(X_train) < 50 or len(X_test) < 20:
            return None

        self._scaler = StandardScaler()
        X_train_s = self._scaler.fit_transform(X_train)
        X_test_s = self._scaler.transform(X_test)

        # Train final model
        model = RandomForestClassifier(
            n_estimators=self._cfg.n_estimators,
            max_depth=self._cfg.max_depth,
            min_samples_leaf=self._cfg.min_child_samples,
            min_samples_split=self._cfg.min_samples_split,
            max_features=self._cfg.max_features,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train_s, y_train)

        # Evaluate on hold-out test
        y_pred = model.predict(X_test_s)
        y_proba = model.predict_proba(X_test_s)[:, 1] if len(set(y_train)) > 1 else [0.5] * len(y_test)

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        try:
            roc_auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            roc_auc = 0.5
        accuracy = accuracy_score(y_test, y_pred)

        # Overfitting check — reject model if gap > 5%
        y_train_pred = model.predict(X_train_s)
        train_prec = precision_score(y_train, y_train_pred, zero_division=0)
        overfit_gap = train_prec - precision
        if overfit_gap > 0.05:
            logger.warning("ML OVERFITTING detected: train_prec=%.3f test_prec=%.3f gap=%.3f — model REJECTED",
                           train_prec, precision, overfit_gap)
            self._metrics = MLMetrics(
                precision=precision, recall=recall, roc_auc=roc_auc,
                accuracy=accuracy, skill_score=0.0,
                train_samples=len(X_train), test_samples=len(X_test),
                feature_importances={},
            )
            return self._metrics

        # Skill score
        skill = 0.40 * precision + 0.25 * recall + 0.25 * roc_auc + 0.10 * accuracy

        # Feature importances
        importances = {}
        for i, name in enumerate(FEATURE_NAMES):
            if i < len(model.feature_importances_):
                importances[name] = float(model.feature_importances_[i])

        # Log top features
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        top5 = ", ".join(f"{n}={v:.3f}" for n, v in sorted_imp[:5])
        logger.info("ML top features: %s", top5)

        metrics = MLMetrics(
            precision=precision,
            recall=recall,
            roc_auc=roc_auc,
            accuracy=accuracy,
            skill_score=skill,
            train_samples=len(X_train),
            test_samples=len(X_test),
            feature_importances=importances,
        )

        # Gate check
        if (precision < self._cfg.min_precision or
                recall < self._cfg.min_recall or
                roc_auc < self._cfg.min_roc_auc or
                skill < self._cfg.min_skill_score):
            logger.warning("ML metrics below threshold: skill=%.3f prec=%.3f rec=%.3f auc=%.3f",
                           skill, precision, recall, roc_auc)
            self._metrics = metrics
            return metrics

        self._model = model
        self._model_version = f"rf_v{int(time.time())}"
        self._metrics = metrics
        self._last_train_ts = int(time.time() * 1000)
        logger.info("ML model trained: skill=%.3f prec=%.3f rec=%.3f auc=%.3f",
                     skill, precision, recall, roc_auc)
        return metrics

    def predict(
        self,
        trade_features: list[float],
    ) -> MLPrediction:
        """Предсказание для нового сигнала.

        Returns MLPrediction с решением allow/reduce/block.
        """
        if not self.is_ready or self._rollout_mode == "off":
            return MLPrediction(
                probability=0.5,
                decision="allow",
                model_version="",
                rollout_mode=self._rollout_mode,
            )

        try:
            import numpy as np
            features_arr = np.array([trade_features], dtype=np.float64)
            features_arr = np.nan_to_num(features_arr, nan=0.0, posinf=0.0, neginf=0.0)
            if self._scaler is not None:
                features_arr = self._scaler.transform(features_arr)
            proba = self._model.predict_proba(features_arr)[0][1]
        except Exception:
            proba = 0.5

        # Decision logic
        cfg = self._cfg
        if proba < cfg.block_threshold:
            decision = "block"
        elif proba < cfg.min_precision:
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
        """Нужно ли переобучение."""
        if self._last_train_ts == 0:
            return True
        days_since = (time.time() * 1000 - self._last_train_ts) / (86400 * 1000)
        return days_since >= self._cfg.retrain_days
