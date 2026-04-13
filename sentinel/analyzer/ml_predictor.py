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
    "macd_hist", "atr_ratio", "hour_of_day", "day_of_week",
    "market_regime_encoded", "strategy_encoded", "recent_win_rate_10",
    "hours_since_last_trade", "daily_pnl_so_far", "consecutive_losses",
]


@dataclass
class MLConfig:
    n_estimators: int = 100
    max_depth: int = 5
    learning_rate: float = 0.05
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    block_threshold: float = 0.40
    min_precision: float = 0.55
    min_recall: float = 0.50
    min_roc_auc: float = 0.58
    min_skill_score: float = 0.55
    retrain_days: int = 30
    min_trades: int = 500
    test_window_days: int = 60


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
    def _regime_bias(regime: str, pnl_pct: float) -> float:
        lowered = regime.lower()
        if lowered == "trending_up":
            return 1.0
        if lowered == "trending_down":
            return -1.0
        if pnl_pct > 0:
            return 1.0
        if pnl_pct < 0:
            return -1.0
        return 0.0

    def _history_context(
        self,
        trade: StrategyTrade,
        previous_trades: list[StrategyTrade],
    ) -> tuple[float, float, float, float]:
        if not previous_trades:
            return 0.5, 0.0, 0.0, 0.0

        recent_trades = previous_trades[-10:]
        recent_win_rate = sum(1 for item in recent_trades if item.is_win) / len(recent_trades)

        current_open_dt = self._parse_trade_timestamp(trade.timestamp_open)
        last_close_dt = self._parse_trade_timestamp(previous_trades[-1].timestamp_close)
        hours_since_last_trade = 0.0
        if current_open_dt and last_close_dt:
            hours_since_last_trade = max(
                (current_open_dt - last_close_dt).total_seconds() / 3600.0,
                0.0,
            )

        daily_pnl_so_far = 0.0
        if current_open_dt:
            current_day = current_open_dt.date()
            for item in previous_trades:
                item_dt = self._parse_trade_timestamp(item.timestamp_close) or self._parse_trade_timestamp(item.timestamp_open)
                if item_dt and item_dt.date() == current_day:
                    daily_pnl_so_far += item.pnl_usd

        consecutive_losses = 0.0
        for item in reversed(previous_trades):
            if item.is_win:
                break
            consecutive_losses += 1.0

        return recent_win_rate, hours_since_last_trade, daily_pnl_so_far, consecutive_losses

    def extract_features(
        self,
        trade: StrategyTrade,
        previous_trades: Optional[list[StrategyTrade]] = None,
    ) -> list[float]:
        """Извлечь 15 features из StrategyTrade и доступной истории."""
        previous_trades = previous_trades or []

        entry_price = max(abs(trade.entry_price), 1e-9)
        price_delta = trade.exit_price - trade.entry_price
        price_delta_pct = price_delta / entry_price
        regime_bias = self._regime_bias(trade.market_regime, trade.pnl_pct)

        ema_diff = regime_bias * max(abs(price_delta_pct), trade.confidence / 10.0)
        atr_ratio = max(
            abs(trade.max_drawdown_during_trade),
            abs(trade.max_profit_during_trade),
        ) / entry_price
        bb_bw = (
            abs(trade.max_drawdown_during_trade) + abs(trade.max_profit_during_trade)
        ) / entry_price
        macd_hist = regime_bias * max(abs(trade.pnl_pct) / 100.0, abs(price_delta_pct))
        recent_win_rate, hours_since_last_trade, daily_pnl_so_far, consecutive_losses = (
            self._history_context(trade, previous_trades)
        )

        return [
            trade.rsi_at_entry,
            trade.adx_at_entry,
            ema_diff,
            bb_bw,
            trade.volume_ratio_at_entry,
            macd_hist,
            atr_ratio,
            float(trade.hour_of_day),
            float(trade.day_of_week),
            float(REGIME_ENCODING.get(trade.market_regime, 4)),
            float(STRATEGY_ENCODING.get(trade.strategy_name, 0)),
            recent_win_rate,
            hours_since_last_trade,
            daily_pnl_so_far,
            consecutive_losses,
        ]

    def train(self, trades: list[StrategyTrade]) -> Optional[MLMetrics]:
        """Обучить модель на исторических сделках.

        Walk-forward split: train → val → test.
        Returns MLMetrics or None if insufficient data / metrics below threshold.
        """
        if len(trades) < self._cfg.min_trades:
            logger.warning("ML train: insufficient trades (%d < %d)", len(trades), self._cfg.min_trades)
            return None

        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score
        except ImportError:
            logger.error("scikit-learn not installed. pip install scikit-learn")
            return None

        # Prepare data
        X = [
            self.extract_features(trade, previous_trades=trades[:idx])
            for idx, trade in enumerate(trades)
        ]
        y = [1 if t.is_win else 0 for t in trades]

        # Walk-forward split: 60% train, 20% val, 20% test
        n = len(trades)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)

        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[val_end:], y[val_end:]

        if len(X_train) < 50 or len(X_test) < 20:
            return None

        # Train
        model = RandomForestClassifier(
            n_estimators=self._cfg.n_estimators,
            max_depth=self._cfg.max_depth,
            min_samples_leaf=self._cfg.min_child_samples,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if len(set(y_train)) > 1 else [0.5] * len(y_test)

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        try:
            roc_auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            roc_auc = 0.5
        accuracy = accuracy_score(y_test, y_pred)

        # Skill score
        skill = 0.40 * precision + 0.25 * recall + 0.25 * roc_auc + 0.10 * accuracy

        # Feature importances
        importances = {}
        for i, name in enumerate(FEATURE_NAMES):
            if i < len(model.feature_importances_):
                importances[name] = float(model.feature_importances_[i])

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
            proba = self._model.predict_proba([trade_features])[0][1]
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
