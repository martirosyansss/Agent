"""
Unit tests for MLPredictor — verifies correctness of all critical paths.

Run: python -m pytest tests/test_ml_predictor.py -v
"""

import pytest
import time
from unittest.mock import MagicMock, patch
from analyzer.ml_predictor import (
    MLPredictor, MLConfig, MLMetrics, MLPrediction,
    FEATURE_NAMES, N_FEATURES, REGIME_ENCODING, STRATEGY_ENCODING,
)
from core.models import StrategyTrade


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _make_trade(**overrides) -> StrategyTrade:
    """Create a StrategyTrade with sensible defaults."""
    defaults = dict(
        trade_id="test_001", symbol="BTCUSDT", strategy_name="ema_crossover_rsi",
        market_regime="trending_up", entry_price=60000.0, exit_price=61000.0,
        quantity=0.01, pnl_usd=10.0, pnl_pct=1.67, is_win=True,
        confidence=0.75, hour_of_day=14, day_of_week=2,
        rsi_at_entry=55.0, adx_at_entry=30.0, volume_ratio_at_entry=1.5,
        news_sentiment=0.2, fear_greed_index=65, trend_alignment=0.7,
        ema_9_at_entry=60500.0, ema_21_at_entry=60000.0,
        bb_bandwidth_at_entry=0.05, macd_histogram_at_entry=100.0,
        atr_at_entry=800.0, timestamp_open="2025-01-01T10:00:00Z",
        timestamp_close="2025-01-01T12:00:00Z",
    )
    defaults.update(overrides)
    return StrategyTrade(**defaults)


@pytest.fixture
def predictor():
    return MLPredictor(MLConfig(min_trades=5))


@pytest.fixture
def sample_trade():
    return _make_trade()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Feature Extraction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestExtractFeatures:
    def test_returns_correct_length(self, predictor, sample_trade):
        features = predictor.extract_features(sample_trade)
        assert len(features) == N_FEATURES, f"Expected {N_FEATURES}, got {len(features)}"

    def test_all_features_are_float(self, predictor, sample_trade):
        features = predictor.extract_features(sample_trade)
        for i, f in enumerate(features):
            assert isinstance(f, (int, float)), f"Feature {i} ({FEATURE_NAMES[i]}) is {type(f)}"

    def test_no_nan_in_features(self, predictor, sample_trade):
        import math
        features = predictor.extract_features(sample_trade)
        for i, f in enumerate(features):
            assert not math.isnan(f), f"Feature {i} ({FEATURE_NAMES[i]}) is NaN"
            assert not math.isinf(f), f"Feature {i} ({FEATURE_NAMES[i]}) is Inf"

    def test_rsi_is_first_feature(self, predictor, sample_trade):
        features = predictor.extract_features(sample_trade)
        assert features[0] == sample_trade.rsi_at_entry

    def test_regime_encoding(self, predictor, sample_trade):
        features = predictor.extract_features(sample_trade)
        assert features[9] == float(REGIME_ENCODING["trending_up"])

    def test_strategy_encoding(self, predictor, sample_trade):
        features = predictor.extract_features(sample_trade)
        assert features[10] == float(STRATEGY_ENCODING["ema_crossover_rsi"])

    def test_fear_greed_normalized(self, predictor, sample_trade):
        features = predictor.extract_features(sample_trade)
        assert features[16] == 0.65  # 65 / 100

    def test_no_forward_looking_bias(self, predictor):
        """N-3: extract_features now delegates to extract_features_batch.

        Batch method uses index-based context: trade at position i sees
        trades 0..i-1. So only trades passed in previous_trades before
        the current trade are used. Callers are responsible for not including
        future trades in the previous_trades list.
        """
        old_trade = _make_trade(
            trade_id="old", timestamp_open="2025-01-01T08:00:00Z",
            timestamp_close="2025-01-01T09:00:00Z", is_win=True,
        )
        current = _make_trade(
            trade_id="current", timestamp_open="2025-01-01T10:00:00Z",
        )
        # Only pass old_trade (correctly excluding future data)
        features = predictor.extract_features(current, [old_trade])
        # Win rate = 1.0 (only old_trade counted, which is a win)
        assert features[11] == 1.0  # recent_win_rate

    def test_zero_entry_price_safe(self, predictor):
        trade = _make_trade(entry_price=0.0)
        features = predictor.extract_features(trade)
        assert len(features) == N_FEATURES  # no crash


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Predict
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPredict:
    def test_predict_not_ready_returns_allow(self, predictor):
        result = predictor.predict([0.0] * N_FEATURES)
        assert result.decision == "allow"
        assert result.probability == 0.5

    def test_predict_off_mode_returns_allow(self, predictor):
        predictor._model = MagicMock()
        predictor._rollout_mode = "off"
        result = predictor.predict([0.0] * N_FEATURES)
        assert result.decision == "allow"

    def test_predict_wrong_feature_count_returns_allow(self, predictor):
        predictor._model = MagicMock()
        predictor._rollout_mode = "block"
        result = predictor.predict([0.0] * 15)  # Wrong count
        assert result.decision == "allow"
        assert result.probability == 0.5

    def test_predict_shadow_mode_always_allows(self, predictor):
        """In shadow mode, even 'block' signals should return 'allow'."""
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = [[0.9, 0.1]]  # Low probability
        predictor._model = mock_model
        predictor._rollout_mode = "shadow"
        result = predictor.predict([0.0] * N_FEATURES)
        assert result.decision == "allow"

    def test_predict_block_mode_blocks(self, predictor):
        """In block mode, low prob should return 'block'."""
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = [[0.7, 0.3]]  # 0.3 < 0.55
        predictor._model = mock_model
        predictor._scaler = None
        predictor._rollout_mode = "block"
        result = predictor.predict([0.0] * N_FEATURES)
        assert result.decision == "block"

    def test_predict_block_mode_allows_high_prob(self, predictor):
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = [[0.2, 0.8]]  # 0.8 > 0.65
        predictor._model = mock_model
        predictor._scaler = None
        predictor._rollout_mode = "block"
        result = predictor.predict([0.0] * N_FEATURES)
        assert result.decision == "allow"

    def test_predict_reduce_zone(self, predictor):
        """Probability between block (thr*0.85) and reduce (thr) should return 'reduce'.
        With default threshold=0.5: block < 0.425, reduce 0.425..0.5, allow > 0.5.
        """
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = [[0.55, 0.45]]  # 0.425 < 0.45 < 0.5
        predictor._model = mock_model
        predictor._scaler = None
        predictor._rollout_mode = "block"
        result = predictor.predict([0.0] * N_FEATURES)
        assert result.decision == "reduce"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Profit Factor Score
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestProfitFactorScore:
    def test_perfect_predictions(self):
        y_pred = [1, 1, 0, 0]
        pnl = [10.0, 5.0, -3.0, -2.0]
        score = MLPredictor._compute_profit_factor_score(y_pred, pnl)
        assert score == 1.0  # All pred=1 are profitable

    def test_all_losing_predictions(self):
        y_pred = [1, 1, 1]
        pnl = [-5.0, -3.0, -2.0]
        score = MLPredictor._compute_profit_factor_score(y_pred, pnl)
        assert score == 0.0

    def test_empty_pnl(self):
        score = MLPredictor._compute_profit_factor_score([1, 0], [])
        assert score == 0.5

    def test_none_pnl(self):
        score = MLPredictor._compute_profit_factor_score([1, 0], None)
        assert score == 0.5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# I/O
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestIO:
    def test_save_not_ready_returns_false(self, predictor, tmp_path):
        assert predictor.save_to_file(tmp_path / "model.pkl") is False

    def test_load_nonexistent_returns_false(self, predictor, tmp_path):
        assert predictor.load_from_file(tmp_path / "nonexistent.pkl") is False

    def test_save_load_roundtrip(self, predictor, tmp_path):
        """Save a model, load it back, verify state is preserved."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        # Train a tiny real model
        X = np.random.rand(20, N_FEATURES)
        y = np.array([1, 0] * 10)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X_s, y)

        predictor._model = model
        predictor._scaler = scaler
        predictor._model_version = "test_v1"
        predictor._metrics = MLMetrics(
            precision=0.7, recall=0.6, roc_auc=0.75,
            accuracy=0.68, skill_score=0.72,
            train_samples=400, test_samples=100,
        )

        path = tmp_path / "model.pkl"
        assert predictor.save_to_file(path) is True
        assert path.exists()

        new_predictor = MLPredictor()
        assert new_predictor.load_from_file(path) is True
        assert new_predictor._model_version == "test_v1"
        assert new_predictor._metrics.precision == 0.7
        assert new_predictor._metrics.skill_score == 0.72
        assert new_predictor.is_ready is True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Config & Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestConfig:
    def test_n_features_matches_feature_names(self):
        assert N_FEATURES == len(FEATURE_NAMES)

    def test_block_threshold_below_reduce(self):
        cfg = MLConfig()
        assert cfg.block_threshold < cfg.reduce_threshold

    def test_rollout_modes(self, predictor):
        for mode in ("off", "shadow", "block"):
            predictor.rollout_mode = mode
            assert predictor.rollout_mode == mode
        predictor.rollout_mode = "invalid"
        assert predictor.rollout_mode == "block"  # unchanged

    def test_needs_retrain_initial(self, predictor):
        assert predictor.needs_retrain() is True

    def test_needs_retrain_fresh(self, predictor):
        predictor._last_train_ts = int(time.time() * 1000)
        assert predictor.needs_retrain() is False
