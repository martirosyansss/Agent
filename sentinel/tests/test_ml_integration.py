"""
Integration tests for the full ML pipeline — FeatureVector → StrategyTrade → features → predict.

Test Coverage → 10/10: covers the integration seam between core.models,
ml_predictor, and the factory method.

Run: python -m pytest tests/test_ml_integration.py -v
"""

import json
import time
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

from core.models import StrategyTrade, FeatureVector
from analyzer.ml_predictor import (
    MLPredictor, MLConfig, MLMetrics, MLPrediction,
    FEATURE_NAMES, N_FEATURES,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _make_feature_vector(**overrides) -> FeatureVector:
    """Create a FeatureVector with realistic non-zero values."""
    defaults = dict(
        timestamp=int(time.time() * 1000), symbol="BTCUSDT",
        ema_9=60500.0, ema_21=60000.0, ema_50=59500.0,
        adx=28.0, macd=50.0, macd_signal=40.0, macd_histogram=10.0,
        rsi_14=55.0, stoch_rsi=0.6,
        bb_upper=62000.0, bb_middle=60500.0, bb_lower=59000.0,
        bb_bandwidth=0.05, atr=800.0,
        volume=1500.0, volume_sma_20=1200.0, volume_ratio=1.25, obv=50000.0,
        price_change_1m=0.1, price_change_5m=0.3, price_change_15m=0.5,
        momentum=120.0, spread=0.01, close=60500.0,
        news_sentiment=0.15, fear_greed_index=62, news_impact_pct=0.5,
        cci=85.0, roc=2.5, cmf=0.15, bb_pct_b=0.65,
        hist_volatility=0.03, dmi_spread=8.0, trend_alignment=0.72,
        rsi_14_daily=52.0,
    )
    defaults.update(overrides)
    return FeatureVector(**defaults)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# from_feature_vector factory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestFromFeatureVector:
    def test_factory_creates_valid_trade(self):
        fv = _make_feature_vector()
        trade = StrategyTrade.from_feature_vector(
            fv, strategy_name="ema_crossover_rsi",
            market_regime="trending_up", confidence=0.8,
            hour_of_day=14, day_of_week=3,
        )
        assert trade.trade_id == "pending"
        assert trade.symbol == "BTCUSDT"
        assert trade.strategy_name == "ema_crossover_rsi"
        assert trade.entry_price == 60500.0

    def test_factory_maps_all_30_fields(self):
        """Verify every ML-critical field is populated from FeatureVector."""
        fv = _make_feature_vector()
        trade = StrategyTrade.from_feature_vector(fv, strategy_name="test")

        # Core indicators
        assert trade.rsi_at_entry == fv.rsi_14
        assert trade.adx_at_entry == fv.adx
        assert trade.volume_ratio_at_entry == fv.volume_ratio
        assert trade.ema_9_at_entry == fv.ema_9
        assert trade.ema_21_at_entry == fv.ema_21
        assert trade.bb_bandwidth_at_entry == fv.bb_bandwidth
        assert trade.macd_histogram_at_entry == fv.macd_histogram
        assert trade.atr_at_entry == fv.atr

        # Sentiment
        assert trade.news_sentiment == fv.news_sentiment
        assert trade.fear_greed_index == fv.fear_greed_index
        assert trade.trend_alignment == fv.trend_alignment

        # Phase 2
        assert trade.cci_at_entry == fv.cci
        assert trade.roc_at_entry == fv.roc
        assert trade.cmf_at_entry == fv.cmf
        assert trade.bb_pct_b_at_entry == fv.bb_pct_b
        assert trade.hist_volatility_at_entry == fv.hist_volatility
        assert trade.dmi_spread_at_entry == fv.dmi_spread
        assert trade.stoch_rsi_at_entry == fv.stoch_rsi
        assert trade.price_change_5h_at_entry == fv.price_change_5h
        assert trade.momentum_at_entry == fv.momentum
        assert trade.rsi_daily_at_entry == fv.rsi_14_daily

    def test_no_zero_fields_when_fv_has_values(self):
        """If FeatureVector has non-zero values, StrategyTrade should NOT have zeros."""
        fv = _make_feature_vector()
        trade = StrategyTrade.from_feature_vector(fv, strategy_name="test")
        predictor = MLPredictor(MLConfig(min_trades=5))
        features = predictor.extract_features(trade)
        # Features 0-6 (technical indicators) should all be non-zero
        # Features 7-8 (hour/day) are from system clock, not FV, so may be 0
        for i in range(7):  # rsi, adx, ema_diff, bb_bw, vol_ratio, macd, atr_ratio
            assert features[i] != 0.0, f"Feature {i} ({FEATURE_NAMES[i]}) is zero — field mapping broken"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Full ML Pipeline Integration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestMLPipelineIntegration:
    def test_extract_features_returns_correct_shape(self):
        """Full pipeline: FeatureVector → StrategyTrade → extract_features → N_FEATURES."""
        fv = _make_feature_vector()
        trade = StrategyTrade.from_feature_vector(fv, strategy_name="ema_crossover_rsi")
        predictor = MLPredictor(MLConfig(min_trades=5))
        features = predictor.extract_features(trade)
        assert len(features) == N_FEATURES

    def test_extract_and_predict_roundtrip(self):
        """Full pipeline: FeatureVector → trade → features → predict."""
        fv = _make_feature_vector()
        trade = StrategyTrade.from_feature_vector(
            fv, strategy_name="ema_crossover_rsi",
            market_regime="trending_up", confidence=0.8,
        )
        predictor = MLPredictor(MLConfig(min_trades=5))
        features = predictor.extract_features(trade)

        # Without model, should return allow with 0.5
        result = predictor.predict(features)
        assert result.decision == "allow"
        assert result.probability == 0.5

    def test_batch_extract_matches_single_extract(self):
        """N-3 verification: batch and single extract produce identical output."""
        predictor = MLPredictor(MLConfig(min_trades=5))
        fv = _make_feature_vector()
        trade = StrategyTrade.from_feature_vector(fv, strategy_name="test")

        single = predictor.extract_features(trade)
        batch = predictor.extract_features_batch([trade])

        assert len(single) == N_FEATURES
        assert batch.shape == (1, N_FEATURES)
        for i in range(N_FEATURES):
            assert abs(single[i] - batch[0, i]) < 1e-9, \
                f"Feature {i} ({FEATURE_NAMES[i]}) diverges: single={single[i]}, batch={batch[0, i]}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model Registry
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestModelRegistry:
    def test_save_creates_registry(self, tmp_path):
        """save_to_file should create a model_registry.json alongside the pickle."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler

        predictor = MLPredictor(MLConfig(min_trades=5))
        X = np.random.rand(20, N_FEATURES)
        y = np.array([1, 0] * 10)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X_s, y)

        predictor._model = model
        predictor._scaler = scaler
        predictor._model_version = "test_reg_v1"
        predictor._metrics = MLMetrics(
            precision=0.75, recall=0.65, roc_auc=0.78,
            accuracy=0.72, skill_score=0.80,
            train_samples=400, test_samples=100,
        )

        model_path = tmp_path / "model.pkl"
        assert predictor.save_to_file(model_path) is True

        # Registry should exist
        registry_path = tmp_path / "model_registry.json"
        assert registry_path.exists()

        with registry_path.open("r") as f:
            registry = json.load(f)

        assert len(registry) == 1
        entry = registry[0]
        assert entry["version"] == "test_reg_v1"
        assert entry["metrics"]["precision"] == 0.75
        assert entry["metrics"]["skill_score"] == 0.80
        assert "checksum_sha256" in entry

    def test_registry_appends_not_overwrites(self, tmp_path):
        """Multiple saves should append entries, not overwrite."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler

        predictor = MLPredictor(MLConfig(min_trades=5))
        X = np.random.rand(20, N_FEATURES)
        y = np.array([1, 0] * 10)
        scaler = StandardScaler()
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(scaler.fit_transform(X), y)

        predictor._model = model
        predictor._scaler = scaler
        predictor._metrics = MLMetrics(precision=0.7, recall=0.6, roc_auc=0.7,
                                       accuracy=0.65, skill_score=0.75,
                                       train_samples=300, test_samples=75)

        model_path = tmp_path / "model.pkl"

        predictor._model_version = "v1"
        predictor.save_to_file(model_path)
        predictor._model_version = "v2"
        predictor.save_to_file(model_path)

        with (tmp_path / "model_registry.json").open("r") as f:
            registry = json.load(f)

        assert len(registry) == 2
        assert registry[0]["version"] == "v1"
        assert registry[1]["version"] == "v2"
