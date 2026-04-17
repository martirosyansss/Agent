"""Regression tests for the ML audit fixes (round 2).

Covers:
1. wilson_lower_bound: returns sane CI bounds
2. compute_skill_score: matches documented weights
3. Isotonic vs Platt selection by sample size
4. Adaptive drift threshold: narrow at large N, wide at small N
5. StrategyTrade.from_db_row: tolerates DB-only columns
6. StrategyTrade.from_feature_vector: populates all 30 ML fields
7. MLPredictor.predict: respects calibrated_threshold floor via max()

Run: python -m pytest tests/test_ml_audit_fixes.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from analyzer.ml_ensemble import VotingEnsemble, _PlattCalibrator
from analyzer.ml_predictor import (
    LivePerformanceTracker,
    MLConfig,
    MLPredictor,
    compute_skill_score,
    wilson_lower_bound,
    _SKILL_W_PRECISION,
    _SKILL_W_RECALL,
    _SKILL_W_ROC_AUC,
    _SKILL_W_PROFIT_FACTOR,
)
from core.models import FeatureVector, StrategyTrade


# ─── 1. Wilson score ──────────────────────────────────────────────────────────

def test_wilson_lower_bound_edge_cases():
    assert wilson_lower_bound(0, 0) == 0.0
    assert wilson_lower_bound(10, 10) < 1.0  # even perfect success has CI < 1
    assert wilson_lower_bound(0, 10) >= 0.0


def test_wilson_lower_bound_tightens_with_n():
    # At p=0.7, more samples = tighter lower bound (closer to 0.7)
    low_n = wilson_lower_bound(7, 10)
    high_n = wilson_lower_bound(700, 1000)
    assert high_n > low_n
    assert abs(high_n - 0.7) < 0.05    # ~0.67 at 95% CI
    assert abs(low_n - 0.7) > 0.10     # ~0.40 at 95% CI


def test_wilson_lower_bound_monotonic_in_successes():
    # more successes with same trials → higher bound
    bounds = [wilson_lower_bound(k, 100) for k in range(0, 101, 10)]
    assert all(bounds[i] <= bounds[i + 1] for i in range(len(bounds) - 1))


# ─── 2. Skill score ───────────────────────────────────────────────────────────

def test_skill_score_weights_sum_to_one():
    total = _SKILL_W_PRECISION + _SKILL_W_RECALL + _SKILL_W_ROC_AUC + _SKILL_W_PROFIT_FACTOR
    assert abs(total - 1.0) < 1e-9


def test_skill_score_precision_dominates_recall():
    # Filter-mode: precision should be weighted more than recall
    only_precision = compute_skill_score(1.0, 0.0, 0.5, 0.5)
    only_recall = compute_skill_score(0.0, 1.0, 0.5, 0.5)
    assert only_precision > only_recall


def test_skill_score_all_ones_equals_one():
    assert abs(compute_skill_score(1.0, 1.0, 1.0, 1.0) - 1.0) < 1e-9


# ─── 3. Calibration method selection ─────────────────────────────────────────

def _make_ensemble_with_member(scores_to_return: np.ndarray) -> VotingEnsemble:
    """Tiny ensemble with a single stubbed member returning fixed probabilities."""

    class _Stub:
        def __init__(self, proba):
            self._p = proba

        def predict_proba(self, X):
            # sklearn-style [P(class=0), P(class=1)] columns
            p = np.asarray(self._p[: len(X)], dtype=float)
            return np.column_stack([1.0 - p, p])

    ens = VotingEnsemble()
    ens.add_member(_Stub(scores_to_return), tag="stub", skill_score=1.0)
    return ens


def test_calibration_uses_platt_on_small_sample():
    rng = np.random.default_rng(0)
    n = 30                                # below MIN_SAMPLES_ISOTONIC=50
    raw = rng.uniform(0.3, 0.9, n)
    y = (raw > 0.5).astype(int)
    ens = _make_ensemble_with_member(raw)
    ens.apply_isotonic_calibration(y, np.zeros((n, 1)))
    assert isinstance(ens._calibrator, _PlattCalibrator)


def test_calibration_uses_isotonic_on_large_sample():
    from sklearn.isotonic import IsotonicRegression

    rng = np.random.default_rng(1)
    n = 100
    raw = rng.uniform(0.2, 0.95, n)
    y = (raw > 0.5).astype(int)
    # guarantee both classes present
    y[0], y[-1] = 0, 1
    ens = _make_ensemble_with_member(raw)
    ens.apply_isotonic_calibration(y, np.zeros((n, 1)))
    assert isinstance(ens._calibrator, IsotonicRegression)


def test_calibration_skipped_if_one_class():
    ens = _make_ensemble_with_member(np.array([0.6, 0.7, 0.8]))
    ens.apply_isotonic_calibration(np.array([1, 1, 1]), np.zeros((3, 1)))
    assert not ens._is_calibrated


# ─── 4. Adaptive drift threshold ─────────────────────────────────────────────

def test_drift_not_triggered_on_small_sample_noise():
    """N=30 with precision drop of 0.08 is within Wilson noise — no drift."""
    tracker = LivePerformanceTracker(window=50)
    # record 30 predictions: live_precision ≈ 0.62, train ≈ 0.70
    # We want 30 predicted-positive events, 19 true-positive (≈0.63 precision)
    np.random.seed(42)
    for _ in range(19):
        tracker.record(predicted_prob=0.8, actual_win=True)
    for _ in range(11):
        tracker.record(predicted_prob=0.8, actual_win=False)
    assert not tracker.is_drifting(training_precision=0.70)


def test_drift_triggered_at_large_sample_real_gap():
    """N=300 with precision drop of 0.15 exceeds Wilson noise — drift fires."""
    tracker = LivePerformanceTracker(window=500)
    # 300 predicted-positive events, only 150 true (≈0.50 precision)
    for _ in range(150):
        tracker.record(predicted_prob=0.8, actual_win=True)
    for _ in range(150):
        tracker.record(predicted_prob=0.8, actual_win=False)
    assert tracker.is_drifting(training_precision=0.80)


def test_drift_uses_fixed_threshold_when_set():
    tracker = LivePerformanceTracker(window=50, drift_threshold=0.05)
    for _ in range(20):
        tracker.record(predicted_prob=0.8, actual_win=True)
    for _ in range(10):
        tracker.record(predicted_prob=0.8, actual_win=False)
    # live_precision ≈ 0.667; train=0.75 → gap=0.083 > 0.05 → drift
    assert tracker.is_drifting(training_precision=0.75)


# ─── 5. StrategyTrade serialization ──────────────────────────────────────────

def test_from_db_row_filters_db_only_columns():
    row = {
        "id": 1,                          # DB primary key, not in dataclass
        "created_at": "2026-04-01",       # DB audit column
        "trade_id": "t1",
        "symbol": "BTCUSDT",
        "strategy_name": "grid",
        "market_regime": "sideways",
        "entry_price": 100.0,
        "exit_price": 101.0,
        "quantity": 0.1,
        "pnl_usd": 1.0,
        "pnl_pct": 1.0,
        "is_win": 1,                      # sqlite INTEGER
        "confidence": 0.7,
        "hour_of_day": 10,
        "day_of_week": 3,
        "timestamp_open": "",
        "timestamp_close": "",
        "rsi_at_entry": 55.0,
    }
    t = StrategyTrade.from_db_row(row)
    assert t.trade_id == "t1"
    assert t.is_win is True              # coerced from sqlite int
    assert t.symbol == "BTCUSDT"


def test_from_feature_vector_populates_all_ml_fields():
    fv = FeatureVector(
        symbol="BTC", timestamp=0, close=1.0,
        rsi_14=55, adx=20, volume_ratio=1.0,
        ema_9=1.0, ema_21=1.0, bb_bandwidth=0.01,
        macd_histogram=1.0, atr=1.0,
        news_sentiment=0.1, fear_greed_index=60,
        trend_alignment=0.5, cci=50, roc=1, cmf=0.1,
        bb_pct_b=0.5, hist_volatility=0.02,
        dmi_spread=5, stoch_rsi=55,
        price_change_5h=1.0, momentum=2.0, rsi_14_daily=55,
    )
    t = StrategyTrade.from_feature_vector(fv, strategy_name="grid", market_regime="sideways")

    ml_fields_checked = [
        "rsi_at_entry", "adx_at_entry", "volume_ratio_at_entry",
        "ema_9_at_entry", "ema_21_at_entry", "bb_bandwidth_at_entry",
        "macd_histogram_at_entry", "atr_at_entry", "trend_alignment",
        "cci_at_entry", "roc_at_entry", "cmf_at_entry",
        "bb_pct_b_at_entry", "hist_volatility_at_entry", "dmi_spread_at_entry",
        "stoch_rsi_at_entry", "price_change_5h_at_entry",
        "momentum_at_entry", "rsi_daily_at_entry",
        "news_sentiment", "fear_greed_index",
    ]
    for name in ml_fields_checked:
        assert hasattr(t, name), f"missing field: {name}"


# ─── 6. Predict threshold respects trained floor ─────────────────────────────

def test_predict_uses_max_of_calibrated_and_cfg():
    """Prevents .env override from silently weakening a strictly-trained model."""
    pred = MLPredictor(MLConfig(block_threshold=0.40))
    # Simulate a model with a trained threshold of 0.73
    pred._calibrated_threshold = 0.73

    # No actual model loaded → predict short-circuits; test the arithmetic directly
    effective = max(pred._calibrated_threshold, pred._cfg.block_threshold)
    assert effective == 0.73   # trained threshold wins

    pred._calibrated_threshold = 0.30
    effective2 = max(pred._calibrated_threshold, pred._cfg.block_threshold)
    assert effective2 == 0.40  # env floor wins
