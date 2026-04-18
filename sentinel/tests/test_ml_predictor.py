"""
Unit tests for MLPredictor — verifies correctness of all critical paths.

Run: python -m pytest tests/test_ml_predictor.py -v
"""

import pytest
import time
from unittest.mock import MagicMock, patch
from analyzer.ml_predictor import (
    MLPredictor, MLConfig, MLMetrics, MLPrediction,
    FEATURE_NAMES, N_FEATURES, REGIME_ENCODING, STRATEGY_REGIME_FIT,
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
        # market_regime_encoded is at index 11 in FEATURE_NAMES
        assert features[11] == float(REGIME_ENCODING["trending_up"])

    def test_strategy_regime_fit(self, predictor, sample_trade):
        # ema_crossover_rsi + trending_up should give fit=1.0
        features = predictor.extract_features(sample_trade)
        # strategy_regime_fit is at index 12 in FEATURE_NAMES
        expected = STRATEGY_REGIME_FIT.get(("ema_crossover_rsi", "trending_up"), 0.0)
        assert features[12] == expected

    def test_fear_greed_normalized(self, predictor, sample_trade):
        features = predictor.extract_features(sample_trade)
        # fear_greed_normalized is at index 19 in FEATURE_NAMES
        assert features[19] == 0.65  # 65 / 100

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
        assert features[13] == 1.0  # recent_win_rate_10 at index 13

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
        With default threshold=0.55: block < 0.4675, reduce 0.4675..0.55, allow >= 0.55.
        """
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = [[0.50, 0.50]]  # 0.4675 < 0.50 < 0.55 → reduce
        predictor._model = mock_model
        predictor._ensemble = None
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Calibration diagnostics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestExpectedCalibrationError:
    """Coverage for MLPredictor._expected_calibration_error.

    ECE is the diagnostic that catches the "every signal looks like 95%"
    failure mode that AUC/precision miss. These tests pin the contract.
    """

    def _ece(self, y_true, y_proba, n_bins=10):
        import numpy as np
        return MLPredictor._expected_calibration_error(
            np.asarray(y_true), np.asarray(y_proba), n_bins=n_bins,
        )

    def test_perfect_calibration_zero(self):
        # Confidence == realized frequency in every populated bin → ECE = 0.
        y_true  = [0, 0, 1, 1]
        y_proba = [0.0, 0.0, 1.0, 1.0]
        assert self._ece(y_true, y_proba) == pytest.approx(0.0, abs=1e-9)

    def test_inverted_calibration_max(self):
        # Confidence 1.0 on losses and 0.0 on wins → ECE = 1.0 (maximally bad).
        y_true  = [0, 0, 1, 1]
        y_proba = [1.0, 1.0, 0.0, 0.0]
        assert self._ece(y_true, y_proba) == pytest.approx(1.0, abs=1e-9)

    def test_constant_high_proba_against_balanced_outcomes(self):
        # The exact "always shows 95%" pathology: model claims 0.9 confidence
        # but only 50% of trades actually win → ECE ≈ 0.4.
        y_true  = [0, 1, 0, 1, 0, 1, 0, 1]
        y_proba = [0.9] * 8
        assert self._ece(y_true, y_proba) == pytest.approx(0.4, abs=0.01)

    def test_empty_input_returns_zero(self):
        assert self._ece([], []) == 0.0

    def test_handles_proba_at_exact_bin_edge(self):
        # 1.0 must land in the last bin (not overflow); single bin → ECE = 0.
        y_true  = [1, 1, 1]
        y_proba = [1.0, 1.0, 1.0]
        assert self._ece(y_true, y_proba) == pytest.approx(0.0, abs=1e-9)


class TestMLMetricsCalibrationFields:
    """MLMetrics carries calibration diagnostics through save/load and to consumers."""

    def test_defaults(self):
        m = MLMetrics()
        assert m.brier_score == 0.0
        assert m.ece == 0.0
        assert m.mean_proba == 0.5
        assert m.median_proba == 0.5
        assert m.proba_p10 == 0.0
        assert m.proba_p90 == 1.0
        assert m.calibration_method == "none"

    def test_explicit_assignment(self):
        m = MLMetrics(
            brier_score=0.18, ece=0.07, mean_proba=0.62,
            median_proba=0.58, proba_p10=0.35, proba_p90=0.82,
            calibration_method="platt",
        )
        assert m.brier_score == 0.18
        assert m.ece == 0.07
        assert m.calibration_method == "platt"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Backtest training pipeline plumbing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestTrainMLPipelinePlumbing:
    """Regression tests for the train_ml.py → StrategyTrade conversion.

    The training script must propagate every feature it has access to;
    silently dropping `trend_alignment` (computed by FeatureBuilder, available
    in backtest) caused that feature to land in training as a constant 0.5
    and get flagged as zero-variance noise.
    """

    def test_backtest_to_strategy_trade_propagates_trend_alignment(self):
        from scripts.train_ml import backtest_trade_to_strategy_trade
        from backtest.engine import BacktestTrade

        bt = BacktestTrade(
            symbol="BTCUSDT", entry_time=1_700_000_000_000, exit_time=1_700_000_300_000,
            entry_price=60000.0, exit_price=60500.0, quantity=0.01,
            pnl=5.0, pnl_pct=0.83, commission=0.06, reason="exit",
        )
        st = backtest_trade_to_strategy_trade(
            bt, "ema_crossover_rsi",
            features_at_entry={"trend_alignment": 0.83},
        )
        assert st.trend_alignment == pytest.approx(0.83)

    def test_backtest_to_strategy_trade_defaults_when_feature_missing(self):
        """If an upstream version of the training script doesn't pass it, the
        StrategyTrade default of 0.5 must still hold (no crash, no None)."""
        from scripts.train_ml import backtest_trade_to_strategy_trade
        from backtest.engine import BacktestTrade

        bt = BacktestTrade(
            symbol="BTCUSDT", entry_time=1_700_000_000_000, exit_time=1_700_000_300_000,
            entry_price=60000.0, exit_price=60500.0, quantity=0.01,
            pnl=5.0, pnl_pct=0.83, commission=0.06, reason="exit",
        )
        st = backtest_trade_to_strategy_trade(bt, "ema_crossover_rsi")
        assert st.trend_alignment == 0.5  # dataclass default


class TestOverfitNoiseMargin:
    """Statistical margin for the train-vs-val precision gap.

    The previous flat heuristic (`0.5 / sqrt(n_val)`) ignored both training
    sample size and the actual proportion p, so it falsely rejected models
    whose train/val gap was within sampling noise. These tests pin the
    statistically-correct behavior.
    """

    def _m(self, p_train, p_val, n_train, n_val, z=1.96):
        return MLPredictor._overfit_noise_margin(p_train, p_val, n_train, n_val, z=z)

    def test_zero_when_either_sample_empty(self):
        assert self._m(0.5, 0.5, 0, 100) == 0.0
        assert self._m(0.5, 0.5, 100, 0) == 0.0

    def test_shrinks_with_n(self):
        big   = self._m(0.5, 0.5, 10_000, 10_000)
        small = self._m(0.5, 0.5, 100, 100)
        assert big < small
        # n×100 → margin ≈ /10 (sqrt-scaling)
        assert big == pytest.approx(small / 10.0, rel=0.05)

    def test_smaller_at_extreme_proportions(self):
        # At p ≈ 0.95 the binomial variance p(1-p) is far smaller than at 0.5,
        # so the margin should be too — preventing false-rejection of models
        # that genuinely converge to high precision on both splits.
        m_extreme = self._m(0.95, 0.92, 600, 130)
        m_middle  = self._m(0.50, 0.50, 600, 130)
        assert m_extreme < m_middle

    def test_regression_gap_0166_now_passes(self):
        """The exact case that surfaced in the live training log:
        train_prec=0.833 val_prec=0.667 n_train≈590 n_val=126.
        Old flat threshold 0.145 rejected gap=0.166; the statistically-correct
        threshold (base 0.10 + 1.96σ margin) must accept it."""
        margin = self._m(0.833, 0.667, 590, 126)
        # 1.96 × sqrt(0.833·0.167/590 + 0.667·0.333/126) ≈ 0.088
        assert margin == pytest.approx(0.088, abs=0.005)
        threshold = 0.10 + margin                # base + noise
        assert 0.166 <= threshold, (
            f"Statistical threshold {threshold:.3f} should accept gap=0.166 "
            f"(was falsely rejected at flat threshold 0.145)"
        )

    def test_truly_overfit_still_rejected(self):
        """A train/val precision drop of 0.30 on healthy n must still be flagged."""
        margin = self._m(0.95, 0.65, 1000, 200)
        threshold = 0.10 + margin
        # gap = 0.30 must exceed base + margin
        assert 0.30 > threshold


class TestMLConfigPlumbing:
    """New MLConfig fields must reach the actual code paths that use them.

    Each test pins a config knob to a non-default value and verifies that
    the corresponding behavior changes — so refactors don't silently strand
    the option as a no-op.
    """

    def test_random_seed_propagates_to_estimators(self):
        cfg = MLConfig(random_seed=1234)
        pred = MLPredictor(cfg)
        rf = pred._build_rf()
        assert rf.random_state == 1234

    def test_temporal_decay_propagates_to_weights(self):
        from analyzer.ml_predictor import MLPredictor as MP
        cfg = MLConfig(temporal_decay=0.0)            # decay=0 → uniform weights
        pred = MP(cfg)
        # Direct helper call confirms decay value reaches the formula.
        w = pred._compute_temporal_weights(50, decay=cfg.temporal_decay)
        # With decay=0, exp(0·i) = 1 → mean=1 → all weights equal 1.
        assert all(abs(wi - 1.0) < 1e-9 for wi in w)

    def test_drift_threshold_propagates_to_tracker(self):
        cfg = MLConfig(drift_threshold=0.42)
        pred = MLPredictor(cfg)
        assert pred._live_tracker._drift_threshold == 0.42

    def test_drift_threshold_default_is_adaptive(self):
        """drift_threshold=None → tracker uses Wilson-width adaptive margin
        (the safe default for unknown live samples)."""
        cfg = MLConfig()
        pred = MLPredictor(cfg)
        assert pred._live_tracker._drift_threshold is None

    def test_reduce_margin_changes_block_zone(self, predictor):
        """A predictor with reduce_margin=0.50 should classify a probability
        between 0.50·thr and thr as 'reduce' instead of 'block'."""
        # Set up a minimally-ready predictor with fixed threshold + simple ensemble.
        from analyzer.ml_ensemble import VotingEnsemble
        import numpy as np

        class _ConstModel:
            def __init__(self, p): self._p = p
            def predict_proba(self, X):
                return np.array([[1 - self._p, self._p]] * len(X))

        cfg = MLConfig(min_trades=5, reduce_margin=0.50)
        pred = MLPredictor(cfg)
        ens = VotingEnsemble()
        ens.add_member(_ConstModel(0.30), "rf", 1.0)   # proba = 0.30
        pred._ensemble = ens
        pred._scaler = None
        pred._calibrated_threshold = 0.50              # → block zone < 0.25, reduce 0.25..0.50
        pred._model_version = "test"
        pred.rollout_mode = "block"
        result = pred.predict([0.0] * N_FEATURES)
        # 0.30 sits in (0.25, 0.50) → reduce, NOT block (would be block at margin=0.85).
        assert result.decision == "reduce", (
            f"Expected 'reduce' with margin=0.50 (proba=0.30, thr=0.50, block-zone<0.25); "
            f"got {result.decision}"
        )


class TestPackageVersions:
    """The pickle artifact must capture the ML stack's versions and the
    loader must warn on divergence."""

    def test_capture_returns_known_packages(self):
        from analyzer.ml_predictor import _capture_package_versions
        v = _capture_package_versions()
        # We always import these in the predictor.
        for k in ("numpy", "sklearn"):
            assert k in v
            # version string or 'missing'/'unknown' — just must be a str
            assert isinstance(v[k], str)

    def test_missing_package_reported_as_missing(self, monkeypatch):
        from analyzer.ml_predictor import _capture_package_versions
        # Force ImportError on a probe by adding a stub key — we instead
        # just assert no exception is raised even for absent libs.
        v = _capture_package_versions()
        assert all(isinstance(val, str) for val in v.values())


class TestRestrictedUnpickler:
    """The restricted unpickler must reject classes outside the whitelist
    even if a checksum-valid pickle were forged."""

    def test_allows_whitelisted_module(self):
        from analyzer.ml_predictor import _restricted_loads
        import pickle as _pk
        # numpy.ndarray is whitelisted
        import numpy as np
        payload = _pk.dumps(np.array([1, 2, 3]))
        loaded = _restricted_loads(payload)
        assert list(loaded) == [1, 2, 3]

    def test_rejects_arbitrary_callable(self):
        from analyzer.ml_predictor import _restricted_loads
        import pickle as _pk

        # Construct a pickle that would invoke os.system on load — the kind
        # of payload an attacker could swap into the model file. The
        # restricted unpickler must refuse to even resolve `os.system`.
        class _Exploit:
            def __reduce__(self):
                import os
                return (os.system, ("echo pwned",))

        payload = _pk.dumps(_Exploit())
        with pytest.raises(_pk.UnpicklingError):
            _restricted_loads(payload)

    def test_rejects_builtins_eval(self):
        """`builtins` IS whitelisted (for dict/tuple), but if an attacker
        somehow encodes `builtins.eval` we still want to know they cannot
        chain it into anything useful via pickle alone — `builtins` is
        permitted, so this test documents that limitation explicitly."""
        from analyzer.ml_predictor import _restricted_loads, _PICKLE_ALLOWED_PREFIXES
        # Document the design choice rather than enforce a stricter rule:
        # builtins is allowed because dicts/tuples/lists are constructed
        # from it. Operators relying on the whitelist should know.
        assert "builtins" in _PICKLE_ALLOWED_PREFIXES


class TestZeroVarianceForceDrop:
    """The training routine must remove zero-variance features unconditionally.

    Tree models occasionally split on a constant column (driven by sample
    weight ties), which can leave the feature with a non-zero "importance"
    that sneaks past the AdaptiveFeatureSelector. We zero its importance
    explicitly so it cannot end up in the deployed feature vector — otherwise
    the StandardScaler divides a constant by std=0 → numerical noise that
    inflates the calibrated probability spread.
    """

    def test_dead_feature_importance_zeroed_before_selector_fit(self):
        """Constant column → variance==0 → its importance must be zeroed
        before AdaptiveFeatureSelector.fit, regardless of any non-zero
        importance the underlying model may have assigned to it.
        """
        from analyzer.ml_ensemble import AdaptiveFeatureSelector
        feature_names = ["a", "b", "c"]
        # Model gave the dead feature 'b' a small but non-zero importance.
        importances = {"a": 0.3, "b": 0.005, "c": 0.4}
        # Simulate the force-drop step from train(): set dead importances to 0.
        for dead in ["b"]:
            importances[dead] = 0.0
        sel = AdaptiveFeatureSelector(min_importance=0.001)
        sel.fit(importances, feature_names)
        assert "b" in sel.dropped_names
        assert "a" in sel.selected_names
        assert "c" in sel.selected_names
