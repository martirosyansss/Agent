"""
Tests for RegimeRouter.

The router is deliberately trainer-agnostic, so these tests inject a fake
``trainer`` callable that returns a trivial ``RegimeModel`` instead of
doing real ML fits. That keeps the tests fast (< 1 s) and lets us assert
on the partitioning / fallback logic directly.
"""
from __future__ import annotations

import numpy as np
import pytest

from analyzer.ml_regime_router import RegimeRouter, RegimeModel


class _FakeEnsemble:
    """Minimal stand-in that returns a fixed probability regardless of input."""
    def __init__(self, proba: float) -> None:
        self._p = proba

    def predict_proba_calibrated(self, X):
        return np.full(len(X), self._p)


def _make_trade(regime: str, rsi: float = 55.0):
    """Build the minimum trade shape the router inspects."""
    from core.models import StrategyTrade
    return StrategyTrade(
        trade_id=f"t_{regime}_{rsi}", symbol="BTCUSDT",
        strategy_name="ema_crossover_rsi",
        market_regime=regime, entry_price=60000.0, exit_price=61000.0,
        quantity=0.01, pnl_usd=10.0, pnl_pct=1.67, is_win=True,
        confidence=0.75, hour_of_day=12, day_of_week=1,
        rsi_at_entry=rsi, adx_at_entry=30.0, volume_ratio_at_entry=1.5,
        news_sentiment=0.0, fear_greed_index=50, trend_alignment=0.0,
        ema_9_at_entry=60500.0, ema_21_at_entry=60000.0,
        bb_bandwidth_at_entry=0.05, macd_histogram_at_entry=100.0,
        atr_at_entry=800.0, timestamp_open="2025-01-01T10:00:00Z",
        timestamp_close="2025-01-01T12:00:00Z",
    )


class TestTrain:
    def test_builds_specialist_per_regime_above_min(self):
        trades = [_make_trade("trending_up") for _ in range(120)] + \
                 [_make_trade("sideways") for _ in range(120)]
        calls: list[tuple[str, int]] = []

        def trainer(subset, regime):
            calls.append((regime, len(subset)))
            return RegimeModel(
                regime=regime,
                ensemble=_FakeEnsemble(0.7 if regime == "trending_up" else 0.3),
                scaler=None, selector=None,
                threshold=0.5, skill_score=0.8, n_train=len(subset),
            )

        router = RegimeRouter(min_trades_per_regime=100)
        stats = router.train(trades, trainer, global_model=RegimeModel(
            regime="__global__", ensemble=_FakeEnsemble(0.5),
            scaler=None, selector=None, threshold=0.5,
        ))
        assert router.is_ready
        assert sorted(router.trained_regimes) == ["sideways", "trending_up"]
        assert len(calls) == 2
        assert stats["trending_up"].trained is True
        assert stats["sideways"].trained is True

    def test_falls_back_to_global_below_min_trades(self):
        trades = [_make_trade("trending_up") for _ in range(30)]
        trainer_calls = {"count": 0}

        def trainer(subset, regime):
            trainer_calls["count"] += 1
            return RegimeModel(regime=regime, ensemble=_FakeEnsemble(0.9),
                               scaler=None, selector=None, threshold=0.5)

        router = RegimeRouter(min_trades_per_regime=100)
        router.train(trades, trainer, global_model=RegimeModel(
            regime="__global__", ensemble=_FakeEnsemble(0.5),
            scaler=None, selector=None, threshold=0.5,
        ))
        assert trainer_calls["count"] == 0  # trainer never called
        assert router.trained_regimes == []

    def test_noise_regimes_always_fall_back(self):
        trades = [_make_trade("unknown") for _ in range(200)] + \
                 [_make_trade("transitioning") for _ in range(200)]
        trainer_calls = []

        def trainer(subset, regime):
            trainer_calls.append(regime)
            return RegimeModel(regime=regime, ensemble=_FakeEnsemble(0.7),
                               scaler=None, selector=None, threshold=0.5)

        router = RegimeRouter(min_trades_per_regime=50)
        router.train(trades, trainer, global_model=RegimeModel(
            regime="__global__", ensemble=_FakeEnsemble(0.5),
            scaler=None, selector=None, threshold=0.5,
        ))
        assert trainer_calls == []  # UNKNOWN regimes never trained
        stats = router.get_regime_stats()
        assert stats["unknown"]["trained"] is False
        assert stats["transitioning"]["trained"] is False

    def test_trainer_returning_none_falls_back(self):
        trades = [_make_trade("trending_up") for _ in range(200)]

        def trainer(subset, regime):
            return None  # simulate trainer failure

        router = RegimeRouter(min_trades_per_regime=100)
        router.train(trades, trainer, global_model=RegimeModel(
            regime="__global__", ensemble=_FakeEnsemble(0.5),
            scaler=None, selector=None, threshold=0.5,
        ))
        assert router.trained_regimes == []

    def test_min_trades_per_regime_validation(self):
        with pytest.raises(ValueError):
            RegimeRouter(min_trades_per_regime=10)


class TestPredict:
    def _mk_router(self):
        trades = [_make_trade("trending_up") for _ in range(120)] + \
                 [_make_trade("sideways") for _ in range(120)]

        def trainer(subset, regime):
            p = 0.85 if regime == "trending_up" else 0.30
            return RegimeModel(regime=regime, ensemble=_FakeEnsemble(p),
                               scaler=None, selector=None, threshold=0.5)

        router = RegimeRouter(min_trades_per_regime=100)
        router.train(trades, trainer, global_model=RegimeModel(
            regime="__global__", ensemble=_FakeEnsemble(0.5),
            scaler=None, selector=None, threshold=0.5,
        ))
        return router

    def test_routes_to_correct_specialist(self):
        router = self._mk_router()
        features = np.zeros(32)
        p_trend, used_trend, thr_trend = router.predict_proba("trending_up", features)
        p_side, used_side, thr_side = router.predict_proba("sideways", features)
        assert used_trend == "trending_up"
        assert used_side == "sideways"
        assert p_trend > p_side
        # Both specialist thresholds came from their RegimeModel.threshold
        assert thr_trend == pytest.approx(0.5)
        assert thr_side == pytest.approx(0.5)

    def test_unknown_regime_falls_back_to_global(self):
        router = self._mk_router()
        features = np.zeros(32)
        p, used, thr = router.predict_proba("not_a_real_regime", features)
        assert used == "global"
        assert p == pytest.approx(0.5)
        assert thr == pytest.approx(0.5)

    def test_no_global_no_specialist_returns_neutral(self):
        router = RegimeRouter(min_trades_per_regime=50)
        # Deliberately don't call train()
        p, used, thr = router.predict_proba("trending_up", np.zeros(32))
        assert used == "none"
        assert p == pytest.approx(0.5)
        assert thr == pytest.approx(0.5)

    def test_specialist_returns_its_own_threshold(self):
        """Regression for M-4: router must surface each specialist's calibrated
        threshold so the caller can honour per-regime calibration."""
        trades = [_make_trade("trending_up") for _ in range(120)] + \
                 [_make_trade("sideways") for _ in range(120)]

        def trainer(subset, regime):
            thr = 0.62 if regime == "trending_up" else 0.48
            return RegimeModel(regime=regime, ensemble=_FakeEnsemble(0.7),
                               scaler=None, selector=None, threshold=thr)

        router = RegimeRouter(min_trades_per_regime=100)
        router.train(trades, trainer, global_model=RegimeModel(
            regime="__global__", ensemble=_FakeEnsemble(0.5),
            scaler=None, selector=None, threshold=0.55,
        ))
        _, _, thr_up = router.predict_proba("trending_up", np.zeros(32))
        _, _, thr_side = router.predict_proba("sideways", np.zeros(32))
        _, _, thr_global = router.predict_proba("unseen", np.zeros(32))
        assert thr_up == pytest.approx(0.62)
        assert thr_side == pytest.approx(0.48)
        assert thr_global == pytest.approx(0.55)


class TestGetRegimeStats:
    def test_stats_serializable(self):
        trades = [_make_trade("trending_up") for _ in range(120)]

        def trainer(subset, regime):
            return RegimeModel(regime=regime, ensemble=_FakeEnsemble(0.7),
                               scaler=None, selector=None, threshold=0.5,
                               skill_score=0.65,
                               metrics_summary={"precision": 0.72, "roc_auc": 0.81})

        router = RegimeRouter(min_trades_per_regime=100)
        router.train(trades, trainer, global_model=RegimeModel(
            regime="__global__", ensemble=_FakeEnsemble(0.5),
            scaler=None, selector=None, threshold=0.5,
        ))
        stats = router.get_regime_stats()
        for _, s in stats.items():
            for k, v in s.items():
                assert isinstance(v, (int, float, str, bool)), f"{k}={v!r}"
