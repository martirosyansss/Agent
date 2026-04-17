"""Tests for the professional risk-control modules.

Covers:
- risk.drawdown_breaker.DrawdownBreaker
- risk.correlation_guard.CorrelationGuard (+ effective_n_positions)
- risk.exposure_caps.ExposureCapGuard
- backtest.execution_model.RealisticExecutionModel
- backtest.walk_forward.probabilistic_sharpe_ratio
- strategy.multi_tf_gate.MultiTFGate
- risk.sentinel.RiskSentinel guard integration
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.events import EventBus
from core.models import Direction, FeatureVector, RiskState, Signal
from risk.correlation_guard import (
    CorrelationConfig,
    CorrelationGuard,
    effective_n_positions,
)
from risk.drawdown_breaker import (
    DrawdownBreaker,
    DrawdownThresholds,
)
from risk.exposure_caps import (
    DEFAULT_ASSET_CLASS_MAP,
    ExposureCapConfig,
    ExposureCapGuard,
    OpenPositionExposure,
)
from risk.sentinel import RiskLimits, RiskSentinel
from risk.state_machine import RiskStateMachine
from backtest.execution_model import (
    ExecutionConfig,
    FillReason,
    RealisticExecutionModel,
)
from backtest.walk_forward import probabilistic_sharpe_ratio
from strategy.multi_tf_gate import (
    MultiTFGate,
    MultiTFGateConfig,
    StrategyType,
    classify_strategy,
)
from risk.regime_gate import RegimeGate, RegimeGateConfig
from risk.news_cooldown import NewsCooldownConfig, NewsCooldownGuard
from risk.liquidity_gate import LiquidityGate, LiquidityGateConfig
from risk.stale_data_gate import StaleDataGate, StaleDataGateConfig
from risk.position_sizer import (
    SizingInput,
    calculate_position_size,
    correlation_factor,
)
from backtest.monte_carlo import (
    MonteCarloAnalyser,
    MonteCarloConfig,
    MonteCarloReport,
)
from monitoring.event_log import EventLog
from risk.decision_tracer import (
    DecisionTrace,
    GateOutcome,
    GateTimer,
    GateVerdict,
    feature_snapshot_dict,
)


# ════════════════════════════════════════════════════════════════════
# DrawdownBreaker
# ════════════════════════════════════════════════════════════════════

class TestDrawdownBreaker:
    def test_no_trip_under_threshold(self):
        b = DrawdownBreaker(DrawdownThresholds(daily_pct=0.05, weekly_pct=0.10, monthly_pct=0.15))
        b.update(1000.0)
        b.update(970.0)  # 3% DD < 5% threshold
        assert b.allows_new_entry()
        assert b.active_trips() == []

    def test_daily_trip(self):
        b = DrawdownBreaker(DrawdownThresholds(daily_pct=0.05))
        b.update(1000.0)
        reason = b.update(940.0)  # 6% DD > 5%
        assert reason is not None and "daily" in reason
        assert not b.allows_new_entry()
        assert "daily" in b.active_trips()

    def test_weekly_trip_independent(self):
        b = DrawdownBreaker(DrawdownThresholds(daily_pct=0.20, weekly_pct=0.05))
        b.update(1000.0)
        b.update(940.0)  # 6% DD; daily threshold is 20% so daily OK; weekly 5% trips
        assert "weekly" in b.active_trips()
        assert not b.allows_new_entry()

    def test_hysteresis_recovery(self):
        b = DrawdownBreaker(DrawdownThresholds(daily_pct=0.05))
        b.update(1000.0)
        b.update(940.0)
        assert not b.allows_new_entry()
        # Recover to within (peak * (1 - 0.05*0.5)) = 1000 * 0.975 = 975
        b.update(980.0)
        assert b.allows_new_entry()

    def test_period_rollover_resets(self):
        # Use injected time to advance past UTC midnight.
        clock = [datetime(2026, 4, 17, 23, 0, 0, tzinfo=timezone.utc).timestamp()]
        b = DrawdownBreaker(
            DrawdownThresholds(daily_pct=0.05),
            time_provider=lambda: clock[0],
        )
        b.update(1000.0)
        b.update(940.0)
        assert not b.allows_new_entry()
        # Next UTC day
        clock[0] = datetime(2026, 4, 18, 1, 0, 0, tzinfo=timezone.utc).timestamp()
        b.update(940.0)  # equity unchanged but new period → fresh peak
        assert b.allows_new_entry()

    def test_force_reset(self):
        b = DrawdownBreaker(DrawdownThresholds(daily_pct=0.05))
        b.update(1000.0)
        b.update(900.0)
        assert not b.allows_new_entry()
        b.force_reset()
        assert b.allows_new_entry()

    def test_invalid_thresholds_raise(self):
        with pytest.raises(ValueError):
            DrawdownThresholds(daily_pct=1.5)
        with pytest.raises(ValueError):
            DrawdownThresholds(weekly_pct=0.0)

    def test_export_restore_roundtrip_preserves_tripped_flag(self):
        """Simulates a restart mid-drawdown: export, new instance, restore."""
        b1 = DrawdownBreaker(DrawdownThresholds(daily_pct=0.05))
        b1.update(1000.0)
        b1.update(940.0)  # 6% drawdown → trips daily
        assert not b1.allows_new_entry()
        blob = b1.export_state()

        b2 = DrawdownBreaker(DrawdownThresholds(daily_pct=0.05))
        assert b2.allows_new_entry()  # fresh instance
        b2.restore_state(blob)
        assert not b2.allows_new_entry()  # tripped flag survives

    def test_restore_tolerates_garbage(self):
        b = DrawdownBreaker(DrawdownThresholds(daily_pct=0.05))
        b.restore_state({"daily": "not-a-dict", "weekly": None, "monthly": 42})
        # Should not crash and should still function.
        b.update(1000.0)
        assert b.allows_new_entry()


# ════════════════════════════════════════════════════════════════════
# CorrelationGuard
# ════════════════════════════════════════════════════════════════════

class TestCorrelationGuard:
    def _walk(self, base: float, drifts: list[float]) -> list[float]:
        out = [base]
        for d in drifts:
            out.append(out[-1] * (1 + d))
        return out

    def test_no_open_positions_approves(self):
        g = CorrelationGuard()
        d = g.check("BTCUSDT", [], {"BTCUSDT": [100.0] * 50})
        assert d.approved

    def test_insufficient_history_skips(self):
        g = CorrelationGuard()
        d = g.check("BTCUSDT", ["ETHUSDT"], {"BTCUSDT": [100, 101], "ETHUSDT": [200, 201]})
        assert d.approved
        assert "Insufficient" in d.reason

    def test_high_correlation_blocks(self):
        # Two perfectly correlated series.
        drifts = [0.01, -0.01, 0.02, -0.005, 0.01, -0.02, 0.03, -0.01] * 5
        prices_a = self._walk(100.0, drifts)
        prices_b = self._walk(50.0, drifts)  # identical drifts → ρ=1
        cfg = CorrelationConfig(threshold=0.7, min_observations=10, max_cluster_size=1)
        g = CorrelationGuard(cfg)
        d = g.check("B", ["A"], {"A": prices_a, "B": prices_b})
        assert not d.approved
        assert "cluster" in d.reason.lower()
        assert "A" in d.cluster

    def test_low_correlation_approves(self):
        drifts_a = [0.01, -0.01, 0.02, -0.005, 0.01, -0.02, 0.03, -0.01] * 5
        drifts_b = [-0.02, 0.03, -0.01, 0.005, -0.01, 0.02, -0.03, 0.01] * 5
        cfg = CorrelationConfig(threshold=0.95, min_observations=10, max_cluster_size=2,
                                min_effective_positions=0.0)
        g = CorrelationGuard(cfg)
        d = g.check("B", ["A"], {
            "A": self._walk(100.0, drifts_a),
            "B": self._walk(50.0, drifts_b),
        })
        assert d.approved

    def test_cluster_transitivity(self):
        # A ≈ B and A ≈ C (both highly correlated to A), but B vs C not directly tested.
        # Cluster should pull C into {A,B} and reject D if it joins via A as well.
        drifts = [0.01, -0.01, 0.02, -0.005, 0.01, -0.02, 0.03, -0.01] * 5
        cfg = CorrelationConfig(threshold=0.8, min_observations=10, max_cluster_size=2,
                                min_effective_positions=0.0)
        g = CorrelationGuard(cfg)
        d = g.check("C", ["A", "B"], {
            "A": self._walk(100.0, drifts),
            "B": self._walk(50.0, drifts),  # ρ(A,B) = 1
            "C": self._walk(200.0, drifts),  # ρ(A,C) = 1
        })
        assert not d.approved

    def test_effective_n_positions_independent(self):
        # 3-asset identity correlation matrix → ENP = 3
        m = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        assert effective_n_positions(m) == pytest.approx(3.0, abs=1e-3)

    def test_effective_n_positions_perfect_correlation(self):
        # All-ones correlation matrix → ENP = 1
        m = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        assert effective_n_positions(m) == pytest.approx(1.0, abs=1e-3)


# ════════════════════════════════════════════════════════════════════
# ExposureCapGuard
# ════════════════════════════════════════════════════════════════════

class TestExposureCapGuard:
    def test_default_class_map_resolves(self):
        g = ExposureCapGuard()
        assert g.asset_class("BTCUSDT") == "L1"
        assert g.asset_class("ARBUSDT") == "L2"
        assert g.asset_class("DOGEUSDT") == "MEME"
        assert g.asset_class("XYZUSDT") == "UNKNOWN"

    def test_under_cap_approves(self):
        g = ExposureCapGuard()
        d = g.check(
            candidate_symbol="BTCUSDT",
            candidate_notional_usd=1000.0,
            equity_usd=10000.0,
            open_positions=[],
        )
        assert d.approved
        assert d.asset_class == "L1"

    def test_over_cap_rejects(self):
        # L1 cap default is 35%. $5000 BTC out of $10000 = 50% → blocked.
        g = ExposureCapGuard()
        d = g.check(
            candidate_symbol="BTCUSDT",
            candidate_notional_usd=5000.0,
            equity_usd=10000.0,
            open_positions=[],
        )
        assert not d.approved
        assert "L1" in d.reason

    def test_cap_considers_existing_positions(self):
        # Already 30% in ETH (L1). Adding 10% in BTC → 40% > 35% cap.
        g = ExposureCapGuard()
        d = g.check(
            candidate_symbol="BTCUSDT",
            candidate_notional_usd=1000.0,
            equity_usd=10000.0,
            open_positions=[OpenPositionExposure(symbol="ETHUSDT", notional_usd=3000.0)],
        )
        assert not d.approved

    def test_meme_cap_strict(self):
        # MEME default cap is 5%.
        g = ExposureCapGuard()
        d = g.check(
            candidate_symbol="DOGEUSDT",
            candidate_notional_usd=600.0,
            equity_usd=10000.0,
            open_positions=[],
        )
        assert not d.approved
        assert "MEME" in d.reason

    def test_snapshot_aggregates_by_class(self):
        g = ExposureCapGuard()
        snap = g.snapshot(
            equity_usd=10000.0,
            open_positions=[
                OpenPositionExposure("BTCUSDT", 2000.0),
                OpenPositionExposure("ETHUSDT", 1500.0),
                OpenPositionExposure("DOGEUSDT", 200.0),
            ],
        )
        assert snap["L1"]["notional_usd"] == 3500.0
        assert snap["L1"]["pct_of_equity"] == 35.0
        assert snap["MEME"]["pct_of_equity"] == 2.0


# ════════════════════════════════════════════════════════════════════
# RealisticExecutionModel
# ════════════════════════════════════════════════════════════════════

class TestRealisticExecutionModel:
    def test_buy_pays_higher_than_reference(self):
        m = RealisticExecutionModel()
        f = m.fill_market_buy(reference_price=100.0, notional_usd=100.0)
        assert f.fill_price > 100.0
        assert f.slippage_bps > 0

    def test_sell_pays_lower_than_reference(self):
        m = RealisticExecutionModel()
        f = m.fill_market_sell(reference_price=100.0, notional_usd=100.0)
        assert f.fill_price < 100.0

    def test_impact_grows_with_size(self):
        m = RealisticExecutionModel()
        small = m.fill_market_buy(100.0, notional_usd=100.0).slippage_bps
        large = m.fill_market_buy(100.0, notional_usd=1_000_000.0).slippage_bps
        assert large > small

    def test_stop_loss_normal_fill(self):
        m = RealisticExecutionModel()
        f = m.fill_stop_loss(stop_price=90.0, candle_open=95.0, candle_low=88.0, notional_usd=100.0)
        assert f is not None
        assert f.reason == FillReason.STOP_LOSS
        # Fill should be slightly below stop_price (slippage).
        assert f.fill_price < 90.0
        assert f.fill_price > 89.0  # but only slightly

    def test_stop_loss_gap_fill_uses_open(self):
        m = RealisticExecutionModel()
        # Overnight gap: open is already below stop.
        f = m.fill_stop_loss(stop_price=90.0, candle_open=80.0, candle_low=75.0, notional_usd=100.0)
        assert f is not None
        assert f.reason == FillReason.STOP_LOSS_GAP
        # Fill should be ≤ open_price (the gap victim).
        assert f.fill_price <= 80.0
        assert "gap" in f.notes.lower()

    def test_stop_loss_no_touch_returns_none(self):
        m = RealisticExecutionModel()
        # Candle stayed above stop.
        f = m.fill_stop_loss(stop_price=90.0, candle_open=95.0, candle_low=92.0, notional_usd=100.0)
        assert f is None

    def test_take_profit_gap_uses_open(self):
        m = RealisticExecutionModel()
        f = m.fill_take_profit(tp_price=110.0, candle_open=120.0, candle_high=125.0, notional_usd=100.0)
        assert f is not None
        assert f.reason == FillReason.TAKE_PROFIT_GAP
        # Fill should be near open price (better than TP), still with sell slippage.
        assert f.fill_price > 110.0
        assert f.fill_price <= 120.0

    def test_slippage_bounded(self):
        # Even an absurd notional shouldn't exceed max_slippage_bps.
        cfg = ExecutionConfig(max_slippage_bps=50.0)
        m = RealisticExecutionModel(cfg)
        f = m.fill_market_buy(100.0, notional_usd=1e12)
        assert f.slippage_bps <= 50.0


# ════════════════════════════════════════════════════════════════════
# Probabilistic Sharpe Ratio
# ════════════════════════════════════════════════════════════════════

class TestPSR:
    def test_psr_rises_with_sample_size(self):
        # Same Sharpe-equivalent series, longer sample → higher PSR.
        small = [0.01, -0.005, 0.012, -0.004, 0.011, -0.003] * 3
        large = small * 10
        assert probabilistic_sharpe_ratio(large) > probabilistic_sharpe_ratio(small)

    def test_psr_returns_zero_for_short_sample(self):
        assert probabilistic_sharpe_ratio([0.01, 0.02]) == 0.0

    def test_psr_in_unit_interval(self):
        rs = [0.005, -0.003, 0.008, -0.002, 0.006, -0.004, 0.007, -0.001] * 5
        psr = probabilistic_sharpe_ratio(rs)
        assert 0.0 <= psr <= 1.0


# ════════════════════════════════════════════════════════════════════
# MultiTFGate
# ════════════════════════════════════════════════════════════════════

class TestMultiTFGate:
    def _fv(self, **kwargs) -> FeatureVector:
        defaults = dict(
            symbol="BTCUSDT",
            timestamp=int(time.time() * 1000),
            close=100.0,
            ema_50=95.0,        # 4h trend up (close > 4h EMA50)
            ema_50_daily=90.0,  # daily trend up
            trend_alignment=0.8,
        )
        defaults.update(kwargs)
        return FeatureVector(**defaults)

    def test_aligned_trend_buy_approves(self):
        gate = MultiTFGate()
        d = gate.check(Direction.BUY, self._fv(), StrategyType.TREND_FOLLOWING)
        assert d.approved

    def test_4h_disagreement_blocks(self):
        gate = MultiTFGate()
        # Close BELOW 4h EMA50 → 4h trend is down, but daily still up.
        d = gate.check(Direction.BUY, self._fv(close=90.0), StrategyType.TREND_FOLLOWING)
        assert not d.approved
        assert "4h_alignment" in d.reason

    def test_daily_disagreement_blocks(self):
        gate = MultiTFGate()
        d = gate.check(
            Direction.BUY,
            self._fv(close=92.0, ema_50_daily=95.0),
            StrategyType.TREND_FOLLOWING,
        )
        assert not d.approved

    def test_low_alignment_score_blocks(self):
        gate = MultiTFGate()
        d = gate.check(
            Direction.BUY,
            self._fv(trend_alignment=0.3),
            StrategyType.TREND_FOLLOWING,
        )
        assert not d.approved
        assert "trend_alignment_score" in d.reason

    def test_mean_reversion_bypasses(self):
        gate = MultiTFGate()
        d = gate.check(Direction.BUY, self._fv(close=80.0), StrategyType.MEAN_REVERSION)
        assert d.approved

    def test_sell_never_blocked(self):
        gate = MultiTFGate()
        d = gate.check(Direction.SELL, self._fv(close=70.0), StrategyType.TREND_FOLLOWING)
        assert d.approved

    def test_missing_data_fail_closed_default(self):
        gate = MultiTFGate()
        d = gate.check(
            Direction.BUY,
            self._fv(ema_50_daily=0.0),
            StrategyType.TREND_FOLLOWING,
        )
        assert not d.approved

    def test_missing_data_fail_open(self):
        gate = MultiTFGate(MultiTFGateConfig(fail_closed_on_missing_data=False))
        d = gate.check(
            Direction.BUY,
            self._fv(ema_50_daily=0.0),
            StrategyType.TREND_FOLLOWING,
        )
        assert d.approved


# ════════════════════════════════════════════════════════════════════
# RiskSentinel guard integration
# ════════════════════════════════════════════════════════════════════

class TestRiskSentinelGuards:
    def _signal(self, **overrides) -> Signal:
        defaults = dict(
            symbol="BTCUSDT",
            direction=Direction.BUY,
            strategy_name="test",
            confidence=0.9,
            suggested_quantity=0.001,
            stop_loss_price=58000.0,
            take_profit_price=64000.0,
            reason="test signal",
            timestamp=int(time.time() * 1000),
        )
        defaults.update(overrides)
        return Signal(**defaults)

    def _sentinel(self) -> RiskSentinel:
        bus = EventBus()
        sm = RiskStateMachine(event_bus=bus, max_daily_loss=500.0)
        limits = RiskLimits(
            max_daily_loss_usd=500.0,
            max_open_positions=10,
            max_total_exposure_pct=200.0,  # not the limit under test here
            max_order_usd=10000.0,
            min_order_usd=1.0,
            max_trades_per_hour=100,
            max_daily_trades=100,
            min_trade_interval_sec=0,
            max_loss_per_trade_pct=10.0,
            max_risk_per_trade_pct=10.0,
            min_rr_ratio=1.0,
        )
        return RiskSentinel(limits=limits, state_machine=sm)

    def test_drawdown_breaker_blocks_buy(self):
        rs = self._sentinel()
        b = DrawdownBreaker(DrawdownThresholds(daily_pct=0.05))
        # Pre-trip: peak at $1000, current at $900 (10% DD).
        b.update(1000.0)
        b.update(900.0)
        assert not b.allows_new_entry()
        rs.attach_drawdown_breaker(b)

        result = rs.check_signal(
            signal=self._signal(),
            daily_pnl=0.0,
            open_positions_count=0,
            total_exposure_pct=0.0,
            balance=900.0,
            current_market_price=60000.0,
        )
        assert not result.approved
        assert "Drawdown" in result.reason

    def test_drawdown_breaker_allows_sell(self):
        rs = self._sentinel()
        b = DrawdownBreaker(DrawdownThresholds(daily_pct=0.05))
        b.update(1000.0); b.update(900.0)
        rs.attach_drawdown_breaker(b)

        result = rs.check_signal(
            signal=self._signal(direction=Direction.SELL),
            daily_pnl=0.0,
            open_positions_count=1,
            total_exposure_pct=10.0,
            balance=900.0,
            current_market_price=60000.0,
        )
        # Sell should bypass the drawdown breaker (closing is always allowed).
        assert result.approved or "Drawdown" not in (result.reason or "")

    def test_correlation_guard_blocks_correlated_entry(self):
        rs = self._sentinel()
        cfg = CorrelationConfig(threshold=0.7, min_observations=10, max_cluster_size=1)
        rs.attach_correlation_guard(CorrelationGuard(cfg))

        # Two perfectly correlated price series.
        drifts = [0.01, -0.01, 0.02, -0.005, 0.01, -0.02, 0.03, -0.01] * 5
        prices_btc = [60000.0]
        prices_eth = [3000.0]
        for d in drifts:
            prices_btc.append(prices_btc[-1] * (1 + d))
            prices_eth.append(prices_eth[-1] * (1 + d))

        result = rs.check_signal(
            signal=self._signal(symbol="ETHUSDT"),
            daily_pnl=0.0,
            open_positions_count=1,
            total_exposure_pct=5.0,
            balance=10000.0,
            current_market_price=3000.0,
            open_symbols={"BTCUSDT"},
            price_history={"BTCUSDT": prices_btc, "ETHUSDT": prices_eth},
        )
        assert not result.approved
        assert "luster" in result.reason or "ffective" in result.reason

    def test_exposure_cap_guard_blocks_oversize_class(self):
        rs = self._sentinel()
        rs.attach_exposure_cap_guard(ExposureCapGuard())

        # 50% L1 exposure — exceeds default 35% cap.
        result = rs.check_signal(
            signal=self._signal(suggested_quantity=0.0833),  # ≈ $5000 at 60k
            daily_pnl=0.0,
            open_positions_count=0,
            total_exposure_pct=0.0,
            balance=10000.0,
            current_market_price=60000.0,
            open_positions_exposure=[],
        )
        assert not result.approved
        assert "L1" in result.reason

    def test_no_guards_back_compat(self):
        # Without guards attached, behaviour matches legacy RiskSentinel.
        rs = self._sentinel()
        result = rs.check_signal(
            signal=self._signal(),
            daily_pnl=0.0,
            open_positions_count=0,
            total_exposure_pct=0.0,
            balance=10000.0,
            current_market_price=60000.0,
        )
        assert result.approved


# ════════════════════════════════════════════════════════════════════
# Strategy classification
# ════════════════════════════════════════════════════════════════════

class TestStrategyClassification:
    def test_known_strategies_mapped(self):
        assert classify_strategy("ema_crossover_rsi") == StrategyType.TREND_FOLLOWING
        assert classify_strategy("macd_divergence") == StrategyType.TREND_FOLLOWING
        assert classify_strategy("bollinger_breakout") == StrategyType.BREAKOUT
        assert classify_strategy("mean_reversion") == StrategyType.MEAN_REVERSION
        assert classify_strategy("dca_bot") == StrategyType.NEUTRAL
        assert classify_strategy("grid_trading") == StrategyType.NEUTRAL

    def test_unknown_defaults_to_trend(self):
        # Unknown → strict gate (TREND_FOLLOWING). Better to false-positive
        # block an unknown than false-negative let it through.
        assert classify_strategy("brand_new_strat") == StrategyType.TREND_FOLLOWING


# ════════════════════════════════════════════════════════════════════
# RegimeGate
# ════════════════════════════════════════════════════════════════════

class TestRegimeGate:
    def _fv(self, regime: str) -> FeatureVector:
        return FeatureVector(
            symbol="BTCUSDT",
            timestamp=int(time.time() * 1000),
            close=100.0,
            ema_50=95.0,
            ema_50_daily=90.0,
            trend_alignment=0.8,
            market_regime=regime,
        )

    def test_trend_blocked_in_trending_down(self):
        rg = RegimeGate()
        d = rg.check("ema_crossover_rsi", self._fv("trending_down"))
        assert not d.approved
        assert "trending_down" in d.reason

    def test_trend_blocked_in_volatile(self):
        rg = RegimeGate()
        d = rg.check("ema_crossover_rsi", self._fv("volatile"))
        assert not d.approved

    def test_trend_blocked_in_transitioning(self):
        rg = RegimeGate()
        d = rg.check("macd_divergence", self._fv("transitioning"))
        assert not d.approved

    def test_trend_allowed_in_trending_up(self):
        rg = RegimeGate()
        d = rg.check("ema_crossover_rsi", self._fv("trending_up"))
        assert d.approved

    def test_mean_reversion_blocked_in_trending_up(self):
        # Don't fade a clear uptrend.
        rg = RegimeGate()
        d = rg.check("mean_reversion", self._fv("trending_up"))
        assert not d.approved

    def test_mean_reversion_allowed_in_trending_down(self):
        # Contrarian opportunity.
        rg = RegimeGate()
        d = rg.check("mean_reversion", self._fv("trending_down"))
        assert d.approved

    def test_breakout_blocked_only_in_transitioning(self):
        rg = RegimeGate()
        assert not rg.check("bollinger_breakout", self._fv("transitioning")).approved
        assert rg.check("bollinger_breakout", self._fv("volatile")).approved
        assert rg.check("bollinger_breakout", self._fv("trending_up")).approved

    def test_neutral_strategies_never_blocked(self):
        rg = RegimeGate()
        for r in ("trending_up", "trending_down", "sideways", "volatile", "transitioning"):
            assert rg.check("dca_bot", self._fv(r)).approved
            assert rg.check("grid_trading", self._fv(r)).approved

    def test_unknown_regime_allows(self):
        rg = RegimeGate()
        d = rg.check("ema_crossover_rsi", self._fv("unknown"))
        assert d.approved


# ════════════════════════════════════════════════════════════════════
# NewsCooldownGuard
# ════════════════════════════════════════════════════════════════════

class TestNewsCooldownGuard:
    def _fv_critical_bearish(self, category: str = "security") -> FeatureVector:
        return FeatureVector(
            symbol="BTCUSDT",
            timestamp=int(time.time() * 1000),
            close=100.0,
            news_critical_alert=True,
            news_composite_score=-0.5,
            news_signal_strength=0.7,
            news_dominant_category=category,
        )

    def _fv_quiet(self) -> FeatureVector:
        return FeatureVector(
            symbol="BTCUSDT",
            timestamp=int(time.time() * 1000),
            close=100.0,
            news_critical_alert=False,
            news_composite_score=0.0,
            news_signal_strength=0.0,
        )

    def test_no_event_allows(self):
        g = NewsCooldownGuard()
        assert g.check(self._fv_quiet()).approved

    def test_critical_event_blocks(self):
        clock = [1_000_000.0]
        g = NewsCooldownGuard(time_provider=lambda: clock[0])
        # First call: records event AND blocks (cooldown starts immediately).
        d = g.check(self._fv_critical_bearish("security"))
        assert not d.approved
        assert d.cooldown_remaining_sec > 0
        # Subsequent quiet calls within cooldown still blocked.
        clock[0] += 60   # 1 minute later
        d2 = g.check(self._fv_quiet())
        assert not d2.approved

    def test_cooldown_expires(self):
        clock = [1_000_000.0]
        g = NewsCooldownGuard(
            NewsCooldownConfig(security_cooldown_sec=300),
            time_provider=lambda: clock[0],
        )
        g.check(self._fv_critical_bearish("security"))
        # Advance past cooldown window
        clock[0] += 301
        d = g.check(self._fv_quiet())
        assert d.approved

    def test_weak_news_does_not_trigger(self):
        # Bearish but weak strength → no cooldown.
        f = FeatureVector(
            symbol="BTCUSDT",
            timestamp=int(time.time() * 1000),
            close=100.0,
            news_critical_alert=True,
            news_composite_score=-0.5,
            news_signal_strength=0.10,  # below min 0.30
        )
        g = NewsCooldownGuard()
        assert g.check(f).approved

    def test_force_reset(self):
        g = NewsCooldownGuard()
        g.check(self._fv_critical_bearish("security"))
        assert not g.check(self._fv_quiet()).approved
        g.force_reset()
        assert g.check(self._fv_quiet()).approved

    def test_regulatory_longer_cooldown_than_security(self):
        # Default config: regulatory > security in duration.
        clock = [1_000_000.0]
        g = NewsCooldownGuard(time_provider=lambda: clock[0])
        g.check(self._fv_critical_bearish("regulatory"))
        # Advance past security-cooldown but not regulatory.
        clock[0] += 5 * 3600  # 5h
        d = g.check(self._fv_quiet())
        assert not d.approved  # regulatory cooldown is 12h


# ════════════════════════════════════════════════════════════════════
# RiskSentinel: integrated multi-TF + regime + news cooldown checks
# ════════════════════════════════════════════════════════════════════

class TestRiskSentinelEntryGuards:
    def _signal(self, strategy="ema_crossover_rsi", direction=Direction.BUY, **fv_overrides):
        defaults = dict(
            symbol="BTCUSDT",
            timestamp=int(time.time() * 1000),
            close=60000.0,
            ema_50=58000.0,
            ema_50_daily=55000.0,
            trend_alignment=0.8,
            market_regime="trending_up",
        )
        defaults.update(fv_overrides)
        fv = FeatureVector(**defaults)
        return Signal(
            symbol="BTCUSDT",
            direction=direction,
            strategy_name=strategy,
            confidence=0.95,
            suggested_quantity=0.001,
            stop_loss_price=58000.0,
            take_profit_price=64000.0,
            reason="test",
            timestamp=int(time.time() * 1000),
            features=fv,
        )

    def _sentinel(self) -> RiskSentinel:
        bus = EventBus()
        sm = RiskStateMachine(event_bus=bus, max_daily_loss=500.0)
        limits = RiskLimits(
            max_daily_loss_usd=500.0,
            max_open_positions=10,
            max_total_exposure_pct=200.0,
            max_order_usd=10000.0,
            min_order_usd=1.0,
            max_trades_per_hour=100,
            max_daily_trades=100,
            min_trade_interval_sec=0,
            max_loss_per_trade_pct=10.0,
            max_risk_per_trade_pct=10.0,
            min_rr_ratio=1.0,
        )
        return RiskSentinel(limits=limits, state_machine=sm)

    def test_multi_tf_blocks_against_daily_trend(self):
        rs = self._sentinel()
        rs.attach_multi_tf_gate(MultiTFGate())
        sig = self._signal(close=50000.0, ema_50_daily=55000.0)  # below daily EMA50
        result = rs.check_signal(
            signal=sig, daily_pnl=0.0, open_positions_count=0,
            total_exposure_pct=0.0, balance=10000.0,
            current_market_price=50000.0,
        )
        assert not result.approved
        assert "Multi-TF" in result.reason

    def test_regime_gate_blocks_trend_in_trending_down(self):
        rs = self._sentinel()
        rs.attach_regime_gate(RegimeGate())
        sig = self._signal(market_regime="trending_down")
        result = rs.check_signal(
            signal=sig, daily_pnl=0.0, open_positions_count=0,
            total_exposure_pct=0.0, balance=10000.0,
            current_market_price=60000.0,
        )
        assert not result.approved
        assert "Regime gate" in result.reason

    def test_news_cooldown_blocks_after_critical_event(self):
        rs = self._sentinel()
        rs.attach_news_cooldown(NewsCooldownGuard())
        sig = self._signal(
            news_critical_alert=True,
            news_composite_score=-0.5,
            news_signal_strength=0.7,
            news_dominant_category="security",
        )
        result = rs.check_signal(
            signal=sig, daily_pnl=0.0, open_positions_count=0,
            total_exposure_pct=0.0, balance=10000.0,
            current_market_price=60000.0,
        )
        assert not result.approved
        assert "News cooldown" in result.reason

    def test_aligned_trend_signal_passes_all_entry_guards(self):
        rs = self._sentinel()
        rs.attach_multi_tf_gate(MultiTFGate())
        rs.attach_regime_gate(RegimeGate())
        rs.attach_news_cooldown(NewsCooldownGuard())
        sig = self._signal()  # all happy defaults
        result = rs.check_signal(
            signal=sig, daily_pnl=0.0, open_positions_count=0,
            total_exposure_pct=0.0, balance=10000.0,
            current_market_price=60000.0,
        )
        assert result.approved

    def test_sell_bypasses_all_entry_guards(self):
        rs = self._sentinel()
        rs.attach_multi_tf_gate(MultiTFGate())
        rs.attach_regime_gate(RegimeGate())
        rs.attach_news_cooldown(NewsCooldownGuard())
        # Worst-case features (would block any BUY) but it's a SELL.
        sig = self._signal(
            direction=Direction.SELL,
            close=50000.0, ema_50_daily=55000.0,
            market_regime="trending_down",
            news_critical_alert=True,
            news_composite_score=-0.5,
            news_signal_strength=0.7,
        )
        result = rs.check_signal(
            signal=sig, daily_pnl=0.0, open_positions_count=0,
            total_exposure_pct=10.0, balance=10000.0,
            current_market_price=50000.0,
        )
        assert result.approved


# ════════════════════════════════════════════════════════════════════
# LiquidityGate
# ════════════════════════════════════════════════════════════════════

class TestLiquidityGate:
    def _fv(self, **overrides) -> FeatureVector:
        defaults = dict(
            symbol="BTCUSDT",
            timestamp=int(time.time() * 1000),
            close=100.0,
            volume_ratio=1.0,
        )
        defaults.update(overrides)
        return FeatureVector(**defaults)

    def test_normal_volume_approves_buy(self):
        g = LiquidityGate()
        d = g.check(Direction.BUY, self._fv(volume_ratio=1.0), 100.0, 5000.0)
        assert d.approved

    def test_thin_volume_blocks_buy(self):
        g = LiquidityGate(LiquidityGateConfig(min_volume_ratio_buy=0.5))
        d = g.check(Direction.BUY, self._fv(volume_ratio=0.3))
        assert not d.approved
        assert "thin" in d.reason.lower()

    def test_oversize_order_blocks(self):
        g = LiquidityGate(LiquidityGateConfig(max_pct_of_recent_volume=0.05))
        # 200 / 1000 = 20% > 5% cap.
        d = g.check(Direction.BUY, self._fv(), 200.0, 1000.0)
        assert not d.approved
        assert "notional" in d.reason

    def test_sell_never_blocked(self):
        g = LiquidityGate(LiquidityGateConfig(min_volume_ratio_buy=0.95))
        d = g.check(Direction.SELL, self._fv(volume_ratio=0.1))
        assert d.approved

    def test_session_window_block(self):
        # Fix UTC hour to 03:00 via injected clock.
        target = datetime(2026, 4, 17, 3, 0, 0, tzinfo=timezone.utc).timestamp()
        g = LiquidityGate(
            LiquidityGateConfig(blocked_utc_hours=(2, 3, 4, 5)),
            time_provider=lambda: target,
        )
        d = g.check(Direction.BUY, self._fv(volume_ratio=1.0))
        assert not d.approved
        assert "UTC hour" in d.reason

    def test_session_window_allows_outside(self):
        target = datetime(2026, 4, 17, 14, 0, 0, tzinfo=timezone.utc).timestamp()
        g = LiquidityGate(
            LiquidityGateConfig(blocked_utc_hours=(2, 3, 4, 5)),
            time_provider=lambda: target,
        )
        assert g.check(Direction.BUY, self._fv(volume_ratio=1.0)).approved


# ════════════════════════════════════════════════════════════════════
# Stale-data gate
# ════════════════════════════════════════════════════════════════════

class TestStaleDataGate:
    def test_fresh_data_approves_buy(self):
        g = StaleDataGate(StaleDataGateConfig(max_age_sec=90.0))
        d = g.check(Direction.BUY, data_age_sec=5.0)
        assert d.approved
        assert d.data_age_sec == 5.0

    def test_stale_data_blocks_buy(self):
        g = StaleDataGate(StaleDataGateConfig(max_age_sec=90.0))
        d = g.check(Direction.BUY, data_age_sec=120.0)
        assert not d.approved
        assert "Stale-data" in d.reason
        assert d.data_age_sec == 120.0

    def test_missing_age_blocks_buy_fail_closed(self):
        g = StaleDataGate()
        assert not g.check(Direction.BUY, data_age_sec=None).approved
        assert not g.check(Direction.BUY, data_age_sec=float("inf")).approved

    def test_sell_always_allowed_even_when_stale(self):
        g = StaleDataGate(StaleDataGateConfig(max_age_sec=30.0))
        assert g.check(Direction.SELL, data_age_sec=999.0).approved
        assert g.check(Direction.SELL, data_age_sec=None).approved

    def test_boundary_equal_to_max_approves(self):
        g = StaleDataGate(StaleDataGateConfig(max_age_sec=90.0))
        assert g.check(Direction.BUY, data_age_sec=90.0).approved

    def test_integration_with_risk_sentinel_blocks_buy(self):
        """Stale gate attached to RiskSentinel rejects BUY via check_signal."""
        bus = EventBus()
        sm = RiskStateMachine(event_bus=bus, max_daily_loss=500.0)
        limits = RiskLimits(
            max_daily_loss_usd=500.0, max_open_positions=10,
            max_total_exposure_pct=200.0, max_order_usd=10000.0,
            min_order_usd=1.0, max_trades_per_hour=100,
            max_daily_trades=100, min_trade_interval_sec=0,
            max_loss_per_trade_pct=10.0, max_risk_per_trade_pct=10.0,
            min_rr_ratio=1.0,
        )
        rs = RiskSentinel(limits=limits, state_machine=sm)
        rs.attach_stale_data_gate(StaleDataGate(StaleDataGateConfig(max_age_sec=60.0)))
        sig = Signal(
            symbol="BTCUSDT", direction=Direction.BUY,
            strategy_name="ema_crossover_rsi", confidence=0.95,
            suggested_quantity=0.001, stop_loss_price=58000.0,
            take_profit_price=64000.0, reason="test",
            timestamp=int(time.time() * 1000),
            features=FeatureVector(
                symbol="BTCUSDT", timestamp=int(time.time() * 1000),
                close=60000.0,
            ),
        )
        result = rs.check_signal(
            signal=sig, daily_pnl=0.0, open_positions_count=0,
            total_exposure_pct=0.0, balance=10000.0,
            current_market_price=60000.0,
            market_data_age_sec=200.0,  # way over the 60s cap
        )
        assert not result.approved
        assert "Stale-data" in result.reason

    def test_integration_sell_bypass_via_risk_sentinel(self):
        """SELL passes even when data is stale and gate is attached."""
        bus = EventBus()
        sm = RiskStateMachine(event_bus=bus, max_daily_loss=500.0)
        limits = RiskLimits(
            max_daily_loss_usd=500.0, max_open_positions=10,
            max_total_exposure_pct=200.0, max_order_usd=10000.0,
            min_order_usd=1.0, max_trades_per_hour=100,
            max_daily_trades=100, min_trade_interval_sec=0,
            max_loss_per_trade_pct=10.0, max_risk_per_trade_pct=10.0,
            min_rr_ratio=1.0,
        )
        rs = RiskSentinel(limits=limits, state_machine=sm)
        rs.attach_stale_data_gate(StaleDataGate(StaleDataGateConfig(max_age_sec=60.0)))
        sig = Signal(
            symbol="BTCUSDT", direction=Direction.SELL,
            strategy_name="ema_crossover_rsi", confidence=0.95,
            suggested_quantity=0.001, stop_loss_price=0.0,
            take_profit_price=0.0, reason="test",
            timestamp=int(time.time() * 1000),
            features=FeatureVector(
                symbol="BTCUSDT", timestamp=int(time.time() * 1000),
                close=60000.0,
            ),
        )
        result = rs.check_signal(
            signal=sig, daily_pnl=0.0, open_positions_count=0,
            total_exposure_pct=0.0, balance=10000.0,
            current_market_price=60000.0,
            market_data_age_sec=9999.0,
        )
        assert result.approved


# ════════════════════════════════════════════════════════════════════
# Position-sizer correlation lookup (rip-out of hardcoded BTC/ETH)
# ════════════════════════════════════════════════════════════════════

class TestPositionSizerCorrLookup:
    def test_legacy_static_btc_eth_still_works(self):
        # No corr_lookup → fallback to static cluster: holding ETH while
        # opening BTC should hit the 0.7 penalty.
        f_with = correlation_factor("BTCUSDT", ["ETHUSDT"])
        f_without = correlation_factor("BTCUSDT", ["DOGEUSDT"])
        assert f_with == 0.70
        assert f_without == 1.0

    def test_injected_corr_lookup_wins_over_static(self):
        # Inject a lookup that says NOTHING is correlated; legacy cluster ignored.
        f = correlation_factor(
            "BTCUSDT", ["ETHUSDT"], corr_lookup=lambda a, b: False,
        )
        assert f == 1.0

    def test_injected_corr_lookup_detects_arbitrary_pair(self):
        # Caller marks DOGE/PEPE as correlated — sizer respects it.
        def _lookup(a: str, b: str) -> bool:
            return {a, b} == {"DOGEUSDT", "PEPEUSDT"}
        f = correlation_factor("DOGEUSDT", ["PEPEUSDT"], corr_lookup=_lookup)
        assert f == 0.70

    def test_calculate_position_size_uses_corr_lookup(self):
        # Verify SizingInput.corr_lookup propagates into sizing.
        result = calculate_position_size(SizingInput(
            balance=10000.0, price=60000.0, atr=600.0,
            win_rate=0.55, avg_win_pct=3.0, avg_loss_pct=2.0,
            sample_size=50, regime_adx=30.0,
            max_position_pct=20.0, max_order_usd=10000.0,
            symbol="BTCUSDT",
            open_symbols=["DOGEUSDT"],
            corr_lookup=lambda a, b: True,  # everything is "correlated"
            stop_loss_pct=2.0,
        ))
        # With penalty applied, budget should be lower than the no-correlation case.
        baseline = calculate_position_size(SizingInput(
            balance=10000.0, price=60000.0, atr=600.0,
            win_rate=0.55, avg_win_pct=3.0, avg_loss_pct=2.0,
            sample_size=50, regime_adx=30.0,
            max_position_pct=20.0, max_order_usd=10000.0,
            symbol="BTCUSDT",
            open_symbols=[],
            stop_loss_pct=2.0,
        ))
        assert result.budget_usd < baseline.budget_usd


# ════════════════════════════════════════════════════════════════════
# Monte Carlo
# ════════════════════════════════════════════════════════════════════

class TestMonteCarlo:
    def _profitable_returns(self) -> list[float]:
        # 60% win rate, +1.5% wins, -1.0% losses → positive expectancy.
        return ([0.015] * 6 + [-0.010] * 4) * 5  # 50 trades

    def _loss_returns(self) -> list[float]:
        # 40% win rate, +1.0% / -1.5% → strongly negative expectancy.
        return ([0.010] * 4 + [-0.015] * 6) * 5

    def test_report_for_profitable_strategy(self):
        mc = MonteCarloAnalyser(MonteCarloConfig(n_simulations=200, seed=42))
        report = mc.analyse(self._profitable_returns())
        assert report.n_simulations == 200
        assert report.expected_return_pct > 0
        # Lower bound of CI may still be positive or near zero — just sanity-check.
        assert report.return_upper_bound_pct > report.return_lower_bound_pct

    def test_report_for_losing_strategy(self):
        mc = MonteCarloAnalyser(MonteCarloConfig(n_simulations=200, seed=42))
        report = mc.analyse(self._loss_returns())
        assert report.expected_return_pct < 0

    def test_probability_of_ruin_high_when_threshold_low(self):
        # Set ruin = 1% — almost any series breaches it.
        mc = MonteCarloAnalyser(MonteCarloConfig(
            n_simulations=200, seed=42, ruin_threshold_pct=1.0
        ))
        report = mc.analyse(self._loss_returns())
        assert report.probability_of_ruin > 0.5

    def test_probability_of_ruin_low_when_threshold_high(self):
        mc = MonteCarloAnalyser(MonteCarloConfig(
            n_simulations=200, seed=42, ruin_threshold_pct=99.0
        ))
        report = mc.analyse(self._profitable_returns())
        assert report.probability_of_ruin < 0.05

    def test_seed_reproducibility(self):
        rs1 = MonteCarloAnalyser(MonteCarloConfig(n_simulations=100, seed=7)).analyse(self._profitable_returns())
        rs2 = MonteCarloAnalyser(MonteCarloConfig(n_simulations=100, seed=7)).analyse(self._profitable_returns())
        assert rs1.expected_return_pct == rs2.expected_return_pct
        assert rs1.expected_max_dd_pct == rs2.expected_max_dd_pct

    def test_block_bootstrap_runs(self):
        mc = MonteCarloAnalyser(MonteCarloConfig(n_simulations=100, seed=1))
        report = mc.analyse(self._profitable_returns(), block_size=5)
        assert report.n_simulations == 100

    def test_empty_returns_yields_empty_report(self):
        mc = MonteCarloAnalyser(MonteCarloConfig(n_simulations=10))
        report = mc.analyse([])
        assert report.n_simulations == 0

    def test_format_report_runs(self):
        mc = MonteCarloAnalyser(MonteCarloConfig(n_simulations=50, seed=1))
        report = mc.analyse(self._profitable_returns())
        text = MonteCarloAnalyser.format_report(report)
        assert "Monte Carlo" in text
        assert "Probability of ruin" in text


# ════════════════════════════════════════════════════════════════════
# Observability: EventLog, DecisionTracer, evaluate_with_trace
# ════════════════════════════════════════════════════════════════════

class TestEventLog:
    def test_emit_in_memory(self, tmp_path):
        log = EventLog(path=None)  # buffer-only, no file
        log.emit("signal_decision", symbol="BTCUSDT", outcome="approved")
        recent = log.recent_events()
        assert len(recent) == 1
        assert recent[0]["type"] == "signal_decision"
        assert recent[0]["symbol"] == "BTCUSDT"
        assert "ts" in recent[0]

    def test_emit_to_file_jsonl(self, tmp_path):
        path = tmp_path / "events.jsonl"
        log = EventLog(path=path)
        log.emit("a", x=1)
        log.emit("b", y="hello")
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        import json as _json
        rec0, rec1 = _json.loads(lines[0]), _json.loads(lines[1])
        assert rec0["type"] == "a" and rec0["x"] == 1
        assert rec1["type"] == "b" and rec1["y"] == "hello"

    def test_recent_filter_by_type(self):
        log = EventLog(path=None)
        log.emit("a"); log.emit("b"); log.emit("a")
        assert len(log.recent_events("a")) == 2
        assert len(log.recent_events("b")) == 1

    def test_buffer_bounded(self):
        log = EventLog(path=None, in_memory_buffer=5)
        for i in range(20):
            log.emit("e", i=i)
        assert len(log.recent_events()) == 5
        # Last event preserved.
        assert log.recent_events()[-1]["i"] == 19

    def test_time_event_records_duration(self):
        log = EventLog(path=None)
        with log.time_event("work", task="x"):
            pass
        ev = log.recent_events()[-1]
        assert ev["type"] == "work"
        assert "duration_ms" in ev
        assert ev["task"] == "x"

    def test_rotation_triggers(self, tmp_path):
        path = tmp_path / "events.jsonl"
        log = EventLog(path=path, max_bytes=100, backup_count=2)
        for _ in range(20):
            log.emit("big", payload="x" * 50)
        # After several rotations the .1/.2 backups should exist.
        assert path.exists() or path.with_suffix(".jsonl.1").exists()


class TestDecisionTracer:
    def test_gate_timer_records_approval(self):
        trace = DecisionTrace()
        with GateTimer(trace, "alpha") as t:
            t.record(approved=True, reason="ok")
        assert len(trace.gates) == 1
        v = trace.gates[0]
        assert v.gate == "alpha"
        assert v.outcome == GateOutcome.APPROVED
        assert v.latency_us >= 0

    def test_gate_timer_records_rejection(self):
        trace = DecisionTrace()
        with GateTimer(trace, "beta") as t:
            t.record(approved=False, reason="bad", payload={"x": 1})
        v = trace.gates[0]
        assert v.outcome == GateOutcome.REJECTED
        assert v.payload == {"x": 1}

    def test_gate_timer_skipped(self):
        trace = DecisionTrace()
        with GateTimer(trace, "gamma") as t:
            t.skipped("not configured")
        assert trace.gates[0].outcome == GateOutcome.SKIPPED

    def test_gate_timer_swallows_exception(self):
        trace = DecisionTrace()
        with GateTimer(trace, "boom") as t:
            raise RuntimeError("kaboom")
        assert len(trace.gates) == 1
        assert trace.gates[0].outcome == GateOutcome.ERROR
        assert "kaboom" in trace.gates[0].reason

    def test_first_rejection_helper(self):
        trace = DecisionTrace()
        for name, ok in [("a", True), ("b", False), ("c", False)]:
            with GateTimer(trace, name) as t:
                t.record(approved=ok)
        first = trace.first_rejection()
        assert first is not None and first.gate == "b"
        all_rej = trace.all_rejections()
        assert [v.gate for v in all_rej] == ["b", "c"]

    def test_to_dict_serialises(self):
        trace = DecisionTrace(symbol="BTC", strategy="test", direction="BUY")
        with GateTimer(trace, "g") as t:
            t.record(False, "nope")
        d = trace.to_dict()
        assert d["symbol"] == "BTC"
        assert d["gates"][0]["gate"] == "g"
        assert d["gates"][0]["outcome"] == "rejected"

    def test_feature_snapshot_dict_handles_none(self):
        assert feature_snapshot_dict(None) == {}

    def test_feature_snapshot_dict_extracts_floats(self):
        fv = FeatureVector(
            symbol="BTC", timestamp=1, close=60_000.5,
            ema_50=58_000.0, rsi_14=55.5, adx=25.0,
            volume_ratio=1.2, market_regime="trending_up",
        )
        snap = feature_snapshot_dict(fv)
        assert snap["close"] == 60_000.5
        assert snap["rsi_14"] == 55.5
        assert snap["market_regime"] == "trending_up"


class TestEvaluateWithTrace:
    def _signal(self, **fv):
        defaults = dict(
            symbol="BTCUSDT", timestamp=int(time.time() * 1000),
            close=60000.0, ema_50=58000.0, ema_50_daily=55000.0,
            trend_alignment=0.8, market_regime="trending_up",
        )
        defaults.update(fv)
        features = FeatureVector(**defaults)
        return Signal(
            symbol="BTCUSDT", direction=Direction.BUY,
            strategy_name="ema_crossover_rsi",
            confidence=0.9, suggested_quantity=0.001,
            stop_loss_price=58000.0, take_profit_price=64000.0,
            reason="t", timestamp=int(time.time() * 1000),
            features=features,
        )

    def _sentinel(self):
        bus = EventBus()
        sm = RiskStateMachine(event_bus=bus, max_daily_loss=500.0)
        limits = RiskLimits(
            max_daily_loss_usd=500.0, max_open_positions=10,
            max_total_exposure_pct=200.0, max_order_usd=10000.0,
            min_order_usd=1.0, max_trades_per_hour=100,
            max_daily_trades=100, min_trade_interval_sec=0,
            max_loss_per_trade_pct=10.0, max_risk_per_trade_pct=10.0,
            min_rr_ratio=1.0,
        )
        return RiskSentinel(limits=limits, state_machine=sm)

    def test_trace_records_all_gates(self):
        rs = self._sentinel()
        rs.attach_multi_tf_gate(MultiTFGate())
        rs.attach_regime_gate(RegimeGate())
        rs.attach_news_cooldown(NewsCooldownGuard())
        result, trace = rs.evaluate_with_trace(
            signal=self._signal(),
            daily_pnl=0.0, open_positions_count=0,
            total_exposure_pct=0.0, balance=10000.0,
            current_market_price=60000.0,
        )
        assert result.approved
        gate_ids = [g.gate for g in trace.gates]
        # All 7 pre-checks + legacy_checks should be recorded
        assert "multi_tf" in gate_ids
        assert "regime" in gate_ids
        assert "news_cooldown" in gate_ids
        assert "legacy_checks" in gate_ids

    def test_short_circuit_stops_after_first_rejection(self):
        rs = self._sentinel()
        rs.attach_multi_tf_gate(MultiTFGate())
        rs.attach_regime_gate(RegimeGate())
        # multi_tf rejects (close < daily EMA50)
        sig = self._signal(close=50000.0, ema_50_daily=55000.0)
        result, trace = rs.evaluate_with_trace(
            signal=sig, daily_pnl=0.0, open_positions_count=0,
            total_exposure_pct=0.0, balance=10000.0,
            current_market_price=50000.0,
            shadow_mode=False,
        )
        assert not result.approved
        # short-circuit: regime gate should NOT have been evaluated
        gate_ids = [g.gate for g in trace.gates]
        assert "multi_tf" in gate_ids
        assert trace.first_rejection().gate == "multi_tf"
        # regime/news/etc. should not have been added after rejection
        rejected_idx = next(i for i, g in enumerate(trace.gates) if g.outcome == GateOutcome.REJECTED)
        assert rejected_idx == len(trace.gates) - 1

    def test_shadow_mode_runs_all_gates_after_rejection(self):
        rs = self._sentinel()
        rs.attach_multi_tf_gate(MultiTFGate())
        rs.attach_regime_gate(RegimeGate())
        rs.attach_news_cooldown(NewsCooldownGuard())
        sig = self._signal(
            close=50000.0, ema_50_daily=55000.0,
            market_regime="trending_down",  # regime would also block
        )
        result, trace = rs.evaluate_with_trace(
            signal=sig, daily_pnl=0.0, open_positions_count=0,
            total_exposure_pct=0.0, balance=10000.0,
            current_market_price=50000.0,
            shadow_mode=True,
        )
        assert not result.approved
        # Both gates evaluated (shadow mode); both record verdicts.
        rejections = trace.all_rejections()
        gate_names = [v.gate for v in rejections]
        assert "multi_tf" in gate_names
        assert "regime" in gate_names

    def test_trace_to_dict_serialisable(self):
        rs = self._sentinel()
        rs.attach_multi_tf_gate(MultiTFGate())
        result, trace = rs.evaluate_with_trace(
            signal=self._signal(),
            daily_pnl=0.0, open_positions_count=0,
            total_exposure_pct=0.0, balance=10000.0,
            current_market_price=60000.0,
        )
        d = trace.to_dict()
        # Must round-trip through JSON.
        import json as _json
        rt = _json.loads(_json.dumps(d, default=str))
        assert rt["symbol"] == "BTCUSDT"
        assert "gates" in rt
        assert "feature_snapshot" in rt
