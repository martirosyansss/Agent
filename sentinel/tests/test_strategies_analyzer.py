"""Тесты Phase 11-15 — Strategies, Analyzer, Live Executor."""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.models import (
    Direction, FeatureVector, MarketRegime, MarketRegimeType, Signal, StrategyTrade,
)


# ══════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════

def make_fv(**overrides) -> FeatureVector:
    """Создать FeatureVector с разумными дефолтами."""
    defaults = dict(
        timestamp=int(time.time() * 1000), symbol="BTCUSDT",
        ema_9=100.0, ema_21=99.0, ema_50=97.0, adx=30.0,
        macd=0.5, macd_signal=0.3, macd_histogram=0.2,
        rsi_14=50.0, stoch_rsi=0.5,
        bb_upper=105.0, bb_middle=100.0, bb_lower=95.0, bb_bandwidth=0.10,
        atr=2.0, volume=1000.0, volume_sma_20=800.0, volume_ratio=1.25,
        obv=50000.0, close=100.0, momentum=0.5, spread=0.01,
        price_change_1m=0.1, price_change_5m=0.3, price_change_15m=-0.5,
        market_regime="trending_up",
    )
    defaults.update(overrides)
    return FeatureVector(**defaults)


def make_trade(
    strategy: str = "ema_crossover_rsi",
    pnl: float = 1.0,
    is_win: bool = True,
    regime: str = "trending_up",
    hour: int = 14,
    day: int = 2,
    rsi: float = 45.0,
    adx: float = 30.0,
    vol_ratio: float = 1.5,
    confidence: float = 0.80,
) -> StrategyTrade:
    return StrategyTrade(
        trade_id=f"t_{int(time.time()*1000)}_{id(pnl)}",
        signal_id="sig_1", symbol="BTCUSDT",
        strategy_name=strategy, market_regime=regime,
        timestamp_open="2026-01-01T10:00", timestamp_close="2026-01-01T14:00",
        entry_price=100.0, exit_price=100.0 + pnl,
        quantity=0.01, pnl_usd=pnl, pnl_pct=pnl,
        is_win=is_win, confidence=confidence,
        hour_of_day=hour, day_of_week=day,
        rsi_at_entry=rsi, adx_at_entry=adx, volume_ratio_at_entry=vol_ratio,
        exit_reason="tp" if is_win else "sl",
        hold_duration_hours=4.0,
        max_drawdown_during_trade=0.5,
        max_profit_during_trade=abs(pnl),
        commission_usd=0.02,
    )


# ══════════════════════════════════════════════════════
# Phase 11: Market Regime
# ══════════════════════════════════════════════════════

class TestMarketRegime:
    def test_trending_up(self):
        from strategy.market_regime import detect_regime, reset_hysteresis
        reset_hysteresis()
        f = make_fv(ema_9=103, ema_21=101, ema_50=99, adx=30)
        # Hysteresis: call 3 times to confirm regime change
        detect_regime(f)
        detect_regime(f)
        r = detect_regime(f)
        assert r.regime == MarketRegimeType.TRENDING_UP

    def test_trending_down(self):
        from strategy.market_regime import detect_regime, reset_hysteresis
        reset_hysteresis()
        f = make_fv(ema_9=95, ema_21=97, ema_50=99, adx=28, close=94)
        detect_regime(f)
        detect_regime(f)
        r = detect_regime(f)
        assert r.regime == MarketRegimeType.TRENDING_DOWN

    def test_sideways(self):
        from strategy.market_regime import detect_regime, reset_hysteresis
        reset_hysteresis()
        f = make_fv(ema_9=100, ema_21=100.5, ema_50=99.5, adx=15, close=100,
                     bb_lower=96, bb_upper=104)
        detect_regime(f)
        detect_regime(f)
        r = detect_regime(f)
        assert r.regime == MarketRegimeType.SIDEWAYS

    def test_volatile(self):
        from strategy.market_regime import detect_regime, reset_hysteresis
        reset_hysteresis()
        # Use non-trending EMAs so volatile is not masked by trend detection
        f = make_fv(atr=5.0, bb_middle=100.0, ema_9=100, ema_21=101, ema_50=99, adx=15)
        detect_regime(f)
        r = detect_regime(f)
        assert r.regime == MarketRegimeType.VOLATILE

    def test_unknown(self):
        from strategy.market_regime import detect_regime, reset_hysteresis
        reset_hysteresis()
        f = make_fv(ema_9=100, ema_21=101, ema_50=99, adx=22,
                     close=110, bb_upper=105, atr=1.0)  # Close > upper BB, mild ADX
        r = detect_regime(f)
        assert r.regime == MarketRegimeType.UNKNOWN


# ══════════════════════════════════════════════════════
# Phase 11: Grid Trading
# ══════════════════════════════════════════════════════

class TestGridTrading:
    def test_grid_buy_signal(self):
        from strategy.grid_trading import GridTrading
        s = GridTrading()
        f = make_fv(close=96.0, bb_lower=95.0, bb_upper=105.0)
        sig = s.generate_signal(f, has_open_position=False)
        assert sig is not None
        assert sig.direction == Direction.BUY
        assert sig.strategy_name == "grid_trading"

    def test_grid_sell_tp(self):
        from strategy.grid_trading import GridTrading
        s = GridTrading()
        # First build grid
        f1 = make_fv(close=96.0, bb_lower=95.0, bb_upper=105.0)
        s.generate_signal(f1, has_open_position=False)
        # Now sell at profit — need >= min_profit_pct + commission (1.2% + 0.2% = 1.4%)
        f2 = make_fv(close=97.5, bb_lower=95.0, bb_upper=105.0)
        sig = s.generate_signal(f2, has_open_position=True, entry_price=96.0)
        assert sig is not None
        assert sig.direction == Direction.SELL

    def test_grid_no_signal_outside_range(self):
        from strategy.grid_trading import GridTrading
        s = GridTrading()
        f = make_fv(close=80.0, bb_lower=95.0, bb_upper=105.0)
        sig = s.generate_signal(f, has_open_position=False)
        assert sig is None  # Price too far below grid

    def test_grid_stop_loss(self):
        from strategy.grid_trading import GridTrading
        s = GridTrading()
        # Wide grid so SL price stays within safety range
        f1 = make_fv(close=92.0, bb_lower=85.0, bb_upper=115.0)
        s.generate_signal(f1, has_open_position=False)
        f2 = make_fv(close=89.0, bb_lower=85.0, bb_upper=115.0)
        sig = s.generate_signal(f2, has_open_position=True, entry_price=95.0)
        assert sig is not None
        assert sig.direction == Direction.SELL
        assert "SL" in sig.reason


# ══════════════════════════════════════════════════════
# Phase 11: Mean Reversion
# ══════════════════════════════════════════════════════

class TestMeanReversion:
    def test_buy_oversold(self):
        from strategy.mean_reversion import MeanReversion
        s = MeanReversion()
        f = make_fv(rsi_14=19.0, close=94.0, bb_lower=95.0, volume_ratio=2.1, ema_50=100.0)
        sig = s.generate_signal(f, has_open_position=False)
        assert sig is not None
        assert sig.direction == Direction.BUY
        assert sig.confidence >= 0.70  # grouped model: correlated indicators don't stack

    def test_no_buy_normal_rsi(self):
        from strategy.mean_reversion import MeanReversion
        s = MeanReversion()
        f = make_fv(rsi_14=50.0)
        sig = s.generate_signal(f, has_open_position=False)
        assert sig is None

    def test_sell_tp(self):
        from strategy.mean_reversion import MeanReversion
        s = MeanReversion()
        f = make_fv(close=109.0)  # +9% >= TP 8%
        sig = s.generate_signal(f, has_open_position=True, entry_price=100.0)
        assert sig is not None
        assert sig.direction == Direction.SELL

    def test_sell_sl(self):
        from strategy.mean_reversion import MeanReversion
        s = MeanReversion()
        f = make_fv(close=95.0)
        sig = s.generate_signal(f, has_open_position=True, entry_price=100.0)
        assert sig is not None
        assert sig.direction == Direction.SELL
        assert "SL" in sig.reason

    def test_sell_overbought(self):
        from strategy.mean_reversion import MeanReversion
        s = MeanReversion()
        f = make_fv(rsi_14=80.0, close=103.0)
        sig = s.generate_signal(f, has_open_position=True, entry_price=100.0)
        assert sig is not None
        assert sig.direction == Direction.SELL


# ══════════════════════════════════════════════════════
# Phase 11: Bollinger Breakout
# ══════════════════════════════════════════════════════

class TestBollingerBreakout:
    def test_buy_breakout(self):
        from strategy.bollinger_breakout import BollingerBreakout
        s = BollingerBreakout()
        f = make_fv(close=106.0, bb_upper=105.0, volume_ratio=2.0,
                     bb_bandwidth=0.03, adx=25, rsi_14=60, ema_9=101, ema_21=100)
        sig = s.generate_signal(f, has_open_position=False)
        assert sig is not None
        assert sig.direction == Direction.BUY

    def test_no_buy_low_volume(self):
        from strategy.bollinger_breakout import BollingerBreakout
        s = BollingerBreakout()
        f = make_fv(close=106.0, bb_upper=105.0, volume_ratio=0.8, adx=25)
        sig = s.generate_signal(f, has_open_position=False)
        assert sig is None

    def test_no_buy_low_adx(self):
        from strategy.bollinger_breakout import BollingerBreakout
        s = BollingerBreakout()
        f = make_fv(close=106.0, bb_upper=105.0, volume_ratio=2.0, adx=15)
        sig = s.generate_signal(f, has_open_position=False)
        assert sig is None

    def test_sell_tp(self):
        from strategy.bollinger_breakout import BollingerBreakout
        s = BollingerBreakout()
        f = make_fv(close=107.0)
        sig = s.generate_signal(f, has_open_position=True, entry_price=100.0)
        assert sig is not None
        assert sig.direction == Direction.SELL

    def test_trailing_stop(self):
        from strategy.bollinger_breakout import BollingerBreakout
        s = BollingerBreakout()
        sym = "BTCUSDT"
        # Simulate price going up then dropping
        # trailing_activate=3%, trailing_stop=1.5%
        s._max_price[sym] = 106.0  # Was at 106
        f = make_fv(close=104.3)  # pnl=4.3%>=3% activate, drop from 106→104.3=1.6%>=1.5% trail
        sig = s.generate_signal(f, has_open_position=True, entry_price=100.0)
        assert sig is not None
        assert sig.direction == Direction.SELL
        assert "trailing" in sig.reason


# ══════════════════════════════════════════════════════
# Phase 11: DCA Bot
# ══════════════════════════════════════════════════════

class TestDCABot:
    def test_buy_signal(self):
        from strategy.dca_bot import DCABot
        s = DCABot()
        f = make_fv(close=100.0)
        sig = s.generate_signal(f, has_open_position=False)
        assert sig is not None
        assert sig.direction == Direction.BUY
        assert sig.confidence == 0.80

    def test_no_buy_too_soon(self):
        from strategy.dca_bot import DCABot
        s = DCABot()
        f = make_fv(close=100.0)
        s.generate_signal(f, has_open_position=False)  # First buy
        sig = s.generate_signal(f, has_open_position=False)  # Too soon
        assert sig is None

    def test_sell_full_tp(self):
        from strategy.dca_bot import DCABot
        s = DCABot()
        f = make_fv(close=109.0)
        sig = s.generate_signal(f, has_open_position=True, entry_price=100.0)
        assert sig is not None
        assert sig.direction == Direction.SELL
        assert "full TP" in sig.reason

    def test_sell_trailing_stop(self):
        """DCA trailing stop replaces old broken partial TP."""
        from strategy.dca_bot import DCABot
        s = DCABot()
        sym = "BTCUSDT"
        # Max was 106 (+6% gain), now dropped to 103 (2.9% from max >= 2.5% trail)
        s._max_price[sym] = 106.0
        f = make_fv(close=103.0)
        sig = s.generate_signal(f, has_open_position=True, entry_price=100.0)
        assert sig is not None
        assert sig.direction == Direction.SELL
        assert "trailing" in sig.reason

    def test_dip_multiplier(self):
        from strategy.dca_bot import DCABot
        s = DCABot()
        f = make_fv(close=100.0, price_change_15m=-6.0)  # -6% dip
        sig = s.generate_signal(f, has_open_position=False)
        assert sig is not None
        assert "2.0x" in sig.reason

    def test_drawdown_stop(self):
        from strategy.dca_bot import DCABot
        s = DCABot()
        f = make_fv(close=84.0)  # -16% from entry
        sig = s.generate_signal(f, has_open_position=True, entry_price=100.0)
        assert sig is not None
        assert "drawdown" in sig.reason


# ══════════════════════════════════════════════════════
# Phase 11: MACD Divergence
# ══════════════════════════════════════════════════════

class TestMACDDivergence:
    def test_no_signal_without_history(self):
        from strategy.macd_divergence import MACDDivergence
        s = MACDDivergence()
        f = make_fv()
        sig = s.generate_signal(f, has_open_position=False)
        assert sig is None

    def test_bullish_divergence_buy(self):
        from strategy.macd_divergence import MACDDivergence
        s = MACDDivergence()
        sym = "BTCUSDT"
        # Swing-point divergence: swing_window=3, min_divergence_bars=10
        # Price swing lows: idx=4 (97), idx=16 (96) → lower-lower
        # MACD swing lows:  idx=4 (-0.6), idx=16 (-0.3) → higher-lower (divergence!)
        # generate_signal appends one more point, total becomes 21
        s._price_history[sym] = [
            105, 103, 100, 98, 97, 98, 100, 103, 105, 106,
            104, 103, 101, 99, 98, 97, 96, 97, 99, 101,
        ]
        s._macd_history[sym] = [
            -0.1, -0.2, -0.4, -0.5, -0.6, -0.5, -0.3, -0.1, 0.0, 0.1,
            -0.05, -0.1, -0.15, -0.2, -0.25, -0.28, -0.3, -0.28, -0.2, -0.1,
        ]
        f = make_fv(close=103.0, rsi_14=25.0, volume_ratio=1.5, macd_histogram=-0.05, macd=-0.05)
        sig = s.generate_signal(f, has_open_position=False)
        assert sig is not None
        assert sig.direction == Direction.BUY

    def test_sell_tp(self):
        from strategy.macd_divergence import MACDDivergence
        s = MACDDivergence()
        f = make_fv(close=108.0)
        sig = s.generate_signal(f, has_open_position=True, entry_price=100.0)
        assert sig is not None
        assert sig.direction == Direction.SELL

    def test_sell_sl(self):
        from strategy.macd_divergence import MACDDivergence
        s = MACDDivergence()
        f = make_fv(close=96.0)
        sig = s.generate_signal(f, has_open_position=True, entry_price=100.0)
        assert sig is not None
        assert sig.direction == Direction.SELL

    def test_no_buy_high_rsi(self):
        from strategy.macd_divergence import MACDDivergence
        s = MACDDivergence()
        sym = "BTCUSDT"
        s._price_history[sym] = [105, 103, 101, 100, 99]
        s._macd_history[sym] = [-0.5, -0.4, -0.3, -0.2, -0.1]
        f = make_fv(close=98.0, rsi_14=50.0, volume_ratio=1.5)  # RSI too high
        sig = s.generate_signal(f, has_open_position=False)
        assert sig is None


# ══════════════════════════════════════════════════════
# Phase 11: Strategy Selector
# ══════════════════════════════════════════════════════

class TestStrategySelector:
    def test_trending_up_allocations(self):
        from strategy.strategy_selector import get_allocations
        r = MarketRegime(regime=MarketRegimeType.TRENDING_UP, adx=30)
        allocs = get_allocations(r)
        ema = next(a for a in allocs if a.strategy_name == "ema_crossover_rsi")
        assert ema.allocation_pct == 25.0
        assert ema.is_active is True

    def test_trending_down_no_ema(self):
        from strategy.strategy_selector import get_allocations
        r = MarketRegime(regime=MarketRegimeType.TRENDING_DOWN, adx=30)
        allocs = get_allocations(r)
        ema = next(a for a in allocs if a.strategy_name == "ema_crossover_rsi")
        assert ema.allocation_pct == 0.0
        assert ema.is_active is False

    def test_sideways_grid_dominant(self):
        from strategy.strategy_selector import get_active_strategies
        r = MarketRegime(regime=MarketRegimeType.SIDEWAYS, adx=15)
        active = get_active_strategies(r)
        assert "grid_trading" in active

    def test_unknown_conservative(self):
        from strategy.strategy_selector import get_strategy_budget_pct
        r = MarketRegime(regime=MarketRegimeType.UNKNOWN)
        total = sum(
            get_strategy_budget_pct(r, name)
            for name in ["ema_crossover_rsi", "grid_trading", "mean_reversion",
                         "bollinger_breakout", "dca_bot", "macd_divergence"]
        )
        assert total <= 20  # Very conservative in unknown

    def test_all_strategies_in_table(self):
        from strategy.strategy_selector import ALL_STRATEGY_NAMES, ALLOCATION_TABLE
        for regime_allocs in ALLOCATION_TABLE.values():
            for name in ALL_STRATEGY_NAMES:
                assert name in regime_allocs


# ══════════════════════════════════════════════════════
# Phase 12: Trade Statistician
# ══════════════════════════════════════════════════════

class TestStatistician:
    def _make_trades(self, n=20):
        trades = []
        for i in range(n):
            is_win = i % 3 != 0  # ~67% win rate
            pnl = 2.0 if is_win else -1.5
            trades.append(make_trade(
                pnl=pnl, is_win=is_win, hour=10 + (i % 8),
                day=i % 7, confidence=0.75 + (i % 5) * 0.03,
                vol_ratio=1.0 + i * 0.1,
            ))
        return trades

    def test_basic_stats(self):
        from analyzer.statistician import Statistician
        s = Statistician()
        trades = self._make_trades(30)
        stats = s.compute_stats(trades)
        assert stats.total_trades == 30
        assert stats.win_rate > 0
        assert stats.total_pnl != 0

    def test_filter_by_strategy(self):
        from analyzer.statistician import Statistician
        s = Statistician()
        trades = self._make_trades(20)
        trades[0] = make_trade(strategy="grid_trading", pnl=3.0)
        stats = s.compute_stats(trades, strategy="grid_trading")
        assert stats.total_trades == 1

    def test_filter_by_regime(self):
        from analyzer.statistician import Statistician
        s = Statistician()
        trades = [make_trade(regime="sideways", pnl=1.0) for _ in range(5)]
        trades += [make_trade(regime="trending_up", pnl=-1.0, is_win=False) for _ in range(5)]
        stats = s.compute_stats(trades, market_regime="sideways")
        assert stats.total_trades == 5

    def test_empty_trades(self):
        from analyzer.statistician import Statistician
        s = Statistician()
        stats = s.compute_stats([])
        assert stats.total_trades == 0
        assert stats.win_rate == 0

    def test_profit_factor(self):
        from analyzer.statistician import Statistician
        s = Statistician()
        trades = [make_trade(pnl=3.0, is_win=True)] * 5 + \
                 [make_trade(pnl=-1.0, is_win=False)] * 5
        stats = s.compute_stats(trades)
        assert stats.profit_factor == pytest.approx(3.0)

    def test_by_strategy(self):
        from analyzer.statistician import Statistician
        s = Statistician()
        trades = [make_trade(strategy="ema_crossover_rsi")] * 5 + \
                 [make_trade(strategy="grid_trading")] * 3
        by_strat = s.compute_by_strategy(trades)
        assert "ema_crossover_rsi" in by_strat
        assert "grid_trading" in by_strat
        assert by_strat["ema_crossover_rsi"].total_trades == 5

    def test_format_report(self):
        from analyzer.statistician import Statistician
        s = Statistician()
        trades = self._make_trades(20)
        stats = s.compute_stats(trades)
        report = s.format_report(stats, "Test Report")
        assert "Test Report" in report
        assert "Win Rate" in report

    def test_max_drawdown(self):
        from analyzer.statistician import Statistician
        s = Statistician()
        # Trades: +2, +2, -3, -3, +1 → cumPnL: 2, 4, 1, -2, -1 → DD = 4-(-2)=6
        trades = [
            make_trade(pnl=2.0, is_win=True),
            make_trade(pnl=2.0, is_win=True),
            make_trade(pnl=-3.0, is_win=False),
            make_trade(pnl=-3.0, is_win=False),
            make_trade(pnl=1.0, is_win=True),
        ]
        stats = s.compute_stats(trades)
        assert stats.max_drawdown == pytest.approx(6.0)

    def test_best_hours(self):
        from analyzer.statistician import Statistician
        s = Statistician()
        trades = [make_trade(hour=10, pnl=5.0)] * 3 + [make_trade(hour=20, pnl=-1.0, is_win=False)] * 3
        stats = s.compute_stats(trades)
        assert stats.best_hours[0] == 10


# ══════════════════════════════════════════════════════
# Phase 13: Optimizer
# ══════════════════════════════════════════════════════

class TestOptimizer:
    def test_frozen_params_blocked(self):
        from analyzer.optimizer import Optimizer
        o = Optimizer()
        trades = [make_trade() for _ in range(200)]
        prop = o.analyze_and_propose(trades, "ema_crossover_rsi", "stop_loss_pct", [4.0, 5.0])
        assert prop is None

    def test_unknown_param_blocked(self):
        from analyzer.optimizer import Optimizer
        o = Optimizer()
        trades = [make_trade() for _ in range(200)]
        prop = o.analyze_and_propose(trades, "ema_crossover_rsi", "magic_param", [1.0])
        assert prop is None

    def test_insufficient_trades(self):
        from analyzer.optimizer import Optimizer
        o = Optimizer()
        trades = [make_trade() for _ in range(10)]
        prop = o.analyze_and_propose(trades, "ema_crossover_rsi", "min_confidence", [0.80])
        assert prop is None

    def test_can_propose_limit(self):
        from analyzer.optimizer import Optimizer, OptimizerConfig
        o = Optimizer(OptimizerConfig(max_changes_per_week=0))
        assert o.can_propose() is False

    def test_apply_and_rollback(self):
        from analyzer.optimizer import Optimizer, OptimizationProposal
        o = Optimizer()
        p = OptimizationProposal(proposal_id="test_1", status="pending", parameter="min_confidence")
        o._history.append(p)
        assert o.apply_proposal("test_1") is True
        assert p.status == "applied"
        assert o.rollback_proposal("test_1") is True
        assert p.status == "rolled_back"

    def test_get_history(self):
        from analyzer.optimizer import Optimizer, OptimizationProposal
        o = Optimizer()
        o._history.append(OptimizationProposal(proposal_id="h1"))
        assert len(o.get_history()) == 1


# ══════════════════════════════════════════════════════
# Phase 14: ML Predictor
# ══════════════════════════════════════════════════════

class TestMLPredictor:
    def test_not_ready_by_default(self):
        from analyzer.ml_predictor import MLPredictor
        ml = MLPredictor()
        assert ml.is_ready is False
        assert ml.rollout_mode == "off"

    def test_predict_when_off(self):
        from analyzer.ml_predictor import MLPredictor
        ml = MLPredictor()
        pred = ml.predict([0.0] * 15)
        assert pred.decision == "allow"
        assert pred.rollout_mode == "off"

    def test_extract_features(self):
        from analyzer.ml_predictor import MLPredictor, FEATURE_NAMES
        ml = MLPredictor()
        trade = make_trade(rsi=45.0, adx=30.0, vol_ratio=1.5)
        # Set raw indicator attributes for new feature extraction
        trade = StrategyTrade(
            trade_id=trade.trade_id, signal_id=trade.signal_id,
            symbol=trade.symbol, strategy_name=trade.strategy_name,
            market_regime=trade.market_regime,
            timestamp_open=trade.timestamp_open, timestamp_close=trade.timestamp_close,
            entry_price=trade.entry_price, exit_price=trade.exit_price,
            quantity=trade.quantity, pnl_usd=trade.pnl_usd, pnl_pct=trade.pnl_pct,
            is_win=trade.is_win, confidence=trade.confidence,
            hour_of_day=trade.hour_of_day, day_of_week=trade.day_of_week,
            rsi_at_entry=45.0, adx_at_entry=30.0, volume_ratio_at_entry=1.5,
            exit_reason=trade.exit_reason, hold_duration_hours=trade.hold_duration_hours,
            max_drawdown_during_trade=trade.max_drawdown_during_trade,
            max_profit_during_trade=trade.max_profit_during_trade,
            commission_usd=trade.commission_usd,
            ema_9_at_entry=102.0, ema_21_at_entry=100.0,
            bb_bandwidth_at_entry=0.08, macd_histogram_at_entry=0.3,
            atr_at_entry=1.5,
        )
        features = ml.extract_features(trade)
        assert len(features) == len(FEATURE_NAMES)
        assert features[0] == 45.0  # rsi
        assert features[1] == 30.0  # adx
        assert features[2] > 0.0    # ema_9_vs_21
        assert features[3] > 0.0    # bb_bandwidth
        assert features[5] > 0.0    # macd_histogram

    def test_extract_features_with_history_context(self):
        from analyzer.ml_predictor import MLPredictor

        ml = MLPredictor()
        prev_1 = make_trade(pnl=1.0, is_win=True)
        prev_1.timestamp_open = "2026-01-01T08:00:00"
        prev_1.timestamp_close = "2026-01-01T09:00:00"

        prev_2 = make_trade(pnl=-0.5, is_win=False)
        prev_2.timestamp_open = "2026-01-01T09:30:00"
        prev_2.timestamp_close = "2026-01-01T10:00:00"

        prev_3 = make_trade(pnl=-0.25, is_win=False)
        prev_3.timestamp_open = "2026-01-01T10:30:00"
        prev_3.timestamp_close = "2026-01-01T11:00:00"

        current = make_trade(pnl=0.75, is_win=True)
        current.timestamp_open = "2026-01-01T12:00:00"
        current.timestamp_close = "2026-01-01T13:00:00"

        features = ml.extract_features(current, previous_trades=[prev_1, prev_2, prev_3])
        # Indices: 13=recent_win_rate_10, 14=hours_since_last_trade,
        #          15=rolling_avg_pnl_pct_20, 16=consecutive_losses
        assert features[13] == pytest.approx(1 / 3, abs=1e-6)   # 1 win / 3 trades
        assert features[14] == pytest.approx(1.0, abs=1e-6)      # 12:00 - 11:00 = 1h
        assert features[15] == pytest.approx(1 / 120, abs=1e-6)  # avg pnl normalized
        assert features[16] == 2.0                                 # 2 consecutive losses

    def test_insufficient_trades_for_train(self):
        from analyzer.ml_predictor import MLPredictor
        ml = MLPredictor()
        trades = [make_trade() for _ in range(50)]
        result = ml.train(trades)
        assert result is None

    def test_needs_retrain_initially(self):
        from analyzer.ml_predictor import MLPredictor
        ml = MLPredictor()
        assert ml.needs_retrain() is True

    def test_rollout_mode_setter(self):
        from analyzer.ml_predictor import MLPredictor
        ml = MLPredictor()
        ml.rollout_mode = "shadow"
        assert ml.rollout_mode == "shadow"
        ml.rollout_mode = "invalid"
        assert ml.rollout_mode == "shadow"  # Unchanged

    def test_train_with_enough_data(self):
        """Train with synthetic data — verify metrics are returned."""
        pytest.importorskip("sklearn")
        from analyzer.ml_predictor import MLPredictor, MLConfig
        ml = MLPredictor(MLConfig(
            min_trades=100, min_precision=0.0, min_recall=0.0,
            min_roc_auc=0.0, min_skill_score=0.0,
        ))
        # Generate enough varied trades
        trades = []
        for i in range(600):
            is_win = i % 2 == 0
            trades.append(make_trade(
                pnl=2.0 if is_win else -1.0,
                is_win=is_win,
                rsi=30 + (i % 40),
                adx=15 + (i % 30),
                vol_ratio=0.5 + (i % 20) * 0.1,
                confidence=0.6 + (i % 10) * 0.03,
                hour=i % 24,
                day=i % 7,
            ))
        metrics = ml.train(trades)
        assert metrics is not None
        assert metrics.train_samples > 0
        assert metrics.test_samples > 0
        assert ml.is_ready is True

    def test_predict_after_train(self):
        """Predict after training."""
        pytest.importorskip("sklearn")
        from analyzer.ml_predictor import MLPredictor, MLConfig
        ml = MLPredictor(MLConfig(
            min_trades=100, min_precision=0.0, min_recall=0.0,
            min_roc_auc=0.0, min_skill_score=0.0,
        ))
        trades = []
        for i in range(600):
            is_win = i % 2 == 0
            trades.append(make_trade(
                pnl=2.0 if is_win else -1.0, is_win=is_win,
                rsi=30 + (i % 40), adx=15 + (i % 30),
            ))
        ml.train(trades)
        ml.rollout_mode = "shadow"
        pred = ml.predict([45.0, 30.0, 0.0, 0.1, 1.5, 0.2, 0.01, 14.0, 3.0, 0.0, 0.0, 0.6, 2.0, 1.0, 0.0])
        assert pred.decision == "allow"  # Shadow always allows
        assert 0.0 <= pred.probability <= 1.0


# ══════════════════════════════════════════════════════
# Phase 15: Live Executor
# ══════════════════════════════════════════════════════

class TestLiveExecutor:
    def test_first_day_max_order(self):
        from execution.live_executor import LiveExecutor
        from core.events import EventBus
        ex = LiveExecutor(EventBus(), first_day_max_order=20.0)
        assert ex._is_first_day is True
        assert ex._get_max_order_usd() == 20.0

    def test_init_without_binance(self):
        from execution.live_executor import LiveExecutor
        from core.events import EventBus
        ex = LiveExecutor(EventBus())
        # Without python-binance installed or valid keys, should handle gracefully
        assert ex._client is None

    @pytest.mark.asyncio
    async def test_rejects_oversized_order(self):
        from execution.live_executor import LiveExecutor
        from core.events import EventBus
        ex = LiveExecutor(EventBus(), first_day_max_order=20.0)
        sig = Signal(
            timestamp=int(time.time()*1000), symbol="BTCUSDT",
            direction=Direction.BUY, confidence=0.80,
            strategy_name="test", reason="test",
        )
        order = await ex.execute_order(sig, quantity=1.0, current_price=100.0)  # $100 > $20
        assert order is None

    @pytest.mark.asyncio
    async def test_rejects_invalid_qty(self):
        from execution.live_executor import LiveExecutor
        from core.events import EventBus
        ex = LiveExecutor(EventBus())
        sig = Signal(
            timestamp=int(time.time()*1000), symbol="BTCUSDT",
            direction=Direction.BUY, confidence=0.80,
            strategy_name="test", reason="test",
        )
        order = await ex.execute_order(sig, quantity=0, current_price=100.0)
        assert order is None
