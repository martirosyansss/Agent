"""Tests for the ``regime_flip_exit`` guard (Phase 3).

Contracts:

* **Narrow-by-design** — only fires on ``trending_down`` + strong ADX.
  Sideways and volatile regimes do NOT trigger the hard exit (those fall
  to Phase 2 stop-tightening instead). A bullish or unknown regime is a
  no-op.
* **Whitelist gate** — strategies not in ``LONG_ONLY_STRATEGIES`` never
  fire. This is the extension point for shorts / market-neutral work:
  they must be opted in explicitly.
* **ADX threshold** — below ``min_adx`` the signal is "weak trend" and
  no exit is forced. Sentinel uses ADX 25 as the textbook trending line.
* **Reason strings are tokenised** — ``reason.split(':')[0]`` groups by
  outcome (``regime_flip_exit`` vs ``weak_trend`` vs ``regime_not_bearish``)
  for clean event-log GROUP BYs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from risk.regime_flip_exit import (
    CONFIRMED_BEARISH_REGIME,
    LONG_ONLY_STRATEGIES,
    should_exit_on_regime_flip,
)


class TestFiresOnly:
    def test_fires_on_trending_down_with_strong_adx(self):
        ok, reason = should_exit_on_regime_flip(
            strategy_name="ema_crossover_rsi",
            current_regime="trending_down",
            adx=30.0,
        )
        assert ok is True
        assert reason.startswith("regime_flip_exit")

    def test_fires_exactly_at_threshold(self):
        # Use the default 25.0 threshold — equality should fire, not reject.
        ok, reason = should_exit_on_regime_flip(
            strategy_name="bollinger_breakout",
            current_regime="trending_down",
            adx=25.0,
        )
        assert ok is True


class TestSkipsOtherRegimes:
    @pytest.mark.parametrize("regime", [
        "trending_up", "sideways", "volatile", "transitioning", "unknown", "",
    ])
    def test_non_bearish_regime_no_exit(self, regime):
        ok, reason = should_exit_on_regime_flip(
            strategy_name="ema_crossover_rsi",
            current_regime=regime,
            adx=40.0,
        )
        assert ok is False
        assert reason.startswith("regime_not_bearish")

    def test_sentinel_constant_matches(self):
        # Guards against a silent rename of MarketRegimeType.TRENDING_DOWN.value.
        assert CONFIRMED_BEARISH_REGIME == "trending_down"


class TestAdxThreshold:
    def test_weak_adx_rejected(self):
        ok, reason = should_exit_on_regime_flip(
            strategy_name="ema_crossover_rsi",
            current_regime="trending_down",
            adx=15.0,
        )
        assert ok is False
        assert reason.startswith("weak_trend")

    def test_custom_min_adx(self):
        ok, reason = should_exit_on_regime_flip(
            strategy_name="ema_crossover_rsi",
            current_regime="trending_down",
            adx=35.0,
            min_adx=40.0,
        )
        assert ok is False
        assert reason.startswith("weak_trend")


class TestWhitelistGate:
    def test_unknown_strategy_does_not_fire(self):
        ok, reason = should_exit_on_regime_flip(
            strategy_name="future_short_strategy",
            current_regime="trending_down",
            adx=35.0,
        )
        assert ok is False
        assert reason.startswith("strategy_not_whitelisted")

    def test_custom_whitelist_used(self):
        # Caller passes its own whitelist — default is ignored.
        ok, reason = should_exit_on_regime_flip(
            strategy_name="ema_crossover_rsi",
            current_regime="trending_down",
            adx=35.0,
            whitelist=frozenset({"only_this_one"}),
        )
        assert ok is False
        assert reason.startswith("strategy_not_whitelisted")

    def test_default_whitelist_includes_all_current_long_strategies(self):
        # If a new Sentinel strategy is added to core.models without
        # updating LONG_ONLY_STRATEGIES it should NOT silently get this
        # guard turned on — this test exists as a tripwire, not a lock.
        for s in {"ema_crossover_rsi", "bollinger_breakout", "mean_reversion",
                  "macd_divergence", "dca_bot", "grid_trading"}:
            assert s in LONG_ONLY_STRATEGIES, f"{s} missing from LONG_ONLY_STRATEGIES"
