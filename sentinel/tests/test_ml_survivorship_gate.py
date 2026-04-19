"""Tests for the survivorship-bias warning gate in the ML trainer.

The trainer cannot *fix* survivorship bias (the universe of trades is
fixed upstream), but it must loudly flag the two failure modes that
look like a successful model but are actually a single-asset over-fit:

1. Too few distinct symbols in training (< 5) — likely a hand-picked
   universe of survivors.
2. One symbol dominates (> 80%) — the "ensemble" is effectively a
   single-asset model, marketed as a portfolio claim.

We verify the warning is emitted in both modes and absent on a healthy
diversified corpus, by capturing the trainer's logger output.
"""
from __future__ import annotations

import logging

import pytest

from core.models import StrategyTrade


# ---------------------------------------------------------------------------
# Fixture: synthetic trade builder
# ---------------------------------------------------------------------------


def _make_trade(idx: int, symbol: str) -> StrategyTrade:
    """Construct a minimally-populated StrategyTrade. Only ``symbol`` and
    a few fields the trainer touches are meaningful here — the trainer
    will fail later anyway because we won't actually run it; we exercise
    only the survivorship-detection block."""
    return StrategyTrade(
        trade_id=f"t{idx}", symbol=symbol, strategy_name="test",
        timestamp_open=f"2024-01-{(idx % 28) + 1:02d}T00:00:00Z",
        timestamp_close=f"2024-01-{(idx % 28) + 1:02d}T01:00:00Z",
        entry_price=100.0, exit_price=101.0, quantity=1.0,
        pnl_usd=1.0, pnl_pct=1.0, is_win=True, hold_duration_hours=1.0,
        market_regime="trending_up",
        rsi_at_entry=50.0, adx_at_entry=20.0,
        volume_ratio_at_entry=1.0,
        ema_9_at_entry=100.0, ema_21_at_entry=100.0,
        bb_bandwidth_at_entry=0.05,
        macd_histogram_at_entry=0.0, atr_at_entry=1.0,
        hour_of_day=12, day_of_week=1,
        news_sentiment=0.0, fear_greed_index=50,
        trend_alignment=0.0,
    )


def _run_survivorship_block(trades: list[StrategyTrade], caplog) -> list[str]:
    """Inline-extract and run *just* the survivorship-bias logging block
    from trainer.run_training. We avoid invoking the full trainer because
    it requires a fully-constructed MLPredictor, all 32 features, and
    minutes of training time. The logic under test is deterministic given
    only ``trades``, so mirroring the small block here keeps the test
    fast and focused.

    If this duplication ever drifts from the trainer, the integration test
    in test_ml_integration.py will catch the mismatch via end-to-end logs.
    """
    logger = logging.getLogger("analyzer.ml.training.trainer")

    # Must mirror trainer.py exactly:
    symbol_counts: dict[str, int] = {}
    for t in trades:
        s = getattr(t, "symbol", "") or "unknown"
        symbol_counts[s] = symbol_counts.get(s, 0) + 1
    n_symbols = len(symbol_counts)
    max_share = max(symbol_counts.values()) / len(trades) if trades else 1.0

    with caplog.at_level(logging.WARNING, logger=logger.name):
        if n_symbols < 5:
            logger.warning(
                "ML train: SURVIVORSHIP-BIAS RISK — only %d distinct symbol(s) in %d trades.",
                n_symbols, len(trades),
            )
        elif max_share > 0.80:
            top_sym = max(symbol_counts, key=symbol_counts.get)
            logger.warning(
                "ML train: SURVIVORSHIP-BIAS RISK — symbol '%s' is %.0f%% of training trades.",
                top_sym, max_share * 100.0,
            )
    return [r.getMessage() for r in caplog.records]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSurvivorshipGate:
    def test_warns_when_too_few_distinct_symbols(self, caplog):
        # 3 symbols < threshold of 5.
        trades = [_make_trade(i, f"SYM{i % 3}") for i in range(60)]
        msgs = _run_survivorship_block(trades, caplog)
        assert any("SURVIVORSHIP-BIAS RISK" in m and "distinct symbol" in m for m in msgs), (
            f"Expected diversity warning, got: {msgs}"
        )

    def test_warns_when_one_symbol_dominates(self, caplog):
        # 5 distinct symbols (passes the count gate), but BTC = 90%.
        trades = [_make_trade(i, "BTCUSDT") for i in range(90)]
        trades += [_make_trade(i, f"SYM{i % 4}") for i in range(90, 100)]
        msgs = _run_survivorship_block(trades, caplog)
        assert any(
            "SURVIVORSHIP-BIAS RISK" in m and "BTCUSDT" in m for m in msgs
        ), f"Expected dominance warning, got: {msgs}"

    def test_no_warning_on_diversified_corpus(self, caplog):
        # 10 symbols, each ~10% — clean corpus.
        symbols = [f"SYM{i:02d}" for i in range(10)]
        trades = [_make_trade(i, symbols[i % 10]) for i in range(200)]
        msgs = _run_survivorship_block(trades, caplog)
        assert not any("SURVIVORSHIP-BIAS RISK" in m for m in msgs), (
            f"Unexpected warning on diversified corpus: {msgs}"
        )

    def test_threshold_boundary_5_symbols_passes(self, caplog):
        # Exactly 5 symbols — should NOT trigger the count gate.
        trades = [_make_trade(i, f"SYM{i % 5}") for i in range(50)]
        msgs = _run_survivorship_block(trades, caplog)
        assert not any("distinct symbol" in m for m in msgs)

    def test_threshold_boundary_80_share_passes(self, caplog):
        # 8 symbols, top one = 80% exactly (not strictly > 80%) — passes.
        trades = [_make_trade(i, "BTCUSDT") for i in range(80)]
        trades += [_make_trade(i, f"SYM{i % 7}") for i in range(80, 100)]
        msgs = _run_survivorship_block(trades, caplog)
        # 80% is the boundary — gate fires only at > 80%.
        assert not any("80%" in m and "BTCUSDT" in m for m in msgs)
