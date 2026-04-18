"""
Shared pytest fixtures for ML-related tests.

Before this file, tests defined synthetic trade factories inline per file
(``_make_trade`` in test_ml_predictor.py, ``_make_feature_vector`` in
test_ml_integration.py, etc.). That was workable while ML tests stayed
small, but the walk-forward / stacking / regime-router tests added in
phases 1–4 all need the same helpers: many-trade chronological fixtures
with controllable win rates, regimes, and label structure. Duplicating
those in every new test file would (a) drift out of sync the moment the
StrategyTrade schema changes, and (b) make it tempting to write each
test with its own quirky data and lose cross-test comparability.

Fixtures live here so each test module can ``pytest.fixture`` them in
without importing private helpers.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.models import StrategyTrade


def make_trade(**overrides) -> StrategyTrade:
    """Factory with the same defaults as the original inline helpers.

    Keeps older tests bit-for-bit identical (they pass the same overrides)
    and gives new tests a single canonical entry point.
    """
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


def make_synthetic_trades(
    n: int = 500,
    win_rate: float = 0.55,
    seed: int = 42,
    regime_mix: dict[str, float] | None = None,
) -> list[StrategyTrade]:
    """Generate ``n`` chronologically ordered synthetic trades.

    The features are perturbed with the RNG so tree models and linear
    models both have something to learn; labels are sampled from a
    feature-conditioned probability so the model's AUC should beat 0.5
    at reasonable N.

    ``regime_mix`` lets tests exercise regime-routing paths by, e.g.,
    passing ``{"trending_up": 0.5, "sideways": 0.5}`` to split the
    history evenly across two regimes.
    """
    rng = np.random.default_rng(seed)
    regimes = regime_mix or {"trending_up": 1.0}
    regime_names = list(regimes.keys())
    regime_probs = np.array(list(regimes.values()), dtype=np.float64)
    regime_probs = regime_probs / regime_probs.sum()

    trades: list[StrategyTrade] = []
    for i in range(n):
        # Feature draws — a handful correlate with label, the rest are noise.
        rsi = float(rng.uniform(20, 80))
        adx = float(rng.uniform(10, 45))
        vol_ratio = float(rng.uniform(0.4, 3.0))
        sentiment = float(rng.uniform(-1.0, 1.0))
        # Label probability: mild signal on rsi+sentiment so models have something to learn
        p_win = 0.25 + 0.5 * ((rsi - 20) / 60) * 0.5 + 0.25 * (sentiment + 1) / 2
        p_win = float(np.clip(p_win * (win_rate / 0.5), 0.1, 0.9))
        is_win = bool(rng.random() < p_win)
        regime = rng.choice(regime_names, p=regime_probs)
        hour = int(rng.integers(0, 24))
        day = int(rng.integers(0, 7))
        pnl_usd = float(rng.normal(15 if is_win else -8, 3))
        # Timestamp strictly increasing so walk-forward can trust row order
        hrs = i
        ts_open = f"2025-01-01T{(hrs % 24):02d}:00:00Z"
        # day shift only, keep it simple
        trades.append(make_trade(
            trade_id=f"syn_{i:05d}",
            market_regime=str(regime),
            rsi_at_entry=rsi,
            adx_at_entry=adx,
            volume_ratio_at_entry=vol_ratio,
            news_sentiment=sentiment,
            hour_of_day=hour,
            day_of_week=day,
            is_win=is_win,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_usd / 1000.0,
            timestamp_open=ts_open,
            timestamp_close=ts_open,
        ))
    return trades


# ─────────────────────────────────────────────
# Pytest fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def sample_trade() -> StrategyTrade:
    """Single trade with defaults."""
    return make_trade()


@pytest.fixture
def synthetic_trades() -> list[StrategyTrade]:
    """500 synthetic trades — default mix (all trending_up)."""
    return make_synthetic_trades(n=500)


@pytest.fixture
def synthetic_trades_multi_regime() -> list[StrategyTrade]:
    """500 trades split across 3 regimes for regime-router tests."""
    return make_synthetic_trades(
        n=500,
        regime_mix={"trending_up": 0.4, "sideways": 0.35, "volatile": 0.25},
    )


@pytest.fixture
def trade_factory():
    """Callable factory for tests that need custom overrides."""
    return make_trade


@pytest.fixture
def synthetic_trades_factory():
    """Callable factory when a test needs a custom n / win_rate / seed."""
    return make_synthetic_trades
