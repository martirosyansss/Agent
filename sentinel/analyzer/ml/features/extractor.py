"""Feature extraction from :class:`StrategyTrade` objects.

Moved out of ``analyzer.ml_predictor`` during the round-10 refactor.
The single-trade and batch paths share a single implementation
(``extract_features`` delegates to ``extract_features_batch`` — the
round-N-3 audit fix that killed training/serving skew).

All outputs are pre-trade features only (no forward-looking bias):

* Technical indicators captured at entry (rsi, adx, volume_ratio, ...)
* Strategy metadata (regime fit score, strategy-specific rolling WR)
* Cyclical time features (hour, day-of-week)
* Historical performance (recent win rate, consecutive losses, rolling PnL)
* Sentiment data (news, fear-greed index)
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Optional

import numpy as np

from core.models import StrategyTrade

from ..domain.constants import (
    FEATURE_NAMES,
    N_FEATURES,
    REGIME_ENCODING,
    STRATEGY_REGIME_FIT,
)

__all__ = [
    "extract_features",
    "extract_features_batch",
    "regime_bias",
    "strategy_regime_fit",
    "parse_trade_timestamp",
    "FEATURE_NAMES",
    "N_FEATURES",
]


def regime_bias(regime: str) -> float:
    """Regime bias using ONLY pre-trade knowledge (no pnl_pct)."""
    lowered = regime.lower()
    if lowered == "trending_up":
        return 1.0
    if lowered == "trending_down":
        return -1.0
    return 0.0


def strategy_regime_fit(strategy_name: str, regime: str) -> float:
    """How well a strategy fits the current market regime, in [-1, 1].

    Looks up ``STRATEGY_REGIME_FIT``; unknown combos return 0.0
    (neutral — no edge, no disadvantage).
    """
    return STRATEGY_REGIME_FIT.get((strategy_name, regime.lower()), 0.0)


def parse_trade_timestamp(value: str) -> Optional[datetime]:
    """Parse ISO-8601 timestamps tolerantly; returns ``None`` on failure."""
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def extract_features(
    trade: StrategyTrade,
    previous_trades: Optional[list[StrategyTrade]] = None,
) -> list[float]:
    """Build the N_FEATURES-long feature vector for a single trade.

    Delegates to :func:`extract_features_batch` on ``previous_trades +
    [trade]`` so training and inference go through exactly the same code
    path — any divergence between the two would cause silent skew
    (and was the N-3 audit finding that forced this consolidation).
    """
    previous_trades = previous_trades or []
    all_trades = previous_trades + [trade]
    X = extract_features_batch(all_trades)
    return X[-1].tolist()


def extract_features_batch(trades: list[StrategyTrade]) -> np.ndarray:
    """Vectorised feature extraction — ~8-15× faster than a per-trade loop.

    Returns a ``(n_trades, N_FEATURES)`` matrix. Assumes ``trades`` is
    sorted chronologically so rolling windows look strictly backward
    (no look-ahead). NaN / inf values from source fields are clamped
    to 0 at the end so sklearn scalers don't blow up.
    """
    n = len(trades)
    X = np.zeros((n, N_FEATURES), dtype=np.float64)

    # Pre-compute vectors that feed rolling calculations. Computing them
    # once upfront keeps the per-trade loop free of inner O(N) work.
    is_win = np.array([1 if t.is_win else 0 for t in trades], dtype=np.float64)

    # C-2 fix: parse every timestamp ONCE (eliminates O(N²) re-parsing).
    parsed_open = [parse_trade_timestamp(t.timestamp_open or "") for t in trades]
    parsed_close = [
        parse_trade_timestamp(t.timestamp_close or t.timestamp_open or "")
        for t in trades
    ]

    # W-2 fix: consecutive-losses in O(N).
    consec_losses = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        consec_losses[i] = 0.0 if is_win[i - 1] == 1 else consec_losses[i - 1] + 1.0

    # v4: per-strategy win index for strategy_specific_wr_10, avoiding O(N²).
    strategy_win_history: dict[str, list[tuple[int, int]]] = defaultdict(list)

    for idx, trade in enumerate(trades):
        entry_price = max(abs(trade.entry_price), 1e-9)
        rb = regime_bias(trade.market_regime)
        ema_9_vs_21 = (trade.ema_9_at_entry - trade.ema_21_at_entry) / entry_price
        atr_safe = max(trade.atr_at_entry, 1e-9)  # safe divisor for ATR-normalisation
        atr_ratio = atr_safe / entry_price

        strat_fit = strategy_regime_fit(trade.strategy_name, trade.market_regime)

        prev_same = strategy_win_history[trade.strategy_name]
        if prev_same:
            last10 = prev_same[-10:]
            strategy_specific_wr = sum(w for _, w in last10) / len(last10)
        else:
            strategy_specific_wr = 0.5

        if idx >= 1:
            start = max(0, idx - 10)
            recent_win_rate = is_win[start:idx].mean()
        else:
            recent_win_rate = 0.5

        if idx >= 1:
            _start20 = max(0, idx - 20)
            _recent_pnl = np.array([t.pnl_pct for t in trades[_start20:idx]], dtype=np.float64)
            rolling_avg_pnl_pct_20 = float(np.mean(_recent_pnl)) / 10.0
        else:
            rolling_avg_pnl_pct_20 = 0.0

        hours_since = 0.0
        if idx > 0 and parsed_open[idx] and parsed_close[idx - 1]:
            delta = (parsed_open[idx] - parsed_close[idx - 1]).total_seconds()
            hours_since = max(delta / 3600.0, 0.0)
        hours_since_clamped = min(hours_since, 72.0)

        cci_norm = max(-3.0, min(float(trade.cci_at_entry or 0) / 200.0, 3.0))

        # Cyclical temporal encoding: sin/cos preserves adjacency (hour 23 ↔ 0)
        _hour = float(trade.hour_of_day or 0)
        _day = float(trade.day_of_week or 0)
        hour_sin = np.sin(2.0 * np.pi * _hour / 24.0)
        hour_cos = np.cos(2.0 * np.pi * _hour / 24.0)
        day_sin = np.sin(2.0 * np.pi * _day / 7.0)
        day_cos = np.cos(2.0 * np.pi * _day / 7.0)

        # NaN-safe indicator reads: ``val or 0`` guards against None propagation
        _fg = float(trade.fear_greed_index or 0) / 100.0
        _dmi = float(trade.dmi_spread_at_entry or 0) / 50.0
        _stoch = float(trade.stoch_rsi_at_entry or 0) / 100.0
        _rsi_d = float(trade.rsi_daily_at_entry or 0) / 100.0
        # ATR-normalised for stationarity (replaces arbitrary /5.0 divisors)
        _pc5h_norm = float(trade.price_change_5h_at_entry or 0) / atr_safe
        _mom_norm = float(trade.momentum_at_entry or 0) / atr_safe

        X[idx] = [
            trade.rsi_at_entry,
            trade.adx_at_entry,
            ema_9_vs_21,
            trade.bb_bandwidth_at_entry,
            trade.volume_ratio_at_entry,
            trade.macd_histogram_at_entry,
            atr_ratio,
            # Cyclical temporal (4 features)
            hour_sin, hour_cos, day_sin, day_cos,
            # Encoding
            float(REGIME_ENCODING.get(trade.market_regime, 5)),
            strat_fit,
            # Historical
            recent_win_rate,
            hours_since_clamped,
            rolling_avg_pnl_pct_20,
            consec_losses[idx],
            strategy_specific_wr,
            # Sentiment / regime
            float(trade.news_sentiment or 0),
            _fg,
            float(trade.trend_alignment or 0),
            rb,
            # Phase 2 (adx_normalized removed — redundant with adx)
            cci_norm,
            float(trade.roc_at_entry or 0) / 10.0,
            float(trade.cmf_at_entry or 0),
            float(trade.bb_pct_b_at_entry or 0),
            float(trade.hist_volatility_at_entry or 0),
            _dmi,
            _stoch,
            _pc5h_norm,
            _mom_norm,
            _rsi_d,
        ]

        # Record AFTER building features (no look-ahead)
        strategy_win_history[trade.strategy_name].append((idx, int(trade.is_win)))

    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
