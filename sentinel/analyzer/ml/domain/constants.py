"""Feature-set + regime-encoding constants.

These live in the domain layer because they are part of the model's
*contract* with the rest of the bot: the strategy emits trades with
specific fields, feature extraction maps them to N_FEATURES columns in
a stable order, and downstream code (persistence, dashboard, tests)
expects that exact shape. Changing any of these without a coordinated
retrain breaks existing saved models.

Moved out of ``analyzer.ml_predictor`` during the round-10 refactor —
the constants themselves are unchanged so importers that use the
compatibility re-exports in ``ml_predictor.py`` stay working.
"""
from __future__ import annotations

# Default temporal-decay factor (per-instance override lives in MLConfig).
# 0.003 ≈ recent 200 trades weighted ~2x more than the oldest in the
# training window.
_TEMPORAL_DECAY: float = 0.003

# Skill-score component weights — a single source of truth shared by
# training, diagnostics, and the dashboard. Precision is weighted 3x
# recall because this model is a *signal filter*: false positives
# (letting bad trades through) cost more than false negatives (blocking
# some good trades — the strategy still has many other chances).
_SKILL_W_PRECISION: float = 0.30
_SKILL_W_RECALL: float = 0.10
_SKILL_W_ROC_AUC: float = 0.35
_SKILL_W_PROFIT_FACTOR: float = 0.25

# Feature-vector length. Incrementing this requires a coordinated
# retrain + model-version bump — existing saved models encode a specific
# (N_FEATURES,) shape inside their StandardScaler and ensemble.
# v5: +2 cyclical temporal (sin/cos), -1 adx_normalized (redundant)
N_FEATURES: int = 32

# Regime → integer code used for the ``market_regime_encoded`` feature
# at index 11 in the feature vector. Stable numbering across releases;
# adding a new regime means appending a new code, never renumbering.
REGIME_ENCODING: dict[str, int] = {
    "trending_up": 0, "trending_down": 1, "sideways": 2,
    "volatile": 3, "transitioning": 4, "unknown": 5,
}

# strategy_regime_fit: how well each strategy fits each market regime
# (-1..1). Replaces a raw strategy_encoded categorical to give the model
# interpretable signal: "EMA crossover works when trending" instead of
# "strategy 0 is historically better". This generalises across regime
# changes rather than memorising per-strategy win rates.
STRATEGY_REGIME_FIT: dict[tuple[str, str], float] = {
    # EMA crossover — momentum strategy, needs a trend
    ("ema_crossover_rsi", "trending_up"):    1.0,
    ("ema_crossover_rsi", "trending_down"):  0.6,
    ("ema_crossover_rsi", "sideways"):      -0.5,
    ("ema_crossover_rsi", "volatile"):      -0.3,
    # Bollinger breakout — volatility expansion, loves squeezes + breakouts
    ("bollinger_breakout", "trending_up"):   0.5,
    ("bollinger_breakout", "trending_down"): 0.5,
    ("bollinger_breakout", "sideways"):      0.3,
    ("bollinger_breakout", "volatile"):      1.0,
    # Mean reversion — range-bound, hostile to trends
    ("mean_reversion", "trending_up"):      -0.7,
    ("mean_reversion", "trending_down"):    -0.7,
    ("mean_reversion", "sideways"):          1.0,
    ("mean_reversion", "volatile"):         -0.3,
    # MACD divergence — trend-following with momentum confirmation
    ("macd_divergence", "trending_up"):      1.0,
    ("macd_divergence", "trending_down"):    0.8,
    ("macd_divergence", "sideways"):        -0.4,
    ("macd_divergence", "volatile"):         0.2,
    # Grid trading — range-bound, symmetric
    ("grid_trading", "trending_up"):        -0.5,
    ("grid_trading", "trending_down"):      -0.5,
    ("grid_trading", "sideways"):            1.0,
    ("grid_trading", "volatile"):            0.0,
    # DCA — long-biased accumulation
    ("dca_bot", "trending_up"):              0.8,
    ("dca_bot", "trending_down"):           -0.3,
    ("dca_bot", "sideways"):                 0.3,
    ("dca_bot", "volatile"):                -0.2,
    # TRANSITIONING regime — dangerous, most strategies perform poorly
    ("ema_crossover_rsi", "transitioning"):  -0.2,
    ("bollinger_breakout", "transitioning"):  0.1,
    ("mean_reversion", "transitioning"):      0.0,
    ("macd_divergence", "transitioning"):    -0.1,
    ("grid_trading", "transitioning"):       -0.3,
    ("dca_bot", "transitioning"):             0.2,
}

# Feature names — ordered to match column indices in the X matrix
# produced by extract_features. Must not be reordered without bumping
# the model schema version; the save/load path assumes column-i maps to
# FEATURE_NAMES[i] for feature-importance diagnostics.
FEATURE_NAMES: list[str] = [
    # Technical (0-6)
    "rsi_14", "adx", "ema_9_vs_21", "bb_bandwidth", "volume_ratio",
    "macd_histogram", "atr_ratio",
    # Temporal — cyclical sin/cos encoding (7-10)
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    # Encoding (11-12)
    "market_regime_encoded",
    "strategy_regime_fit",
    # Historical (13-17)
    "recent_win_rate_10", "hours_since_last_trade",
    "rolling_avg_pnl_pct_20", "consecutive_losses",
    "strategy_specific_wr_10",
    # Sentiment / regime (18-21)
    "news_sentiment", "fear_greed_normalized", "trend_alignment",
    "regime_bias",
    # Phase 2: Enhanced indicators (22-31)
    "cci", "roc", "cmf", "bb_pct_b", "hist_volatility",
    "dmi_spread", "stoch_rsi", "price_change_5h_norm",
    "momentum_norm", "rsi_daily",
]
