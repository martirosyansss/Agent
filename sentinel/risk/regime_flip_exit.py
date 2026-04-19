"""
Regime-Flip Exit — hard cut when market turns confirmed-bearish on a long.

Phase 2 re-evaluation already tightens the stop when the regime is adverse,
which gives the market a chance to shake and recover. Phase 3 is the *hard*
cut: when the regime is **confirmed** bearish (trending_down with a strong
ADX) we don't wait for the stop to get hit — we close the position now.

The distinction matters for multi-day holds: by the time a tightened stop
fires, the average long has already given back most of its open profit.
A regime-flip exit locks in the remaining gain before the next leg down.

Why narrower criteria than Phase 2:

- Phase 2 fires on *any* adverse regime (trending_down, sideways, volatile)
  because tightening the stop is cheap — if the market recovers, the stop
  never hits and nothing happens.
- Phase 3 only fires on ``trending_down`` with ``ADX ≥ 25``. A premature
  hard exit throws away real edge, so we require the trend classifier to
  be both directional *and* strong.

Long-only: every current Sentinel strategy is long-only, so we check the
whitelist for defensive future-proofing (a new short strategy must not be
force-exited by this guard) rather than because of ambiguity today.
"""

from __future__ import annotations


# Strategies that this guard applies to. A short strategy (or anything
# else that would benefit from a bearish regime) must NOT be in this set.
LONG_ONLY_STRATEGIES: frozenset[str] = frozenset({
    "ema_crossover_rsi",
    "bollinger_breakout",
    "mean_reversion",
    "macd_divergence",
    "dca_bot",
    "grid_trading",
})


# Confirmed-bearish signal — strictly narrower than the "adverse" set used
# by Phase 2 re-evaluation. Sideways and volatile regimes do NOT warrant
# a hard exit because neither implies *directional* downside.
CONFIRMED_BEARISH_REGIME: str = "trending_down"


def should_exit_on_regime_flip(
    strategy_name: str,
    current_regime: str,
    adx: float,
    *,
    min_adx: float = 25.0,
    whitelist: frozenset[str] = LONG_ONLY_STRATEGIES,
) -> tuple[bool, str]:
    """Decide whether an open long position should be force-closed now.

    Args:
        strategy_name: Strategy that opened the position. Used to skip
            strategies outside ``whitelist`` (non-long-only).
        current_regime: Current ``MarketRegimeType.value`` string for
            the symbol — see ``core.models.MarketRegimeType``.
        adx: Current ADX on the signal timeframe. Proxy for trend strength.
        min_adx: Threshold ADX value required to treat the trend as
            strong enough to force-exit. 25 is the textbook "trending"
            cut-off for ADX.
        whitelist: Strategies this guard is allowed to act on. Opt-out
            by passing a set that doesn't contain the strategy name.

    Returns:
        ``(should_exit, reason)``. ``reason`` is human-readable and safe
        to embed in a ``Signal.reason`` / ``Position.close_reason`` field
        — the event-log consumer groups by the leading token.
    """
    if strategy_name not in whitelist:
        return (False, f"strategy_not_whitelisted:{strategy_name}")

    if current_regime != CONFIRMED_BEARISH_REGIME:
        return (False, f"regime_not_bearish:{current_regime or 'unknown'}")

    if adx < min_adx:
        return (False, f"weak_trend:adx={adx:.1f}<{min_adx:.1f}")

    return (True, f"regime_flip_exit:bearish_trend adx={adx:.1f}")
