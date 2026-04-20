"""
Базовый класс стратегии.

Все торговые стратегии SENTINEL наследуют BaseStrategy.
"""

from __future__ import annotations

import math
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

from core.models import FeatureVector, Signal


# ──────────────────────────────────────────────
# News time decay tracking
# ──────────────────────────────────────────────
# Category-specific half-lives (seconds). Different news categories exhibit
# very different price-impact persistence — empirically:
#   • security (hacks, exploits): acute, very fast burn-off — ~30 min
#   • breaking / technical: ~1 h
#   • technical (generic default): ~2 h
#   • adoption: ~4 h (positive narratives compound)
#   • regulatory / macro: ~6 h (slow-burn, reflex reactions dominate day 1)
# These are calibrated against Tetlock-style event-study return persistence
# and the intuition that the information-to-noise ratio of a piece of news
# decays roughly geometrically with category-specific λ = ln(2)/τ.
_NEWS_CATEGORY_HALF_LIFE_SEC: dict[str, float] = {
    "security": 1800.0,      # 30 min — hacks/exploits: fast reaction, fast fade
    "breaking": 3600.0,      # 1 hour
    "technical": 7200.0,     # 2 hours (baseline)
    "adoption": 14400.0,     # 4 hours
    "regulatory": 21600.0,   # 6 hours — slow burn
    "macro": 21600.0,        # 6 hours — slow burn
}
_NEWS_DECAY_HALF_LIFE_SEC: float = 7200.0  # default (2h) for unknown categories
_NEWS_MAX_AGE_SEC: float = 14400.0         # 4 hours: ignore news older than 4h

_last_news_update_ts: float = 0.0


def update_news_timestamp() -> None:
    """Call when fresh news data is received from NewsCollector."""
    global _last_news_update_ts
    _last_news_update_ts = time.time()


def _news_decay_factor(category: str = "") -> float:
    """Exponential decay factor for news impact, category-aware.

    decay(t) = 2^(-age / τ_category)

    Returns 1.0 for fresh news, decays toward 0 as news ages. After
    _NEWS_MAX_AGE_SEC, returns 0 — news is considered stale regardless
    of category.

    Args:
        category: one of the keys in _NEWS_CATEGORY_HALF_LIFE_SEC;
                  falls back to the 2h default when unknown/empty.
    """
    if _last_news_update_ts <= 0:
        return 0.5  # no timestamp = assume moderately stale
    age = time.time() - _last_news_update_ts
    if age >= _NEWS_MAX_AGE_SEC:
        return 0.0  # completely stale
    half_life = _NEWS_CATEGORY_HALF_LIFE_SEC.get(
        category, _NEWS_DECAY_HALF_LIFE_SEC
    )
    return math.pow(2.0, -age / half_life)


def news_confidence_adjustment(
    f: FeatureVector,
    direction: str = "buy",
    strategy_type: str = "trend",
) -> tuple[float, str]:
    """
    Professional news-based confidence adjustment with time decay.

    Uses composite_score, signal_strength, urgency, category, and Fear&Greed
    to compute a confidence delta for trading signals.

    Args:
        f: Current FeatureVector with news fields.
        direction: "buy" or "sell".
        strategy_type: "trend" | "breakout" | "divergence" | "mean_reversion"
            - trend/breakout: news supports momentum direction
            - divergence: news is secondary (reversal-focused)
            - mean_reversion: contrarian — moderate bearish = buy opportunity

    Returns:
        (delta, reason) where delta is confidence adjustment (-0.30 .. +0.15)
        and reason is a string for signal logging.
    """
    delta = 0.0
    parts: list[str] = []

    score = f.news_composite_score
    strength = f.news_signal_strength

    # Apply category-aware time decay to news impact (Fear & Greed stays live).
    # Security events (hacks/exploits) decay fastest; regulatory/macro decay
    # slowest — see _NEWS_CATEGORY_HALF_LIFE_SEC.
    decay = _news_decay_factor(f.news_dominant_category)
    score *= decay
    strength *= decay
    if decay < 0.1:
        parts.append("stale")

    # If news not actionable — only Fear & Greed extremes matter
    if not f.news_actionable:
        fgi_delta = _fear_greed_adjustment(f.fear_greed_index, direction, strategy_type)
        if fgi_delta != 0:
            parts.append(f"F&G={f.fear_greed_index}")
        reason = f"news={score:+.2f}(weak)"
        if parts:
            reason += " " + " ".join(parts)
        return round(fgi_delta, 3), reason

    news_bullish = score > 0.1
    news_bearish = score < -0.1

    if strategy_type == "mean_reversion":
        # Contrarian: moderate bearish = buy opportunity, bullish = less room to revert
        if direction == "buy":
            if news_bearish and not f.news_critical_alert:
                # Fear is our friend (but not panic from real disaster)
                delta += min(0.12, abs(score) * strength * 0.3)
                parts.append("contrarian_buy")
            elif news_bullish:
                # Bullish market = less room for mean reversion
                delta -= min(0.10, score * strength * 0.25)
                parts.append("less_reversion_room")
        else:  # sell
            if news_bullish:
                delta += min(0.10, score * strength * 0.25)
                parts.append("momentum_exit")
            elif news_bearish:
                delta -= min(0.08, abs(score) * strength * 0.2)
                parts.append("contrarian_hold")
    else:
        # Trend-following / breakout / divergence — news supports direction
        weight_map = {"trend": 0.30, "breakout": 0.35, "divergence": 0.20}
        w = weight_map.get(strategy_type, 0.30)
        penalty_w = w * 1.15  # slight asymmetry (was 1.3x — too aggressive)

        if direction == "buy":
            if news_bullish:
                delta += min(0.15, score * strength * w)
            elif news_bearish:
                delta += max(-0.20, score * strength * penalty_w)
        else:  # sell
            if news_bearish:
                delta += min(0.15, abs(score) * strength * w)
            elif news_bullish:
                delta += max(-0.15, -score * strength * penalty_w)

    # Critical alert amplification
    if f.news_critical_alert:
        delta *= 1.5
        parts.append("CRITICAL")

    # Category weight — regulatory/macro/security are strongest market movers
    cat = f.news_dominant_category
    if cat in ("regulatory", "macro") and abs(score) > 0.2:
        delta *= 1.2
        parts.append(f"cat={cat}")
    elif cat == "security" and abs(score) > 0.15:
        delta *= 1.3  # security events (hacks, exploits) move markets fast
        parts.append("cat=security")

    # Fear & Greed overlay
    fgi_delta = _fear_greed_adjustment(f.fear_greed_index, direction, strategy_type)
    delta += fgi_delta
    if fgi_delta != 0:
        parts.append(f"F&G={f.fear_greed_index}")

    # Clamp — wider range for critical events (black swan protection)
    delta = max(-0.30, min(0.15, delta))

    reason = f"news={score:+.2f}(str={strength:.0%})"
    if parts:
        reason += " " + " ".join(parts)

    return round(delta, 3), reason


def _fear_greed_adjustment(fgi: int, direction: str, strategy_type: str) -> float:
    """Fear & Greed Index adjustment per strategy type."""
    if strategy_type == "mean_reversion":
        # Contrarian: extreme fear = strong buy, extreme greed = risky buy
        if direction == "buy":
            if fgi <= 10:
                return 0.08
            elif fgi <= 20:
                return 0.05
            elif fgi >= 85:
                return -0.08
            elif fgi >= 75:
                return -0.03
        else:
            if fgi >= 85:
                return 0.05
            elif fgi <= 15:
                return -0.05
    else:
        if direction == "buy":
            if fgi <= 15:
                return 0.05  # extreme fear = contrarian opportunity
            elif fgi >= 85:
                return -0.05  # bubble territory
        else:
            if fgi >= 85:
                return 0.05  # smart to sell in greed
            elif fgi <= 15:
                return -0.05  # selling into panic = risky
    return 0.0


def news_should_block_entry(f: FeatureVector) -> tuple[bool, str]:
    """
    Check if critical news should prevent opening a new position.

    Blocks entry when:
    - Critical alert + strongly bearish + high strength (black swan)
    - Security category + bearish + actionable (exchange hack, exploit)

    Returns:
        (should_block, reason)
    """
    score = f.news_composite_score
    strength = f.news_signal_strength

    # Black swan: critical + strongly bearish
    if f.news_critical_alert and score < -0.3 and strength > 0.3:
        return True, f"BLOCKED: critical bearish news (score={score:+.2f}, str={strength:.0%})"

    # Security event (hack/exploit) + bearish
    if f.news_dominant_category == "security" and score < -0.2 and f.news_actionable:
        return True, f"BLOCKED: security event (score={score:+.2f}, cat=security)"

    return False, ""


def news_should_accelerate_exit(
    f: FeatureVector, pnl_pct: float,
) -> tuple[bool, float, str]:
    """
    Check if news warrants accelerated position exit.

    Returns:
        (should_exit, confidence, reason)
    """
    score = f.news_composite_score
    strength = f.news_signal_strength

    # Critical bearish alert → immediate exit regardless of PnL
    if f.news_critical_alert and score < -0.3 and strength > 0.3:
        conf = min(0.92, 0.80 + strength * 0.15)
        return True, conf, f"NEWS EXIT: critical bearish (score={score:+.2f}, pnl={pnl_pct:+.1f}%)"

    # Security event → emergency exit
    if f.news_dominant_category == "security" and score < -0.2 and f.news_actionable:
        return True, 0.88, f"NEWS EXIT: security event (score={score:+.2f}, pnl={pnl_pct:+.1f}%)"

    # Strong bearish + small profit → take profits early before reversal
    if score < -0.25 and strength > 0.4 and f.news_actionable and 0 < pnl_pct < 3.0:
        return True, 0.75, f"NEWS EXIT: bearish, locking profit (score={score:+.2f}, pnl={pnl_pct:+.1f}%)"

    return False, 0.0, ""


def news_adjust_sl_tp(
    f: FeatureVector,
    close: float,
    sl_pct: float,
    tp_pct: float,
) -> tuple[float, float]:
    """
    Adjust SL/TP based on news-driven expected volatility.

    - High impact news → widen both (expect bigger moves)
    - Critical bearish → tighten SL (protect capital)
    - Strong bullish + actionable → extend TP (let winners run)

    Returns:
        (sl_price, tp_price)
    """
    sl_mult = 1.0
    tp_mult = 1.0

    impact = abs(f.news_impact_pct)
    score = f.news_composite_score

    # High-impact news → expect bigger moves, widen range
    if impact > 2.0:
        sl_mult = 1.15
        tp_mult = 1.25
    elif impact > 1.0:
        sl_mult = 1.08
        tp_mult = 1.12

    # Critical bearish → tighten SL to protect capital (overrides widening)
    if f.news_critical_alert and score < -0.2:
        sl_mult = 0.75

    # Strong bullish + actionable → extend TP
    if score > 0.3 and f.news_actionable:
        tp_mult = max(tp_mult, 1.15)

    sl_price = close * (1 - sl_pct * sl_mult / 100)
    tp_price = close * (1 + tp_pct * tp_mult / 100)

    return sl_price, tp_price


def grouped_confidence(
    groups: list[list[tuple[bool, float]]],
    correlation_penalty: float = 0.0,
) -> float:
    """Grouped evidence model — prevents over-counting correlated indicators.

    Each group contains (condition, bonus) pairs of correlated indicators.
    Within a group, only the BEST matching bonus is taken (not sum).

    Between groups:
      - correlation_penalty = 0 (default): naive sum — treats groups as fully
        independent. Mathematically equivalent to assuming mutual information
        I(Gᵢ; Gⱼ) = 0, which is rarely true in practice (trend and momentum
        indicators share common regime drivers, etc.).

      - correlation_penalty ∈ (0, 1): apply diminishing-returns attenuation.
        Groups are sorted by strength and the k-th strongest contributes
        bₖ · (1 − penalty)^(k−1). Equivalent to modelling pairwise
        correlation ρ between groups and down-weighting redundant evidence
        using a geometric decay — stronger signal dominates, weaker signals
        add fractionally. Typical ρ ∈ [0.1, 0.2] in practice.

    The two regimes coincide exactly when only one group fires, preserving
    the original behaviour for any single-group configuration (and therefore
    all existing tests that exercise that path).

    Args:
        groups: List of evidence groups.
        correlation_penalty: Pairwise group correlation ρ in [0, 1). 0 keeps
            legacy behaviour. Recommended: 0.10–0.20 for realistically
            correlated groups; 0 for strictly orthogonal evidence.

    Example:
        groups = [
            # Group A: Trend (EMA50, MACD, trend_alignment — all measure trend)
            [(close > ema50, 0.10), (macd > 0, 0.10), (trend_align >= 0.8, 0.10)],
            # Group B: Momentum (RSI, Ichimoku, Williams — all measure momentum)
            [(rsi < 50, 0.10), (above_cloud, 0.08), (williams < -20, 0.05)],
            # Group C: Strength (volume, ADX — both measure conviction)
            [(vol > 2.0, 0.10), (adx > 25, 0.05)],
        ]

    Returns:
        Combined evidence score (≥ 0).
    """
    if not 0.0 <= correlation_penalty < 1.0:
        raise ValueError(
            f"correlation_penalty must be in [0, 1), got {correlation_penalty}"
        )

    bests: list[float] = []
    for group in groups:
        best = 0.0
        for condition, bonus in group:
            if condition and bonus > best:
                best = bonus
        if best > 0:
            bests.append(best)

    if not bests:
        return 0.0
    if correlation_penalty == 0.0 or len(bests) == 1:
        return sum(bests)

    # Sort strongest-first, apply geometric attenuation — this corresponds to
    # marginal information gain under the assumption each additional (partially
    # correlated) group adds (1 − ρ) new information relative to the previous.
    bests.sort(reverse=True)
    total = 0.0
    attenuation = 1.0
    for b in bests:
        total += b * attenuation
        attenuation *= (1.0 - correlation_penalty)
    return total


def _shrinkage_win_rate(
    observed_wr: float,
    sample_size: int,
    prior_wr: float = 0.5,
    prior_weight: int = 30,
) -> float:
    """Bayesian shrinkage of observed win-rate toward a neutral prior.

    Implements the conjugate Beta-Binomial posterior mean:
        ŵ = (n·p̂ + k·w₀) / (n + k)
    where n = sample_size, p̂ = observed_wr, w₀ = prior_wr, k = prior_weight.
    This is mathematically equivalent to a Beta(k·w₀, k·(1-w₀)) prior updated
    with n Bernoulli trials, then taking the posterior mean.

    Rationale:
        Raw sample win-rates have variance p(1-p)/n. For n=10, a 70% WR has
        ~14% stderr — wildly unreliable. Shrinking toward 0.5 with k=30
        pseudo-trades gives a James-Stein-type estimator that dominates the
        MLE under squared-error loss when the true rate is near 0.5.

    Args:
        observed_wr: raw sample win-rate ∈ [0, 1]
        sample_size: number of trades observed (n)
        prior_wr: centre of shrinkage (default 0.5 = no edge)
        prior_weight: strength of prior in pseudo-trades (default 30)

    Returns:
        Shrunk posterior-mean win-rate ∈ [0, 1].
    """
    if sample_size <= 0:
        return prior_wr
    return (sample_size * observed_wr + prior_weight * prior_wr) / (
        sample_size + prior_weight
    )


def adaptive_min_confidence(
    base: float,
    regime: str,
    strategy_type: str = "trend",
    recent_win_rate: float | None = None,
    sample_size: int = 0,
) -> float:
    """Adjust min_confidence threshold based on regime + recent performance.

    Two adjustment axes, additive in log-odds-like fashion:

    1. Regime + strategy-type (existing, market-structure driven):
       - Trending_up + trend: lower bar (signals are higher quality)
       - Trending_down + LONG: very high bar (fighting the trend)
       - Sideways: higher bar for trend, lower for mean_reversion/grid
       - Volatile: high bar across the board
       - Transitioning: danger zone, highest bar

    2. Recent win-rate (new, performance-feedback driven):
       Shrinks the observed WR toward 0.5 with a Beta-Binomial posterior
       (prior_weight=20 pseudo-trades). If the shrunk posterior indicates
       no edge (< 0.45), raise the bar to demand stronger signals. If WR
       is genuinely strong (> 0.60 after shrinkage), lower the bar modestly
       to let the validated model trade more often. Ignored until ≥ 5
       trades are available (below that, the shrinkage already pins the
       estimate near 0.5 and the adjustment is negligible).

    Args:
        base: baseline min_confidence from strategy config
        regime: detected market regime
        strategy_type: "trend" | "breakout" | "divergence" | "mean_reversion" | "grid"
        recent_win_rate: rolling WR from last N trades, None if unknown
        sample_size: number of trades in the rolling window
    """
    # Base adjustments by regime
    regime_adj: dict[str, float] = {
        "trending_up": -0.05,
        "trending_down": +0.10,
        "sideways": +0.03,
        "volatile": +0.05,
        "transitioning": +0.08,
        "unknown": +0.05,
    }

    # Strategy-specific override in favorable regimes
    strategy_regime_bonus: dict[tuple[str, str], float] = {
        ("trend", "trending_up"): -0.03,       # trend + trending_up = ideal
        ("breakout", "volatile"): -0.03,        # breakout loves volatility
        ("mean_reversion", "sideways"): -0.05,  # MR + sideways = ideal
        ("mean_reversion", "trending_down"): -0.03,  # contrarian opportunity
        ("grid", "sideways"): -0.05,            # grid + sideways = ideal
    }

    adj = regime_adj.get(regime, 0.0)
    adj += strategy_regime_bonus.get((strategy_type, regime), 0.0)

    # Performance-feedback adjustment (shrunk for small-sample bias)
    if recent_win_rate is not None and sample_size >= 5:
        shrunk = _shrinkage_win_rate(
            recent_win_rate, sample_size, prior_wr=0.5, prior_weight=20
        )
        if shrunk < 0.40:
            adj += 0.08   # strong negative signal — demand higher confidence
        elif shrunk < 0.45:
            adj += 0.04
        elif shrunk > 0.60:
            adj -= 0.03   # validated edge — allow slightly lower bar
        elif shrunk > 0.55:
            adj -= 0.015

    return max(0.50, min(base + adj, 0.90))


def _emit_strategy_decision(strategy: object, features: FeatureVector, signal: Optional[Signal]) -> None:
    """Emit ``strategy_decision`` event with the strategy's pre-risk verdict.

    Skips HOLDs (signal is None) — every tick produces a HOLD for most
    strategies and emitting them would drown out the log. Dashboard's
    "Strategy Decisions" view reads these to show what the strategy
    *wanted to do* before risk gates filtered it.

    Failure-tolerant: telemetry must never break the strategy pipeline.
    """
    if signal is None:
        return
    try:
        from monitoring.event_log import EventType, get_event_log
        from risk.decision_tracer import feature_snapshot_dict

        name = getattr(strategy, "NAME", "") or type(strategy).__name__
        get_event_log().emit(
            EventType.STRATEGY_DECISION,
            strategy=name,
            symbol=getattr(features, "symbol", "") or "",
            direction=getattr(signal.direction, "value", str(signal.direction)),
            confidence=round(float(signal.confidence), 4),
            reason=getattr(signal, "reason", "") or "",
            signal_id=getattr(signal, "signal_id", "") or "",
            suggested_quantity=getattr(signal, "suggested_quantity", 0.0),
            stop_loss=getattr(signal, "stop_loss_price", 0.0),
            take_profit=getattr(signal, "take_profit_price", 0.0),
            feature_snapshot=feature_snapshot_dict(features),
        )
    except Exception:
        pass


class BaseStrategy(ABC):
    """Абстрактный базовый класс для торговых стратегий."""

    NAME: str = "base"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Wrap generate_signal with per-symbol lock for thread-safety AND
        # post-call strategy_decision emit — the dashboard needs to see the
        # raw strategy verdict (reason + features) BEFORE risk gates filter it.
        original = cls.__dict__.get("generate_signal")
        if original is not None:
            def _locked_generate(self, features, has_open_position=False, entry_price=None, _orig=original):
                if not hasattr(self, "_symbol_locks"):
                    self._symbol_locks = {}
                lock = self._symbol_locks.setdefault(features.symbol, threading.Lock())
                with lock:
                    sig = _orig(self, features, has_open_position, entry_price)
                _emit_strategy_decision(self, features, sig)
                return sig
            cls.generate_signal = _locked_generate  # type: ignore[assignment]

    def __init__(self):
        self._symbol_locks: dict[str, threading.Lock] = {}

    @abstractmethod
    def generate_signal(
        self,
        features: FeatureVector,
        has_open_position: bool = False,
        entry_price: float | None = None,
    ) -> Optional[Signal]:
        """
        Проанализировать индикаторы и вернуть сигнал.

        Args:
            features: Текущий снимок индикаторов.
            has_open_position: True если уже есть открытая позиция для этого символа.
            entry_price: Цена входа в текущую позицию (для SL/TP).

        Returns:
            Signal или None (≡ HOLD).
        """
        ...
