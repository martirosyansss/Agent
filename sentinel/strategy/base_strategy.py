"""
Базовый класс стратегии.

Все торговые стратегии SENTINEL наследуют BaseStrategy.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

from core.models import FeatureVector, Signal


# ──────────────────────────────────────────────
# News time decay tracking
# ──────────────────────────────────────────────
_last_news_update_ts: float = 0.0
_NEWS_DECAY_HALF_LIFE_SEC: float = 7200.0  # 2 hours: impact halves every 2h
_NEWS_MAX_AGE_SEC: float = 14400.0         # 4 hours: ignore news older than 4h


def update_news_timestamp() -> None:
    """Call when fresh news data is received from NewsCollector."""
    global _last_news_update_ts
    _last_news_update_ts = time.time()


def _news_decay_factor() -> float:
    """Exponential decay factor for news impact based on age.

    Returns 1.0 for fresh news, decays toward 0 as news ages.
    After 4h (MAX_AGE), returns 0 — news is stale.
    """
    if _last_news_update_ts <= 0:
        return 0.5  # no timestamp = assume moderately stale
    age = time.time() - _last_news_update_ts
    if age >= _NEWS_MAX_AGE_SEC:
        return 0.0  # completely stale
    # Exponential decay: factor = 2^(-age / half_life)
    import math
    return math.pow(2.0, -age / _NEWS_DECAY_HALF_LIFE_SEC)


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

    # Apply time decay to news impact (except Fear & Greed which is always current)
    decay = _news_decay_factor()
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
        penalty_w = w * 1.3  # penalties are stronger than boosts

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


class BaseStrategy(ABC):
    """Абстрактный базовый класс для торговых стратегий."""

    NAME: str = "base"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Wrap generate_signal with per-symbol lock for thread-safety
        original = cls.__dict__.get("generate_signal")
        if original is not None:
            def _locked_generate(self, features, has_open_position=False, entry_price=None, _orig=original):
                if not hasattr(self, "_symbol_locks"):
                    self._symbol_locks = {}
                lock = self._symbol_locks.setdefault(features.symbol, threading.Lock())
                with lock:
                    return _orig(self, features, has_open_position, entry_price)
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
