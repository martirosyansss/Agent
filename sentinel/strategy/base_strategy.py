"""
Базовый класс стратегии.

Все торговые стратегии SENTINEL наследуют BaseStrategy.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Optional

from core.models import FeatureVector, Signal


def news_confidence_adjustment(f: FeatureVector, direction: str = "buy") -> tuple[float, str]:
    """
    Professional news-based confidence adjustment.
    
    Uses composite_score, signal_strength, urgency, category, and Fear&Greed
    to compute a confidence delta for trading signals.
    
    Args:
        f: Current FeatureVector with news fields.
        direction: "buy" or "sell" — determines how news aligns with signal.
        
    Returns:
        (delta, reason) where delta is confidence adjustment (-0.20 .. +0.15)
        and reason is a string for signal logging.
    """
    delta = 0.0
    parts: list[str] = []
    
    # Only act on actionable news signals
    if not f.news_actionable:
        # Still apply Fear & Greed contrarian for extreme values
        if f.fear_greed_index <= 15 and direction == "buy":
            delta += 0.05
            parts.append(f"F&G={f.fear_greed_index}(extreme_fear)")
        elif f.fear_greed_index >= 85 and direction == "buy":
            delta -= 0.05
            parts.append(f"F&G={f.fear_greed_index}(extreme_greed)")
        elif f.fear_greed_index <= 15 and direction == "sell":
            delta -= 0.05  # selling into extreme fear = risky
        elif f.fear_greed_index >= 85 and direction == "sell":
            delta += 0.05  # selling at extreme greed = smart
        reason = f"news={f.news_composite_score:+.2f}(weak)"
        if parts:
            reason += " " + " ".join(parts)
        return delta, reason
    
    score = f.news_composite_score
    strength = f.news_signal_strength

    # Direction alignment: does news agree with our trade direction?
    news_bullish = score > 0.1
    news_bearish = score < -0.1
    
    if direction == "buy":
        if news_bullish:
            # News agrees with buy → boost
            delta += min(0.15, score * strength * 0.3)
        elif news_bearish:
            # News contradicts buy → penalty (stronger)
            delta += max(-0.20, score * strength * 0.4)
    else:  # sell
        if news_bearish:
            # News agrees with sell → boost
            delta += min(0.15, abs(score) * strength * 0.3)
        elif news_bullish:
            # News contradicts sell → penalty
            delta += max(-0.20, -score * strength * 0.4)
    
    # Critical alert: amplify (both ways)
    if f.news_critical_alert:
        delta *= 1.5
        parts.append("CRITICAL")
    
    # Category bonuses for regulatory/macro (strongest market movers)
    if f.news_dominant_category in ("regulatory", "macro") and abs(score) > 0.2:
        delta *= 1.2
        parts.append(f"cat={f.news_dominant_category}")
    
    # Fear & Greed contrarian overlay
    if f.fear_greed_index <= 15:
        if direction == "buy":
            delta += 0.05  # extreme fear = contrarian buy
        parts.append(f"F&G={f.fear_greed_index}")
    elif f.fear_greed_index >= 85:
        if direction == "buy":
            delta -= 0.05  # extreme greed = caution
        elif direction == "sell":
            delta += 0.05
        parts.append(f"F&G={f.fear_greed_index}")
    
    # Clamp
    delta = max(-0.20, min(0.15, delta))
    
    reason = f"news={score:+.2f}(str={strength:.0%})"
    if parts:
        reason += " " + " ".join(parts)
    
    return round(delta, 3), reason


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
