"""
News cooldown — block new entries for N hours after a critical bearish event.

The existing ``base_strategy.news_should_block_entry`` only blocks while the
critical-alert flag is currently TRUE. Critical alerts often have a short
half-life: the news vendor flips the flag back off as the headline ages,
even though markets keep re-pricing on event memory for hours.

This guard records the WALL-CLOCK time of every critical bearish event
seen in the feature stream and refuses BUY entries until the configured
cooldown elapses since the most recent event. SELL is never blocked.

The guard is observation-driven: every call to ``check`` updates internal
state from the passed FeatureVector. There is no external scheduler — it
just remembers what it has seen.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

from core.models import FeatureVector

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NewsCooldownConfig:
    """Cooldown durations per news category. Anything not listed defaults
    to ``default_cooldown_sec``. Zero disables cooldown for that category.

    Calibration: matches event-study return persistence. Security incidents
    (exchange hacks) reprice for 4-6h then mean-revert hard, so 4h cooldown.
    Regulatory/macro decisions take 12-24h to digest fully.
    """
    security_cooldown_sec: float = 4 * 3600        # 4h after a hack/exploit
    regulatory_cooldown_sec: float = 12 * 3600     # 12h after regulatory event
    macro_cooldown_sec: float = 12 * 3600          # 12h after macro event
    default_cooldown_sec: float = 2 * 3600         # 2h baseline
    # Triggering thresholds (must hold to start a cooldown).
    min_signal_strength: float = 0.30
    max_score_for_bearish: float = -0.25


@dataclass
class NewsCooldownDecision:
    approved: bool
    reason: str = ""
    last_event_age_sec: Optional[float] = None
    cooldown_remaining_sec: float = 0.0


class NewsCooldownGuard:
    """Records critical bearish events and blocks entries during cooldown."""

    def __init__(
        self,
        config: Optional[NewsCooldownConfig] = None,
        time_provider: Optional[Callable[[], float]] = None,
    ) -> None:
        self._cfg = config or NewsCooldownConfig()
        self._time = time_provider or time.time
        # Track the timestamp + cooldown duration of the most recent event
        # per category. Map category → (event_ts, cooldown_sec).
        self._last_events: dict[str, tuple[float, float]] = {}

    def _category_cooldown(self, category: str) -> float:
        if category == "security":
            return self._cfg.security_cooldown_sec
        if category == "regulatory":
            return self._cfg.regulatory_cooldown_sec
        if category == "macro":
            return self._cfg.macro_cooldown_sec
        return self._cfg.default_cooldown_sec

    def _is_event_triggering(self, features: FeatureVector) -> bool:
        """A feature snapshot triggers cooldown when news is critical AND
        bearish AND strong enough to matter."""
        if not getattr(features, "news_critical_alert", False):
            return False
        score = getattr(features, "news_composite_score", 0.0)
        strength = getattr(features, "news_signal_strength", 0.0)
        return (
            score <= self._cfg.max_score_for_bearish
            and strength >= self._cfg.min_signal_strength
        )

    def check(self, features: FeatureVector) -> NewsCooldownDecision:
        """Update internal state from features and decide whether to allow entry.

        Even when the call passes, it may have just RECORDED a new event from
        the current features (if they describe a critical bearish situation),
        and subsequent calls within the cooldown window will be blocked.
        """
        now = self._time()

        # Step 1: record any new triggering event from this feature snapshot.
        if self._is_event_triggering(features):
            category = getattr(features, "news_dominant_category", "") or "default"
            cooldown_sec = self._category_cooldown(category)
            self._last_events[category] = (now, cooldown_sec)
            logger.warning(
                "News cooldown started: category=%s for %.0fs (score=%.2f, strength=%.2f)",
                category, cooldown_sec,
                getattr(features, "news_composite_score", 0.0),
                getattr(features, "news_signal_strength", 0.0),
            )

        # Step 2: check the longest-remaining cooldown across all categories.
        worst_remaining = 0.0
        worst_category = ""
        worst_age = None
        for category, (event_ts, cd_sec) in self._last_events.items():
            age = now - event_ts
            remaining = cd_sec - age
            if remaining > worst_remaining:
                worst_remaining = remaining
                worst_category = category
                worst_age = age

        if worst_remaining > 0:
            return NewsCooldownDecision(
                approved=False,
                reason=(
                    f"News cooldown active: {worst_category} event {worst_age:.0f}s ago, "
                    f"{worst_remaining / 60:.1f}m remaining"
                ),
                last_event_age_sec=worst_age,
                cooldown_remaining_sec=worst_remaining,
            )

        return NewsCooldownDecision(
            approved=True,
            reason="No active news cooldown",
        )

    def force_reset(self, category: Optional[str] = None) -> None:
        """Operator-only: clear cooldowns. Useful for testing and manual override."""
        if category:
            self._last_events.pop(category, None)
        else:
            self._last_events.clear()
        logger.warning("News cooldown reset: %s", category or "all")

    def snapshot(self) -> dict:
        now = self._time()
        return {
            category: {
                "event_ts": ev_ts,
                "age_sec": now - ev_ts,
                "cooldown_sec": cd,
                "remaining_sec": max(0.0, cd - (now - ev_ts)),
            }
            for category, (ev_ts, cd) in self._last_events.items()
        }
