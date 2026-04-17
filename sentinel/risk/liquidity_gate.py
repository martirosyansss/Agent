"""
Liquidity gate — block entries when the order would dominate available liquidity
or when the market session is unusually thin.

Two independent checks:

1. **Volume floor.** A BUY into a candle whose volume is well below the
   rolling average pays disproportionate slippage and gets terrible fills.
   Block when ``volume_ratio < min_volume_ratio_buy``. Strategies often
   already include a volume filter; this is a defence-in-depth against
   strategies that don't (and protects all of them at the risk layer).

2. **Order-vs-volume cap.** Even in normal volume, an order that's too
   large relative to the candle's actual traded notional will move the
   tape against itself. Block when
   ``candidate_notional_usd > max_pct_of_recent_volume × recent_notional``.

3. **Session window (opt-in).** UTC-hour blacklist for sessions where
   crypto liquidity is empirically thinnest (Asian dead-hours 02:00-05:00
   UTC). Disabled by default — enable when backtests show those hours
   produce worse fills.

SELL is never blocked — exit liquidity is always preferable to staying in
a deteriorating position.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from core.models import Direction, FeatureVector

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiquidityGateConfig:
    # Hard floor on volume_ratio (current volume / 20-bar SMA).
    # 0.4 = current volume is at least 40% of recent average.
    min_volume_ratio_buy: float = 0.4

    # Cap order notional as fraction of the most recent candle's notional.
    # 0.05 = order may be up to 5% of last bar's traded value.
    max_pct_of_recent_volume: float = 0.05

    # Optional UTC-hour blacklist. Empty = disabled.
    blocked_utc_hours: tuple[int, ...] = field(default_factory=tuple)


@dataclass
class LiquidityDecision:
    approved: bool
    reason: str = ""
    volume_ratio: float = 0.0
    order_pct_of_volume: float = 0.0


class LiquidityGate:
    """Stateless liquidity check. Inject candle volume separately when known —
    most call sites can rely on the FeatureVector's ``volume_ratio`` alone."""

    def __init__(
        self,
        config: Optional[LiquidityGateConfig] = None,
        time_provider: Optional[Callable[[], float]] = None,
    ) -> None:
        self._cfg = config or LiquidityGateConfig()
        self._time = time_provider

    def _utc_hour(self) -> int:
        ts = self._time() if self._time else __import__("time").time()
        return datetime.fromtimestamp(ts, tz=timezone.utc).hour

    def check(
        self,
        direction: Direction,
        features: FeatureVector,
        candidate_notional_usd: float = 0.0,
        recent_candle_notional_usd: float = 0.0,
    ) -> LiquidityDecision:
        """Approve / reject a candidate entry on liquidity grounds.

        ``candidate_notional_usd`` and ``recent_candle_notional_usd`` are
        optional — when the caller doesn't provide them, the order-vs-volume
        cap is skipped and only the volume_ratio floor is enforced.
        """
        # Never block exits.
        if direction != Direction.BUY:
            return LiquidityDecision(approved=True, reason="Non-BUY not gated")

        vr = getattr(features, "volume_ratio", 0.0)

        # 1. Volume floor.
        if vr > 0 and vr < self._cfg.min_volume_ratio_buy:
            return LiquidityDecision(
                approved=False,
                reason=(
                    f"Liquidity gate: volume_ratio {vr:.2f} < "
                    f"{self._cfg.min_volume_ratio_buy:.2f} (thin market)"
                ),
                volume_ratio=vr,
            )

        # 2. Order vs recent volume.
        if candidate_notional_usd > 0 and recent_candle_notional_usd > 0:
            order_pct = candidate_notional_usd / recent_candle_notional_usd
            if order_pct > self._cfg.max_pct_of_recent_volume:
                return LiquidityDecision(
                    approved=False,
                    reason=(
                        f"Liquidity gate: order {order_pct * 100:.1f}% of last bar's "
                        f"notional > cap {self._cfg.max_pct_of_recent_volume * 100:.1f}%"
                    ),
                    volume_ratio=vr,
                    order_pct_of_volume=order_pct,
                )

        # 3. Session window.
        if self._cfg.blocked_utc_hours:
            hour = self._utc_hour()
            if hour in self._cfg.blocked_utc_hours:
                return LiquidityDecision(
                    approved=False,
                    reason=f"Liquidity gate: trading blocked at UTC hour {hour:02d}",
                    volume_ratio=vr,
                )

        return LiquidityDecision(
            approved=True,
            reason="Liquidity OK",
            volume_ratio=vr,
        )
