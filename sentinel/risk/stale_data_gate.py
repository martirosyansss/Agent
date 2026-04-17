"""
Stale Data Gate — block BUY entries when market data is too old.

The WebSocket collector publishes ``last_data_age_sec`` to the dashboard but
nothing gates trading on it. If the WS connection freezes (silent failure —
no disconnect event, but no trade/candle messages either), the reconnect
watchdog in ``collector/binance_ws.py`` will eventually force a reconnect
after ``MAX_DATA_AGE_SEC``, but between the freeze and the forced reconnect
strategies keep producing signals on the last known candle snapshot. Those
signals hit a real exchange at a real *current* price — the gap between
internal features and real market price can be large on a fast move.

This gate refuses BUY when the global data age exceeds ``max_age_sec``.
SELL is never blocked: closing a stale-indicator position is still safer
than holding it.

Design mirrors ``LiquidityGate`` / ``NewsCooldownGuard``:
- Stateless, single ``check()`` method returning an explicit Decision.
- SELL bypasses the gate (symmetric with every other entry gate).
- Fail-closed: when age is unknown / infinite, reject BUY.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from core.models import Direction

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StaleDataGateConfig:
    """Maximum acceptable WS data age (seconds) before BUY is blocked.

    Default is slightly above the collector's forced-reconnect threshold
    (``MAX_DATA_AGE_SEC = 60`` in constants.py): the collector will attempt
    reconnection first, and the gate fires only if the freeze persists.
    """
    max_age_sec: float = 90.0


@dataclass
class StaleDataDecision:
    approved: bool
    reason: str = ""
    data_age_sec: float = 0.0


class StaleDataGate:
    """Stateless WS-freshness gate."""

    def __init__(self, config: Optional[StaleDataGateConfig] = None) -> None:
        self._cfg = config or StaleDataGateConfig()

    def check(
        self,
        direction: Direction,
        data_age_sec: Optional[float],
    ) -> StaleDataDecision:
        """Approve / reject a candidate entry on data-freshness grounds.

        ``data_age_sec`` is the global age of the most recent WS message,
        as reported by ``BinanceWebSocketCollector.last_data_age_sec``.
        ``None`` or ``float('inf')`` is treated as "no data ever seen" and
        rejects BUY (fail-closed).
        """
        if direction != Direction.BUY:
            return StaleDataDecision(approved=True, reason="Non-BUY not gated")

        if data_age_sec is None or data_age_sec != data_age_sec or data_age_sec == float("inf"):
            return StaleDataDecision(
                approved=False,
                reason="Stale-data gate: no market data yet (fail-closed)",
                data_age_sec=float("inf"),
            )

        if data_age_sec > self._cfg.max_age_sec:
            return StaleDataDecision(
                approved=False,
                reason=(
                    f"Stale-data gate: market data age {data_age_sec:.1f}s > "
                    f"{self._cfg.max_age_sec:.1f}s (WS frozen?)"
                ),
                data_age_sec=data_age_sec,
            )

        return StaleDataDecision(
            approved=True,
            reason=f"Data fresh ({data_age_sec:.1f}s <= {self._cfg.max_age_sec:.1f}s)",
            data_age_sec=data_age_sec,
        )
