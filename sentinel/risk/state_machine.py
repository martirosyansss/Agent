"""
Risk State Machine — режимы безопасности SENTINEL.

NORMAL → REDUCED → SAFE → STOP
Каждый переход понижает торговые возможности.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from core.constants import EVENT_RISK_STATE_CHANGED
from core.events import EventBus
from core.models import RiskState

logger = logging.getLogger(__name__)


class RiskStateMachine:
    """State Machine: NORMAL → REDUCED → SAFE → STOP."""

    def __init__(
        self,
        event_bus: EventBus,
        max_daily_loss: float = 50.0,
    ) -> None:
        self._event_bus = event_bus
        self._max_daily_loss = max_daily_loss
        self._state = RiskState.NORMAL
        self._last_change_ts: int = 0

    @property
    def state(self) -> RiskState:
        return self._state

    def evaluate(self, daily_pnl: float) -> RiskState:
        """Пересчитать состояние на основе дневного PnL.

        Пороги:
          loss > 30% лимита → REDUCED
          loss > 70% лимита → SAFE
          loss > 100% лимита → STOP
        """
        loss = abs(min(daily_pnl, 0))
        limit = self._max_daily_loss

        if loss >= limit:
            new_state = RiskState.STOP
        elif loss >= limit * 0.7:
            new_state = RiskState.SAFE
        elif loss >= limit * 0.3:
            new_state = RiskState.REDUCED
        else:
            new_state = RiskState.NORMAL

        return new_state

    async def update(self, daily_pnl: float) -> Optional[RiskState]:
        """Обновить состояние. Возвращает новый state если изменился."""
        new_state = self.evaluate(daily_pnl)
        if new_state != self._state:
            old = self._state
            self._state = new_state
            self._last_change_ts = int(time.time() * 1000)
            logger.warning("Risk state: %s → %s (daily PnL: $%.2f)", old.value, new_state.value, daily_pnl)
            await self._event_bus.emit(EVENT_RISK_STATE_CHANGED, old, new_state, f"Daily PnL: ${daily_pnl:.2f}")
            return new_state
        return None

    def reset(self) -> None:
        """Ручной сброс в NORMAL (начало нового дня)."""
        self._state = RiskState.NORMAL
        logger.info("Risk state manually reset to NORMAL")
