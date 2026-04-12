"""
Kill Switch — аварийная остановка.

Вызывается:
1. Автоматически при STOP state
2. Вручную через Telegram: /kill
3. Вручную через Dashboard: EMERGENCY STOP
"""

from __future__ import annotations

import logging
from typing import Callable, Coroutine, Optional

from core.constants import EVENT_EMERGENCY_STOP
from core.events import EventBus

logger = logging.getLogger(__name__)


class KillSwitch:
    """Аварийная остановка SENTINEL."""

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._activated = False

        # Callbacks, устанавливаются из main
        self.on_close_all_positions: Optional[Callable[[], Coroutine]] = None
        self.on_cancel_all_orders: Optional[Callable[[], Coroutine]] = None
        self.on_stop_trading: Optional[Callable[[], Coroutine]] = None

    @property
    def is_activated(self) -> bool:
        return self._activated

    async def activate(self, reason: str = "Manual kill") -> None:
        """Активировать аварийную остановку."""
        if self._activated:
            logger.warning("Kill switch already activated")
            return

        self._activated = True
        logger.critical("KILL SWITCH ACTIVATED: %s", reason)

        # 1. Отменить все ордера
        if self.on_cancel_all_orders:
            try:
                await self.on_cancel_all_orders()
            except Exception as exc:
                logger.error("Failed to cancel orders: %s", exc)

        # 2. Закрыть все позиции
        if self.on_close_all_positions:
            try:
                await self.on_close_all_positions()
            except Exception as exc:
                logger.error("Failed to close positions: %s", exc)

        # 3. Остановить торговлю
        if self.on_stop_trading:
            try:
                await self.on_stop_trading()
            except Exception as exc:
                logger.error("Failed to stop trading: %s", exc)

        # 4. Оповестить систему
        await self._event_bus.emit(EVENT_EMERGENCY_STOP, reason)

    def reset(self) -> None:
        """Сброс kill switch (после ручной проверки)."""
        self._activated = False
        logger.info("Kill switch reset")
