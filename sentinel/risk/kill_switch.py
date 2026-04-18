"""
Kill Switch — аварийная остановка.

Вызывается:
1. Автоматически при STOP state
2. Вручную через Telegram: /kill
3. Вручную через Dashboard: EMERGENCY STOP
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Coroutine, Optional

from core.constants import EVENT_EMERGENCY_STOP
from core.events import EventBus
from monitoring.event_log import EventType, emit_component_error, get_event_log

logger = logging.getLogger(__name__)

_KILL_STEP_TIMEOUT = 10  # seconds per step
_CLOSE_POSITIONS_RETRIES = 3  # retry close_positions on failure


class KillSwitch:
    """Аварийная остановка SENTINEL."""

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._activated = False
        self._errors: list[str] = []

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
        self._errors = []
        logger.critical("KILL SWITCH ACTIVATED: %s", reason)
        try:
            get_event_log().emit(
                EventType.GUARD_TRIPPED,
                guard="kill_switch",
                reason=reason,
            )
        except Exception:
            pass

        # 1. Отменить все ордера
        if self.on_cancel_all_orders:
            try:
                await asyncio.wait_for(self.on_cancel_all_orders(), timeout=_KILL_STEP_TIMEOUT)
            except asyncio.TimeoutError:
                self._errors.append("cancel_orders timed out")
                logger.error("Kill switch: cancel orders timed out after %ds", _KILL_STEP_TIMEOUT)
            except Exception as exc:
                self._errors.append(f"cancel_orders: {exc}")
                logger.error("Failed to cancel orders: %s", exc)

        # 2. Закрыть все позиции (с retry)
        if self.on_close_all_positions:
            close_ok = False
            for attempt in range(1, _CLOSE_POSITIONS_RETRIES + 1):
                try:
                    await asyncio.wait_for(self.on_close_all_positions(), timeout=_KILL_STEP_TIMEOUT)
                    close_ok = True
                    break
                except asyncio.TimeoutError:
                    self._errors.append(f"close_positions timed out (attempt {attempt})")
                    logger.error("Kill switch: close positions timed out (attempt %d/%d)", attempt, _CLOSE_POSITIONS_RETRIES)
                except Exception as exc:
                    self._errors.append(f"close_positions attempt {attempt}: {exc}")
                    logger.error("Failed to close positions (attempt %d/%d): %s", attempt, _CLOSE_POSITIONS_RETRIES, exc)
            if not close_ok:
                logger.critical("KILL SWITCH: ALL %d CLOSE ATTEMPTS FAILED — positions may still be open!", _CLOSE_POSITIONS_RETRIES)
                emit_component_error(
                    "kill_switch.close_positions",
                    f"all {_CLOSE_POSITIONS_RETRIES} close attempts failed — positions may still be open",
                    severity="critical",
                    retries=_CLOSE_POSITIONS_RETRIES,
                    errors=list(self._errors),
                )

        # 3. Остановить торговлю (always attempt even if prior steps failed)
        if self.on_stop_trading:
            try:
                await asyncio.wait_for(self.on_stop_trading(), timeout=_KILL_STEP_TIMEOUT)
            except asyncio.TimeoutError:
                self._errors.append("stop_trading timed out")
                logger.error("Kill switch: stop trading timed out after %ds", _KILL_STEP_TIMEOUT)
            except Exception as exc:
                self._errors.append(f"stop_trading: {exc}")
                logger.error("Failed to stop trading: %s", exc)

        if self._errors:
            logger.critical("Kill switch completed WITH ERRORS: %s", self._errors)
            emit_component_error(
                "kill_switch.activate",
                f"kill switch finished with {len(self._errors)} error(s)",
                severity="critical",
                errors=list(self._errors),
            )
        else:
            logger.info("Kill switch completed successfully")

        # 4. Оповестить систему
        await self._event_bus.emit(EVENT_EMERGENCY_STOP, reason)

    def reset(self) -> None:
        """Сброс kill switch (после ручной проверки)."""
        self._activated = False
        self._errors = []
        logger.info("Kill switch reset")
