"""
Лёгкая pub/sub-шина для обмена событиями между модулями.

Использование:
    bus = EventBus()
    bus.subscribe("new_candle", my_handler)
    await bus.emit("new_candle", candle)
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

Handler = Callable[..., Coroutine[Any, Any, None]]


class EventBus:
    """Асинхронная шина событий (in-process pub/sub)."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Handler]] = defaultdict(list)

    def subscribe(self, event: str, handler: Handler) -> None:
        """Подписать обработчик на событие."""
        self._subscribers[event].append(handler)

    def unsubscribe(self, event: str, handler: Handler) -> None:
        """Отписать обработчик от события."""
        try:
            self._subscribers[event].remove(handler)
        except ValueError:
            pass

    async def emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Отправить событие всем подписчикам (параллельно)."""
        handlers = self._subscribers.get(event, [])
        if not handlers:
            return
        results = await asyncio.gather(
            *(h(*args, **kwargs) for h in handlers),
            return_exceptions=True,
        )
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "Event handler %s for '%s' raised: %s",
                    handlers[idx].__qualname__,
                    event,
                    result,
                )
