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
        self._event_counts: dict[str, int] = defaultdict(int)
        self._event_last_ts: dict[str, float] = {}

    def get_event_stats(self) -> dict[str, dict]:
        """Статистика по всем событиям."""
        import time
        now = time.time()
        return {
            event: {
                "count": self._event_counts.get(event, 0),
                "last_ago_sec": round(now - self._event_last_ts[event], 1) if event in self._event_last_ts else None,
            }
            for event in set(list(self._event_counts.keys()) + list(self._subscribers.keys()))
        }

    def subscribe(self, event: str, handler: Handler) -> None:
        """Подписать обработчик на событие."""
        self._subscribers[event].append(handler)

    def unsubscribe(self, event: str, handler: Handler) -> None:
        """Отписать обработчик от события."""
        try:
            self._subscribers[event].remove(handler)
        except ValueError:
            logger.debug("Handler %s was not subscribed to '%s'", handler.__qualname__, event)

    # Events where handler failures must be surfaced (not silently swallowed)
    CRITICAL_EVENTS = frozenset({
        "order_filled", "position_opened", "position_closed",
    })

    async def emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Отправить событие всем подписчикам (параллельно)."""
        import time
        self._event_counts[event] += 1
        self._event_last_ts[event] = time.time()

        handlers = self._subscribers.get(event, [])
        if not handlers:
            return
        results = await asyncio.gather(
            *(h(*args, **kwargs) for h in handlers),
            return_exceptions=True,
        )
        critical_failures: list[tuple[str, Exception]] = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                handler_name = handlers[idx].__qualname__
                logger.error(
                    "Event handler %s for '%s' raised: %s",
                    handler_name,
                    event,
                    result,
                )
                if event in self.CRITICAL_EVENTS:
                    critical_failures.append((handler_name, result))
        if critical_failures:
            raise RuntimeError(
                f"Critical handler(s) failed for '{event}': "
                + ", ".join(f"{name}: {err}" for name, err in critical_failures)
            )
