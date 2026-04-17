"""
In-memory rolling close-price cache used by the correlation guard.

The collector and feature builder already pull recent candles per symbol
on every loop tick. Rather than re-querying SQLite or the WebSocket buffer
each time the risk pipeline asks for correlation, we keep a single bounded
deque per symbol that the main loop updates as features are built.

Thread-safety: not needed — Sentinel's main loop is single-threaded asyncio.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable


class PriceHistoryCache:
    """Bounded close-price cache, ``{symbol → deque[float]}``.

    Used as the price source for ``risk.correlation_guard.CorrelationGuard``.
    Default ``max_history=240`` covers ~10 days of 1h closes — long enough
    for a stable Pearson correlation, short enough that regime shifts move
    the metric within a few sessions.
    """

    def __init__(self, max_history: int = 240) -> None:
        self._max = max_history
        self._cache: dict[str, deque[float]] = {}

    def update_from_candles(self, symbol: str, closes: Iterable[float]) -> None:
        """Replace the cache for ``symbol`` with the latest closes (in order).

        The main loop calls this once per feature build, passing the same
        ``closes_1h`` list it just gave to the indicators. Cheap; we don't
        deduplicate against existing entries because the input is already
        the canonical recent window.
        """
        buf = self._cache.get(symbol)
        if buf is None:
            buf = deque(maxlen=self._max)
            self._cache[symbol] = buf
        buf.clear()
        for c in closes:
            if c is not None and c > 0:
                buf.append(float(c))

    def push_close(self, symbol: str, close: float) -> None:
        """Append a single close — useful from tick handlers."""
        if close is None or close <= 0:
            return
        buf = self._cache.get(symbol)
        if buf is None:
            buf = deque(maxlen=self._max)
            self._cache[symbol] = buf
        buf.append(float(close))

    def snapshot(self) -> dict[str, list[float]]:
        """Materialised copy suitable to pass to CorrelationGuard.check()."""
        return {sym: list(buf) for sym, buf in self._cache.items() if buf}

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._cache and bool(self._cache[symbol])

    def __len__(self) -> int:
        return len(self._cache)
