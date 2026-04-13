"""
Binance WebSocket collector — приём рыночных данных в реальном времени.

Подписки:
  - @trade — сырые сделки
  - @kline_1m, @kline_5m, @kline_15m, @kline_1h, @kline_4h, @kline_1d — свечи

Автоматический reconnect с экспоненциальной задержкой (1s → 60s).
Heartbeat / stale-data detection встроено.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

import websockets
from loguru import logger

from collector.data_validator import validate_candle, validate_trade
from core.constants import (
    EVENT_NEW_CANDLE,
    EVENT_NEW_TRADE,
    WS_PING_INTERVAL_SEC,
    WS_RECONNECT_DELAYS,
    WS_STALE_DATA_TIMEOUT_SEC,
)
from core.events import EventBus
from core.models import Candle, MarketTrade
from database.repository import Repository

log = logger.bind(module="collector")

BINANCE_WS_BASE = "wss://stream.binance.com:9443/ws"
BINANCE_STREAM_BASE = "wss://stream.binance.com:9443/stream?streams="


class BinanceWebSocketCollector:
    """Подключается к Binance WS, парсит данные, валидирует, сохраняет, эмитит события."""

    def __init__(
        self,
        symbols: list[str],
        repo: Repository,
        bus: EventBus,
    ) -> None:
        self._symbols = [s.lower() for s in symbols]
        self._repo = repo
        self._bus = bus
        self._running = False
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._last_data_ts: float = 0.0
        self._reconnect_idx = 0
        # Counters
        self._msg_count: int = 0
        self._trade_count: int = 0
        self._candle_count: int = 0
        self._candle_closed_count: int = 0
        self._last_prices: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Запуск сбора данных (вызывать через asyncio.create_task)."""
        self._running = True
        log.info("Collector запущен для {}", self._symbols)
        while self._running:
            try:
                await self._connect_and_listen()
            except (
                websockets.ConnectionClosed,
                websockets.InvalidURI,
                OSError,
                asyncio.TimeoutError,
            ) as e:
                if not self._running:
                    break
                delay = self._next_reconnect_delay()
                log.warning("WS отключён ({}). Reconnect через {} сек...", type(e).__name__, delay)
                await asyncio.sleep(delay)
            except Exception as e:
                if not self._running:
                    break
                log.error("Unexpected collector error: {}", e)
                await asyncio.sleep(5)

    async def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False
        if self._ws:
            await self._ws.close()
        log.info("Collector остановлен")

    @property
    def last_data_age_sec(self) -> float:
        """Возраст последних полученных данных в секундах."""
        if self._last_data_ts == 0:
            return float("inf")
        return time.time() - self._last_data_ts

    @property
    def stats(self) -> dict:
        """Текущие метрики коллектора."""
        return {
            "connected": self._ws is not None and self._running,
            "msg_count": self._msg_count,
            "trade_count": self._trade_count,
            "candle_count": self._candle_count,
            "candle_closed": self._candle_closed_count,
            "last_prices": dict(self._last_prices),
            "symbols": [s.upper() for s in self._symbols],
            "data_age_sec": round(self.last_data_age_sec, 1) if self.last_data_age_sec != float("inf") else None,
        }

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _build_stream_url(self) -> str:
        """Формирует combined-stream URL для всех подписок."""
        streams: list[str] = []
        for sym in self._symbols:
            streams.append(f"{sym}@trade")
            for interval in ("1m", "5m", "15m", "1h", "4h", "1d"):
                streams.append(f"{sym}@kline_{interval}")
        return BINANCE_STREAM_BASE + "/".join(streams)

    async def _connect_and_listen(self) -> None:
        url = self._build_stream_url()
        log.info("Подключаюсь к Binance WS: {} потоков", len(self._symbols) * 7)

        async with websockets.connect(
            url,
            ping_interval=WS_PING_INTERVAL_SEC,
            ping_timeout=WS_PING_INTERVAL_SEC * 2,
            close_timeout=10,
        ) as ws:
            self._ws = ws
            self._reconnect_idx = 0  # reset on successful connect
            log.info("✅ Binance WS подключён")

            stale_checker = asyncio.create_task(self._stale_data_checker())
            try:
                async for raw_msg in ws:
                    if not self._running:
                        break
                    self._last_data_ts = time.time()
                    await self._handle_message(raw_msg)
            finally:
                stale_checker.cancel()

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    async def _handle_message(self, raw: str) -> None:
        self._msg_count += 1
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("Невалидный JSON от WS")
            return

        # Combined stream формат: {"stream": "...", "data": {...}}
        data = msg.get("data", msg)
        event_type = data.get("e")

        if event_type == "trade":
            await self._handle_trade(data)
        elif event_type == "kline":
            await self._handle_kline(data)

    async def _handle_trade(self, data: dict) -> None:
        try:
            trade = MarketTrade(
                timestamp=data["T"],
                symbol=data["s"],
                price=float(data["p"]),
                quantity=float(data["q"]),
                is_buyer_maker=data["m"],
            )
        except (KeyError, ValueError, TypeError) as e:
            log.warning("Failed to parse trade data: {}", e)
            return
        if not validate_trade(trade):
            return
        self._trade_count += 1
        self._last_prices[trade.symbol] = trade.price

        # Сохраняем в БД (async thread)
        await asyncio.to_thread(self._repo.insert_trade, trade)
        # Эмитим событие для других модулей
        await self._bus.emit(EVENT_NEW_TRADE, trade)

    async def _handle_kline(self, data: dict) -> None:
        try:
            k = data["k"]
            candle = Candle(
                timestamp=k["t"],
                symbol=k["s"],
                interval=k["i"],
                open=float(k["o"]),
                high=float(k["h"]),
                low=float(k["l"]),
                close=float(k["c"]),
                volume=float(k["v"]),
                trades_count=k.get("n", 0),
            )
        except (KeyError, ValueError, TypeError) as e:
            log.warning("Failed to parse kline data: {}", e)
            return
        if not validate_candle(candle):
            return
        self._candle_count += 1

        # Upsert свечу (обновляется до закрытия)
        await asyncio.to_thread(self._repo.upsert_candle, candle)

        # Эмитим событие только для закрытой свечи
        is_closed = k.get("x", False)
        if is_closed:
            self._candle_closed_count += 1
            await self._bus.emit(EVENT_NEW_CANDLE, candle)

    # ------------------------------------------------------------------
    # Reconnect & stale data
    # ------------------------------------------------------------------

    def _next_reconnect_delay(self) -> int:
        delay = WS_RECONNECT_DELAYS[min(self._reconnect_idx, len(WS_RECONNECT_DELAYS) - 1)]
        self._reconnect_idx += 1
        return delay

    async def _stale_data_checker(self) -> None:
        """Фоновая проверка: если данные не приходят > N секунд — reconnect."""
        while self._running:
            await asyncio.sleep(WS_STALE_DATA_TIMEOUT_SEC)
            age = self.last_data_age_sec
            if age > WS_STALE_DATA_TIMEOUT_SEC:
                log.warning(
                    "Stale data: {} сек без данных (лимит {}с). Пере-подключаюсь...",
                    int(age), WS_STALE_DATA_TIMEOUT_SEC,
                )
                if self._ws:
                    await self._ws.close()
                break
