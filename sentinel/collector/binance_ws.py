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
        await self._backfill_gaps()
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
    # Backfill gaps via REST API
    # ------------------------------------------------------------------

    async def _backfill_gaps(self) -> None:
        """При старте докачивает пропущенные свечи через Binance REST API.

        Для каждого символа и интервала проверяет последнюю свечу в БД.
        Если с момента последней свечи прошло больше одного периода —
        значит бот был выключен и в БД есть дыра. Докачиваем через REST.

        Максимум 1000 свечей за один запрос (ограничение Binance).
        При 3-дневном простое: 72 свечи по 1h — один запрос.
        """
        import requests as _requests

        INTERVALS = ["1m", "5m", "15m", "1h", "4h", "1d"]
        INTERVAL_MS: dict[str, int] = {
            "1m":     60_000,
            "5m":    300_000,
            "15m":   900_000,
            "1h":  3_600_000,
            "4h": 14_400_000,
            "1d": 86_400_000,
        }
        BINANCE_REST = "https://api.binance.com/api/v3/klines"
        now_ms = int(time.time() * 1000)

        for symbol in self._symbols:
            sym_upper = symbol.upper()
            for interval in INTERVALS:
                try:
                    last = await asyncio.to_thread(
                        self._repo.get_latest_candle, sym_upper, interval
                    )
                except Exception as e:
                    log.warning("Backfill: не удалось получить последнюю свечу {}/{}: {}", sym_upper, interval, e)
                    continue

                interval_ms = INTERVAL_MS[interval]

                # Для коротких интервалов (1m/5m/15m) — подгружаем начальный
                # набор свечей даже если истории нет, чтобы дашборд сразу
                # показывал заполненный график.
                SHORT_INTERVALS = {"1m", "5m", "15m"}
                # Кол-во свечей для начальной загрузки (совпадает с лимитами дашборда)
                INITIAL_COUNTS = {"1m": 150, "5m": 150, "15m": 120}

                if not last:
                    if interval in SHORT_INTERVALS:
                        # Нет истории — подгружаем начальный набор
                        count = INITIAL_COUNTS.get(interval, 150)
                        start_ts = now_ms - count * interval_ms
                        log.info(
                            "Backfill: {}/{} — нет истории, загружаю {} свечей...",
                            sym_upper, interval, count,
                        )
                    else:
                        continue  # для 1h/4h/1d — скип (download_history.py)
                else:
                    last_ts: int = last["timestamp"]
                    gap_ms = now_ms - last_ts

                    if gap_ms < interval_ms * 2:
                        continue  # пропущено < 2 свечей — не считаем дырой

                    missing_count = int(gap_ms // interval_ms)
                    log.info(
                        "Backfill: {}/{} — обнаружена дыра ~{} свечей ({:.1f}ч), докачиваю...",
                        sym_upper, interval, missing_count, gap_ms / 3_600_000,
                    )
                    start_ts = last_ts + interval_ms

                try:
                    params = {
                        "symbol": sym_upper,
                        "interval": interval,
                        "startTime": start_ts,
                        "endTime": now_ms,
                        "limit": 1000,
                    }
                    resp = await asyncio.to_thread(
                        lambda p=params: _requests.get(BINANCE_REST, params=p, timeout=10)
                    )
                    if resp.status_code != 200:
                        log.warning("Backfill REST ошибка {}: {} {}", resp.status_code, sym_upper, interval)
                        continue

                    raw = resp.json()
                    if not raw:
                        continue

                    from core.models import Candle as _Candle
                    candles = [
                        _Candle(
                            timestamp=int(k[0]),
                            symbol=sym_upper,
                            interval=interval,
                            open=float(k[1]),
                            high=float(k[2]),
                            low=float(k[3]),
                            close=float(k[4]),
                            volume=float(k[5]),
                            trades_count=int(k[8]),
                        )
                        for k in raw
                    ]

                    inserted = await asyncio.to_thread(
                        self._repo.upsert_candles_batch, candles
                    )
                    log.info("Backfill: {}/{} — сохранено {} свечей ✅", sym_upper, interval, inserted)

                except Exception as e:
                    log.warning("Backfill: {}/{} — ошибка при докачке: {}", sym_upper, interval, e)

                await asyncio.sleep(0.3)  # rate-limit protection

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
