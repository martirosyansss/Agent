"""
Binance Futures Collector — сбор On-Chain метрик (Funding Rate, Open Interest).
Использует бесплатные публичные эндпоинты Binance.
"""

from __future__ import annotations

import asyncio
from typing import Optional
from loguru import logger
import aiohttp

class BinanceFuturesCollector:
    def __init__(self, symbols: list[str], update_interval: int = 180):
        # We need symbols in format like BTCUSDT
        self._symbols = [s.upper() for s in symbols]
        self._update_interval = update_interval
        self._running = False
        self._session: Optional[aiohttp.ClientSession] = None
        
        # In-memory storage of metrics
        self._funding_rates: dict[str, float] = {}
        self._open_interests: dict[str, float] = {}

    async def start(self) -> None:
        """Запуск фонового сбора данных."""
        self._running = True
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        logger.info("BinanceFuturesCollector запущен для {}", self._symbols)
        
        while self._running:
            try:
                await self._fetch_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("BinanceFuturesCollector error: {}", e)
            
            # Спим заданное количество секунд между обновлениями
            for _ in range(self._update_interval):
                if not self._running:
                    break
                await asyncio.sleep(1.0)

    async def stop(self) -> None:
        """Остановка фонового сбора."""
        self._running = False
        if self._session:
            await self._session.close()
        logger.info("BinanceFuturesCollector остановлен")

    async def _fetch_open_interest_single(self, sym: str, sem: asyncio.Semaphore) -> None:
        """Скачивает Open Interest для одной монеты с контролем rate limit через семафор."""
        async with sem:
            if not self._running or not self._session:
                return
            try:
                async with self._session.get(f"https://fapi.binance.com/fapi/v1/openInterest?symbol={sym}") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self._open_interests[sym] = float(data.get("openInterest", 0.0))
            except Exception as e:
                logger.debug("Failed to fetch fapi openInterest for {}: {}", sym, e)

    async def _fetch_data(self) -> None:
        if not self._session:
            return
            
        # 1. Загрузка Premium Index (содержит Funding Rate для всех монет разом, O(1) запрос)
        try:
            async with self._session.get("https://fapi.binance.com/fapi/v1/premiumIndex") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for item in data:
                        sym = item.get("symbol")
                        if sym in self._symbols:
                            self._funding_rates[sym] = float(item.get("lastFundingRate", 0.0))
        except Exception as e:
            logger.debug("Failed to fetch fapi premiumIndex: {}", e)

        # 2. Загрузка Open Interest через asyncio.gather + Semaphore (устранение "Bottle-neck" O(N) лупа)
        # 5 параллельных потоков гарантируют обход ограничений Binance (Weight Limits),
        # но при этом загружают данные мгновенно, не "вешая" цикл на 0.5s * N.
        sem = asyncio.Semaphore(5)
        tasks = [self._fetch_open_interest_single(sym, sem) for sym in self._symbols]
        await asyncio.gather(*tasks, return_exceptions=True)

    def get_metrics(self, symbol: str) -> tuple[float, float]:
        """Возвращает (funding_rate, open_interest) для символа."""
        sym = symbol.upper()
        return self._funding_rates.get(sym, 0.0), self._open_interests.get(sym, 0.0)

    @property
    def stats(self) -> dict:
        """Возвращает текущие метрики для всех символов."""
        return {
            "funding_rates": self._funding_rates,
            "open_interests": self._open_interests,
            "symbols": self._symbols,
            "running": self._running,
        }
