"""
Сбор крипто-новостей с бесплатных API и оценка их влияния на курс.

Два режима анализа:
1. LLM (Groq / Llama 3.3 70B) — точный контекстный анализ, если задан GROQ_API_KEY
2. Keyword fallback — если ключа нет, работает на словарях

Источники:
- RSS: CoinDesk, CoinTelegraph, Decrypt (через rss2json, бесплатный)
- Fear & Greed Index (alternative.me, бесплатный)

Каждая новость получает:
- sentiment_score: от -1.0 (крайне негативно) до +1.0 (крайне позитивно)
- impact_pct: предполагаемое изменение курса BTC в %
- direction: "bullish" / "bearish" / "neutral"
- llm_reasoning: объяснение от LLM (только в LLM-режиме)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


# ── Ключевые слова для sentiment-анализа ──────────

_BULLISH_KEYWORDS: dict[str, float] = {
    # Сильно бычьи (0.6-1.0)
    "etf approved": 0.9,
    "etf approval": 0.9,
    "spot etf": 0.8,
    "bitcoin etf": 0.8,
    "institutional adoption": 0.7,
    "mass adoption": 0.7,
    "strategic reserve": 0.8,
    "nation adopt": 0.8,
    "country adopt": 0.8,
    "legal tender": 0.85,
    "all-time high": 0.7,
    "ath": 0.6,
    "halving": 0.6,
    "supply shock": 0.7,
    "whale accumulation": 0.6,
    "whales buying": 0.65,
    "trillion": 0.5,
    "billion inflow": 0.7,
    "massive inflow": 0.7,
    "record inflow": 0.7,
    # Умеренно бычьи (0.3-0.5)
    "bullish": 0.5,
    "rally": 0.5,
    "surge": 0.5,
    "pump": 0.4,
    "breakout": 0.45,
    "moon": 0.3,
    "soar": 0.45,
    "gain": 0.3,
    "recovery": 0.4,
    "upgrade": 0.35,
    "partnership": 0.35,
    "integration": 0.3,
    "launch": 0.3,
    "listing": 0.35,
    "adoption": 0.4,
    "accumulate": 0.4,
    "inflow": 0.4,
    "buy": 0.25,
    "positive": 0.3,
    "growth": 0.35,
    "support": 0.25,
    "staking": 0.2,
    "defi growth": 0.3,
}

_BEARISH_KEYWORDS: dict[str, float] = {
    # Сильно медвежьи (-0.6 to -1.0)
    "sec lawsuit": -0.8,
    "sec charges": -0.85,
    "sec enforcement": -0.7,
    "ban crypto": -0.9,
    "crypto ban": -0.9,
    "exchange hack": -0.85,
    "hacked": -0.7,
    "exploit": -0.65,
    "rug pull": -0.9,
    "ponzi": -0.8,
    "fraud": -0.75,
    "scam": -0.7,
    "bankruptcy": -0.85,
    "insolvent": -0.85,
    "collapse": -0.8,
    "crash": -0.7,
    "plunge": -0.65,
    "outflow": -0.5,
    "massive outflow": -0.7,
    "record outflow": -0.7,
    "delisting": -0.65,
    # Умеренно медвежьи (-0.3 to -0.5)
    "bearish": -0.5,
    "dump": -0.5,
    "sell-off": -0.5,
    "selloff": -0.5,
    "decline": -0.4,
    "drop": -0.35,
    "fall": -0.3,
    "regulation": -0.3,
    "crackdown": -0.5,
    "investigation": -0.4,
    "lawsuit": -0.5,
    "fine": -0.35,
    "penalty": -0.35,
    "restriction": -0.35,
    "warning": -0.25,
    "concern": -0.2,
    "risk": -0.15,
    "bubble": -0.4,
    "overvalued": -0.3,
    "liquidation": -0.45,
    "liquidated": -0.45,
    "whale selling": -0.5,
    "mt.gox": -0.5,
    "mt gox": -0.5,
}

# Множители влияния для разных монет упомянутых в новости
_COIN_IMPACT_MULT: dict[str, float] = {
    "bitcoin": 1.0,
    "btc": 1.0,
    "ethereum": 0.8,
    "eth": 0.8,
    "solana": 0.5,
    "sol": 0.5,
    "bnb": 0.4,
    "xrp": 0.4,
    "cardano": 0.3,
    "ada": 0.3,
    "dogecoin": 0.2,
    "doge": 0.2,
}


@dataclass
class NewsItem:
    """Одна новость с оценкой влияния."""
    title: str
    source: str
    url: str
    published_at: int  # unix timestamp
    sentiment_score: float = 0.0  # -1.0 .. +1.0
    impact_pct: float = 0.0  # estimated BTC price change %
    direction: str = "neutral"  # bullish / bearish / neutral
    coins_mentioned: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    llm_reasoning: str = ""  # пояснение от LLM (пусто если keyword-режим)
    analysis_mode: str = "keyword"  # "llm" или "keyword"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MarketSentiment:
    """Общий рыночный sentiment."""
    fear_greed_index: int = 50  # 0-100
    fear_greed_label: str = "Neutral"
    overall_score: float = 0.0  # -1.0 .. +1.0
    news_count: int = 0
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0
    updated_at: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class NewsCollector:
    """Сбор и анализ крипто-новостей."""

    FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1&format=json"
    COINGECKO_TRENDING_URL = "https://api.coingecko.com/api/v3/search/trending"
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

    # RSS через rss2json (бесплатный, до 10000 запросов/день)
    RSS_FEEDS = [
        ("CoinDesk", "https://api.rss2json.com/v1/api.json?rss_url=https://www.coindesk.com/arc/outboundfeeds/rss/"),
        ("CoinTelegraph", "https://api.rss2json.com/v1/api.json?rss_url=https://cointelegraph.com/rss"),
        ("Decrypt", "https://api.rss2json.com/v1/api.json?rss_url=https://decrypt.co/feed"),
        ("Bitcoin.com", "https://api.rss2json.com/v1/api.json?rss_url=https://news.bitcoin.com/feed/"),
        ("NewsBTC", "https://api.rss2json.com/v1/api.json?rss_url=https://www.newsbtc.com/feed/"),
        ("Bitcoinist", "https://api.rss2json.com/v1/api.json?rss_url=https://bitcoinist.com/feed/"),
        ("CryptoSlate", "https://api.rss2json.com/v1/api.json?rss_url=https://cryptoslate.com/feed/"),
        ("TheBlock", "https://api.rss2json.com/v1/api.json?rss_url=https://www.theblock.co/rss.xml"),
        ("CryptoPotato", "https://api.rss2json.com/v1/api.json?rss_url=https://cryptopotato.com/feed/"),
        ("UToday", "https://api.rss2json.com/v1/api.json?rss_url=https://u.today/rss"),
    ]

    MAX_NEWS = 200          # макс новостей в памяти
    MAX_AGE_DAYS = 3        # хранить за последние N дней
    ITEMS_PER_FEED = 20     # статей с каждого RSS

    # Промпт для batch-анализа новостей
    _LLM_SYSTEM_PROMPT = """You are a crypto market analyst. Analyze each news headline for its impact on Bitcoin/crypto prices.

For EACH news item, respond with a JSON object containing:
- "sentiment": float from -1.0 (extremely bearish) to +1.0 (extremely bullish)
- "impact_pct": estimated BTC price change in % (e.g. +2.5 or -1.3). Use realistic values: most news = 0.1-1%, major events = 1-5%, black swans = 5-15%
- "direction": "bullish", "bearish", or "neutral"  
- "coins": list of mentioned/affected ticker symbols (e.g. ["BTC", "ETH"])
- "reasoning": 1 sentence in Russian explaining why this affects the price

IMPORTANT:
- Understand CONTEXT. "ETF rejected" is bearish even though "ETF" alone might seem bullish
- "Bitcoin drops to X" is bearish. "Bitcoin surges to X" is bullish
- Regulatory crackdowns = bearish. Adoption news = bullish
- Consider the MAGNITUDE: SEC lawsuit > minor partnership
- If news is not crypto-related or has no price impact, set impact_pct to 0 and direction to "neutral"

Respond ONLY with a JSON array, one object per news item, in the same order as input. No markdown, no explanation outside JSON."""

    def __init__(self, update_interval: int = 300, groq_api_key: str = "") -> None:
        self._update_interval = update_interval
        self._news: list[NewsItem] = []
        self._sentiment = MarketSentiment()
        self._last_fetch: float = 0
        self._running = False
        self._session: Optional[aiohttp.ClientSession] = None
        self._groq_api_key = groq_api_key
        self._llm_available = bool(groq_api_key)
        self._llm_failures = 0  # подряд неудач LLM, после 3 → fallback

    async def start(self) -> None:
        """Запустить фоновый сбор новостей."""
        self._running = True
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        mode = "LLM (Groq)" if self._llm_available else "keyword"
        asyncio.create_task(self._loop())
        logger.info("NewsCollector started (interval=%ds, mode=%s)", self._update_interval, mode)

    async def stop(self) -> None:
        """Остановить сбор."""
        self._running = False
        if self._session:
            await self._session.close()
            self._session = None

    async def _loop(self) -> None:
        """Основной цикл сбора."""
        while self._running:
            try:
                await self._fetch_all()
            except Exception as e:
                logger.error("News fetch error: %s", e)
            await asyncio.sleep(self._update_interval)

    async def _fetch_all(self) -> None:
        """Получить все данные параллельно."""
        tasks = [
            self._fetch_rss_news(),
            self._fetch_fear_greed(),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, Exception):
                logger.warning("News source failed: %s", r)

        self._last_fetch = time.time()
        self._update_overall_sentiment()

    async def _fetch_rss_news(self) -> None:
        """Получить новости с RSS-лент через rss2json."""
        if not self._session:
            return

        all_items: list[NewsItem] = []

        for source_name, feed_url in self.RSS_FEEDS:
            try:
                async with self._session.get(feed_url) as resp:
                    if resp.status != 200:
                        logger.debug("RSS %s returned %d", source_name, resp.status)
                        continue
                    data = await resp.json()
            except Exception as e:
                logger.debug("RSS %s failed: %s", source_name, e)
                continue

            if data.get("status") != "ok":
                continue

            items = data.get("items", [])
            for article in items[:self.ITEMS_PER_FEED]:
                title = article.get("title", "")
                description = article.get("description", "")
                url = article.get("link", "")
                pub_date = article.get("pubDate", "")
                categories = article.get("categories", [])

                # Парсим дату (rss2json даёт "2026-04-13 10:00:00")
                published = int(time.time())
                if pub_date:
                    try:
                        from datetime import datetime
                        dt = datetime.strptime(pub_date[:19], "%Y-%m-%d %H:%M:%S")
                        published = int(dt.timestamp())
                    except (ValueError, TypeError):
                        pass

                # Анализ sentiment
                # Убираем HTML тэги из description
                clean_desc = re.sub(r'<[^>]+>', '', description)
                text = f"{title} {clean_desc}".lower()
                sentiment, impact = self._analyze_sentiment(text)
                coins = self._extract_coins(text)
                direction = "bullish" if sentiment > 0.15 else "bearish" if sentiment < -0.15 else "neutral"

                item = NewsItem(
                    title=title,
                    source=source_name,
                    url=url,
                    published_at=published,
                    sentiment_score=round(sentiment, 3),
                    impact_pct=round(impact, 2),
                    direction=direction,
                    coins_mentioned=coins,
                    categories=[str(c).strip() for c in categories if c] if isinstance(categories, list) else [],
                )
                all_items.append(item)

        if all_items:
            # Фильтруем: только за последние N дней
            cutoff = int(time.time()) - self.MAX_AGE_DAYS * 86400
            all_items = [n for n in all_items if n.published_at >= cutoff]

            # Дедупликация: объединяем с существующими (по URL)
            existing_urls = {n.url for n in self._news if n.url}
            new_items = [n for n in all_items if n.url not in existing_urls]

            # Сохраняем старые (ещё в окне) + новые
            old_valid = [n for n in self._news if n.published_at >= cutoff]
            merged = old_valid + new_items
            # Сортируем по дате (свежие первые)
            merged.sort(key=lambda n: n.published_at, reverse=True)
            merged = merged[:self.MAX_NEWS]

            # LLM-анализ только для новых (ещё не проанализированных)
            unanalyzed = [n for n in merged if n.analysis_mode != "llm"]
            if unanalyzed and self._llm_available and self._llm_failures < 3:
                success = await self._analyze_batch_llm(unanalyzed)
                if success:
                    logger.info("LLM analyzed %d new items", len(unanalyzed))
                else:
                    logger.info("LLM failed, using keyword analysis as fallback")

            self._news = merged
            logger.info(
                "Fetched %d news items (bull=%d, bear=%d, neutral=%d, mode=%s)",
                len(self._news),
                sum(1 for n in self._news if n.direction == "bullish"),
                sum(1 for n in self._news if n.direction == "bearish"),
                sum(1 for n in self._news if n.direction == "neutral"),
                "llm" if any(n.analysis_mode == "llm" for n in self._news) else "keyword",
            )

    async def _fetch_fear_greed(self) -> None:
        """Получить Fear & Greed Index."""
        if not self._session:
            return

        try:
            async with self._session.get(self.FEAR_GREED_URL) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()
        except Exception as e:
            logger.warning("Fear & Greed fetch failed: %s", e)
            return

        items = data.get("data", [])
        if items:
            fgi = items[0]
            self._sentiment.fear_greed_index = int(fgi.get("value", 50))
            self._sentiment.fear_greed_label = fgi.get("value_classification", "Neutral")

    async def _analyze_batch_llm(self, items: list[NewsItem]) -> bool:
        """
        Отправить пачку новостей в Groq LLM для анализа.
        Обрабатывает частями по 10 чтобы не превысить лимит токенов.
        Returns True при успехе хотя бы одной пачки, False при полном провале.
        """
        if not self._session or not self._llm_available:
            return False

        BATCH_SIZE = 10
        any_success = False

        for batch_start in range(0, len(items), BATCH_SIZE):
            batch = items[batch_start:batch_start + BATCH_SIZE]
            success = await self._analyze_single_batch(batch)
            if success:
                any_success = True
            else:
                # При провале одной пачки не ломаем остальные
                logger.debug("LLM batch %d failed, items keep keyword analysis", batch_start)

        if any_success:
            self._llm_failures = 0
        return any_success

    async def _analyze_single_batch(self, items: list[NewsItem]) -> bool:
        """Отправить одну пачку (≤10) новостей в Groq."""
        headlines = [f"{i+1}. [{item.source}] {item.title}" for i, item in enumerate(items)]
        user_msg = "Analyze these crypto news headlines:\n" + "\n".join(headlines)

        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": self._LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.1,
            "max_tokens": 4000,
        }
        headers = {
            "Authorization": f"Bearer {self._groq_api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with self._session.post(
                self.GROQ_API_URL, json=payload, headers=headers
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning("Groq API error %d: %s", resp.status, body[:200])
                    self._llm_failures += 1
                    if self._llm_failures >= 3:
                        logger.warning("LLM failed %d times, switching to keyword mode", self._llm_failures)
                        self._llm_available = False
                    return False

                data = await resp.json()
        except Exception as e:
            logger.warning("Groq API request failed: %s", e)
            self._llm_failures += 1
            if self._llm_failures >= 3:
                self._llm_available = False
            return False

        # Парсим ответ LLM
        try:
            content = data["choices"][0]["message"]["content"].strip()
            # Убираем возможное markdown-обёртывание
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

            analyses = json.loads(content)
            if not isinstance(analyses, list):
                logger.warning("LLM returned non-list JSON")
                return False

        except (KeyError, json.JSONDecodeError, IndexError) as e:
            logger.warning("Failed to parse LLM response: %s", e)
            self._llm_failures += 1
            if self._llm_failures >= 3:
                self._llm_available = False
            return False

        # Применяем результаты LLM к элементам
        for i, item in enumerate(items):
            if i >= len(analyses):
                break
            a = analyses[i]
            try:
                item.sentiment_score = round(max(-1.0, min(1.0, float(a.get("sentiment", 0)))), 3)
                item.impact_pct = round(max(-15.0, min(15.0, float(a.get("impact_pct", 0)))), 2)
                item.direction = str(a.get("direction", "neutral"))
                if item.direction not in ("bullish", "bearish", "neutral"):
                    item.direction = "neutral"
                item.coins_mentioned = [str(c).upper() for c in a.get("coins", [])]
                item.llm_reasoning = str(a.get("reasoning", ""))
                item.analysis_mode = "llm"
            except (TypeError, ValueError):
                pass  # оставляем дефолтные значения

        self._llm_failures = 0
        logger.info("LLM analyzed %d/%d news items", min(len(analyses), len(items)), len(items))
        return True

    def _analyze_sentiment(self, text: str) -> tuple[float, float]:
        """
        Анализ sentiment текста.
        Returns: (sentiment_score, impact_pct)
        """
        scores: list[float] = []

        # Проверяем бычьи слова (word boundary для коротких слов)
        for keyword, weight in _BULLISH_KEYWORDS.items():
            if len(keyword) <= 4:
                if re.search(r'\b' + re.escape(keyword) + r'\b', text):
                    scores.append(weight)
            elif keyword in text:
                scores.append(weight)

        # Проверяем медвежьи слова
        for keyword, weight in _BEARISH_KEYWORDS.items():
            if len(keyword) <= 4:
                if re.search(r'\b' + re.escape(keyword) + r'\b', text):
                    scores.append(weight)
            elif keyword in text:
                scores.append(weight)

        if not scores:
            return 0.0, 0.0

        # Средний score с бустом от сильных сигналов
        avg_score = sum(scores) / len(scores)

        # Сильные сигналы усиливают оценку
        max_abs = max(abs(s) for s in scores)
        if max_abs > 0.7:
            avg_score = avg_score * 0.6 + (max_abs * (1 if avg_score >= 0 else -1)) * 0.4

        # Ограничиваем -1..+1
        sentiment = max(-1.0, min(1.0, avg_score))

        # Оценка влияния на цену (грубая — база на историческом анализе)
        # Типичное движение BTC на новости: 0.5%-5%
        # Большинство новостей: 0.1%-1%
        base_impact = abs(sentiment) * 3.0  # макс ~3% для очень сильных

        # Множитель от количества совпавших слов (больше слов = сильнее сигнал)
        word_mult = min(len(scores) / 3.0, 2.0)
        impact = base_impact * (0.5 + word_mult * 0.25)

        # Ограничиваем максимальное влияние
        impact = min(impact, 5.0)

        return sentiment, impact if sentiment > 0 else -impact

    def _extract_coins(self, text: str) -> list[str]:
        """Извлечь упомянутые монеты."""
        found = set()
        for coin in _COIN_IMPACT_MULT:
            # Ищем как целое слово
            if re.search(r'\b' + re.escape(coin) + r'\b', text):
                # Нормализуем к тикеру
                normalized = coin.upper()
                if coin == "bitcoin":
                    normalized = "BTC"
                elif coin == "ethereum":
                    normalized = "ETH"
                elif coin == "solana":
                    normalized = "SOL"
                elif coin == "cardano":
                    normalized = "ADA"
                elif coin == "dogecoin":
                    normalized = "DOGE"
                found.add(normalized)
        return sorted(found)

    def _update_overall_sentiment(self) -> None:
        """Обновить общий sentiment по всем новостям."""
        if not self._news:
            return

        bullish = sum(1 for n in self._news if n.direction == "bullish")
        bearish = sum(1 for n in self._news if n.direction == "bearish")
        neutral = sum(1 for n in self._news if n.direction == "neutral")

        scores = [n.sentiment_score for n in self._news]
        # Взвешиваем более свежие новости сильнее
        now = time.time()
        weighted_sum = 0.0
        weight_total = 0.0
        for n in self._news:
            age_hours = (now - n.published_at) / 3600
            weight = max(0.05, 1.0 - age_hours / 72)  # свежие (0h) = 1.0, 3 дня = 0.05
            weighted_sum += n.sentiment_score * weight
            weight_total += weight

        overall = weighted_sum / weight_total if weight_total > 0 else 0.0

        self._sentiment.overall_score = round(overall, 3)
        self._sentiment.news_count = len(self._news)
        self._sentiment.bullish_count = bullish
        self._sentiment.bearish_count = bearish
        self._sentiment.neutral_count = neutral
        self._sentiment.updated_at = int(time.time())

    # ── Public API ────────────────────────────────

    def get_news(self, limit: int = 200) -> list[dict]:
        """Получить последние новости с оценками."""
        items = sorted(self._news, key=lambda n: abs(n.impact_pct), reverse=True)
        return [n.to_dict() for n in items[:limit]]

    def get_sentiment(self) -> dict:
        """Получить текущий market sentiment."""
        return self._sentiment.to_dict()

    def get_impact_summary(self) -> dict:
        """Сводка влияния новостей на рынок."""
        if not self._news:
            return {
                "status": "no_data",
                "message": "Ожидание загрузки новостей...",
            }

        high_impact = [n for n in self._news if abs(n.impact_pct) >= 1.5]
        medium_impact = [n for n in self._news if 0.5 <= abs(n.impact_pct) < 1.5]

        avg_impact = sum(n.impact_pct for n in self._news) / len(self._news) if self._news else 0

        return {
            "status": "ok",
            "high_impact_count": len(high_impact),
            "medium_impact_count": len(medium_impact),
            "avg_impact_pct": round(avg_impact, 2),
            "overall_direction": "bullish" if avg_impact > 0.2 else "bearish" if avg_impact < -0.2 else "neutral",
            "fear_greed": self._sentiment.fear_greed_index,
            "fear_greed_label": self._sentiment.fear_greed_label,
        }
