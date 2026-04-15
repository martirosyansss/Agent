"""
Сбор крипто-новостей с бесплатных API и оценка их влияния на курс.

Режим анализа: LLM (Groq primary → OpenRouter fallback).
Без LLM новости остаются нейтральными (keyword отключён).

Источники:
- RSS: 10 крипто-изданий (через rss2json, бесплатный)
- Fear & Greed Index (alternative.me, бесплатный)

Улучшения v2:
- Fuzzy-дедупликация заголовков (n-gram similarity)
- Backup LLM: OpenRouter (если Groq лимит исчерпан)
- Анализ полного текста (title + description)
- Вес источника (trust score)
- Группировка одинаковых событий

Каждая новость получает:
- sentiment_score: от -1.0 (крайне негативно) до +1.0 (крайне позитивно)
- impact_pct: предполагаемое изменение курса BTC в %
- direction: "bullish" / "bearish" / "neutral"
- llm_reasoning: объяснение от LLM
- source_trust: коэффициент доверия источника
- event_group: ID группы события (дублирующие новости → одна группа)
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
import feedparser

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
    llm_reasoning: str = ""  # пояснение от LLM
    analysis_mode: str = "pending"  # "llm" или "pending"
    description: str = ""  # краткое описание статьи (для LLM)
    source_trust: float = 1.0  # коэффициент доверия источника 0.5-1.0
    event_group: str = ""  # ID группы события (дублирующие новости)
    # Pro-level fields
    urgency: str = "low"  # "critical" / "high" / "medium" / "low"
    confidence: float = 0.5  # LLM уверенность в оценке 0.0-1.0
    category: str = "other"  # macro / regulatory / adoption / technical / security / market / defi / other
    impact_timeframe: str = "hours"  # "minutes" / "hours" / "days" / "weeks"
    effective_impact: float = 0.0  # impact с учётом decay, trust, confidence, consensus

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
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

    # Коэффициент доверия источника (0.5 = низкий, 1.0 = высокий)
    SOURCE_TRUST: dict[str, float] = {
        "CoinDesk": 1.0,
        "TheBlock": 1.0,
        "CoinTelegraph": 0.95,
        "Decrypt": 0.9,
        "CryptoSlate": 0.85,
        "Bitcoin.com": 0.8,
        "CryptoPotato": 0.75,
        "UToday": 0.7,
        "Bitcoinist": 0.7,
        "NewsBTC": 0.65,
    }

    # Порог похожести заголовков для ДЕДУПЛИКАЦИИ (0-1, выше = строже)
    SIMILARITY_THRESHOLD = 0.60

    # RSS напрямую (без rss2json — парсим через feedparser)
    RSS_FEEDS = [
        ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
        ("CoinTelegraph", "https://cointelegraph.com/rss"),
        ("Decrypt", "https://decrypt.co/feed"),
        ("Bitcoin.com", "https://news.bitcoin.com/feed/"),
        ("NewsBTC", "https://www.newsbtc.com/feed/"),
        ("Bitcoinist", "https://bitcoinist.com/feed/"),
        ("CryptoSlate", "https://cryptoslate.com/feed/"),
        ("TheBlock", "https://www.theblock.co/rss.xml"),
        ("CryptoPotato", "https://cryptopotato.com/feed/"),
        ("UToday", "https://u.today/rss"),
    ]

    MAX_NEWS = 200          # макс новостей в памяти
    MAX_AGE_DAYS = 3        # хранить за последние N дней
    ITEMS_PER_FEED = 20     # статей с каждого RSS

    # Промпт для batch-анализа новостей (professional-grade)
    _LLM_SYSTEM_PROMPT = """You are a senior crypto market analyst at a quantitative trading fund. Your job is to assess news impact on BTC/crypto prices with institutional precision.

For EACH news item, respond with a JSON object:
- "sentiment": float -1.0 to +1.0 (extremely bearish to extremely bullish)
- "impact_pct": estimated BTC price change in % within the impact_timeframe. CALIBRATION:
  * Noise/irrelevant: 0.0%
  * Minor (partnership, listing, small fund move): 0.1-0.5%
  * Moderate (regulatory guidance, ETF flow, whale move): 0.5-2.0%
  * Major (ETF approval/denial, country ban, exchange hack): 2.0-5.0%
  * Black swan (systemic failure, war, global ban): 5.0-15.0%
- "direction": "bullish", "bearish", or "neutral"
- "coins": list of directly affected tickers (e.g. ["BTC", "ETH"])
- "category": one of: "macro" (economic/geopolitical), "regulatory" (SEC, laws, bans), "adoption" (institutional/retail adoption), "technical" (protocol upgrades, forks), "security" (hacks, exploits), "market" (whale moves, liquidations, flows), "defi" (DeFi/yield/staking), "other"
- "urgency": "critical" (act immediately), "high" (within hours), "medium" (within day), "low" (informational)
- "confidence": 0.0-1.0 how confident you are in the assessment. Lower if: clickbait, vague, opinion piece, unverified rumor. Higher if: official announcement, on-chain data, regulatory filing
- "impact_timeframe": "minutes" (flash crash/pump), "hours" (same day), "days" (1-3 days), "weeks" (structural shift)
- "reasoning": 1 concise sentence in Russian explaining the MECHANISM (cause → effect on price)

CRITICAL RULES:
- Read BOTH headline AND summary. Clickbait headlines often contradict the content
- "ETF rejected" is BEARISH. "ETF approved" is BULLISH. Context > keywords
- Regulatory clarity (clear rules) can be bullish even if it seems restrictive
- Distinguish between REALIZED events (already priced in) and NEW information
- Old news rehashed = lower impact. "Bitcoin hits $X" after it already did = near zero
- Correlation: if ETH-specific news, BTC impact is ~0.3x. If altcoin-only, BTC impact ≈ 0
- Whale/flow data is medium impact unless it's record-breaking
- Opinion pieces and price predictions = low confidence, low impact

Respond ONLY with a JSON array. No markdown, no text outside JSON."""

    def __init__(self, update_interval: int = 300, groq_api_key: str = "", openrouter_api_key: str = "", db=None) -> None:
        self._update_interval = update_interval
        self._news: list[NewsItem] = []
        self._sentiment = MarketSentiment()
        self._last_fetch: float = 0
        self._running = False
        self._session: Optional[aiohttp.ClientSession] = None
        self._groq_api_key = groq_api_key
        self._openrouter_api_key = openrouter_api_key
        self._llm_available = bool(groq_api_key) or bool(openrouter_api_key)
        self._groq_available = bool(groq_api_key)
        self._openrouter_available = bool(openrouter_api_key)
        self._llm_failures = 0  # подряд неудач LLM, после 3 → переключаемся
        self._task: Optional[asyncio.Task] = None
        self._db = db  # Database instance for caching LLM results

    async def start(self) -> None:
        """Запустить фоновый сбор новостей."""
        self._running = True
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        providers = []
        if self._groq_available:
            providers.append("Groq")
        if self._openrouter_available:
            providers.append("OpenRouter")
        mode = " + ".join(providers) if providers else "no LLM"
        self._cleanup_cache()
        self._task = asyncio.create_task(self._loop())
        cache_info = f", db_cache={'on' if self._db else 'off'}"
        logger.info("NewsCollector started (interval=%ds, llm=%s%s)", self._update_interval, mode, cache_info)

    async def stop(self) -> None:
        """Остановить сбор."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
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
        """Получить новости с RSS-лент напрямую через feedparser."""
        if not self._session:
            return

        all_items: list[NewsItem] = []

        for idx, (source_name, feed_url) in enumerate(self.RSS_FEEDS):
            # Задержка между запросами
            if idx > 0:
                await asyncio.sleep(1.0)
            try:
                async with self._session.get(
                    feed_url,
                    headers={"User-Agent": "SENTINEL/1.5 CryptoNewsBot"},
                ) as resp:
                    if resp.status == 429:
                        logger.warning("RSS %s rate-limited (429), skipping", source_name)
                        continue
                    if resp.status != 200:
                        logger.debug("RSS %s returned %d", source_name, resp.status)
                        continue
                    raw_xml = await resp.text()
            except Exception as e:
                logger.debug("RSS %s failed: %s", source_name, e)
                continue

            # Парсим XML через feedparser в executor (CPU-bound)
            loop = asyncio.get_event_loop()
            try:
                feed = await loop.run_in_executor(None, feedparser.parse, raw_xml)
            except Exception as e:
                logger.debug("RSS %s parse error: %s", source_name, e)
                continue

            entries = feed.get("entries", [])
            for article in entries[:self.ITEMS_PER_FEED]:
                title = article.get("title", "")
                description = article.get("summary", article.get("description", ""))
                url = article.get("link", "")

                # Парсим дату — feedparser даёт struct_time или строку
                published = int(time.time())
                published_parsed = article.get("published_parsed")
                if published_parsed:
                    try:
                        import calendar
                        published = int(calendar.timegm(published_parsed))
                    except (TypeError, ValueError, OverflowError):
                        pass

                categories = [t.get("term", "") for t in article.get("tags", []) if t.get("term")]

                # Убираем HTML тэги из description
                clean_desc = re.sub(r'<[^>]+>', '', description)
                # Обрезаем description до 300 символов для LLM
                short_desc = clean_desc[:300].strip()
                text = f"{title} {clean_desc}".lower()
                coins = self._extract_coins(text)
                trust = self.SOURCE_TRUST.get(source_name, 0.7)

                item = NewsItem(
                    title=title,
                    source=source_name,
                    url=url,
                    published_at=published,
                    sentiment_score=0.0,
                    impact_pct=0.0,
                    direction="neutral",
                    coins_mentioned=coins,
                    categories=[str(c).strip() for c in categories if c],
                    description=short_desc,
                    source_trust=trust,
                )
                all_items.append(item)

            logger.debug("RSS %s: %d entries parsed", source_name, len(entries))

        if all_items:
            # Фильтруем: только за последние N дней
            cutoff = int(time.time()) - self.MAX_AGE_DAYS * 86400
            all_items = [n for n in all_items if n.published_at >= cutoff]

            # Дедупликация: URL + fuzzy title similarity
            existing_urls = {n.url for n in self._news if n.url}
            new_items = []
            for item in all_items:
                if item.url in existing_urls:
                    continue
                # Fuzzy проверка — не дубль ли существующей новости
                if self._is_duplicate_title(item.title, self._news + new_items):
                    continue
                new_items.append(item)

            # Сохраняем старые (ещё в окне) + новые
            old_valid = [n for n in self._news if n.published_at >= cutoff]
            merged = old_valid + new_items
            # Сортируем по дате (свежие первые)
            merged.sort(key=lambda n: n.published_at, reverse=True)
            merged = merged[:self.MAX_NEWS]

            # Группировка событий (похожие новости → один event_group)
            self._group_events(merged)

            # Восстановить из кэша то, что уже анализировалось
            cached = self._load_from_cache(merged)

            # LLM-анализ только для реально новых (нет в кэше и не проанализированы)
            unanalyzed = [n for n in merged if n.analysis_mode != "llm"]
            if unanalyzed and self._llm_available and self._llm_failures < 3:
                success = await self._analyze_batch_llm(unanalyzed)
                if success:
                    logger.info("LLM analyzed %d new items (cached=%d)", len(unanalyzed), cached)
                    # Сохранить свежие результаты в кэш
                    just_analyzed = [n for n in unanalyzed if n.analysis_mode == "llm"]
                    self._save_to_cache(just_analyzed)
                else:
                    logger.info("LLM unavailable, %d items unanalyzed (cached=%d)", len(unanalyzed), cached)
            elif unanalyzed and (not self._llm_available or self._llm_failures >= 3):
                logger.info("LLM unavailable, %d items waiting (cached=%d)", len(unanalyzed), cached)

            # Compute effective impact (decay × confidence × consensus)
            self._compute_effective_impacts(merged)

            self._news = merged
            analyzed = sum(1 for n in self._news if n.analysis_mode == "llm")
            critical = sum(1 for n in self._news if n.urgency in ("critical", "high"))
            logger.info(
                "Fetched %d news (bull=%d bear=%d neutral=%d llm=%d critical=%d)",
                len(self._news),
                sum(1 for n in self._news if n.direction == "bullish"),
                sum(1 for n in self._news if n.direction == "bearish"),
                sum(1 for n in self._news if n.direction == "neutral"),
                analyzed,
                critical,
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
        """Отправить одну пачку (≤10) новостей в LLM. Groq primary → OpenRouter fallback."""
        # Формируем сообщение с title + description
        lines = []
        for i, item in enumerate(items):
            entry = f"{i+1}. [{item.source}] {item.title}"
            if item.description:
                entry += f"\n   Summary: {item.description}"
            lines.append(entry)
        user_msg = "Analyze these crypto news items:\n" + "\n".join(lines)

        # Пробуем Groq первым
        if self._groq_available:
            data = await self._call_llm_api(
                url=self.GROQ_API_URL,
                api_key=self._groq_api_key,
                model="llama-3.3-70b-versatile",
                user_msg=user_msg,
                provider="Groq",
            )
            if data is not None:
                return self._apply_llm_results(data, items)
            # Groq не сработал — пробуем OpenRouter
            logger.info("Groq failed, trying OpenRouter fallback")

        # OpenRouter fallback
        if self._openrouter_available:
            data = await self._call_llm_api(
                url=self.OPENROUTER_API_URL,
                api_key=self._openrouter_api_key,
                model="meta-llama/llama-3.3-70b-instruct:free",
                user_msg=user_msg,
                provider="OpenRouter",
            )
            if data is not None:
                return self._apply_llm_results(data, items)

        # Оба провалились
        self._llm_failures += 1
        if self._llm_failures >= 5:
            self._llm_available = False
            logger.warning("All LLMs failed %d times, disabling LLM", self._llm_failures)
        return False

    async def _call_llm_api(self, url: str, api_key: str, model: str, user_msg: str, provider: str) -> Optional[dict]:
        """Вызвать LLM API (Groq или OpenRouter). Возвращает parsed JSON или None."""
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": self._LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.1,
            "max_tokens": 4000,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with self._session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning("%s API error %d: %s", provider, resp.status, body[:200])
                    return None
                data = await resp.json()
        except Exception as e:
            logger.warning("%s API request failed: %s", provider, e)
            return None

        # Парсим ответ LLM
        try:
            content = data["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

            analyses = json.loads(content)
            if not isinstance(analyses, list):
                logger.warning("%s returned non-list JSON", provider)
                return None
            return analyses

        except (KeyError, json.JSONDecodeError, IndexError) as e:
            logger.warning("Failed to parse %s response: %s", provider, e)
            return None

    def _apply_llm_results(self, analyses: list, items: list[NewsItem]) -> bool:
        """Применить результаты LLM к новостям. Учитывает source_trust + confidence."""
        _VALID_CATEGORIES = {"macro", "regulatory", "adoption", "technical", "security", "market", "defi", "other"}
        _VALID_URGENCY = {"critical", "high", "medium", "low"}
        _VALID_TIMEFRAME = {"minutes", "hours", "days", "weeks"}

        for i, item in enumerate(items):
            if i >= len(analyses):
                break
            a = analyses[i]
            try:
                raw_sentiment = max(-1.0, min(1.0, float(a.get("sentiment", 0))))
                raw_impact = max(-15.0, min(15.0, float(a.get("impact_pct", 0))))
                confidence = max(0.0, min(1.0, float(a.get("confidence", 0.5))))

                # Корректируем на вес источника × confidence
                trust = item.source_trust
                item.sentiment_score = round(raw_sentiment * trust, 3)
                item.impact_pct = round(raw_impact * trust, 2)
                item.direction = str(a.get("direction", "neutral"))
                if item.direction not in ("bullish", "bearish", "neutral"):
                    item.direction = "neutral"
                item.coins_mentioned = [str(c).upper() for c in a.get("coins", [])]
                item.llm_reasoning = str(a.get("reasoning", ""))
                item.analysis_mode = "llm"

                # Pro fields
                item.confidence = confidence
                cat = str(a.get("category", "other")).lower()
                item.category = cat if cat in _VALID_CATEGORIES else "other"
                urg = str(a.get("urgency", "low")).lower()
                item.urgency = urg if urg in _VALID_URGENCY else "low"
                tf = str(a.get("impact_timeframe", "hours")).lower()
                item.impact_timeframe = tf if tf in _VALID_TIMEFRAME else "hours"

            except (TypeError, ValueError):
                pass

        self._llm_failures = 0
        logger.info("LLM analyzed %d/%d items", min(len(analyses), len(items)), len(items))
        return True

    # ── Fuzzy дедупликация ────────────────────────

    # Стоп-слова которые не несут смысла для сравнения
    _STOP_WORDS = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "as", "its", "it", "this", "that",
        "and", "or", "but", "not", "has", "have", "had", "be", "been", "will",
        "can", "could", "would", "should", "may", "might", "do", "does", "did",
        "just", "now", "new", "says", "said", "after", "before", "into", "over",
        "up", "down", "out", "about", "here", "heres", "what", "why", "how",
    })

    @classmethod
    def _title_keywords(cls, title: str) -> set[str]:
        """Извлечь значимые слова из заголовка (без стоп-слов и чисел)."""
        clean = re.sub(r'[^\w\s]', '', title.lower())
        words = clean.split()
        return {w for w in words if w not in cls._STOP_WORDS and len(w) > 1 and not w.isdigit()}

    @classmethod
    def _title_similarity(cls, t1: str, t2: str) -> float:
        """Jaccard similarity между значимыми словами двух заголовков. 0-1."""
        kw1 = cls._title_keywords(t1)
        kw2 = cls._title_keywords(t2)
        if not kw1 or not kw2:
            return 0.0
        intersection = kw1 & kw2
        union = kw1 | kw2
        return len(intersection) / len(union) if union else 0.0

    def _is_duplicate_title(self, title: str, existing: list[NewsItem]) -> bool:
        """Проверить, есть ли похожий заголовок в списке."""
        for n in existing:
            if self._title_similarity(title, n.title) >= self.SIMILARITY_THRESHOLD:
                return True
        return False

    # ── Группировка событий ───────────────────────

    def _group_events(self, items: list[NewsItem]) -> None:
        """
        Группировать похожие новости в события.
        Новости с similarity >= 0.4 попадают в одну группу.
        Внутри группы: лучший source_trust получает полный вес,
        остальные получают пониженный impact (чтобы не мультиплицировать).
        """
        GROUP_THRESHOLD = 0.25  # ниже чем dedup — группируем даже при частичном сходстве
        group_id = 0
        assigned: set[int] = set()

        for i, item_i in enumerate(items):
            if i in assigned:
                continue
            # Собираем группу
            group = [i]
            assigned.add(i)
            for j in range(i + 1, len(items)):
                if j in assigned:
                    continue
                if self._title_similarity(item_i.title, items[j].title) >= GROUP_THRESHOLD:
                    group.append(j)
                    assigned.add(j)

            if len(group) > 1:
                group_id += 1
                gid = f"evt_{group_id}"
                # Сортируем по trust (лучший первый)
                group.sort(key=lambda idx: items[idx].source_trust, reverse=True)
                for _, idx in enumerate(group):
                    items[idx].event_group = gid
                    # NOTE: Don't reduce impact_pct/sentiment_score here.
                    # The consensus mechanism in _compute_effective_impacts()
                    # handles cross-source weighting correctly — reducing here
                    # would double-penalize duplicates.

    # ── Effective Impact (professional scoring) ───

    # Temporal decay half-lives per timeframe (hours)
    _TIMEFRAME_HALFLIFE: dict[str, float] = {
        "minutes": 0.5,   # decays fast
        "hours": 3.0,
        "days": 12.0,
        "weeks": 48.0,
    }

    # Urgency multipliers
    _URGENCY_MULT: dict[str, float] = {
        "critical": 1.5,
        "high": 1.2,
        "medium": 1.0,
        "low": 0.7,
    }

    def _compute_effective_impacts(self, items: list[NewsItem]) -> None:
        """
        Вычислить effective_impact для каждой новости.
        
        effective_impact = impact_pct × temporal_decay × confidence × urgency_mult × consensus_boost
        
        Это единственная метрика, используемая для торговых решений.
        """
        now = time.time()

        # Step 1: Cross-source consensus per event group
        group_consensus = self._compute_consensus(items)

        for item in items:
            if item.analysis_mode != "llm":
                item.effective_impact = 0.0
                continue

            # 1. Temporal decay (exponential)
            age_hours = max(0, (now - item.published_at) / 3600)
            halflife = self._TIMEFRAME_HALFLIFE.get(item.impact_timeframe, 3.0)
            decay = 0.5 ** (age_hours / halflife) if halflife > 0 else 0.0

            # 2. Urgency multiplier
            urgency_mult = self._URGENCY_MULT.get(item.urgency, 1.0)

            # 3. Confidence filter (low confidence → diminished impact)
            conf_mult = max(0.2, item.confidence)

            # 4. Consensus boost (multiple sources agree → stronger signal)
            consensus_mult = group_consensus.get(item.event_group, 1.0) if item.event_group else 1.0

            # Composite
            raw = item.impact_pct
            effective = raw * decay * urgency_mult * conf_mult * consensus_mult
            item.effective_impact = round(effective, 3)

    def _compute_consensus(self, items: list[NewsItem]) -> dict[str, float]:
        """
        Cross-source consensus: если N разных источников подтверждают
        одно событие с одинаковым direction → boost.
        
        Returns: {event_group_id: consensus_multiplier}
        """
        groups: dict[str, list[NewsItem]] = {}
        for item in items:
            if item.event_group and item.analysis_mode == "llm":
                groups.setdefault(item.event_group, []).append(item)

        consensus: dict[str, float] = {}
        for gid, members in groups.items():
            if len(members) < 2:
                consensus[gid] = 1.0
                continue

            # Count unique sources
            unique_sources = len({m.source for m in members})

            # Direction agreement: are they all pointing the same way?
            directions = [m.direction for m in members if m.direction != "neutral"]
            if not directions:
                consensus[gid] = 1.0
                continue

            bullish = sum(1 for d in directions if d == "bullish")
            bearish = sum(1 for d in directions if d == "bearish")
            agreement = max(bullish, bearish) / len(directions)

            # Boost: 2 sources → 1.15, 3 → 1.25, 4+ → 1.35 (capped)
            # But only if agreement >= 70%
            if agreement >= 0.7:
                source_boost = min(1.0 + unique_sources * 0.1, 1.35)
            else:
                # Conflicting signals → dampen
                source_boost = 0.8

            consensus[gid] = round(source_boost, 2)

        return consensus

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
        """Обновить общий sentiment по всем новостям (professional scoring)."""
        if not self._news:
            return

        # Refresh effective_impact (temporal decay changes every cycle)
        self._compute_effective_impacts(self._news)

        bullish = sum(1 for n in self._news if n.direction == "bullish")
        bearish = sum(1 for n in self._news if n.direction == "bearish")
        neutral = sum(1 for n in self._news if n.direction == "neutral")

        # Weighted average using effective_impact × confidence × trust
        now = time.time()
        weighted_sum = 0.0
        weight_total = 0.0
        for n in self._news:
            if n.analysis_mode != "llm":
                continue
            age_hours = (now - n.published_at) / 3600
            halflife = self._TIMEFRAME_HALFLIFE.get(n.impact_timeframe, 3.0)
            decay = 0.5 ** (age_hours / halflife) if halflife > 0 else 0.0
            weight = decay * n.source_trust * n.confidence
            weighted_sum += n.sentiment_score * weight
            weight_total += weight

        overall = weighted_sum / weight_total if weight_total > 0 else 0.0

        self._sentiment.overall_score = round(overall, 3)
        self._sentiment.news_count = len(self._news)
        self._sentiment.bullish_count = bullish
        self._sentiment.bearish_count = bearish
        self._sentiment.neutral_count = neutral
        self._sentiment.updated_at = int(time.time())

    # ── DB cache ──────────────────────────────────

    def _save_to_cache(self, items: list[NewsItem]) -> int:
        """Сохранить LLM-проанализированные новости в БД. Возвращает кол-во сохранённых."""
        if not self._db or not items:
            return 0
        saved = 0
        now = int(time.time())
        for n in items:
            if n.analysis_mode != "llm" or not n.url:
                continue
            try:
                self._db.execute(
                    "INSERT OR REPLACE INTO news_cache "
                    "(url, title, source, published_at, sentiment_score, impact_pct, "
                    "direction, coins_mentioned, llm_reasoning, urgency, confidence, "
                    "category, impact_timeframe, cached_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        n.url, n.title, n.source, n.published_at,
                        n.sentiment_score, n.impact_pct, n.direction,
                        json.dumps(n.coins_mentioned), n.llm_reasoning,
                        n.urgency, n.confidence, n.category, n.impact_timeframe, now,
                    ),
                )
                saved += 1
            except Exception as e:
                logger.debug("Cache save error for %s: %s", n.url[:60], e)
        if saved:
            self._db.commit()
            logger.info("Saved %d analyzed news to cache", saved)
        return saved

    def _load_from_cache(self, items: list[NewsItem]) -> int:
        """Восстановить LLM-результаты из кэша для items. Возвращает кол-во восстановленных."""
        if not self._db or not items:
            return 0
        urls = [n.url for n in items if n.url and n.analysis_mode != "llm"]
        if not urls:
            return 0
        # Забираем всё из кэша за последние MAX_AGE_DAYS
        cutoff = int(time.time()) - self.MAX_AGE_DAYS * 86400
        try:
            rows = self._db.fetchall(
                "SELECT url, sentiment_score, impact_pct, direction, coins_mentioned, "
                "llm_reasoning, urgency, confidence, category, impact_timeframe "
                "FROM news_cache WHERE published_at >= ?",
                (cutoff,),
            )
        except Exception as e:
            logger.warning("Cache load error: %s", e)
            return 0
        cache_map = {row["url"]: row for row in rows}
        restored = 0
        for n in items:
            if n.analysis_mode == "llm" or not n.url:
                continue
            row = cache_map.get(n.url)
            if not row:
                continue
            n.sentiment_score = row["sentiment_score"]
            n.impact_pct = row["impact_pct"]
            n.direction = row["direction"]
            try:
                n.coins_mentioned = json.loads(row["coins_mentioned"])
            except (json.JSONDecodeError, TypeError):
                pass
            n.llm_reasoning = row["llm_reasoning"] or ""
            n.urgency = row["urgency"] or "low"
            n.confidence = row["confidence"] if row["confidence"] is not None else 0.5
            n.category = row["category"] or "other"
            n.impact_timeframe = row["impact_timeframe"] or "hours"
            n.analysis_mode = "llm"
            restored += 1
        if restored:
            logger.info("Restored %d news from cache (skipped LLM)", restored)
        return restored

    def _cleanup_cache(self) -> None:
        """Удалить устаревшие записи из кэша."""
        if not self._db:
            return
        cutoff = int(time.time()) - self.MAX_AGE_DAYS * 86400 * 2  # 2x window
        try:
            self._db.execute("DELETE FROM news_cache WHERE published_at < ?", (cutoff,))
            self._db.commit()
        except Exception:
            pass

    # ── Public API ────────────────────────────────

    def get_news(self, limit: int = 200) -> list[dict]:
        """Получить последние новости с оценками."""
        items = sorted(self._news, key=lambda n: abs(n.impact_pct), reverse=True)
        return [n.to_dict() for n in items[:limit]]

    def get_sentiment(self) -> dict:
        """Получить текущий market sentiment."""
        return self._sentiment.to_dict()

    def get_impact_summary(self) -> dict:
        """Сводка влияния новостей на рынок (professional-grade)."""
        if not self._news:
            return {
                "status": "no_data",
                "message": "Ожидание загрузки новостей...",
            }

        analyzed = [n for n in self._news if n.analysis_mode == "llm"]
        high_impact = [n for n in analyzed if abs(n.effective_impact) >= 1.0]
        medium_impact = [n for n in analyzed if 0.3 <= abs(n.effective_impact) < 1.0]
        critical_news = [n for n in analyzed if n.urgency in ("critical", "high")]
        event_groups = len({n.event_group for n in self._news if n.event_group})

        avg_eff_impact = sum(n.effective_impact for n in analyzed) / len(analyzed) if analyzed else 0

        # Category breakdown
        cat_counts: dict[str, int] = {}
        for n in analyzed:
            cat_counts[n.category] = cat_counts.get(n.category, 0) + 1

        # Consensus direction (what % of analyzed agree on direction)
        if analyzed:
            bull_pct = sum(1 for n in analyzed if n.direction == "bullish") / len(analyzed)
            bear_pct = sum(1 for n in analyzed if n.direction == "bearish") / len(analyzed)
        else:
            bull_pct = bear_pct = 0.0

        return {
            "status": "ok",
            "total_news": len(self._news),
            "llm_analyzed": len(analyzed),
            "high_impact_count": len(high_impact),
            "medium_impact_count": len(medium_impact),
            "critical_count": len(critical_news),
            "avg_effective_impact": round(avg_eff_impact, 3),
            "overall_direction": "bullish" if avg_eff_impact > 0.15 else "bearish" if avg_eff_impact < -0.15 else "neutral",
            "bull_pct": round(bull_pct * 100, 1),
            "bear_pct": round(bear_pct * 100, 1),
            "consensus_strength": round(max(bull_pct, bear_pct) * 100, 1),
            "event_groups": event_groups,
            "category_breakdown": cat_counts,
            "fear_greed": self._sentiment.fear_greed_index,
            "fear_greed_label": self._sentiment.fear_greed_label,
            # Legacy compat
            "avg_impact_pct": round(avg_eff_impact, 2),
            "high_impact_count": len(high_impact),
        }

    def get_news_signal(self) -> dict:
        """
        Агрегированный торговый сигнал от новостей.
        
        Используется стратегиями для корректировки confidence.
        
        Returns:
            composite_score: -1.0..+1.0 (weighted effective_impact, учтены decay/trust/confidence/consensus)
            signal_strength: 0.0..1.0 (сила сигнала: 0=нет данных, 1=очень сильный consensus)
            bias: "bullish"/"bearish"/"neutral" 
            critical_alert: True если есть critical urgency news
            dominant_category: str — самая частая категория
            actionable: bool — достаточно ли strong для торговой коррекции
        """
        analyzed = [n for n in self._news if n.analysis_mode == "llm"]
        if not analyzed:
            return {
                "composite_score": 0.0,
                "signal_strength": 0.0,
                "bias": "neutral",
                "critical_alert": False,
                "dominant_category": "other",
                "actionable": False,
            }

        # Composite score = weighted sum of effective_impact × sign(sentiment)
        total_weight = 0.0
        weighted_score = 0.0
        for n in analyzed:
            w = abs(n.effective_impact) * n.confidence
            direction_sign = 1.0 if n.direction == "bullish" else (-1.0 if n.direction == "bearish" else 0.0)
            weighted_score += direction_sign * abs(n.effective_impact) * n.confidence
            total_weight += w

        composite = weighted_score / total_weight if total_weight > 0 else 0.0
        composite = max(-1.0, min(1.0, composite))

        # Signal strength: based on agreement + volume of data
        directions = [n.direction for n in analyzed if n.direction != "neutral"]
        if directions:
            majority = max(
                sum(1 for d in directions if d == "bullish"),
                sum(1 for d in directions if d == "bearish"),
            )
            agreement = majority / len(directions)
        else:
            agreement = 0.0

        # More analyzed items + higher agreement → stronger signal
        data_depth = min(len(analyzed) / 20.0, 1.0)  # saturates at 20 items
        strength = agreement * data_depth * min(total_weight / 5.0, 1.0)
        strength = min(1.0, strength)

        # Critical alert
        has_critical = any(n.urgency == "critical" for n in analyzed)

        # Dominant category
        cats: dict[str, float] = {}
        for n in analyzed:
            cats[n.category] = cats.get(n.category, 0) + abs(n.effective_impact)
        dominant = max(cats, key=cats.get) if cats else "other"

        # Actionable: composite strong enough AND sufficient agreement
        actionable = abs(composite) > 0.15 and strength > 0.3

        bias = "bullish" if composite > 0.1 else "bearish" if composite < -0.1 else "neutral"

        return {
            "composite_score": round(composite, 3),
            "signal_strength": round(strength, 3),
            "bias": bias,
            "critical_alert": has_critical,
            "dominant_category": dominant,
            "actionable": actionable,
        }
