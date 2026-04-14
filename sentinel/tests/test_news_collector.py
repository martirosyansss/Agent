"""NewsCollector resilience tests (LLM cooldown / recovery)."""

import asyncio
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collector.news_collector import NewsCollector, NewsItem


@pytest.mark.skip(reason="LLM cooldown logic was refactored out of NewsCollector")
class TestNewsCollectorCooldown:
    def test_provider_cooldown_after_rate_limit(self):
        c = NewsCollector(groq_api_key="x")
        assert c._provider_in_cooldown("Groq") is False

        c._set_provider_cooldown("Groq", seconds=30, status_code=429)
        assert c._provider_in_cooldown("Groq") is True

    def test_llm_reenabled_after_disable_cooldown(self):
        c = NewsCollector(groq_api_key="x")
        c._llm_available = False
        c._llm_failures = 5
        c._llm_disabled_until = time.time() - 1

        c._restore_llm_if_cooldown_passed()

        assert c._llm_available is True
        assert c._llm_failures == 0
        assert c._llm_disabled_until == 0.0

    @pytest.mark.asyncio
    async def test_rate_limited_batch_does_not_increment_failures(self):
        c = NewsCollector(groq_api_key="x")

        async def fake_call(**kwargs):
            c._set_provider_cooldown("Groq", seconds=60, status_code=429)
            c._last_llm_error_kind = "rate_limited"
            return None

        c._call_llm_api = fake_call  # type: ignore[method-assign]

        item = NewsItem(
            title="t",
            source="s",
            url="u",
            published_at=int(time.time()),
        )

        ok = await c._analyze_single_batch([item])
        assert ok is False
        assert c._llm_failures == 0
