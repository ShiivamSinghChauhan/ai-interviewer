import asyncio
import time

import httpx
import pytest

from app.llm.keypool import KeyPool, ProviderError, request


def replies(*statuses: int, retry_after: str | None = None):
    """A fake `send(key)` that answers with these statuses in order, and records which keys it saw."""
    queue = list(statuses)
    seen = []

    async def send(key: str) -> httpx.Response:
        seen.append(key)
        headers = {"retry-after": retry_after} if retry_after else {}
        return httpx.Response(queue.pop(0), text="rate limit per minute", headers=headers)

    return send, seen


def test_rate_limit_benches_the_key_for_that_model_only():
    pool = KeyPool("groq", ["k1", "k2"])
    send, seen = replies(429, 200, retry_after="7")
    response = asyncio.run(request(pool, send, model="qwen"))
    assert response.status_code == 200
    assert seen == ["k1", "k2"]
    assert [k.value for k in pool.candidates("qwen")] == ["k2"]
    assert [k.value for k in pool.candidates("llama")] == ["k2", "k1"]  # starts with the key that worked


def test_a_dead_key_is_benched_for_every_model():
    pool = KeyPool("groq", ["k1", "k2"])
    send, _ = replies(401, 200)
    asyncio.run(request(pool, send, model="qwen"))
    assert [k.value for k in pool.candidates("anything")] == ["k2"]


def test_daily_quota_waits_longer_than_a_minute_limit():
    pool = KeyPool("openrouter", ["k1"])
    key = pool.keys[0]
    pool.fail(key, 429, "Rate limit exceeded: free-models-per-day", model="gemma")
    assert key.model_benched["gemma"] - time.monotonic() > 3000  # an hour, not the 10 s of a minute limit


def test_server_errors_stop_at_once():
    pool = KeyPool("groq", ["k1", "k2"])
    send, seen = replies(500)
    with pytest.raises(ProviderError):
        asyncio.run(request(pool, send))
    assert seen == ["k1"]  # a 500 isn't the key's fault, so the other key isn't burned on it


def test_no_usable_key():
    pool = KeyPool("groq", ["k1"])
    send, _ = replies(429, retry_after="60")
    with pytest.raises(ProviderError):
        asyncio.run(request(pool, send, model="qwen"))
    with pytest.raises(ProviderError, match="no usable key"):
        asyncio.run(request(pool, send, model="qwen"))
