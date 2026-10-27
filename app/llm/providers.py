"""Chat models behind OpenAI-compatible APIs, plus Groq Whisper and ScaleDown summaries.

A chain is a list like "groq:qwen/qwen3.8-27b,openrouter:google/gemma-4-31b-it:free". Every entry tries each
of its keys in turn, then the next entry gets a go. Extra request parameters ride along after a pipe:
"groq:openai/gpt-oss-120b|reasoning_effort=low".
"""
import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator

import httpx
import numpy as np

from app.audio.wav import wav_bytes
from app.config import settings
from app.llm.keypool import KEY_ERRORS, KeyPool, ProviderError, request

log = logging.getLogger("llm")

BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "openrouter": "https://openrouter.ai/api/v1",
}
POOLS = {name: KeyPool(name, keys) for name, keys in settings.api_keys.items()}
USAGE: dict[str, int] = {}  # Jev/ScaleDown tokens spent since the server started, shown after a session
_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))


class LLMUnavailable(Exception):
    pass


def parse_entry(entry: str) -> tuple[str, str, dict]:
    """Split "groq:openai/gpt-oss-120b|reasoning_effort=low" into provider, model and extra params."""
    head, *params = entry.split("|")
    provider, model = head.split(":", 1)
    extra = {}
    for param in params:
        name, raw = (x.strip() for x in param.split("=", 1))
        extra[name] = _cast(raw)
    return provider.strip(), model.strip(), extra


def _cast(raw: str):
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    return raw


async def stream_chat(messages: list[dict], chain: list[str] | None = None, temperature: float = 0.7,
                      max_tokens: int = 400, first_token_timeout: float | None = None,
                      **params) -> AsyncIterator[str]:
    """Yield text deltas from the first entry of the chain that answers.

    first_token_timeout is for live replies: free tiers sometimes queue a request for 20-30 s, and switching
    to another model beats waiting.
    """
    chain = chain or settings.llm_chain
    errors = []
    for entry in chain:
        provider, model, extra = parse_entry(entry)
        pool = POOLS.get(provider)
        if not pool:
            continue
        timeout = httpx.Timeout(first_token_timeout, connect=5.0) if first_token_timeout else _client.timeout
        body = {"model": model, "messages": messages, "stream": True, "temperature": temperature,
                "max_tokens": max_tokens, **params, **extra}
        for key in pool.candidates(model):
            started = False
            try:
                async with _client.stream("POST", f"{BASE_URLS[provider]}/chat/completions", json=body,
                                          headers={"Authorization": f"Bearer {key.value}"},
                                          timeout=timeout) as response:
                    if response.status_code != 200:
                        text = (await response.aread()).decode(errors="ignore")[:300]
                        errors.append(f"{provider}/{model} {response.status_code}: {text}")
                        if response.status_code in KEY_ERRORS:
                            pool.fail(key, response.status_code, text, response.headers, model)
                            continue
                        break
                    pool.ok(key)
                    async for delta in _sse_deltas(response):
                        started = True
                        yield delta
                    return
            except (httpx.TransportError, httpx.TimeoutException) as e:
                if started:
                    raise LLMUnavailable(f"{provider} stream broke mid-answer: {e}") from e
                if isinstance(e, httpx.TimeoutException) and first_token_timeout:
                    errors.append(f"{provider}/{model}: nothing in {first_token_timeout:.0f}s")
                else:
                    errors.append(f"{provider}/{model}: {type(e).__name__}")
                break
    has_keys = any(POOLS.get(parse_entry(entry)[0]) for entry in chain)
    raise LLMUnavailable("; ".join(errors) or ("every key is cooling down" if has_keys else "no LLM keys set"))


async def _sse_deltas(response: httpx.Response) -> AsyncIterator[str]:
    """Text deltas from an OpenAI-style server-sent event stream."""
    async for line in response.aiter_lines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            delta = json.loads(data)["choices"][0]["delta"].get("content")
        except (json.JSONDecodeError, KeyError, IndexError, TypeError, AttributeError):
            continue
        if delta:
            yield delta


def cooldown_left(chain: list[str]) -> float:
    """Seconds until some key in this chain is usable again."""
    now = time.monotonic()
    waits = [key.free_at(model) - now
             for provider, model, _ in map(parse_entry, chain) if POOLS.get(provider)
             for key in POOLS[provider].keys]
    return max(0.0, min(waits)) if waits else 0.0


async def complete(messages: list[dict], chain: list[str] | None = None, temperature: float = 0.4,
                   max_tokens: int = 2000, retries: int = 0, **params) -> str:
    chain = chain or settings.llm_chain
    for attempt in range(retries + 1):
        try:
            return "".join([d async for d in stream_chat(messages, chain, temperature, max_tokens, **params)])
        except LLMUnavailable:
            if attempt == retries:
                raise
            # rate-limited keys come back on their own: wait for the first one instead of giving up
            await asyncio.sleep(min(65.0, max(3.0 * (attempt + 1), cooldown_left(chain) + 0.5)))


async def complete_json(messages: list[dict], chain: list[str], temperature: float = 0.3,
                        max_tokens: int = 3000, retries: int = 1, **params):
    """Like complete(), but broken or empty JSON also moves on to the next model. Reasoning models sometimes
    spend the whole budget thinking and return nothing."""
    errors = []
    for entry in chain:
        try:
            return extract_json(await complete(messages, [entry], temperature, max_tokens, retries, **params))
        except (LLMUnavailable, ValueError) as e:
            errors.append(f"{entry.split('|')[0]}: {str(e)[:160]}")
    raise LLMUnavailable(" | ".join(errors) or "empty chain")


def extract_json(text: str):
    """The first JSON object or array in a reply, ignoring code fences and chatter around it."""
    body = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    for opener, closer in ("{}", "[]"):
        start, end = body.find(opener), body.rfind(closer)
        if 0 <= start < end:
            try:
                return json.loads(body[start:end + 1])
            except json.JSONDecodeError:
                pass
    raise ValueError(f"no JSON in reply: {text[:200]}")


async def whisper_transcribe(samples: np.ndarray, prompt: str = "") -> str | None:
    """A second, more careful transcript of a finished answer. None when Groq isn't available."""
    if not POOLS["groq"] or len(samples) < 8000:
        return None
    wav = wav_bytes(samples)
    form = {"model": settings.whisper_model, "language": "en", "prompt": prompt[:800], "temperature": "0"}
    try:
        response = await request(POOLS["groq"], lambda key: _client.post(
            f"{BASE_URLS['groq']}/audio/transcriptions", headers={"Authorization": f"Bearer {key}"},
            data=form, files={"file": ("answer.wav", wav, "audio/wav")}), settings.whisper_model)
    except ProviderError as e:
        log.warning("whisper failed: %s", e)
        return None
    return response.json().get("text", "").strip()


async def scaledown_summarize(text: str, instructions: str) -> str | None:
    if not POOLS["scaledown"]:
        return None
    body = {"model": "summarize", "max_tokens": 400,
            "messages": [{"role": "system", "content": instructions}, {"role": "user", "content": text}]}
    try:
        response = await request(POOLS["scaledown"], lambda key: _client.post(
            "https://api.scaledown.xyz/v1/chat/completions", json=body, headers={"x-api-key": key}))
        data = response.json()
        USAGE["scaledown"] = USAGE.get("scaledown", 0) + int((data.get("usage") or {}).get("prompt_tokens", 0))
        return json.loads(data["choices"][0]["message"]["content"]).get("summary") or None
    except (ProviderError, KeyError, IndexError, json.JSONDecodeError) as e:
        log.warning("scaledown summary failed: %s", e)
        return None
