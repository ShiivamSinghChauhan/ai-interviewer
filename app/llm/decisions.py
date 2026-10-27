"""Quick typed judgements (yes/no probability, 0-3 score, pick one) from Jev, or ScaleDown's look-alike API.

Both take {model, state, questions} and return {answers, usage}, so callers don't care which one answered.
"""
import asyncio
import logging
from functools import partial

import httpx

from app.config import settings
from app.llm.keypool import ProviderError, request
from app.llm.providers import POOLS, USAGE, parse_entry

log = logging.getLogger("decisions")

ENDPOINTS = {
    "jev": ("https://api.typesafe.ai/v1/systemone", lambda k: {"Authorization": f"Bearer {k}"}),
    "scaledown": ("https://api.scaledown.xyz/v1/scaledown", lambda k: {"x-api-key": k}),
}
_client = httpx.AsyncClient(timeout=httpx.Timeout(6.0, connect=3.0))


def available() -> bool:
    return any(POOLS.get(parse_entry(e)[0]) for e in settings.decisions_chain)


async def decide(state: str, questions: dict, timeout: float = 4.0) -> dict | None:
    """{question_name: answer} or None if nobody answered in time."""
    try:
        return await asyncio.wait_for(_decide(state, questions), timeout)
    except TimeoutError:
        log.warning("decision timed out after %.1fs (%s)", timeout, ",".join(questions))
        return None


async def _decide(state: str, questions: dict) -> dict | None:
    for provider, model, _ in map(parse_entry, settings.decisions_chain):
        if not POOLS.get(provider):
            continue
        body = {"model": model, "state": {"text": state}, "questions": questions}
        try:
            response = await request(POOLS[provider], partial(_post, provider, body))
        except ProviderError as e:
            log.warning("%s", e)
            continue
        data = response.json()
        USAGE[provider] = USAGE.get(provider, 0) + int((data.get("usage") or {}).get("input_tokens", 0))
        return data.get("answers") or {}
    return None


def _post(provider: str, body: dict, key: str):
    url, auth = ENDPOINTS[provider]
    return _client.post(url, json=body, headers=auth(key))


def noul(instructions: str, true: str = "", false: str = "") -> dict:
    q = {"type": "noul", "instructions": instructions}
    if true and false:
        q["criteria"] = {"true": true, "false": false}
    return q


def score(instructions: str, levels: list[str]) -> dict:
    return {"type": "score", "instructions": instructions, "criteria": levels}
