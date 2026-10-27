"""Check every API key in .env and show which of the configured models each provider really has.

    python -m scripts.check_keys
"""
import asyncio
import time

import httpx

from app.config import settings
from app.llm.providers import BASE_URLS, parse_entry

DECISION_PROBE = {"done": {"type": "noul", "instructions": "The speaker finished their sentence."}}


def tag(key: str) -> str:
    return f"...{key[-4:]}"


async def list_models(client: httpx.AsyncClient, provider: str, key: str) -> tuple[str, set[str]]:
    r = await client.get(f"{BASE_URLS[provider]}/models", headers={"Authorization": f"Bearer {key}"})
    if r.status_code != 200:
        return f"HTTP {r.status_code}: {r.text[:120]}", set()
    ids = {m["id"].removeprefix("models/") for m in r.json().get("data", [])}
    return "ok", ids


async def probe_decision(client: httpx.AsyncClient, provider: str, model: str, key: str) -> str:
    url = {"jev": "https://api.typesafe.ai/v1/systemone", "scaledown": "https://api.scaledown.xyz/v1/scaledown"}[provider]
    headers = {"Authorization": f"Bearer {key}"} if provider == "jev" else {"x-api-key": key}
    body = {"model": model, "state": {"text": "Candidate: I used LoRA because full fine-tuning was too expensive."},
            "questions": DECISION_PROBE}
    t0 = time.perf_counter()
    r = await client.post(url, json=body, headers=headers)
    ms = (time.perf_counter() - t0) * 1000
    if r.status_code != 200:
        return f"HTTP {r.status_code}: {r.text[:160]}"
    data = r.json()
    ans = (data.get("answers") or {}).get("done", {})
    return f"ok in {ms:.0f} ms (P={ans.get('noul')}, usage={data.get('usage')})"


async def main() -> None:
    async with httpx.AsyncClient(timeout=20) as client:
        available: dict[str, set[str]] = {}
        for provider in ("groq", "gemini", "openrouter"):
            keys = settings.api_keys[provider]
            if not keys:
                print(f"{provider:10s} no keys")
                continue
            for k in keys:
                status, ids = await list_models(client, provider, k)
                print(f"{provider:10s} {tag(k)}  {status}  ({len(ids)} models)")
                available.setdefault(provider, set()).update(ids)

        for name, chain in (("LLM_CHAIN", settings.llm_chain), ("REPORT_LLM_CHAIN", settings.report_chain)):
            print(f"\n{name}:")
            for entry in chain:
                provider, model, _ = parse_entry(entry)
                ids = available.get(provider)
                mark = "?" if ids is None else ("ok" if model in ids else "MISSING")
                print(f"  {mark:8s} {entry}")
        groq = available.get("groq", set())
        if groq:
            whisper = sorted(m for m in groq if "whisper" in m)
            chat = sorted(m for m in groq if "whisper" not in m and "tts" not in m and "guard" not in m)
            print("\nGroq chat models:", ", ".join(chat))
            print("Groq whisper models:", ", ".join(whisper))

        print("\nDECISIONS_CHAIN:")
        for entry in settings.decisions_chain:
            provider, model, _ = parse_entry(entry)
            keys = settings.api_keys.get(provider) or []
            if not keys:
                print(f"  {provider:10s} no keys")
            for k in keys:
                print(f"  {provider:10s} {tag(k)}  {await probe_decision(client, provider, model, k)}")

        print()
        for k in settings.api_keys["elevenlabs"]:
            r = await client.get("https://api.elevenlabs.io/v1/user/subscription", headers={"xi-api-key": k})
            if r.status_code == 200:
                d = r.json()
                print(f"elevenlabs {tag(k)}  ok  tier={d.get('tier')}  characters used "
                      f"{d.get('character_count')}/{d.get('character_limit')}")
            else:
                print(f"elevenlabs {tag(k)}  HTTP {r.status_code}: {r.text[:160]}")
        if not settings.api_keys["elevenlabs"]:
            print("elevenlabs no keys")
        for k in settings.api_keys.get("fish", []):
            r = await client.get("https://api.fish.audio/model", params={"title": "Indian", "language": "en",
                                 "page_size": 3}, headers={"Authorization": f"Bearer {k}"})
            found = f"{r.json().get('total', '?')} Indian voices in library" if r.status_code == 200 else r.text[:160]
            print(f"fish       {tag(k)}  HTTP {r.status_code}  {found}")
        if not settings.api_keys.get("fish"):
            print("fish       no keys (free s2.1-pro-free model: get a key at fish.audio/app/api-keys)")

        if settings.api_keys["scaledown"]:
            k = settings.api_keys["scaledown"][0]
            r = await client.post("https://api.scaledown.xyz/v1/chat/completions", headers={"x-api-key": k}, json={
                "model": "summarize", "max_tokens": 60,
                "messages": [{"role": "user", "content": "The candidate explained LoRA, QLoRA and why rank 16 "
                                                         "was enough for their classification fine-tune."}]})
            print(f"\nScaleDown summarize: HTTP {r.status_code} {r.text[:160]}")


if __name__ == "__main__":
    asyncio.run(main())
