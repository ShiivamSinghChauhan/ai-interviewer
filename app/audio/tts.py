"""Voices: edge-tts by default (free, fast), Fish Audio or ElevenLabs when picked, Kokoro offline.

Fish and ElevenLabs fall back to edge when they fail, and edge falls back to Kokoro, so a dead key or a
network hiccup never leaves the partner speechless mid-session.
"""
import asyncio
import contextlib
import hashlib
import logging
import math
import re
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

import httpx
import numpy as np

from app.audio import cues
from app.audio.wav import wav_bytes
from app.config import DATA_DIR, load, settings
from app.llm.keypool import request
from app.llm.providers import POOLS

log = logging.getLogger("tts")

ENGINES = ("edge", "fish", "elevenlabs", "kokoro")
KOKORO_SIDS = {"hm_omega": 33, "hm_psi": 34, "hf_alpha": 31, "hf_beta": 32}  # Hindi voices = Indian English
SOUTH_ASIAN = ("indian", "pakistani", "bangladeshi", "sri lankan", "nepali", "south asian", "hindi", "tamil")
SILENCE = wav_bytes(np.zeros(9600, dtype=np.float32), 24000)


@dataclass(frozen=True)
class Voice:
    engine: str = "edge"  # one of ENGINES
    id: str = ""          # engine-specific voice id; empty means the engine's default
    speed: float = 1.0
    fallback: str = ""    # the edge voice that stands in when Fish or ElevenLabs fails


def _lowered(items) -> set[str]:
    return {item.lower() for item in items}


class Kokoro:
    def __init__(self, model_dir: str, threads: int):
        import sherpa_onnx

        d = Path(model_dir)
        lexicons = sorted(d.glob("lexicon-*.txt"), key=lambda p: (not p.name.startswith("lexicon-us"), p.name))
        model = sherpa_onnx.OfflineTtsKokoroModelConfig(
            model=str(sorted(d.glob("*.onnx"))[0]), voices=str(d / "voices.bin"), tokens=str(d / "tokens.txt"),
            data_dir=str(d / "espeak-ng-data"), dict_dir=str(d / "dict") if (d / "dict").is_dir() else "",
            lexicon=",".join(map(str, lexicons)))
        self.tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(kokoro=model, num_threads=threads), max_num_sentences=1))
        self.lock = threading.Lock()

    def wav(self, text: str, voice: str, speed: float) -> bytes:
        with self.lock:
            audio = self.tts.generate(text, sid=KOKORO_SIDS.get(voice, 33), speed=speed)
        return wav_bytes(np.asarray(audio.samples, dtype=np.float32), audio.sample_rate)


class EdgeStream:
    """edge-tts over one websocket that stays open. A fresh connection costs about a second per sentence;
    reusing it gets the first audio in 250-550 ms. This talks to edge-tts internals, so the edge-tts version
    is pinned in requirements.txt."""

    CONFIG = ('{"context":{"synthesis":{"audio":{"metadataoptions":{"sentenceBoundaryEnabled":"false",'
              '"wordBoundaryEnabled":"false"},"outputFormat":"audio-24khz-48kbitrate-mono-mp3"}}}}\r\n')
    IDLE_RECONNECT_S = 45  # the server drops idle sockets without telling anyone

    def __init__(self):
        self.session = self.ws = None
        self.last_used = 0.0
        self.lock = asyncio.Lock()

    async def _connect(self) -> None:
        import aiohttp
        import edge_tts.communicate as C

        await self.close()
        self.session = aiohttp.ClientSession(trust_env=True)
        self.ws = await self.session.ws_connect(
            f"{C.WSS_URL}&ConnectionId={C.connect_id()}&Sec-MS-GEC={C.DRM.generate_sec_ms_gec()}"
            f"&Sec-MS-GEC-Version={C.SEC_MS_GEC_VERSION}",
            compress=15, headers=C.DRM.headers_with_muid(C.WSS_HEADERS), ssl=C._SSL_CTX)

    async def close(self) -> None:
        for conn in (self.ws, self.session):
            try:
                if conn is not None:
                    await conn.close()
            except Exception:
                pass  # already gone
        self.ws = self.session = None

    async def warm(self) -> None:
        async with self.lock:
            stale = time.monotonic() - self.last_used > self.IDLE_RECONNECT_S
            if self.ws is None or self.ws.closed or stale:
                try:
                    await self._connect()
                except Exception as e:
                    log.warning("edge-tts warm-up failed: %s", e)

    async def synth(self, text: str, voice: str, speed: float) -> bytes:
        async with self.lock:
            self.last_used = time.monotonic()
            for attempt in (1, 2):
                try:
                    if self.ws is None or self.ws.closed:
                        await self._connect()
                    return await asyncio.wait_for(self._request(text, voice, speed), 8)
                except Exception:
                    await self.close()  # stale socket: reconnect once
                    if attempt == 2:
                        raise

    async def _request(self, text: str, voice: str, speed: float) -> bytes:
        import aiohttp
        import edge_tts.communicate as C
        from edge_tts.data_classes import TTSConfig

        cfg = TTSConfig(voice, f"{round((speed - 1) * 100):+d}%", "+0%", "+0Hz", "SentenceBoundary")
        ssml = C.mkssml(cfg, C.escape(C.remove_incompatible_characters(text)))
        await self.ws.send_str(f"X-Timestamp:{C.date_to_string()}\r\nContent-Type:application/json; charset=utf-8"
                               f"\r\nPath:speech.config\r\n\r\n{self.CONFIG}")
        await self.ws.send_str(C.ssml_headers_plus_data(C.connect_id(), C.date_to_string(), ssml))
        out = bytearray()
        async for msg in self.ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                out += C.get_headers_and_data(msg.data, int.from_bytes(msg.data[:2], "big"))[1]
            elif msg.type == aiohttp.WSMsgType.TEXT:
                if "Path:turn.end" in msg.data:
                    break
            else:
                raise ConnectionError(f"edge-tts socket {msg.type}")
        if not out:
            raise RuntimeError("edge-tts returned no audio")
        return bytes(out)


class ElevenLabs:
    """Free keys only get the premade voices, so those come with an accent tag ([strong Indian accent]) that
    the v3/v4 models act out without reading it aloud. Paid keys also get South Asian library voices."""

    BASE = "https://api.elevenlabs.io"

    def __init__(self):
        self.pool = POOLS["elevenlabs"]
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0))
        self.default_voice = settings.elevenlabs_voice
        self.added: dict[str, str] = {}
        self.tier = ""

    @property
    def available(self) -> bool:
        return bool(self.pool)

    def _get(self, path: str, **params):
        return request(self.pool, lambda key: self.client.get(f"{self.BASE}{path}", params=params,
                                                               headers={"xi-api-key": key}))

    async def warm(self) -> None:
        try:
            await self._get("/v1/user")  # opens the TLS connection before the first sentence needs it
        except Exception as e:
            log.warning("ElevenLabs warm-up failed: %s", e)

    async def voices(self) -> list[dict]:
        if not self.tier:
            self.tier = (await self._get("/v1/user/subscription")).json().get("tier", "free")
        south_asian, premade, seen = [], [], set()
        for v in (await self._get("/v2/voices", page_size=100)).json().get("voices", []):
            if self.tier == "free" and v.get("category") in ("professional", "famous", "high_quality"):
                continue
            labels = v.get("labels") or {}
            tag = f"{labels.get('gender', '')}, {labels.get('age', '')}".strip(", ")
            seen.add(v["voice_id"])
            if any(a in f"{labels.get('accent')} {v.get('description')}".lower() for a in SOUTH_ASIAN):
                south_asian.append({"id": v["voice_id"], "name": f"{v['name']} ({tag}, {labels.get('accent')})"})
            elif labels.get("language", "en") == "en":
                premade.append({"id": f"accent:{v['voice_id']}", "name": f"{v['name'].split(' - ')[0]} ({tag})"})
        library = []
        if self.tier != "free":
            for accent in ("indian", "pakistani", "sri lankan", "bangladeshi"):
                found = await self._get("/v1/shared-voices", accent=accent, language="en", page_size=30,
                                        use_cases="conversational")
                library += [{"id": f"lib:{v['public_owner_id']}:{v['voice_id']}",
                             "name": f"{v['name']} ({v.get('gender', '')}, {v.get('accent')}) - library"}
                            for v in found.json().get("voices", []) if v["voice_id"] not in seen]
        return south_asian + library + premade

    async def _resolve(self, voice: str) -> str:
        """A library voice has to be added to the account once before it can speak."""
        if not voice.startswith("lib:"):
            return voice
        if voice not in self.added:
            _, owner, voice_id = voice.split(":", 2)
            try:
                added = await request(self.pool, lambda key: self.client.post(
                    f"{self.BASE}/v1/voices/add/{owner}/{voice_id}", headers={"xi-api-key": key},
                    json={"new_name": f"Practice {voice_id[:6]}"}))
                self.added[voice] = added.json().get("voice_id", voice_id)
            except Exception:
                self.added[voice] = voice_id  # probably added already
        return self.added[voice]

    async def synth(self, text: str, voice: str, speed: float, previous: str = "") -> bytes:
        if not voice:
            if not self.default_voice:
                self.default_voice = ((await self.voices()) or [{"id": ""}])[0]["id"]
            voice = self.default_voice
        if voice.startswith("accent:") and settings.elevenlabs_accent_tag:
            text = f"{settings.elevenlabs_accent_tag} {text}"
        body = {"text": text, "model_id": settings.elevenlabs_model, "previous_text": previous[-400:],
                "voice_settings": {"stability": settings.elevenlabs_stability, "similarity_boost": 0.75,
                                   "speed": max(0.7, min(1.2, speed)), "use_speaker_boost": True}}
        if settings.elevenlabs_model != "eleven_multilingual_v2":
            body["language_code"] = "en"
        voice_id = await self._resolve(voice.removeprefix("accent:"))
        response = await request(self.pool, lambda key: self.client.post(
            f"{self.BASE}/v1/text-to-speech/{voice_id}/stream", params={"output_format": settings.elevenlabs_format},
            headers={"xi-api-key": key}, json=body))
        return response.content


class Fish:
    """Fish Audio S2.1. `s2.1-pro-free` costs nothing through the API (fair use, until Nov 30 2026) and any
    public library voice can be used, so Indian-English voices don't need a paid plan."""

    BASE = "https://api.fish.audio"

    def __init__(self):
        self.pool = POOLS["fish"]
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=5.0))
        self.default_voice = settings.fish_voice
        self.cache: dict[str, list[dict]] = {}

    @property
    def available(self) -> bool:
        return bool(self.pool)

    def _search(self, **params):
        return request(self.pool, lambda key: self.client.get(f"{self.BASE}/model", params=params,
                                                               headers={"Authorization": f"Bearer {key}"}))

    async def warm(self) -> None:
        try:
            await self._search(page_size=1)
        except Exception as e:
            log.warning("Fish warm-up failed: %s", e)

    async def voices(self, accent: str | None = None) -> list[dict]:
        """Library voices for an accent, best first. The library is full of celebrity clones and anime
        characters, so the filters under `fish:` in common.yaml do most of the work here."""
        cfg = load("common").get("fish", {})
        accent = accent or cfg.get("default_accent", "Indian")
        if accent in self.cache:
            return self.cache[accent]
        excluded_tags = _lowered(cfg.get("exclude_tags", []))
        preferred_tags = _lowered(cfg.get("prefer_tags", []))
        excluded_words = _lowered(cfg.get("exclude_title_words", []))
        generic_words = _lowered(cfg.get("generic_title_words", []))
        found, seen = [], set()
        for tag in cfg.get("accent_tags", {}).get(accent, [f"{accent} Accent"]):
            try:
                page = await self._search(tag=tag, language=cfg.get("language", "en"), page_size=60,
                                          sort_by="task_count")
            except Exception as e:
                log.warning("Fish voice search failed: %s", e)
                continue
            for v in page.json().get("items", []):
                tags, title = _lowered(v.get("tags") or []), (v.get("title") or "").lower()
                words = set(re.findall(r"[a-z]+", title))
                uses = v.get("task_count", 0)
                if (v["_id"] in seen or tags & excluded_tags or words & excluded_words
                        or title.strip() in ("", "me") or uses < cfg.get("min_uses", 10)
                        or any(name.lower() in title for name in cfg.get("exclude_titles", []))):
                    continue
                seen.add(v["_id"])
                gender = "female" if "female" in tags else "male" if "male" in tags else ""
                age = next((t for t in ("young", "middle-aged", "old") if t in tags), "")
                # descriptive titles ("calm Indian male voice") rarely belong to a cloned celebrity
                score = (3 * min(len(words & generic_words), 2) + len(tags & preferred_tags)
                         + math.log10(max(uses, 1)))
                described = ", ".join(x for x in (gender, age) if x)
                found.append({"id": v["_id"], "gender": gender, "score": score,
                              "name": f"{v.get('title', '?')} ({described}; {uses:,} uses)"})
        found.sort(key=lambda v: -v["score"])
        self.cache[accent] = found
        return found

    async def find_voice(self, accent: str, gender: str, age: int | None = None) -> str:
        pinned = (load("common").get("fish", {}).get("pinned") or {}).get(f"{accent}-{gender}")
        if pinned:
            return pinned
        voices = await self.voices(accent) or await self.voices()
        matches = [v for v in voices if v["gender"] == gender] or voices
        if age:
            bucket = "young" if age < 32 else "middle-aged" if age < 58 else "old"
            matches = [v for v in matches if bucket in v["name"]] or matches
        return matches[0]["id"] if matches else ""

    async def synth(self, text: str, voice: str, speed: float) -> bytes:
        if not voice and not self.default_voice:
            self.default_voice = ((await self.voices()) or [{"id": ""}])[0]["id"]
        body = {"text": text, "format": "mp3", "mp3_bitrate": 64, "latency": settings.fish_latency,
                "normalize": True, "prosody": {"speed": speed, "volume": 0}}
        if voice or self.default_voice:
            body["reference_id"] = voice or self.default_voice
        response = await request(self.pool, lambda key: self.client.post(
            f"{self.BASE}/v1/tts", json=body, headers={"Authorization": f"Bearer {key}", "model": settings.fish_model}))
        return response.content


class TTS:
    HEDGE_AFTER = 0.7       # edge usually answers in ~440 ms, but sometimes takes 2-5 s
    PREMIUM_DEADLINE = 6.0  # Fish/ElevenLabs slower than this on one sentence: use edge instead...
    PREMIUM_BENCH_S = 30    # ...and keep using it for a while, so every sentence doesn't wait 6 s
    CACHE_DIR = DATA_DIR / "cache" / "tts"
    CACHE_MAX_CHARS = 24
    OPENERS = ["Okay,", "Okay.", "Got it,", "Got it.", "Okay, got it.", "Okay, makes sense.", "Right,", "Hmm,",
               "Hmm, okay.", "I see.", "Alright,", "Sure,", "Interesting.", "Makes sense.", "Take your time."]

    def __init__(self):
        self.edge, self.edge_backup = EdgeStream(), EdgeStream()
        self.eleven, self.fish = ElevenLabs(), Fish()
        self.down_until: dict[str, float] = {}
        self.kokoro_available = Path(settings.kokoro_dir).is_dir()
        self._kokoro: Kokoro | None = None
        self._kokoro_lock = asyncio.Lock()

    def ready(self, engine: str) -> bool:
        """Whether an engine can speak here at all: its keys are set, or its model is downloaded."""
        return {"edge": True, "fish": self.fish.available, "elevenlabs": self.eleven.available,
                "kokoro": self.kokoro_available}.get(engine, False)

    async def kokoro(self) -> Kokoro | None:
        """Loaded on first use only: it's slow on a laptop CPU and mostly an emergency fallback."""
        async with self._kokoro_lock:
            if self._kokoro is None and self.kokoro_available:
                try:
                    self._kokoro = await asyncio.to_thread(Kokoro, settings.kokoro_dir, settings.kokoro_threads)
                except Exception as e:
                    log.warning("Kokoro unavailable: %s", e)
                    self.kokoro_available = False
        return self._kokoro

    async def synth(self, text: str, voice: Voice, previous: str = "") -> tuple[bytes, str]:
        """(audio, "mp3" | "wav"). Short phrases are cached on disk per voice: most replies open with
        "Okay," or "Got it.", and a cached opener plays at once instead of after a one-second round trip."""
        if len(text) > self.CACHE_MAX_CHARS:
            audio, fmt, _ = await self._synth(text, voice, previous)
            return audio, fmt
        model = {"elevenlabs": settings.elevenlabs_model, "fish": settings.fish_model}.get(voice.engine, "")
        key = (f"{voice.engine}|{voice.id}|{model}|{settings.elevenlabs_accent_tag}|{voice.speed}|"
               f"{text.strip().lower()}")
        stem = hashlib.sha1(key.encode()).hexdigest()
        for fmt in ("mp3", "wav"):
            if (cached := self.CACHE_DIR / f"{stem}.{fmt}").exists():
                return cached.read_bytes(), fmt
        audio, fmt, engine = await self._synth(text, voice, previous)
        if engine == voice.engine:  # never file a fallback voice under this one
            self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            (self.CACHE_DIR / f"{stem}.{fmt}").write_bytes(audio)
        return audio, fmt

    async def prewarm(self, voice: Voice) -> None:
        """Synthesize the usual openers ahead of time, two at a time."""
        slots = asyncio.Semaphore(2)

        async def one(text: str) -> None:
            async with slots:
                with contextlib.suppress(Exception):  # a missing opener is synthesized live later
                    await self.synth(text, voice)
        await asyncio.gather(*map(one, self.OPENERS))

    async def _synth(self, text: str, voice: Voice, previous: str) -> tuple[bytes, str, str]:
        """(audio, format, engine that actually spoke)."""
        if voice.engine in ("fish", "elevenlabs"):
            if time.monotonic() >= self.down_until.get(voice.engine, 0):
                try:
                    audio = await asyncio.wait_for(self._premium(text, voice, previous), self.PREMIUM_DEADLINE)
                    return audio, "mp3", voice.engine
                except Exception as e:
                    log.warning("%s failed (%s); using edge-tts for %ds", voice.engine,
                                str(e)[:120] or type(e).__name__, self.PREMIUM_BENCH_S)
                    self.down_until[voice.engine] = time.monotonic() + self.PREMIUM_BENCH_S
            voice = replace(voice, engine="edge", id=voice.fallback or settings.edge_voice)

        text = cues.plain(text)  # plain voices would read "[sigh]" out loud
        if not text:
            return SILENCE, "wav", voice.engine
        if voice.engine == "edge" or not self.kokoro_available:
            edge_voice = voice.id if voice.engine == "edge" and voice.id else settings.edge_voice
            try:
                return await self._edge(text, edge_voice, voice.speed), "mp3", "edge"
            except Exception as e:
                if not self.kokoro_available:
                    raise
                log.warning("edge-tts failed (%s); falling back to Kokoro", e)
                voice = replace(voice, engine="kokoro", id=settings.kokoro_voice)
        kokoro = await self.kokoro()
        if kokoro is None:
            raise RuntimeError("no TTS engine available")
        return await asyncio.to_thread(kokoro.wav, text, voice.id, voice.speed), "wav", "kokoro"

    def _premium(self, text: str, voice: Voice, previous: str):
        if voice.engine == "fish":
            return self.fish.synth(text, voice.id, voice.speed)
        return self.eleven.synth(text, voice.id, voice.speed, previous)

    async def warm(self, engine: str = "edge") -> None:
        """Edge is warmed whatever the engine: it's the fallback, and a cold fallback is a slow one."""
        jobs = [self.edge.warm(), self.edge_backup.warm()]
        if engine == "elevenlabs" and self.eleven.available:
            jobs.append(self.eleven.warm())
        elif engine == "fish" and self.fish.available:
            jobs.append(self.fish.warm())
        await asyncio.gather(*jobs)

    async def _edge(self, text: str, voice: str, speed: float) -> bytes:
        """If the main socket is slow, race the same sentence on the spare one and keep the winner."""
        first = asyncio.create_task(self.edge.synth(text, voice, speed))
        done, _ = await asyncio.wait({first}, timeout=self.HEDGE_AFTER)
        if done:
            return first.result()
        second = asyncio.create_task(self.edge_backup.synth(text, voice, speed))
        pending = {first, second}
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task.exception() is None:
                    for loser in pending:
                        loser.cancel()
                    asyncio.create_task((self.edge_backup if task is first else self.edge).close())
                    if task is second:
                        self.edge, self.edge_backup = self.edge_backup, self.edge
                    return task.result()
        raise first.exception() or second.exception()
