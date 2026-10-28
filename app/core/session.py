"""One live voice session over a WebSocket.

The interviewer (app/modes/interview) subclasses BaseSession and fill in the hooks. The base class does
the rest: it feeds the microphone to the AudioWorker, streams the partner's reply into the voice sentence by
sentence, handles interruptions and silences, and writes the report at the end. A session started through the
REST API (app/api.py) brings its own setup and hands its result back to the API.
"""
import asyncio
import contextlib
import json
import logging
import math
import struct
import time
from dataclasses import dataclass, field

from fastapi import WebSocket

from app.api import AgentSession, book
from app.audio import cues
from app.audio.tts import Voice
from app.config import load, settings
from app.core import analysis
from app.core.reply import SentenceChunker, parse_meta
from app.core.store import sessions
from app.core.turns import FRAME_SEC, ONSET_FRAMES, AudioWorker, TurnConfig, UserTurn
from app.llm.providers import USAGE, LLMUnavailable, complete, stream_chat, whisper_transcribe

log = logging.getLogger("session")

SORRY = "Sorry, I lost you for a second there. Could you say that again?"


def patience_labels() -> dict:
    return {key: preset["label"] for key, preset in load("common")["patience"].items()}


def _log_crash(task: asyncio.Task) -> None:
    if not task.cancelled() and task.exception():
        log.error("background task crashed", exc_info=task.exception())


@dataclass
class _Reply:
    """Bookkeeping for one reply while it's being spoken."""
    started: float  # when the user's turn was committed
    clips: list[int] = field(default_factory=list)
    spoken: list[str] = field(default_factory=list)
    first_token_ms: float | None = None
    first_tts_ms: float | None = None


class BaseSession:
    mode = "base"

    def __init__(self, ws: WebSocket, models, job: AgentSession | None = None):
        """`models` holds the shared STT, Smart Turn, speaker gate and TTS (see main.Models); `job` is set when
        the session was created through the API."""
        self.ws, self.models, self.job = ws, models, job
        self.events: asyncio.Queue = asyncio.Queue()
        self.worker = AudioWorker(models.stt, models.smart_turn, models.speaker, asyncio.get_running_loop(),
                                  self.events.put_nowait)
        self.send_lock = asyncio.Lock()
        self.common = load("common")
        self.analysis_cfg = self.common.get("analysis", {})
        self.limits = self.common.get("session", {})

        self.voice = Voice(settings.tts_engine)
        self.chain: list[str] = settings.llm_chain
        self.temperature, self.max_tokens = 0.75, 400

        self.history: list[dict] = []  # the conversation so far, as LLM messages
        self.summary = ""              # older history, folded into notes by _compact()
        self.turns: list[UserTurn] = []
        self.opened_at = time.monotonic()
        self.started_at: float | None = None
        self.response_task: asyncio.Task | None = None
        self.clip_seq = 0
        self.clips: dict[int, dict] = {}    # every clip sent: its text, and whether it started playing
        self.current_clips: list[int] = []  # the clips of the reply being spoken
        self.merge_next = False   # the next user turn continues the last one: we replied too early
        self.ending = False       # the partner said goodbye; end once it has played
        self.ended = False
        self.user_active = False
        self.compacting = False
        self.partner_done_at: float | None = None
        self.pending_latency: float | None = None
        self.idle_step = 0
        self.background: set[asyncio.Task] = set()
        self._commit_at = time.monotonic()  # when the turn being answered was handed to the LLM

    # --- hooks for the modes ---

    async def setup(self, msg: dict) -> None:
        """Read the setup screen; pick the voice, the model and the turn-taking preset."""

    def system_prompt(self) -> str:
        return ""

    def opening(self) -> str | None:
        """An instruction for the partner's first line, or None when the user speaks first."""
        return None

    async def ready(self) -> None:
        """Awaited before the first line (the interview finishes its preparation here)."""

    def before_reply(self) -> None:
        """Called before every reply is written."""

    def on_reply(self, spoken: str, meta: dict) -> None:
        """The partner finished writing a reply; `meta` is its ##META line."""

    async def on_user_turn(self, turn: UserTurn) -> None:
        """The user finished a turn."""

    def on_idle(self, idle_s: float) -> tuple[str, str] | None:
        """Silence after the partner spoke: ("say", line) speaks a fixed line, ("llm", instruction) a fresh one."""
        return None

    def time_limit_min(self) -> float:
        return 60

    def memory_cfg(self) -> dict:
        return {"trigger": 20, "keep": 10, "prompt": "Summarize this conversation in 5-8 bullets."}

    def normalize(self, text: str) -> str:
        """Clean up what the speech recognizer heard."""
        return text

    def whisper_hint(self) -> str:
        """Vocabulary for the second transcription pass."""
        return ""

    def cue_hint(self) -> str:
        """How this partner acts when the voice can act."""
        return ""

    def speakable(self, sentence: str) -> str:
        """Last touches to a sentence before it's voiced."""
        return sentence

    async def mode_report(self, transcript: str) -> str:
        """The mode's own feedback, in Markdown."""
        return ""

    def extra_record(self) -> dict:
        """Extra fields for session.json."""
        return {}

    async def control(self, msg: dict) -> None:
        """Messages from the page that the base class doesn't know."""

    # --- helpers for the modes ---

    def use_llm(self, cfg: dict, temperature: float, max_tokens: int) -> None:
        llm = cfg.get("llm", {})
        self.chain = llm.get("chain") or settings.llm_chain
        self.temperature = llm.get("temperature", temperature)
        self.max_tokens = llm.get("max_tokens", max_tokens)

    async def write_report(self, cfg: dict, prompt: str, max_tokens: int = 2500) -> str:
        try:
            return await complete([{"role": "user", "content": prompt}], cfg.get("chain") or settings.report_chain,
                                  temperature=cfg.get("temperature", 0.4), max_tokens=max_tokens)
        except LLMUnavailable as e:
            return f"_Couldn't write the report ({e})._"

    def apply_turn_config(self, preset: str, manual: bool = False) -> None:
        presets = self.common["patience"]
        values = presets.get(preset) or presets["patient"]
        known = TurnConfig.__dataclass_fields__
        self.worker.cfg = TurnConfig(**{k: v for k, v in values.items() if k in known}, manual=manual)

    async def set_voice(self, engine: str, voice: str, speed: float, accent: str = "", gender: str = "",
                        age: int | None = None) -> None:
        """An empty voice means "Auto": edge picks the voice for the accent and gender, Fish searches its library
        for one. An engine that can't speak here (no key, no model) is swapped for edge-tts, which needs neither."""
        if not self.models.tts.ready(engine):
            engine, voice = "edge", ""
        edge_voice = (self.common["accents"].get(accent) or {}).get(gender) or settings.edge_voice
        if not voice:
            voice = {"edge": edge_voice, "kokoro": settings.kokoro_voice}.get(engine, "")
        if engine == "fish" and not voice and accent:
            try:
                voice = await self.models.tts.fish.find_voice(accent, gender, age)
            except Exception as e:
                log.warning("Fish voice match failed, using the default voice: %s", e)
        self.voice = Voice(engine, voice, max(0.7, min(1.3, float(speed or 1.0))), fallback=edge_voice)

    @staticmethod
    def capped(minutes: float) -> float:
        """A public deployment caps the length, so one visitor can't hold the only slot for long."""
        cap = settings.max_session_min
        return min(minutes, cap) if cap else minutes

    # --- transport ---

    async def send(self, msg: dict) -> None:
        async with self.send_lock:
            with contextlib.suppress(Exception):  # the page is gone; run() cleans up
                await self.ws.send_text(json.dumps(msg, default=str))

    async def send_clip(self, clip_id: int, fmt: str, audio: bytes, text: str) -> None:
        """Audio goes out as one binary frame: a 4-byte header length, a JSON header, then the audio."""
        header = json.dumps({"type": "clip", "id": clip_id, "fmt": fmt, "text": text}).encode()
        async with self.send_lock:
            with contextlib.suppress(Exception):
                await self.ws.send_bytes(struct.pack("<I", len(header)) + header + audio)

    async def close(self) -> None:
        with contextlib.suppress(Exception):
            await self.ws.close()

    def spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self.background.add(task)
        task.add_done_callback(self.background.discard)
        task.add_done_callback(_log_crash)
        return task

    async def run(self) -> None:
        self.worker.start()
        controls: asyncio.Queue = asyncio.Queue()
        loops = [asyncio.create_task(self._consume_events()), asyncio.create_task(self._tick()),
                 asyncio.create_task(self._consume_controls(controls))]
        try:
            # Never stop reading: "begin" can wait minutes for the interview prep, and a socket nobody reads
            # stops answering keepalive pings, so uvicorn closes it while the mic keeps streaming.
            while (msg := await self.ws.receive())["type"] != "websocket.disconnect":
                if msg.get("bytes") is not None:
                    self.worker.push(msg["bytes"])
                elif msg.get("text"):
                    controls.put_nowait(msg["text"])
        finally:
            self.worker.stop()
            for task in (*loops, self.response_task):
                if task:
                    task.cancel()
            peak = self.worker.peak
            log.info("%s closed after %.0f s: %.0f s of mic audio, loudest %s, %d answer(s)%s", self.mode,
                     time.monotonic() - self.opened_at, self.worker.received_s,
                     f"{20 * math.log10(peak):.0f} dBFS" if peak else "silence", len(self.turns),
                     "" if self.started_at else ", never started")
            if self.started_at and not self.ended:  # the page closed mid-session: keep what there is
                self.ended = True
                await self.save(write_reports=False)

    async def _consume_controls(self, controls: asyncio.Queue) -> None:
        """The page's control messages, one at a time and in order."""
        while True:
            text = await controls.get()
            try:
                await self._control(json.loads(text))
            except Exception:
                log.exception("control message failed")

    async def _control(self, msg: dict) -> None:
        match msg.get("type"):
            case "setup":
                await self.setup(self.job.request if self.job else msg)
                self.worker.enroll_mode = True
            case "skip_enroll":
                self.worker.enroll_mode = False
                log.info("voice check skipped")
            case "mic":  # which microphone the page opened: tells a silent mic from a deaf recognizer
                log.info("mic: %s", {k: msg.get(k) for k in ("label", "muted", "track_rate", "rate", "state")})
            case "begin" if not self.started_at:
                await self._begin()
            case "end_turn":
                self.worker.push(b"__end__")
            case "patience":
                self.apply_turn_config(msg.get("value", "patient"), bool(msg.get("manual")))
            case "clip_start":
                if clip := self.clips.get(msg.get("id")):
                    clip["started"] = True
                    self.worker.ai_speaking = True
            case "playback_idle":
                if not self.response_task or self.response_task.done():
                    self._partner_finished()
            case "end":
                self.spawn(self.finish())
            case _:
                await self.control(msg)

    async def _begin(self) -> None:
        self.worker.enroll_mode = False
        self.spawn(self._warm())
        await self.ready()
        self.worker.push(b"__reset__")  # whatever they said while the prep finished isn't an answer
        self.spawn(self._warm())  # sockets can go stale during a long interview prep
        self.started_at = self._commit_at = time.monotonic()
        if self.job:
            book.started(self.job)
        await self.send({"type": "started", "duration_min": self.time_limit_min()})
        if opening := self.opening():
            self._add("user", opening)
            self.start_reply()
        else:
            self.partner_done_at = time.monotonic()
            await self.send({"type": "your_turn"})

    async def _warm(self) -> None:
        if self.voice.engine != "kokoro":
            await self.models.tts.warm(self.voice.engine)
        if self.voice.engine in ("elevenlabs", "fish"):
            await self.models.tts.prewarm(self.voice)

    # --- events from the audio worker ---

    async def _consume_events(self) -> None:
        while True:
            event = await self.events.get()
            try:
                await self._on_event(event)
            except Exception:
                log.exception("event %s failed", event.get("kind"))

    async def _on_event(self, ev: dict) -> None:
        match ev.pop("kind"):
            case "partial":
                await self.send({"type": "partial", "text": self.normalize(ev["text"]), "tentative": ev["barge_in"]})
            case "turn_started" if not ev["barge_in"]:
                await self._on_turn_started()
            case "barge_in":
                self.pending_latency = 0.0
                await self._interrupted()
            case "turn_ended":
                await self._on_turn_ended(ev)
            case "turn_discarded":
                await self._on_turn_discarded(ev["reason"])
            case "debug":
                await self.send({"type": "debug", **ev})
            case "enrolled":
                log.info("voice check passed (%d words)", len(ev["text"].split()))
                await self.send({"type": "enrolled", "text": self.normalize(ev["text"]), "voiced": ev["voiced"],
                                 "speaker_gate": self.models.speaker is not None})
            case "enroll_failed":
                log.info("voice check too short (%d words)", len(ev.get("text", "").split()))
                await self.send({"type": "enroll_failed", **ev})

    async def _on_turn_started(self) -> None:
        self.user_active = True
        self.idle_step = 0
        if self.partner_done_at is not None and not self.worker.ai_speaking:
            # the VAD noticed them a few frames after they actually started
            waited = time.monotonic() - self.partner_done_at - ONSET_FRAMES * FRAME_SEC
            self.pending_latency = round(max(0.0, waited), 2)
        if self.voice.engine != "kokoro":
            self.spawn(self.models.tts.warm(self.voice.engine))
        replying = self.response_task and not self.response_task.done()
        if self.started_at and replying and not any(self.clips[c]["started"] for c in self.current_clips):
            # they weren't done after all: drop the reply before it's heard and keep listening
            self.response_task.cancel()
            self.merge_next = True
            self.worker.ai_speaking = False
            await self.send({"type": "stop_audio"})
        await self.send({"type": "user_speaking"})

    async def _on_turn_discarded(self, reason: str) -> None:
        self.user_active = False
        if self.merge_next and self.started_at and not self.ending:
            self.merge_next = False  # the reply we dropped for them was dropped for nothing: give it again
            self._commit_at = time.monotonic()
            self.start_reply()
        if reason == "background":
            await self.send({"type": "notice", "text": "Ignored a voice that didn't match yours."})
        await self.send({"type": "user_idle", "reason": reason})

    async def _on_turn_ended(self, ev: dict) -> None:
        self.user_active = False
        text = self.normalize(ev["text"])
        info = {key: ev.get(key) for key in ("reason", "smart_turn", "semantic", "silence", "voiced", "pauses")}
        await self.send({"type": "user_final", "text": text, "info": info})
        if not self.started_at or self.ended:
            return
        merge = self.merge_next and bool(self.turns)
        self.merge_next = False
        if merge:
            turn = self.turns[-1]
            turn.merge(text, ev["audio"], ev["voiced"], ev.get("pauses") or [])
        else:
            turn = UserTurn(text=text, audio=ev["audio"], voiced=ev["voiced"], partner_said=self.last_partner_line(),
                            pauses=list(ev.get("pauses") or []), latency=self.pending_latency,
                            minute=round((time.monotonic() - self.started_at) / 60, 2))
            self.turns.append(turn)
        self.pending_latency = None
        self._add("user", text)
        self._commit_at = time.monotonic()
        self.spawn(self._analyse_turn(turn))
        await self.on_user_turn(turn)
        if not self.ending:
            self.start_reply()

    async def _analyse_turn(self, turn: UserTurn) -> None:
        """A second, careful transcript (it keeps the fillers) and the delivery metrics, off the critical path."""
        revision = turn.revision
        if settings.refine_transcripts:
            prompt = f"{self.analysis_cfg.get('verbatim_prompt', '')} {self.whisper_hint()}"
            verbatim = await whisper_transcribe(turn.audio, prompt)
            if verbatim and turn.revision == revision:
                turn.verbatim = self.normalize(verbatim)  # jargon fixes only; the disfluencies stay
        if self.analysis_cfg.get("enabled", True) and turn.revision == revision:
            turn.metrics = analysis.turn_metrics(turn, self.analysis_cfg)

    # --- the partner talking ---

    def start_reply(self, extra: str = "") -> None:
        if self.response_task and not self.response_task.done():
            self.response_task.cancel()
        self.response_task = asyncio.create_task(self._reply(extra))
        self.response_task.add_done_callback(_log_crash)

    def acting(self) -> bool:
        """True when the voice performs [bracket] cues (Fish, ElevenLabs v3/v4) and acting is switched on."""
        return (self.common.get("expressive", {}).get("enabled", True)
                and cues.supports(self.voice.engine, settings.elevenlabs_model))

    def for_voice(self, text: str) -> str:
        """What the TTS gets: jargon respelled, and cues kept only when the voice performs them."""
        say = self.common.get("pronounce") or {}
        text = cues.respell(text, {**(say.get("all") or {}), **(say.get(self.voice.engine) or {})})
        if not self.acting():
            return cues.plain(text)  # may be empty: a lone "[sigh]" must not be read out as "sigh"
        return cues.normalise(text, int(self.common["expressive"].get("max_cues_per_sentence", 3)))

    async def _reply(self, extra: str = "") -> None:
        """Stream the LLM's reply and speak it sentence by sentence while it's still being written."""
        self.before_reply()
        reply = _Reply(started=self._commit_at)
        self.current_clips = reply.clips
        sentences: asyncio.Queue[str | None] = asyncio.Queue()
        speaker = asyncio.create_task(self._speak(sentences, reply))
        await self.send({"type": "ai_thinking"})
        try:
            full = await self._write(self._messages(extra), sentences, reply)
            await sentences.put(None)
            await speaker
        except asyncio.CancelledError:
            speaker.cancel()
            raise
        log.info("said: %s", full.replace("\n", " ")[:600])
        spoken, meta = parse_meta(full)
        spoken = cues.plain(spoken)
        self._add("assistant", spoken)
        self.worker.last_question = spoken
        await self.send({"type": "ai_text", "text": spoken})
        self.on_reply(spoken, meta)
        if meta.get("end") is True:
            self.ending = True
        self.spawn(self._compact())
        if not reply.clips:
            self._partner_finished()

    def _messages(self, extra: str) -> list[dict]:
        prompt = self.system_prompt()
        if self.acting():
            prompt += "\n\n" + self.common["expressive"]["instructions"].format(hint=self.cue_hint()).strip()
        messages = [{"role": "system", "content": prompt}, *self.history]
        if extra:
            messages.append({"role": "user", "content": extra})
        return messages

    async def _write(self, messages: list[dict], sentences: asyncio.Queue, reply: _Reply) -> str:
        """Stream the LLM into the sentence queue and return the whole text."""
        full = ""
        chunker = SentenceChunker()
        try:
            async for delta in stream_chat(messages, self.chain, self.temperature, self.max_tokens,
                                           first_token_timeout=float(self.limits.get("first_token_timeout_s", 6))):
                full += delta
                if reply.first_token_ms is None:
                    reply.first_token_ms = (time.monotonic() - reply.started) * 1000
                for sentence in chunker.push(delta):
                    await sentences.put(self.speakable(sentence))
            for sentence in chunker.flush():
                await sentences.put(self.speakable(sentence))
        except LLMUnavailable as e:
            log.error("LLM unavailable: %s", e)
            await self.send({"type": "notice", "text": f"All LLM providers failed: {str(e)[:300]}"})
            full = SORRY
            await sentences.put(full)
        return full

    async def _speak(self, sentences: asyncio.Queue, reply: _Reply) -> None:
        carry = ""  # a cue that arrived as a "sentence" of its own waits for the next one
        while (sentence := await sentences.get()) is not None:
            sentence, carry = f"{carry} {sentence}".strip(), ""
            if not cues.plain(sentence):
                carry = sentence
                continue
            if not (tts_text := self.for_voice(sentence)):
                continue
            if tts_text != cues.plain(sentence):
                log.info("voiced as: %s", tts_text)
            try:
                tts_started = time.monotonic()
                audio, fmt = await self.models.tts.synth(tts_text, self.voice, " ".join(reply.spoken))
            except Exception as e:
                log.warning("TTS failed: %s", e)
                await self.send({"type": "notice", "text": f"TTS failed: {e}"})
                continue
            if reply.first_tts_ms is None:
                reply.first_tts_ms = (time.monotonic() - tts_started) * 1000
            caption = cues.plain(sentence)
            clip_id = self._new_clip(caption)
            reply.clips.append(clip_id)
            reply.spoken.append(caption)
            self.worker.set_ai_text(" ".join(reply.spoken))
            if len(reply.clips) == 1:
                self.worker.ai_speaking = True
                total_ms = (time.monotonic() - reply.started) * 1000
                log.info("reply: first token %.0f ms, tts %.0f ms, total %.0f ms",
                         reply.first_token_ms or -1, reply.first_tts_ms or -1, total_ms)
                await self.send({"type": "metrics", "response_ms": round(total_ms)})
            await self.send_clip(clip_id, fmt, audio, caption)

    def _new_clip(self, text: str) -> int:
        self.clip_seq += 1
        self.clips[self.clip_seq] = {"text": text, "started": False}
        return self.clip_seq

    async def say(self, text: str) -> None:
        """A fixed line, like a nudge or "Hello? Are you there?"."""
        try:
            audio, fmt = await self.models.tts.synth(self.for_voice(self.speakable(text)) or text, self.voice)
        except Exception as e:
            log.warning("couldn't say %r: %s", text, e)
            return
        text = cues.plain(text)
        clip_id = self._new_clip(text)
        self.current_clips = [clip_id]
        self.worker.set_ai_text(text)
        self._add("assistant", text)
        await self.send({"type": "ai_text", "text": text})
        await self.send_clip(clip_id, fmt, audio, text)

    def _partner_finished(self) -> None:
        self.worker.ai_speaking = False
        self.partner_done_at = time.monotonic()
        self.spawn(self.finish() if self.ending and not self.ended else self.send({"type": "your_turn"}))

    async def _interrupted(self) -> None:
        """They talked over the partner for real: stop, and remember only what they actually heard."""
        await self.send({"type": "stop_audio"})
        heard = " ".join(self.clips[c]["text"] for c in self.current_clips if self.clips[c]["started"])
        cut = f"{heard} [cut off by the user]" if heard else ""
        if self.response_task and not self.response_task.done():
            self.response_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):  # a crash is logged by _log_crash
                await self.response_task
            if cut:
                self._add("assistant", cut)
        elif cut and self.history and self.history[-1]["role"] == "assistant":
            self.history[-1]["content"] = cut
        self.ending = False
        self.worker.ai_speaking = False
        self.user_active = True
        await self.send({"type": "user_speaking", "interrupt": True})

    async def _tick(self) -> None:
        """Once a second: long silences, the time limit, and connections that never start."""
        setup_timeout = self.capped(float(self.limits.get("setup_timeout_min", 20))) * 60
        while True:
            await asyncio.sleep(1)
            now = time.monotonic()
            if not self.started_at:
                if now - self.opened_at > setup_timeout:
                    await self.send({"type": "error", "text": "The session never started, so it was closed."})
                    await self.close()
                    return
                continue
            if self.ended:
                return
            busy = (self.user_active or self.worker.turn is not None or self.worker.ai_speaking
                    or (self.response_task and not self.response_task.done()))
            if not busy and self.partner_done_at and (action := self.on_idle(now - self.partner_done_at)):
                kind, text = action
                self.idle_step += 1
                if kind == "say":
                    await self.say(text)
                else:
                    self._commit_at = now
                    self.start_reply(text)
            overtime = float(self.limits.get("overtime_min", 5))
            if (now - self.started_at) / 60 > self.time_limit_min() + overtime and not self.ending:
                self.ending = True
                self.spawn(self.finish())

    # --- memory ---

    def _add(self, role: str, content: str) -> None:
        if not content:
            return
        if self.history and self.history[-1]["role"] == role:
            self.history[-1]["content"] += " " + content
        else:
            self.history.append({"role": role, "content": content})

    def last_partner_line(self) -> str:
        return next((m["content"] for m in reversed(self.history) if m["role"] == "assistant"), "")

    async def summarize(self, text: str, prompt: str) -> str | None:
        try:
            return await complete([{"role": "system", "content": prompt}, {"role": "user", "content": text}],
                                  self.chain, 0.2, 400)
        except LLMUnavailable:
            return None

    async def _compact(self) -> None:
        """Long sessions: fold the oldest turns into a running summary, so prompts stay small."""
        memory = self.memory_cfg()
        if self.compacting or len(self.history) <= memory["trigger"]:
            return
        self.compacting = True
        try:
            cut = len(self.history) - memory["keep"]
            while cut < len(self.history) and self.history[cut]["role"] != "user":
                cut += 1
            old = self.history[:cut]
            text = "\n".join(f"{'Partner' if m['role'] == 'assistant' else 'User'}: {m['content']}" for m in old)
            notes = await self.summarize(text, memory["prompt"])
            if notes and self.history[:cut] == old:
                self.summary = (self.summary + "\n" + notes).strip()[-4000:]
                self.history = self.history[cut:]
        finally:
            self.compacting = False

    # --- the end ---

    def transcript(self) -> str:
        """For the report: restarts, repeated words and fillers are taken out, so it quotes what was meant."""
        fillers = self.analysis_cfg.get("fillers", [])
        lines = []
        for turn in self.turns:
            if turn.partner_said:
                lines.append(f"Partner: {turn.partner_said}")
            lines.append(f"User: {analysis.clean_disfluencies(turn.best_text, fillers)}")
        return "\n".join(lines)

    async def finish(self) -> None:
        if self.ended:
            return
        self.ended = True
        if self.response_task and not self.response_task.done() and not self.ending:
            self.response_task.cancel()
        await self.send({"type": "ended"})
        if pending := [t for t in self.background if t is not asyncio.current_task()]:
            await asyncio.wait(pending, timeout=30)
        await self.send({"type": "status", "text": "Writing the report..."})
        result = await self.save(write_reports=True)
        if self.job and not self.job.show_report:
            result = {"submitted": True}  # the report is for whoever created the session, not for the person
        await self.send({"type": "report", **result})
        await self.send({"type": "usage", **USAGE})
        await self.close()  # frees the live slot for the next person

    async def save(self, write_reports: bool) -> dict:
        delivery = analysis.summarize(self.turns) if self.analysis_cfg.get("enabled", True) else {}
        report = ""
        if write_reports:
            report = await self._report() if self.turns else "The session ended before anything was said."
        record = {
            "mode": self.mode,
            "minutes": round((time.monotonic() - self.started_at) / 60, 1) if self.started_at else 0,
            "delivery": delivery, "report_markdown": report,
            "summary": self.summary, "history": self.history,
            "turns": [turn.to_json() for turn in self.turns],
            **self.extra_record(),
        }
        folder = sessions.save(self.mode, record, [turn.audio for turn in self.turns],
                               report if write_reports else None)
        if self.job:
            book.finished(self.job, record, completed=write_reports)
        return {"markdown": report, "delivery": delivery, "folder": folder}

    async def _report(self) -> str:
        try:
            return await self.mode_report(self.transcript())
        except Exception as e:  # the record is worth keeping even without a report
            log.exception("report failed")
            return f"_Report failed: {e}_"
