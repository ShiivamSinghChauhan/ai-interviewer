"""Who is talking, and are they done? Runs in its own thread, because every model call here blocks.

The hard part is not cutting people off. People pause to think, restart sentences or get stuck on a word, and
a pause is not the end of a turn, so three signals vote, cheapest first:
  1. silence from Silero VAD (32 ms frames)
  2. Smart Turn on the audio: did they *sound* finished?
  3. Jev on the transcript: is the *thought* finished? (only asked when 2 isn't sure, or for long answers)
A trailing "and..." or "so..." and a "let me think" buy extra time. Space ends the turn by hand, and pauses
inside a turn are recorded for the speech report.
"""
import asyncio
import logging
import queue
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field, fields

import numpy as np

from app.audio.models import FRAME, SR, SileroVAD, SmartTurn, SpeakerGate, StreamingSTT
from app.config import settings
from app.llm import decisions

log = logging.getLogger("turns")

FRAME_SEC = FRAME / SR
SPEECH_PROB, SILENCE_PROB = 0.5, 0.35  # VAD: above counts as speech, below as silence, in between as neither
ONSET_FRAMES = 3                       # ~100 ms of speech opens a turn...
ONSET_FRAMES_OVER_PARTNER = 8          # ...but ~250 ms while the partner is talking (their echo isn't you)
SURE, LIKELY = 0.85, 0.5               # Smart Turn probability that they finished
TRAILING_CAP = 0.45                    # "and..." can sound finished, but it never counts as LIKELY
ECHO_SHARE = 0.6                       # this share of words also in the partner's line = their echo
GATE_MIN_S = 0.8                       # the voice match needs about this much audio

TURN_DONE_QUESTION = {
    "done": decisions.noul(
        "The speaker has finished their thought and expects the other person to reply now. "
        "Note: people pause to think, repeat words and break off words mid-sentence; that does NOT mean "
        "they are finished.",
        "Finished; the other person should reply now",
        "Still mid-thought: the sentence or idea is incomplete and they will likely continue"),
}
BACKCHANNEL = {
    "yeah", "yes", "yep", "yup", "okay", "ok", "right", "sure", "mhm", "mm", "mmm", "hmm", "uh", "huh",
    "uhhuh", "alright", "cool", "great", "nice", "correct", "exactly", "true", "fine", "got", "it", "i",
    "see", "understood", "oh", "ah", "no", "thanks", "thank", "you", "good", "perfect", "makes", "sense",
}
TRAILING = {
    "and", "so", "but", "because", "um", "uh", "like", "the", "a", "an", "of", "to", "with", "or",
    "which", "that", "is", "are", "basically", "actually", "then", "if", "when", "where", "we", "i",
    "my", "for", "in", "on", "by", "also", "plus", "its", "their", "this",
}
THINKING = re.compile(
    r"(let me think|give me a (sec|second|moment|minute)|one (sec|second|moment)|let me recall|"
    r"how do i (put|explain)|let me see|thinking about it|hmm+)\W*$", re.IGNORECASE)


@dataclass
class TurnConfig:
    """Seconds of silence. The values come from the patience presets in common.yaml."""
    pause_to_check: float = 0.3
    answer_gap: float = 1.4      # silence needed after a confident "done"
    max_pause: float = 4.0       # longest silence tolerated when the signals disagree
    thinking_pause: float = 10.0
    manual: bool = False         # only Space ends the turn
    long_pause: float = 1.0      # pauses this long inside a turn are recorded
    quick_replies: bool = False  # off: a short phrase may just be the start of a longer answer


@dataclass
class Signals:
    """Everything the end-of-turn vote looks at."""
    silence: float
    smart_turn: float               # P(finished), from the audio
    semantic: float | None = None   # P(finished), from the words; None if not asked or no answer
    semantic_pending: bool = False
    trailing: bool = False          # ends on "and", "so", a comma...
    thinking: bool = False          # "let me think"
    question: bool = False
    quick: bool = False             # a short reply that may be answered without the usual gap


def end_reason(s: Signals, cfg: TurnConfig) -> str | None:
    """Why the turn is over, or None to keep listening. The first matching rule wins."""
    sounded = min(s.smart_turn, TRAILING_CAP) if s.trailing else s.smart_turn
    meant = s.semantic
    gap = 0.0 if s.quick else cfg.answer_gap  # people pause between sentences of one long answer

    if meant is not None and meant < 0.2:  # the words say "not done": only a long, sure pause overrides that
        if sounded >= SURE and s.silence >= max(1.6, gap + 0.4):
            return "smart_turn_override"
    elif sounded >= SURE and s.silence >= gap:
        return "smart_turn"
    elif s.question and sounded >= LIKELY and s.silence >= min(gap, 0.8):
        return "question"
    elif meant is not None and meant >= 0.75 and s.silence >= max(0.5, gap):
        return "semantic"
    elif sounded >= LIKELY and meant is not None and meant >= 0.4 and s.silence >= gap:
        return "smart_turn+semantic"
    elif sounded >= LIKELY and meant is None and not s.semantic_pending and s.silence >= max(0.6, gap):
        return "smart_turn_wait"
    elif sounded >= LIKELY and s.semantic_pending and s.silence >= max(1.2, gap):
        return "semantic_timeout"

    if s.silence >= (cfg.thinking_pause if s.thinking else cfg.max_pause):
        return "max_pause"
    return None


def words_of(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]", " ", text.lower()).replace("'", "").split()


@dataclass
class UserTurn:
    """A finished answer, as kept for the reports."""
    text: str                       # live transcript, jargon fixed
    audio: np.ndarray
    voiced: float                   # seconds of speech
    partner_said: str = ""          # the line it answers
    pauses: list[float] = field(default_factory=list)
    latency: float | None = None    # seconds from the partner finishing to the user starting
    minute: float = 0.0             # when in the session it started
    verbatim: str | None = None     # Whisper's second transcript, which keeps the disfluencies
    metrics: dict | None = None     # see analysis.turn_metrics
    notes: dict = field(default_factory=dict)  # extras from the mode, e.g. the interview's grades
    revision: int = 0               # bumped when a later fragment is merged in

    @property
    def best_text(self) -> str:
        return self.verbatim or self.text

    def merge(self, text: str, audio: np.ndarray, voiced: float, pauses: list[float]) -> None:
        """They carried on after the partner had started to reply: it's all one answer."""
        self.text += " " + text
        self.audio = np.concatenate([self.audio, audio])
        self.voiced += voiced
        self.pauses += pauses
        self.revision += 1

    def to_json(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name not in ("audio", "revision")}


@dataclass
class Turn:
    """A turn in progress."""
    stream: object  # the STT stream
    frames: list
    barge_in: bool  # started while the partner was talking
    started: float = field(default_factory=time.monotonic)
    pending: list = field(default_factory=list)  # frames not yet fed to the STT
    voiced_frames: int = 0
    silence_frames: int = 0
    text: str = ""
    confirmed: bool = False  # a barge-in that turned out to be a real interruption
    pauses: list = field(default_factory=list)  # durations (s) of long silences inside the turn
    # verdicts, cleared whenever they start talking again
    smart_turn: float | None = None
    trailing: bool = False
    thinking: bool = False
    question: bool = False
    quick: bool = False
    semantic_job: Future | None = None
    semantic: float | None = None  # -1 when the check failed
    semantic_text: str = ""        # the transcript the semantic check saw
    semantic_retried: bool = False


class AudioWorker(threading.Thread):
    def __init__(self, stt: StreamingSTT, smart_turn: SmartTurn, speaker: SpeakerGate | None,
                 loop: asyncio.AbstractEventLoop, emit: Callable[[dict], None]):
        super().__init__(daemon=True)
        self.stt, self.smart_turn, self.speaker = stt, smart_turn, speaker
        self.vad = SileroVAD(settings.vad_model)
        self.cfg = TurnConfig()
        self.loop, self._emit = loop, emit
        self.q: queue.Queue = queue.Queue()
        self.leftover = np.zeros(0, dtype=np.float32)
        self.preroll: deque = deque(maxlen=10)  # 320 ms before speech onset
        self.speech_run = 0
        self.turn: Turn | None = None

        # set by the session
        self.ai_speaking = False
        self.ai_words: set[str] = set()
        self.last_question = ""
        self.enroll_mode = False
        self.enrolled: np.ndarray | None = None
        self.received_s, self.peak = 0.0, 0.0  # how much mic audio arrived, and how loud it got (for the log)

    def emit(self, kind: str, **data) -> None:
        self.loop.call_soon_threadsafe(self._emit, {"kind": kind, **data})

    def push(self, pcm16: bytes) -> None:
        self.q.put(pcm16)

    def stop(self) -> None:
        self.q.put(None)

    def set_ai_text(self, text: str) -> None:
        self.ai_words = set(words_of(text))

    def run(self) -> None:
        while (item := self.q.get()) is not None:
            if item == b"__end__":  # the user pressed "I'm done"
                if self.turn is not None and not (self.turn.barge_in and not self.turn.confirmed):
                    self._end_turn(self.turn, "manual", self.turn.silence_frames * FRAME_SEC)
                continue
            if item == b"__reset__":
                self.turn, self.speech_run = None, 0
                self.preroll.clear()
                continue
            samples = np.frombuffer(item, dtype=np.int16).astype(np.float32) / 32768.0
            self.received_s += len(samples) / SR
            self.peak = max(self.peak, float(np.abs(samples).max(initial=0.0)))
            buf = np.concatenate([self.leftover, samples])
            n = len(buf) // FRAME
            for i in range(n):
                try:
                    self._frame(buf[i * FRAME:(i + 1) * FRAME])
                except Exception:  # one bad frame must not kill the audio thread
                    log.exception("frame processing failed")
                    self.turn = None
            self.leftover = buf[n * FRAME:]

    def _frame(self, frame: np.ndarray) -> None:
        p = self.vad(frame)
        t = self.turn
        if t is None:
            self.preroll.append(frame)
            self.speech_run = self.speech_run + 1 if p >= SPEECH_PROB else 0
            if self.speech_run >= (ONSET_FRAMES_OVER_PARTNER if self.ai_speaking else ONSET_FRAMES):
                self._start_turn()
            return

        t.frames.append(frame)
        t.pending.append(frame)
        if p >= SPEECH_PROB:
            if t.silence_frames * FRAME_SEC >= self.cfg.long_pause:
                t.pauses.append(round(t.silence_frames * FRAME_SEC, 2))
            if t.silence_frames >= 4:  # they carried on after a pause: earlier verdicts are stale
                t.smart_turn, t.semantic_job, t.semantic = None, None, None
            t.silence_frames = 0
            t.voiced_frames += 1
        elif p < SILENCE_PROB:
            t.silence_frames += 1

        if len(t.pending) >= 3:  # feed the STT every ~96 ms
            text = self.stt.feed(t.stream, np.concatenate(t.pending))
            t.pending.clear()
            if text != t.text:
                t.text = text
                self.emit("partial", text=text, barge_in=t.barge_in and not t.confirmed)

        silence = t.silence_frames * FRAME_SEC
        if t.barge_in and not t.confirmed:
            self._check_barge_in(t, silence)
        else:
            self._check_end(t, silence)

    def _start_turn(self) -> None:
        frames = list(self.preroll)
        self.preroll.clear()
        self.turn = Turn(stream=self.stt.new_stream(), frames=frames, pending=list(frames),
                         barge_in=self.ai_speaking and not self.enroll_mode, voiced_frames=self.speech_run)
        self.speech_run = 0
        self.emit("turn_started", barge_in=self.turn.barge_in)

    def _check_barge_in(self, t: Turn, silence: float) -> None:
        """Talking over the partner: a real interruption, a quick "mm-hmm", their echo, or someone else?"""
        if not self.ai_speaking:  # the partner finished while they were talking: it's a normal turn now
            t.barge_in = False
            self.emit("turn_started", barge_in=False)
            return
        words = words_of(t.text)
        echo = sum(w in self.ai_words for w in words) / len(words) if words else 0.0
        content = [w for w in words if w not in BACKCHANNEL]
        voiced = t.voiced_frames * FRAME_SEC
        # Fast path: 0.7 s of speech that clearly matches the enrolled voice. A quick "mm-hmm" never gets
        # that far, and the partner's own echo doesn't match the user.
        recheck = self._gate_ready() and voiced >= 0.7 and len(t.frames) % 8 == 0  # every ~250 ms
        if recheck and echo < ECHO_SHARE and self._similarity(t) >= settings.speaker_reject_below + 0.1:
            self._confirm_barge_in(t)
            return
        long_enough = len(t.frames) * FRAME_SEC >= GATE_MIN_S
        substantial = len(content) >= 2 or len(words) >= 4 or voiced > 2.0
        if long_enough and words and echo < ECHO_SHARE and substantial:
            if self._voice_ok(t, strict=True):
                self._confirm_barge_in(t)
            else:
                self._discard(t, "background")
            return
        if silence >= 0.6 or time.monotonic() - t.started > 5:
            self._discard(t, "echo" if echo >= ECHO_SHARE else "backchannel")

    def _confirm_barge_in(self, t: Turn) -> None:
        t.confirmed, t.barge_in = True, False
        self.emit("barge_in", text=t.text)

    def _check_end(self, t: Turn, silence: float) -> None:
        if silence < self.cfg.pause_to_check:
            return
        if t.voiced_frames < 5 and silence > 0.5:  # a click or a cough
            self._discard(t, "noise")
            return
        if self.enroll_mode:  # mic check: just wait for a clear pause after the sentence
            if silence >= 1.5:
                self._end_turn(t, "enroll", silence)
            return
        if self.cfg.manual:  # they end their own turn; only a very long silence ends it for them
            if silence >= 30:
                self._end_turn(t, "manual_timeout", silence)
            return

        if t.smart_turn is None:
            self._judge_audio(t)
        if t.semantic_job is not None and t.semantic is None and t.semantic_job.done():
            self._collect_semantic(t, silence)

        signals = Signals(silence=silence, smart_turn=t.smart_turn,
                          semantic=t.semantic if t.semantic is not None and t.semantic >= 0 else None,
                          semantic_pending=t.semantic_job is not None and t.semantic is None,
                          trailing=t.trailing, thinking=t.thinking, question=t.question, quick=t.quick)
        if reason := end_reason(signals, self.cfg):
            self._end_turn(t, reason, silence)

    def _judge_audio(self, t: Turn) -> None:
        """First check after a pause: Smart Turn on the audio, plus cheap clues from the words so far."""
        t.smart_turn = self.smart_turn.probability(np.concatenate(t.frames))
        words = words_of(t.text)
        t.trailing = bool(words) and (words[-1] in TRAILING or t.text.rstrip().endswith(","))
        t.thinking = bool(THINKING.search(t.text))
        t.question = t.text.rstrip().endswith("?")
        t.quick = self.cfg.quick_replies and (t.question or len(words) <= 6)
        unsure = 0.3 <= t.smart_turn < SURE or (t.trailing and t.smart_turn >= LIKELY)
        # long answers: also ask whether the *answer* is complete, not just the sentence
        if (unsure or not t.quick) and t.text and decisions.available():
            self._ask_semantic(t)
        self.emit("debug", smart_turn=round(t.smart_turn, 3), trailing=t.trailing, thinking=t.thinking)

    def _ask_semantic(self, t: Turn) -> None:
        t.semantic_text = t.text
        state = (f"The other person last said: {self.last_question}\n"
                 f"The speaker so far (live transcript): {t.text}")
        t.semantic_job = asyncio.run_coroutine_threadsafe(
            decisions.decide(state, TURN_DONE_QUESTION, timeout=2.0), self.loop)

    def _collect_semantic(self, t: Turn, silence: float) -> None:
        try:
            answer = t.semantic_job.result()
            t.semantic = float(answer["done"]["noul"]) if answer and "done" in answer else -1.0
        except Exception:
            t.semantic = -1.0
        if t.semantic >= 0:
            self.emit("debug", semantic_done=round(t.semantic, 3))
        # The live transcript lags ~0.5 s, so the first check may have judged a cut-off sentence.
        # If it said "not done" and the words have caught up since, ask once more with the full text.
        if 0 <= t.semantic < 0.4 and t.text != t.semantic_text and not t.semantic_retried and silence >= 0.7:
            t.semantic_retried = True
            t.semantic = None
            self._ask_semantic(t)

    def _end_turn(self, t: Turn, reason: str, silence: float) -> None:
        if t.pending:
            self.stt.feed(t.stream, np.concatenate(t.pending))
            t.pending.clear()
        text = self.stt.finish(t.stream, silence)
        audio = np.concatenate(t.frames)
        self.turn = None
        if not text.strip():
            self._discard(t, "no_words", reset=False)
            return
        if self.enroll_mode:
            self._enroll(audio, text, t.voiced_frames * FRAME_SEC)
            return
        if not self._voice_ok(t):
            self._discard(t, "background", reset=False)
            return
        self.emit("turn_ended", text=text, audio=audio, reason=reason, smart_turn=t.smart_turn, pauses=t.pauses,
                  semantic=t.semantic, silence=round(silence, 2), voiced=round(t.voiced_frames * FRAME_SEC, 2))

    def _discard(self, t: Turn, reason: str, reset: bool = True) -> None:
        if reset:
            self.turn = None
        self.emit("turn_discarded", reason=reason, text=t.text)

    def _gate_ready(self) -> bool:
        return bool(settings.speaker_gate and self.speaker and self.enrolled is not None)

    def _similarity(self, t: Turn) -> float:
        voiced = t.frames[:len(t.frames) - t.silence_frames] or t.frames
        audio = np.concatenate(voiced)[-SR * 4:]  # the last 4 s, about 60 ms of CPU
        similarity = float(np.dot(self.speaker.embed(audio), self.enrolled))
        self.emit("debug", speaker_sim=round(similarity, 3))
        return similarity

    def _voice_ok(self, t: Turn, strict: bool = False) -> bool:
        if not self._gate_ready():
            return True
        if len(t.frames) * FRAME_SEC < GATE_MIN_S:
            return not strict
        threshold = settings.speaker_reject_below + (0.1 if strict else 0.0)
        return self._similarity(t) >= threshold

    def _enroll(self, audio: np.ndarray, text: str, voiced: float) -> None:
        if voiced < 2.5:
            self.emit("enroll_failed", text=text, reason="Too short - please read the whole sentence.")
            return
        if self.speaker:
            self.enrolled = self.speaker.embed(audio)
        self.emit("enrolled", text=text, voiced=round(voiced, 1), audio=audio)
