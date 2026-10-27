"""A fake user with a fake stammer talks to the running server, so you don't have to.

    python -m app.main                                   # terminal 1
    python -m scripts.smoke_test --mode phone --report   # terminal 2

The voice comes from edge-tts and is streamed in real time as 16 kHz PCM. The script prints what the
server heard, why it decided the turn was over, and how long the reply took.
"""
import argparse
import asyncio
import json
import re
import struct
import subprocess
import sys
import time

import edge_tts
import numpy as np
import websockets

VOICE = "en-IN-NeerjaNeural"
MIC_CHECK = ("Hello, my name is Sahil. I like talking about my work, my hobbies, and the places I have visited. "
             "Today I want to practise speaking English with a new person.")
# repetitions, and "||" is a 1.8 s silent block in the middle of an answer
ANSWERS = {
    "conversation": [
        "I I think the most important thing in life is... || to have people who understand you.",
        "When I was in college, I I wanted to become a, a teacher, but || my father wanted me to do engineering.",
        "Honestly, I am still not sure. Maybe || happiness is just being at peace with yourself.",
    ],
    "interview": [
        "Sure. I am a machine learning engineer with two years of experience. || "
        "Recently I built a RAG chatbot over insurance documents.",
        "We used QLoRA with four bit quantization so the fine-tune fit on a single GPU. "
        "The rank was sixteen and we evaluated with an LLM as a judge.",
        "For serving we used vLLM because paged attention and continuous batching gave much better throughput.",
    ],
    "phone": [
        "Hello? Yes, this is Sahil speaking. || Who is this?",
        "Okay. My flat is B 304, Green Park Society. || The gate is opposite the petrol pump.",
        "Yes, use the second gate. || The guard will let you in. Thank you, bye.",
    ],
}
SETUPS = {
    "conversation": {"persona": {"name": "Kabir", "age": 24, "gender": "male", "accent": "Indian",
                                 "occupation": "philosophy student", "mode": "deep", "scene": "cafe",
                                 "patience": "very_patient", "duration_min": 10,
                                 "traits": {"warmth": 5, "humor": 4, "energy": 4, "curiosity": 4}}},
    "interview": {"setup": {"candidate_name": "Sahil", "duration_min": 10, "style": "neutral",
                            "resume": "ML engineer, 2 years. RAG chatbot over insurance PDFs (LangChain, Qdrant). "
                                      "QLoRA fine-tune of Llama 3 8B.",
                            "jd": "AI Engineer at a fintech: RAG over financial documents, LoRA fine-tuning, "
                                  "vLLM serving, evaluation."},
                  "repos": [], "overrides": {"prep": {"steps": {"critique": {"enabled": False}}}}},
    "phone": {"setup": {"scenario": "delivery_lost", "difficulty": "medium", "duration_min": 5}},
}


async def tts_pcm(text: str, pause: float = 0.45) -> np.ndarray:
    """Synthesize sentence by sentence and join with human-like pauses (edge-tts pauses ~1 s)."""
    if "||" in text:
        segs = [await tts_pcm(t.strip(), pause) for t in text.split("||")]
        out = [segs[0]]
        for seg in segs[1:]:
            out += [np.zeros(int(16000 * 1.8), dtype=np.int16), seg]
        return np.concatenate(out)
    parts = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        pcm = await _tts_one(sentence)
        loud = np.nonzero(np.abs(pcm) > 600)[0]
        if len(loud):
            pcm = pcm[max(0, loud[0] - 800): loud[-1] + 800]
        parts += [pcm, np.zeros(int(16000 * pause), dtype=np.int16)]
    return np.concatenate(parts[:-1])


async def _tts_one(text: str) -> np.ndarray:
    mp3 = bytearray()
    async for c in edge_tts.Communicate(text, VOICE).stream():
        if c["type"] == "audio":
            mp3 += c["data"]
    pcm = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-f", "s16le", "-ar", "16000",
                          "-ac", "1", "pipe:1"], input=bytes(mp3), capture_output=True, check=True).stdout
    return np.frombuffer(pcm, dtype=np.int16)


def clip_seconds(fmt: str, audio: bytes) -> float:
    if fmt == "wav":
        return (len(audio) - 44) / 2 / 24000
    return len(audio) * 8 / 48000  # edge-tts mp3 is 48 kbps


class Candidate:
    def __init__(self, ws):
        self.ws = ws
        self.events: asyncio.Queue = asyncio.Queue()
        self.playing_until = 0.0
        self.idle = asyncio.Event()
        self.first_clip_at = None
        self.ai_text = ""

    async def reader(self):
        async for msg in self.ws:
            if isinstance(msg, bytes):
                n = struct.unpack("<I", msg[:4])[0]
                h = json.loads(msg[4:4 + n])
                dur = clip_seconds(h["fmt"], msg[4 + n:])
                if self.first_clip_at is None:
                    self.first_clip_at = time.monotonic()
                self.idle.clear()
                start = max(time.monotonic(), self.playing_until)
                self.playing_until = start + dur
                asyncio.get_running_loop().call_at(
                    asyncio.get_running_loop().time() + (start - time.monotonic()),
                    lambda i=h["id"]: asyncio.ensure_future(self.ws.send(json.dumps({"type": "clip_start", "id": i}))))
                asyncio.get_running_loop().call_at(
                    asyncio.get_running_loop().time() + (self.playing_until - time.monotonic()), self._maybe_idle)
                continue
            m = json.loads(msg)
            await self.events.put(m)

    def _maybe_idle(self):
        if time.monotonic() >= self.playing_until - 0.01:
            asyncio.ensure_future(self.ws.send(json.dumps({"type": "playback_idle"})))
            self.idle.set()

    async def wait_for(self, kind: str, timeout: float = 60):
        end = time.monotonic() + timeout
        while True:
            m = await asyncio.wait_for(self.events.get(), max(0.1, end - time.monotonic()))
            if m["type"] == "prep":
                print(f"   prep {m['step']:24s} {m['status']:8s} {m['detail'][:110]}")
            if m["type"] in ("notice", "debug", "partial"):
                if m["type"] == "notice":
                    print("   notice:", m["text"][:200])
                if m["type"] == "debug":
                    print("   debug:", {k: v for k, v in m.items() if k != "type"})
            if m["type"] == "ai_text":
                self.ai_text = m["text"]
            if m["type"] == kind:
                return m

    async def speak(self, pcm: np.ndarray, tail: float = 3.0):
        frame = 512
        pcm = np.concatenate([pcm, np.zeros(int(16000 * tail), dtype=np.int16)])
        t0 = time.monotonic()
        for i in range(0, len(pcm), frame):
            await self.ws.send(pcm[i:i + frame].tobytes())
            target = t0 + (i + frame) / 16000
            await asyncio.sleep(max(0, target - time.monotonic()))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="conversation", choices=["conversation", "interview", "phone"])
    ap.add_argument("--url", default="ws://127.0.0.1:8000/ws/")
    ap.add_argument("--answers", type=int, default=3)
    ap.add_argument("--no-enroll", action="store_true")
    ap.add_argument("--tts", default="edge", choices=["edge", "fish", "elevenlabs", "kokoro"])
    ap.add_argument("--patience", default="very_patient", choices=["very_patient", "patient", "normal"])
    ap.add_argument("--report", action="store_true", help="end the interview and print the report")
    ap.add_argument("--style", help="interview style (friendly/neutral/tough) or phone difficulty (easy/medium/hard)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # reports contain non-ASCII punctuation

    print("synthesizing candidate speech...")
    mic = await tts_pcm(MIC_CHECK)
    answers = [await tts_pcm(a) for a in ANSWERS[args.mode][:args.answers]]

    async with websockets.connect(args.url + args.mode, max_size=None) as ws:
        c = Candidate(ws)
        reader = asyncio.create_task(c.reader())
        setup = json.loads(json.dumps(SETUPS[args.mode]))
        inner = setup.get("persona") or setup.get("setup")
        inner.update({"tts_engine": args.tts, "patience": args.patience})
        if args.style:
            inner["difficulty" if args.mode == "phone" else "style"] = args.style
        await ws.send(json.dumps({"type": "setup", **setup}))
        ready = await c.wait_for("ready", 120)
        if args.mode == "phone":
            print("[call]", json.dumps(ready.get("call"))[:300])
        if args.no_enroll:
            await ws.send(json.dumps({"type": "skip_enroll"}))
        else:
            asyncio.create_task(c.speak(mic, tail=1.5))
            m = await c.wait_for("enrolled", 60)
            print(f"[mic check] heard: {m['text']}")
        await ws.send(json.dumps({"type": "begin"}))
        await c.wait_for("started", 600)
        await c.wait_for("ai_text", 60)
        await c.idle.wait()
        print(f"[partner] {c.ai_text}")

        for i, pcm in enumerate(answers, 1):
            await asyncio.sleep(0.7)
            c.first_clip_at = None
            voiced = np.nonzero(np.abs(pcm) > 600)[0]  # ignore edge-tts trailing silence
            speech_end = time.monotonic() + (voiced[-1] if len(voiced) else len(pcm)) / 16000
            speaking = asyncio.create_task(c.speak(pcm, tail=7.0))
            m = await c.wait_for("user_final", 60)
            turn_ms = (time.monotonic() - speech_end) * 1000
            print(f"\n[answer {i}] heard: {m['text']}")
            print(f"   end-of-turn after {turn_ms:.0f} ms of silence via {m['info']}")
            metrics = await c.wait_for("metrics", 60)
            await c.wait_for("ai_text", 60)
            gap = (c.first_clip_at - speech_end) * 1000 if c.first_clip_at else float("nan")
            print(f"   server response_ms={metrics['response_ms']}  silence gap heard by candidate ~{gap:.0f} ms")
            await speaking
            await c.idle.wait()
            print(f"[partner] {c.ai_text}")

        if args.report:
            await ws.send(json.dumps({"type": "end"}))
            r = await c.wait_for("report", 300)
            print("\n" + r["markdown"][:2500])
            sp = {k: v for k, v in (r.get("speech") or {}).items() if k != "per_turn"}
            print("\n--- speech analysis ---\n" + json.dumps(sp, indent=1)[:2500])
            print("\n" + (r.get("speech_markdown") or "")[:2000])
            print("saved:", r["folder"])
        reader.cancel()


if __name__ == "__main__":
    asyncio.run(main())
