"""A closed browser tab or a failing voice must never crash the session."""
import asyncio

from app.audio.tts import TTS, Voice
from app.core.session import BaseSession


class ClosedSocket:
    async def send_text(self, text: str) -> None:
        raise RuntimeError("websocket is closed")

    async def send_bytes(self, data: bytes) -> None:
        raise RuntimeError("websocket is closed")

    async def close(self) -> None:
        raise RuntimeError("already closed")


def test_sending_to_a_closed_page_is_harmless():
    session = BaseSession.__new__(BaseSession)  # no models needed for the transport
    session.ws = ClosedSocket()

    async def run():
        session.send_lock = asyncio.Lock()
        await session.send({"type": "notice", "text": "hi"})
        await session.send_clip(1, "mp3", b"\x00", "hi")
        await session.close()

    asyncio.run(run())


def test_prewarming_survives_a_dead_voice():
    tts = TTS()
    attempts = []

    async def broken_synth(text: str, voice: Voice, previous: str = "") -> tuple[bytes, str]:
        attempts.append(text)
        raise ConnectionError("no route to the TTS")

    tts.synth = broken_synth
    asyncio.run(tts.prewarm(Voice("fish", "some-voice")))
    assert len(attempts) == len(TTS.OPENERS)
