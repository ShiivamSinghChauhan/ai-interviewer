import asyncio
from types import SimpleNamespace

from app.config import load
from app.core.session import BaseSession


class FakeTTS:
    """Edge always works; Fish only with keys; ElevenLabs and Kokoro never."""

    def __init__(self, fish_keys: bool):
        self.fish_keys = fish_keys
        self.fish = SimpleNamespace(find_voice=self.find_voice)

    def ready(self, engine: str) -> bool:
        return engine == "edge" or (engine == "fish" and self.fish_keys)

    async def find_voice(self, accent: str, gender: str, age: int | None) -> str:
        return f"fish:{accent}:{gender}"


def voice(fish_keys: bool, *args):
    session = BaseSession.__new__(BaseSession)  # no socket and no audio worker: only the voice choice
    session.common, session.models = load("common"), SimpleNamespace(tts=FakeTTS(fish_keys))
    asyncio.run(session.set_voice(*args))
    return session.voice


def test_fish_finds_a_library_voice_and_keeps_a_matching_edge_voice_in_reserve():
    v = voice(True, "fish", "", 1.0, "British", "female", 40)
    assert (v.engine, v.id, v.fallback) == ("fish", "fish:British:female", "en-GB-SoniaNeural")


def test_an_engine_without_keys_becomes_edge_with_the_same_accent_and_gender():
    v = voice(False, "fish", "", 1.0, "British", "female", 40)
    assert (v.engine, v.id) == ("edge", "en-GB-SoniaNeural")


def test_a_voice_id_from_another_engine_is_not_reused():
    v = voice(True, "elevenlabs", "a1b2c3", 1.0, "American", "male")
    assert (v.engine, v.id) == ("edge", "en-US-AndrewNeural")


def test_unknown_engine_and_speed_limits():
    v = voice(True, "karaoke", "", 3.0, "Indian", "female")
    assert (v.engine, v.id, v.speed) == ("edge", "en-IN-NeerjaNeural", 1.3)
