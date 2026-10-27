"""Settings come from two places.

`.env` holds secrets and machine details: API keys, model paths, thread counts. `config/*.yaml` holds
behaviour: prompts, presets, timings. The YAML is re-read for every session, so edits apply without a
restart, and the setup screen can override any key for a single session.
"""
import copy
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
load_dotenv(ROOT / ".env")
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(ROOT / "models")))

PROVIDERS = ("groq", "gemini", "openrouter", "jev", "scaledown", "elevenlabs", "fish")


def _env(name: str, default=""):
    """An environment variable, cast to the type of its default."""
    raw = os.getenv(name)
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return type(default)(raw)


def _env_list(name: str, default: str = "") -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


def _find_stt_model() -> str:
    found = sorted(MODELS_DIR.glob("sherpa-onnx-nemotron-speech-streaming-en-0.6b-*"))
    return os.getenv("STT_MODEL_DIR") or (str(found[0]) if found else "")


@dataclass
class Settings:
    # Several comma-separated keys per provider. The pool sticks to one until it fails (app/llm/keypool.py).
    api_keys: dict[str, list[str]] = field(
        default_factory=lambda: {p: _env_list(f"{p.upper()}_API_KEYS") for p in PROVIDERS})
    github_token: str = _env("GITHUB_TOKEN")  # private repos only; never used by a public deployment

    llm_chain: list[str] = field(default_factory=lambda: _env_list(
        "LLM_CHAIN", "groq:qwen/qwen3.8-27b,groq:openai/gpt-oss-120b|reasoning_effort=low,"
                     "openrouter:google/gemma-4-31b-it:free"))
    report_chain: list[str] = field(default_factory=lambda: _env_list(
        "REPORT_LLM_CHAIN", "groq:openai/gpt-oss-120b|reasoning_effort=medium,groq:qwen/qwen3.8-27b,"
                            "openrouter:google/gemma-4-31b-it:free"))
    decisions_chain: list[str] = field(default_factory=lambda: _env_list(
        "DECISIONS_CHAIN", "jev:jev-latest,scaledown:classify-1"))
    whisper_model: str = _env("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")
    refine_transcripts: bool = _env("REFINE_WITH_GROQ_WHISPER", True)

    stt_dir: str = field(default_factory=_find_stt_model)
    stt_threads: int = _env("STT_THREADS", 3)
    vad_model: str = str(MODELS_DIR / "silero_vad.onnx")
    smart_turn_model: str = str(MODELS_DIR / "smart-turn-v3.2-cpu.onnx")
    speaker_model: str = str(MODELS_DIR / "speaker_titanet_small_en.onnx")
    speaker_gate: bool = _env("SPEAKER_GATE", True)
    speaker_reject_below: float = _env("SPEAKER_REJECT_BELOW", 0.45)

    tts_engine: str = _env("TTS_ENGINE", "edge")  # edge | fish | elevenlabs | kokoro
    edge_voice: str = _env("EDGE_VOICE", "en-IN-PrabhatNeural")
    kokoro_dir: str = _env("KOKORO_DIR", str(MODELS_DIR / "kokoro-multi-lang-v1_0"))
    kokoro_voice: str = _env("KOKORO_VOICE", "hm_omega")
    kokoro_threads: int = _env("TTS_THREADS", 3)
    elevenlabs_model: str = _env("ELEVENLABS_MODEL", "eleven_v4_turbo")
    elevenlabs_voice: str = _env("ELEVENLABS_VOICE")
    elevenlabs_stability: float = _env("ELEVENLABS_STABILITY", 0.45)
    elevenlabs_accent_tag: str = _env("ELEVENLABS_ACCENT_TAG", "[strong Indian accent]")
    elevenlabs_format: str = _env("ELEVENLABS_FORMAT", "mp3_44100_64")
    fish_model: str = _env("FISH_MODEL", "s2.1-pro-free")
    fish_voice: str = _env("FISH_VOICE")
    fish_latency: str = _env("FISH_LATENCY", "balanced")  # low | balanced | normal

    host: str = _env("HOST", "127.0.0.1")
    port: int = _env("PORT", 8000)
    sessions_dir: Path = DATA_DIR / "sessions"

    # A public deployment keeps no transcripts or recordings, can't edit config/*.yaml from the browser,
    # and only reads repositories from public git hosts.
    public_demo: bool = _env("PUBLIC_DEMO", False)
    max_session_min: float = _env("MAX_SESSION_MIN", 0.0)  # 0 = whatever the setup screen asks for

    # The REST API (app/api.py). With no tokens it is open, which is fine on a laptop and wrong on a server.
    api_tokens: list[str] = field(default_factory=lambda: _env_list("API_TOKENS"))
    webhook_secret: str = _env("WEBHOOK_SECRET")
    public_url: str = _env("PUBLIC_URL")  # for join links; by default the address the request came in on
    api_sessions_dir: Path = DATA_DIR / "api_sessions"


settings = Settings()


def load(name: str, overrides: dict | None = None) -> dict:
    """config/<name>.yaml with per-session overrides merged on top."""
    data = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8")) or {}
    return deep_merge(data, overrides or {})


def deep_merge(base: dict, extra: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged
