"""End-of-session speech analysis shared by every mode: how much you stammered and how nervous you sounded,
with trends. All numbers are estimates for practice feedback, not a clinical assessment.

Inputs per user turn: the live transcript, a "verbatim" Whisper transcript that keeps disfluencies, the turn
audio (pitch, loudness, prolongations), pauses inside the turn, and the response latency. A calm baseline
comes from the mic-check reading.
"""
import asyncio
import json
import logging
import re

import numpy as np

from app.audio import prosody
from app.config import settings
from app.core.turns import UserTurn
from app.llm import decisions
from app.llm.providers import LLMUnavailable, complete

log = logging.getLogger("analysis")

PART_WORD = re.compile(r"\b(\w{1,3})-(?:\s*\1-)*\s*\1\w*", re.IGNORECASE)  # "w-w-what", "I-I", "b- because"
WHOLE_WORD = re.compile(r"\b(\w+)(?:[\s,.]+\1\b)+", re.IGNORECASE)          # "I I I", "the, the"
BROKEN = re.compile(r"\w+(?:--|—)(?=\s|$)")                                 # "wan-- want"


def clean_disfluencies(text: str, fillers: list[str]) -> str:
    """What the speaker meant, without stammer repetitions or fillers: for grammar and content feedback."""
    text = BROKEN.sub("", text)                                             # "want-- wanted" -> "wanted"
    text = PART_WORD.sub(lambda m: m.group(0).split("-")[-1].strip(), text)  # "w-w-what" -> "what"
    text = WHOLE_WORD.sub(lambda m: m.group(1), text)                       # "I I wanted" -> "I wanted"
    if fillers:
        pattern = r"\b(?:" + "|".join(map(re.escape, fillers)) + r")\b[,.]?\s*"
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", text).strip()


def syllables(text: str) -> int:
    total = 0
    for word in re.findall(r"[a-z]+", text.lower()):
        groups = re.findall(r"[aeiouy]+", word)
        silent_e = word.endswith("e") and len(groups) > 1 and not word.endswith("le")
        total += max(1, len(groups) - silent_e)
    return total


def _count(words: list[str], text: str) -> int:
    return sum(len(re.findall(rf"\b{re.escape(w)}\b", text)) for w in words)


def text_disfluencies(text: str, cfg: dict) -> dict:
    lowered = f" {text.lower()} "
    whole = [m.group(0) for m in WHOLE_WORD.finditer(text)]
    return {
        "part_word": len(PART_WORD.findall(text)), "whole_word": len(whole), "broken": len(BROKEN.findall(text)),
        "fillers": _count(cfg.get("fillers", []), lowered),
        "markers": _count(cfg.get("discourse_markers", []), lowered),
        "examples": [m.group(0) for m in PART_WORD.finditer(text)][:3] + whole[:3],
    }


def turn_metrics(turn: UserTurn, cfg: dict) -> dict:
    """Called once per user turn, in a worker thread: the acoustics take ~70 ms per 6 s of audio."""
    text = turn.best_text
    found = text_disfluencies(text, cfg)
    fillers = r"\b(" + "|".join(map(re.escape, cfg.get("fillers", ["um"]))) + r")\b"
    clean = re.sub(fillers, " ", text.lower())
    syl = syllables(clean)
    acoustic = prosody.features(turn.audio, cfg) if turn.audio is not None else {}
    blocks = [p for p in turn.pauses if p >= cfg.get("block_min_s", 1.0)]
    events = found["part_word"] + found["whole_word"] + found["broken"] + len(acoustic.get("prolongations", []))
    if cfg.get("count_blocks_as_stutters", True):
        events += len(blocks)
    voiced = max(turn.voiced or 0.1, 0.1)
    return {
        "words": len(re.findall(r"[A-Za-z']+", clean)), "syllables": syl, "stutter_events": events,
        "pct_ss": round(100 * events / max(syl, 1), 1), "rate_sps": round(syl / voiced, 2),
        "blocks": blocks, "latency": turn.latency, **found, **acoustic,
    }


def acoustic_baseline(audio: np.ndarray, text: str, voiced: float, cfg: dict) -> dict:
    acoustic = prosody.features(audio, cfg)
    return {"f0_median": acoustic.get("f0_median"), "loudness_db": acoustic.get("loudness_db"),
            "rate_sps": round(syllables(text) / max(voiced, 0.1), 2)}


def _band(value: float, bands: list) -> str:
    """bands: [[0, "calm"], [30, "slightly nervous"], ...] -> the label of the highest band reached."""
    label = bands[0][1]
    for low, name in bands:
        if value >= low:
            label = name
    return label


def _scale(value, calm, nervous) -> float | None:
    if value is None or calm == nervous:
        return None
    return float(np.clip((value - calm) / (nervous - calm), 0, 1))


def _indicators(turns: list[UserTurn], baseline: dict) -> dict:
    """The raw nervousness signals, each compared with the calm baseline where there is one."""
    ms = [t.metrics for t in turns if t.metrics]
    latencies = [m["latency"] for m in ms if m.get("latency") is not None and m["latency"] >= 0]
    words = sum(m["words"] for m in ms) or 1
    minutes = sum(t.voiced for t in turns) / 60 or 1 / 60
    pitches = [m["f0_median"] for m in ms if m.get("f0_median")]
    loudness = [m["loudness_db"] for m in ms if m.get("loudness_db") is not None]
    rate = float(np.median([m["rate_sps"] for m in ms]))
    base_pitch, base_rate, base_loud = baseline.get("f0_median"), baseline.get("rate_sps"), baseline.get("loudness_db")
    return {
        "response_latency_s": float(np.median(latencies)) if latencies else None,
        "filler_per_100_words": 100 * sum(m["fillers"] for m in ms) / words,
        "pitch_raise_st": 12 * np.log2(np.median(pitches) / base_pitch) if pitches and base_pitch else None,
        "rate_change_pct": abs(rate - base_rate) / base_rate * 100 if base_rate else None,
        "quieter_db": base_loud - float(np.median(loudness)) if loudness and base_loud is not None else None,
        "short_answers_words": words / len(ms),
        "long_pauses_per_min": sum(len(m["blocks"]) for m in ms) / minutes,
    }


def nervousness(turns: list[UserTurn], baseline: dict, cfg: dict) -> dict:
    """A 0-100 score: each indicator is placed between its "calm" and "nervous" values, then weighted."""
    if not any(t.metrics for t in turns):
        return {}
    values = _indicators(turns, baseline)
    parts, total, weights = {}, 0.0, 0.0
    for name, spec in cfg.get("nervousness", {}).items():
        level = _scale(values.get(name), spec["calm"], spec["nervous"])
        if level is None:
            continue
        parts[name] = {"value": round(float(values[name]), 2), "level": round(level, 2)}
        total += level * spec.get("weight", 1)
        weights += spec.get("weight", 1)
    score = round(100 * total / weights) if weights else None
    return {"score": score, "label": _band(score, cfg["nervousness_bands"]) if score is not None else None,
            "indicators": parts}


def _pct_stuttered(turns: list[UserTurn]) -> float:
    ms = [t.metrics for t in turns if t.metrics]
    return round(100 * sum(m["stutter_events"] for m in ms) / (sum(m["syllables"] for m in ms) or 1), 1)


def summarize(turns: list[UserTurn], baseline: dict, cfg: dict) -> dict:
    ms = [t.metrics for t in turns if t.metrics]
    if not ms:
        return {}
    pct = _pct_stuttered(turns)
    out = {
        "turns": len(ms),
        "stammer": {
            "pct_syllables_stuttered": pct,
            "label": _band(pct, cfg["severity_bands"]),
            "repetitions_part_word": sum(m["part_word"] for m in ms),
            "repetitions_whole_word": sum(m["whole_word"] for m in ms),
            "broken_words": sum(m["broken"] for m in ms),
            "possible_prolongations": sum(len(m.get("prolongations", [])) for m in ms),
            "possible_blocks": sum(len(m["blocks"]) for m in ms),
            "fillers": sum(m["fillers"] for m in ms),
            "examples": [e for m in ms for e in m["examples"]][:8],
        },
        "nervousness": nervousness(turns, baseline, cfg),
        "baseline": baseline,
        "per_turn": [{"turn": i, "pct_ss": m["pct_ss"], "latency": m.get("latency"), "words": m["words"],
                      "f0": m.get("f0_median")} for i, m in enumerate(ms, 1)],
    }
    if len(turns) >= 4:  # did they settle in?
        first, second = turns[:len(turns) // 2], turns[len(turns) // 2:]
        out["trend"] = {
            "first_half": nervousness(first, baseline, cfg).get("score"),
            "second_half": nervousness(second, baseline, cfg).get("score"),
            "pct_ss_first_half": _pct_stuttered(first),
            "pct_ss_second_half": _pct_stuttered(second),
        }
    return out


async def jev_confidence(turns: list[UserTurn]) -> float | None:
    """Optional: Jev reads each answer for hedging, apologising and trailing off. Returns the mean, 0-3."""
    if not decisions.available():
        return None
    question = {"confidence": decisions.score(
        "How confident and composed does the speaker come across in this reply, judging only by the wording "
        "(hedging, apologising, trailing off, self-corrections)? Repeated words are a stammer; ignore them.",
        ["Very hesitant", "Somewhat hesitant", "Fairly confident", "Confident"])}
    answers = [t.best_text for t in turns if len(t.best_text.split()) >= 5][-12:]
    results = await asyncio.gather(*(decisions.decide(f"Speaker said: {a}", question, timeout=10) for a in answers))
    scores = [r["confidence"]["score"] for r in results if r and "confidence" in r]
    return round(float(np.mean(scores)), 2) if scores else None


async def narrative(summary: dict, turns: list[UserTurn], mode: str, cfg: dict) -> str:
    """The speech coach's notes, written by the LLM from the numbers and a few verbatim answers."""
    if not summary:
        return ""
    metrics = json.dumps({k: v for k, v in summary.items() if k != "per_turn"}, default=str)
    excerpts = "\n".join(f"- {t.best_text}" for t in turns[:10])
    prompt = cfg["prompt"].format(mode=mode, metrics=metrics, excerpts=excerpts)
    try:
        return await complete([{"role": "user", "content": prompt}], cfg.get("chain") or settings.report_chain,
                              temperature=0.4, max_tokens=1200)
    except LLMUnavailable as e:
        log.warning("speech narrative failed: %s", e)
        return ""
