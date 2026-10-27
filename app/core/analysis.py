"""Delivery metrics for the person talking to the agent: pace, response time, pauses, fillers, answer length.

They describe how things were said, never what: the agent's own report covers the content. Inputs per turn are
the transcript (Whisper's careful second pass when there is one, because it keeps the fillers), the seconds of
speech, the pauses inside the turn and the response time.
"""
import re
import statistics

from app.core.turns import UserTurn

PART_WORD = re.compile(r"\b(\w{1,3})-(?:\s*\1-)*\s*\1\w*", re.IGNORECASE)  # "w-w-what", "I-I", "b- because"
WHOLE_WORD = re.compile(r"\b(\w+)(?:[\s,.]+\1\b)+", re.IGNORECASE)          # "I I I", "the, the"
BROKEN = re.compile(r"\w+(?:--|—)(?=\s|$)")                                 # "wan-- want"


def clean_disfluencies(text: str, fillers: list[str]) -> str:
    """What the speaker meant, without restarts, repeated words or fillers: what the reports quote."""
    text = BROKEN.sub("", text)                                             # "want-- wanted" -> "wanted"
    text = PART_WORD.sub(lambda m: m.group(0).split("-")[-1].strip(), text)  # "w-w-what" -> "what"
    text = WHOLE_WORD.sub(lambda m: m.group(1), text)                       # "I I wanted" -> "I wanted"
    if fillers:
        pattern = r"\b(?:" + "|".join(map(re.escape, fillers)) + r")\b[,.]?\s*"
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", text).strip()


def count_fillers(text: str, fillers: list[str]) -> int:
    if not fillers:
        return 0
    return len(re.findall(r"\b(?:" + "|".join(map(re.escape, fillers)) + r")\b", text, flags=re.IGNORECASE))


def turn_metrics(turn: UserTurn, cfg: dict) -> dict:
    """One answer. The pace counts words per minute of speech, so long pauses don't drag it down."""
    fillers = cfg.get("fillers", [])
    words = len(re.findall(r"[A-Za-z']+", clean_disfluencies(turn.best_text, fillers)))
    return {
        "words": words,
        "wpm": round(60 * words / max(turn.voiced or 0.0, 0.1)),
        "fillers": count_fillers(turn.best_text, fillers),
        "long_pauses": sum(pause >= cfg.get("long_pause_s", 1.0) for pause in turn.pauses),
        "voiced_s": round(turn.voiced or 0.0, 1),
        "latency": turn.latency,
    }


def summarize(turns: list[UserTurn]) -> dict:
    """The whole session: medians for pace and response time, rates for pauses and fillers."""
    ms = [t.metrics for t in turns if t.metrics]
    if not ms:
        return {}
    words = sum(m["words"] for m in ms)
    talk_s = sum(m["voiced_s"] for m in ms)
    latencies = [m["latency"] for m in ms if m["latency"] is not None and m["latency"] >= 0]
    return {
        "answers": len(ms),
        "talk_time_s": round(talk_s),
        "words_per_minute": round(statistics.median(m["wpm"] for m in ms)),
        "response_time_s": round(statistics.median(latencies), 2) if latencies else None,
        "long_pauses_per_min": round(sum(m["long_pauses"] for m in ms) / max(talk_s / 60, 1 / 60), 1),
        "fillers_per_100_words": round(100 * sum(m["fillers"] for m in ms) / max(words, 1), 1),
        "words_per_answer": round(words / len(ms)),
        "per_turn": [{"turn": i, "words": m["words"], "wpm": m["wpm"], "latency": m["latency"],
                      "long_pauses": m["long_pauses"]} for i, m in enumerate(ms, 1)],
    }
