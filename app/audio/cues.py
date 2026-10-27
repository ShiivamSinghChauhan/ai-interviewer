"""Delivery cues ("acting directions") inside spoken text.

Fish Audio S2.x and ElevenLabs v3/v4 perform free-form [bracket] cues - [emphasis], [sigh], [chuckling],
[in a hurry tone], [doubtful], [break] - without reading them aloud. Other engines would read them, so they
are stripped there. Angle-bracket tags (<emphasis>x</emphasis>) are converted to bracket cues: measured on
Fish S2.1, `<emphasis>` worked but `<stress>` was read aloud, while `[emphasis]` lengthened the word and
raised its pitch by 3-6 semitones.
"""
import re

BRACKET = re.compile(r"\[[^\]\[]{1,40}\]\s*")
ANGLE_PAIR = re.compile(r"<\s*([a-zA-Z][\w -]{0,30}?)\s*>(.*?)<\s*/\s*\1\s*>", re.DOTALL)
ANGLE_ANY = re.compile(r"</?\s*[a-zA-Z][\w -]{0,30}\s*/?>")


def supports(engine: str, elevenlabs_model: str = "") -> bool:
    if engine == "fish":
        return True
    return engine == "elevenlabs" and elevenlabs_model.startswith(("eleven_v3", "eleven_v4"))


# SSML-ish names the LLM may use -> what Fish S2 performs reliably
ALIASES = {"stress": "emphasis", "strong": "emphasis", "em": "emphasis", "b": "emphasis", "pause": "break",
           "long pause": "long-break", "whisper": "whispering", "laugh": "laughing", "sighs": "sighing"}


def normalise(text: str, max_cues: int = 3) -> str:
    """<emphasis>x</emphasis> -> [emphasis] x; drop leftover angle tags; keep at most max_cues cues."""
    t = ANGLE_PAIR.sub(lambda m: f"[{ALIASES.get(m.group(1).strip().lower(), m.group(1).strip())}] {m.group(2)}", text)
    t = ANGLE_ANY.sub("", t)
    count = 0

    def keep(m):
        nonlocal count
        count += 1
        return m.group(0) if count <= max_cues else ""
    t = BRACKET.sub(keep, t)
    return re.sub(r"\s{2,}", " ", t).strip()


def respell(text: str, mapping: dict) -> str:
    """Say words the way they should sound ("Qdrant" -> "quadrant"); whole words, case-sensitive."""
    if not mapping:
        return text
    words = "|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True))  # longest first
    pattern = re.compile(r"(?<![\w-])(" + words + r")(?![\w-])")
    return pattern.sub(lambda m: str(mapping[m.group(1)]), text)


def plain(text: str) -> str:
    """Plain words only (captions, transcripts, engines without cue support)."""
    t = ANGLE_PAIR.sub(lambda m: m.group(2), text)
    t = ANGLE_ANY.sub("", t)
    t = BRACKET.sub("", t)
    t = re.sub(r"\s+([.,!?;:])", r"\1", t)
    return re.sub(r"\s{2,}", " ", t).strip()
