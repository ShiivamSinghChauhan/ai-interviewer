"""What the partner's streamed reply turns into: sentences for the voice, and the hidden `##META` line.

The partner may end a reply with `##META {...}`. That line is never spoken: it goes to the mode
(on_reply), and `"end": true` ends the session once the reply has finished playing.
"""
import json
import re

SENTENCE_END = re.compile(r"[.!?]+[\"')\]]?\s+")
META = re.compile(r"##\s*META\s*(\{.*\})", re.DOTALL)


def parse_meta(text: str) -> tuple[str, dict]:
    """Split a full reply into (spoken text, metadata)."""
    spoken = text.split("##", 1)[0].strip()
    match = META.search(text)
    try:
        return spoken, json.loads(match.group(1)) if match else {}
    except json.JSONDecodeError:
        return spoken, {}


class SentenceChunker:
    """Cuts the streamed reply into sentences for the voice. The first piece is kept short ("Okay, got it.")
    so audio starts while the rest is still being written."""

    MIN_CHARS = 14        # later sentences shorter than this wait for the next one
    MIN_FIRST_CHARS = 5
    FIRST_CLAUSE_MAX = 30  # "Okay, so..." - a comma this early may end the first piece

    def __init__(self):
        self.buf = ""
        self.first = True
        self.closed = False

    def push(self, delta: str) -> list[str]:
        """Add streamed text; return the sentences that are complete now."""
        if self.closed:
            return []
        self.buf += delta
        if "##" in self.buf:  # the metadata line has started: nothing after it is spoken
            self.buf = self.buf.split("##", 1)[0]
            return self._split() + self.flush()
        return self._split()

    def flush(self) -> list[str]:
        """Whatever is left once the stream ends."""
        rest = self.buf.strip().rstrip("#").strip()
        self.buf, self.closed = "", True
        return [rest] if rest else []

    def _split(self) -> list[str]:
        out = []
        while (end := self._next_cut()) is not None:
            out.append(self.buf[:end].strip())
            self.buf, self.first = self.buf[end:], False
        return out

    def _next_cut(self) -> int | None:
        min_chars = self.MIN_FIRST_CHARS if self.first else self.MIN_CHARS
        end = next((m.end() for m in SENTENCE_END.finditer(self.buf)
                    if len(self.buf[:m.end()].strip()) >= min_chars), None)
        if not self.first:
            return end
        comma = self.buf.find(", ", 4)
        if 0 < comma <= self.FIRST_CLAUSE_MAX and (end is None or end > comma + 2) and len(self.buf) > comma + 25:
            return comma + 2
        if end is None and len(self.buf) > 70:  # a long first sentence: cut at a late comma rather than wait
            comma = self.buf.rfind(", ", 35)
            return comma + 2 if comma > 0 else None
        return end
