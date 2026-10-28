"""Finished sessions on disk: data/sessions/<mode>/<started>/ holds session.json, report.md and the person's
audio (audio/U1.wav, U2.wav...). A public deployment keeps nothing."""
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from app.audio.wav import wav_bytes
from app.config import settings


class SessionStore:
    def __init__(self, root: Path, enabled: bool = True):
        self.root = root
        self.enabled = enabled

    def save(self, mode: str, record: dict, recordings: list[np.ndarray], report: str | None = None) -> str:
        """Write one session and return its folder, or "" when nothing is kept."""
        if not self.enabled:
            return ""
        folder = self.root / mode / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        (folder / "audio").mkdir(parents=True, exist_ok=True)
        for i, audio in enumerate(recordings, 1):
            (folder / "audio" / f"U{i}.wav").write_bytes(wav_bytes(audio))
        (folder / "session.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        if report is not None:
            (folder / "report.md").write_text(report, encoding="utf-8")
        return str(folder)

    def list(self, limit: int = 200) -> list[dict]:
        """The newest sessions, summarised for the history table."""
        if not self.enabled:
            return []
        rows = []
        files = sorted(self.root.glob("*/*/session.json"), key=lambda f: f.parent.name, reverse=True)
        for file in files[:limit]:
            try:
                record = json.loads(file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            delivery = record.get("delivery") or {}
            rows.append({
                "mode": record.get("mode"), "id": file.parent.name, "minutes": record.get("minutes"),
                "title": _title(record), "turns": len(record.get("turns", [])),
                "has_report": bool(record.get("report_markdown")),
                "wpm": delivery.get("words_per_minute"), "response_s": delivery.get("response_time_s"),
            })
        return rows

    def get(self, mode: str, session_id: str) -> dict | None:
        file = self.root / mode / session_id / "session.json"
        if not self.enabled or not file.resolve().is_relative_to(self.root.resolve()) or not file.exists():
            return None
        record = json.loads(file.read_text(encoding="utf-8"))
        return {"markdown": record.get("report_markdown", ""), "delivery": record.get("delivery") or {},
                "folder": str(file.parent), "mode": record.get("mode")}


def _title(record: dict) -> str:
    """The role the interview was for."""
    return (record.get("setup") or {}).get("role") or ""


sessions = SessionStore(settings.sessions_dir, enabled=not settings.public_demo)
