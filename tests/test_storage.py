"""Where things are kept, and what a public deployment refuses to do."""
import json

import numpy as np
import pytest

from app.config import deep_merge, settings
from app.core.store import SessionStore
from app.modes.interview import repos


def test_deep_merge_overrides_without_touching_the_base():
    base = {"prep": {"enabled": True, "steps": {"jd": {"temperature": 0.3}}}}
    merged = deep_merge(base, {"prep": {"steps": {"jd": {"temperature": 0.9}}}})
    assert merged["prep"] == {"enabled": True, "steps": {"jd": {"temperature": 0.9}}}
    assert base["prep"]["steps"]["jd"]["temperature"] == 0.3


def test_sessions_round_trip_newest_first(tmp_path):
    store = SessionStore(tmp_path)
    for mode, stamp in (("phone", "2026-01-01_10-00-00"), ("interview", "2026-02-01_10-00-00")):
        folder = tmp_path / mode / stamp
        folder.mkdir(parents=True)
        (folder / "session.json").write_text(json.dumps({"mode": mode, "report_markdown": "ok"}))
    assert [row["mode"] for row in store.list()] == ["interview", "phone"]
    assert store.get("phone", "2026-01-01_10-00-00")["markdown"] == "ok"


def test_saving_writes_the_audio(tmp_path):
    folder = SessionStore(tmp_path).save("phone", {"mode": "phone"}, [np.zeros(1600, dtype=np.float32)], "# hi")
    assert (tmp_path / "phone").exists() and folder
    assert len(list((tmp_path / "phone").glob("*/audio/U1.wav"))) == 1


def test_a_disabled_store_keeps_nothing(tmp_path):
    store = SessionStore(tmp_path, enabled=False)
    assert store.save("phone", {"mode": "phone"}, [np.zeros(1600, dtype=np.float32)]) == ""
    assert not any(tmp_path.iterdir())
    assert store.list() == []


def test_session_ids_cannot_escape_the_folder(tmp_path):
    (tmp_path / "secret.json").write_text("{}")
    assert SessionStore(tmp_path / "sessions").get("..", "..") is None


def test_repo_links_are_normalised():
    assert repos.normalize("git@github.com:me/proj.git") == "https://github.com/me/proj"
    assert repos.normalize("https://github.com/me/proj/tree/main/src") == "https://github.com/me/proj"


def test_public_demo_reads_only_public_repos(monkeypatch):
    monkeypatch.setattr(settings, "public_demo", True)
    repos.check_allowed("https://github.com/me/proj")
    for source in ("/home/user/app", "https://internal.example.com/repo", "file:///etc"):
        with pytest.raises(ValueError):
            repos.check_allowed(repos.normalize(source))
    monkeypatch.setattr(settings, "github_token", "secret")
    assert "secret" not in repos._with_token("https://github.com/me/private")


def test_readme_is_read_first_and_tests_last():
    files = ["tests/test_x.py", "src/model.py", "README.md", "pyproject.toml", "notes.txt"]
    assert sorted(files, key=repos._priority) == ["README.md", "pyproject.toml", "src/model.py", "notes.txt",
                                                  "tests/test_x.py"]
