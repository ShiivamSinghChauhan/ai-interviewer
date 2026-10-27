"""Publish the app as a Hugging Face Space (Docker SDK, free CPU hardware).

    pip install -r requirements-dev.txt
    python -m scripts.deploy_space you/practice-studio --secrets

Needs HF_TOKEN, a write token from huggingface.co/settings/tokens, in .env or the environment. --secrets copies
the *_API_KEYS from .env into the Space's secrets, which only the running app can read. Every run mirrors the
project into the Space in one commit, and the Space rebuilds itself.
"""
import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from app.config import PROVIDERS, ROOT

SHIPPED = ["app", "web", "config", "scripts", "Dockerfile", "requirements.txt", "pyproject.toml"]
SKIPPED = shutil.ignore_patterns("__pycache__", "*.bak", "*.pyc")
SECRETS = [f"{provider.upper()}_API_KEYS" for provider in PROVIDERS]
VARIABLES = ["LLM_CHAIN", "REPORT_LLM_CHAIN", "DECISIONS_CHAIN", "TTS_ENGINE", "FISH_MODEL", "ELEVENLABS_MODEL"]
SPACE_CARD = """---
title: Practice Studio
emoji: 🎙️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Voice practice for interviews, calls and small talk
---

Voice practice for job interviews, phone calls and small talk, built to be patient with stammering.
{source}
"""


def stage(folder: Path, source_url: str) -> None:
    for name in SHIPPED:
        path = ROOT / name
        if path.is_dir():
            shutil.copytree(path, folder / name, ignore=SKIPPED)
        else:
            shutil.copy2(path, folder / name)
    source = f"\nSource code and docs: {source_url}\n" if source_url else ""
    (folder / "README.md").write_text(SPACE_CARD.format(source=source), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("space", help="owner/name, e.g. shiivamchauhan/practice-studio")
    ap.add_argument("--secrets", action="store_true", help="copy the API keys from .env into the Space's secrets")
    ap.add_argument("--source", default="", help="link to the source code, shown on the Space's page")
    args = ap.parse_args()

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)
    if args.secrets:
        for name in SECRETS:
            if value := os.getenv(name):
                api.add_space_secret(args.space, name, value)
                print(f"secret   {name}")
        for name in VARIABLES:
            if value := os.getenv(name):
                api.add_space_variable(args.space, name, value)
                print(f"variable {name}")
    with tempfile.TemporaryDirectory() as folder:
        stage(Path(folder), args.source)
        api.upload_folder(folder_path=folder, repo_id=args.space, repo_type="space",
                          delete_patterns=[f"{d}/*" for d in SHIPPED if (ROOT / d).is_dir()],  # files removed here
                          commit_message="Deploy from scripts/deploy_space.py")
    owner, name = args.space.split("/")
    print(f"\nbuilding: https://huggingface.co/spaces/{args.space}")
    print(f"live at:  https://{owner}-{name}.hf.space (once the build finishes)")


if __name__ == "__main__":
    main()
