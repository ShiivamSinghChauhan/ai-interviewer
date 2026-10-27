"""Study the repositories behind the candidate's projects, and remember what was learned.

Reports are cached per repo + commit in data/repo_reports (index.json). Before downloading anything the
current commit is looked up with `git ls-remote`, so an unchanged repo is never analysed twice. All limits,
models and prompts come from the `repos:` section of config/interview.yaml.
"""
import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import stat
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from urllib.parse import urlparse

from app.config import ROOT, settings
from app.llm.providers import LLMUnavailable, complete, complete_json

log = logging.getLogger("repos")
Progress = Callable[[str, str], Awaitable[None]]  # stage, detail
PUBLIC_HOSTS = ("github.com", "gitlab.com", "bitbucket.org", "codeberg.org", "huggingface.co")


async def _quiet(stage: str, detail: str) -> None:
    pass


def normalize(source: str) -> str:
    """One spelling per repo: a clean https URL, or an absolute folder path."""
    source = source.strip().strip('"').rstrip("/")
    if re.match(r"^(https?://|git@)", source):
        source = re.sub(r"\.git$", "", source)
        source = re.sub(r"^git@github\.com:", "https://github.com/", source)
        return re.sub(r"/(tree|blob)/[^/]+.*$", "", source)  # a link to a branch or a file
    return str(Path(source).expanduser().resolve())


def is_remote(source: str) -> bool:
    return bool(re.match(r"^https?://", source))


def check_allowed(source: str) -> None:
    """A public deployment reads public repos on well-known hosts only: no server folders, no private repos."""
    if not settings.public_demo:
        return
    host = urlparse(source).hostname or ""
    if not is_remote(source) or host.removeprefix("www.") not in PUBLIC_HOSTS:
        raise ValueError(f"this demo only reads public repositories on {', '.join(PUBLIC_HOSTS)}")


def slug(source: str) -> str:
    parts = [p for p in re.split(r"[/\\:]+", source) if p][-2:]
    return re.sub(r"[^A-Za-z0-9_.-]", "_", "-".join(parts))[:60] or "repo"


def _with_token(url: str) -> str:
    if settings.github_token and not settings.public_demo and url.startswith("https://github.com/"):
        return url.replace("https://", f"https://x-access-token:{settings.github_token}@", 1)
    return url


async def _git(*args: str, cwd: str | None = None, timeout: float = 120) -> str:
    proc = await asyncio.create_subprocess_exec("git", *args, cwd=cwd, stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
    except TimeoutError:
        proc.kill()
        raise RuntimeError(f"git {args[0]} timed out after {timeout:.0f}s") from None
    if proc.returncode != 0:
        raise RuntimeError(f"git {args[0]} failed: {err.decode(errors='ignore')[:300]}")
    return out.decode(errors="ignore")


async def fingerprint(source: str, cfg: dict) -> str:
    """The commit id for remotes (without downloading anything); commit or content hash for local folders."""
    if is_remote(source):
        out = await _git("ls-remote", _with_token(source) + ".git", "HEAD", timeout=60)
        return out.split()[0] if out.strip() else "unknown"
    path = Path(source)
    if (path / ".git").exists():
        head = (await _git("rev-parse", "HEAD", cwd=str(path))).strip()
        dirty = await _git("status", "--porcelain", cwd=str(path))
        return head + ("-" + hashlib.sha1(dirty.encode()).hexdigest()[:8] if dirty.strip() else "")
    digest = hashlib.sha1()
    for file in sorted(_candidate_files(path, cfg)):
        info = file.stat()
        digest.update(f"{file.relative_to(path)}|{info.st_size}|{int(info.st_mtime)}".encode())
    return "dir-" + digest.hexdigest()[:16]


class ReportStore:
    def __init__(self, cfg: dict):
        self.dir = ROOT / cfg.get("reports_dir", "data/repo_reports")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.json"

    def _index(self) -> dict:
        try:
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_index(self, index: dict) -> None:
        self.index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    def get(self, key: str) -> tuple[dict | None, dict | None]:
        """(report, index entry), or (None, None)."""
        entry = self._index().get(key)
        if not entry:
            return None, None
        try:
            return json.loads((self.dir / entry["file"]).read_text(encoding="utf-8")), entry
        except FileNotFoundError:
            return None, None

    def put(self, key: str, sha: str, report: dict) -> None:
        index = self._index()
        name = f"{slug(key)}-{sha[:10]}.json"
        (self.dir / name).write_text(json.dumps(report, indent=2), encoding="utf-8")
        index[key] = {"sha": sha, "file": name, "created": time.time(), "summary": report.get("summary", "")[:200]}
        self._save_index(index)

    def delete(self, key: str) -> bool:
        index = self._index()
        entry = index.pop(key, None)
        if entry:
            (self.dir / entry["file"]).unlink(missing_ok=True)
            self._save_index(index)
        return bool(entry)

    def list(self) -> list[dict]:
        return [{"source": key, **entry}
                for key, entry in sorted(self._index().items(), key=lambda item: -item[1]["created"])]


def _rmtree(path: Path) -> None:
    def make_writable(func, p, _):  # git objects are read-only on Windows
        os.chmod(p, stat.S_IWRITE)
        func(p)
    if path.exists():
        shutil.rmtree(path, onexc=make_writable)


async def fetch(source: str, cfg: dict, progress: Progress = _quiet) -> Path:
    if not is_remote(source):
        path = Path(source)
        if not path.is_dir():
            raise FileNotFoundError(f"folder not found: {source}")
        return path
    dest = ROOT / cfg.get("workdir", "data/repos") / slug(source)
    _rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    await progress("clone", f"downloading {source}")
    args = ["clone", "--depth", str(cfg.get("clone_depth", 1)), "--single-branch"]
    if cfg.get("blob_limit"):
        args.append(f"--filter=blob:limit={cfg['blob_limit']}")
    await _git(*args, _with_token(source) + ".git", str(dest), timeout=cfg.get("timeout_s", 600))
    return dest


def _candidate_files(root: Path, cfg: dict) -> list[Path]:
    extensions = {e.lower() for e in cfg.get("include_ext", [])}
    names = set(cfg.get("include_names", []))
    skipped = set(cfg.get("exclude_dirs", []))
    found = []
    for folder, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skipped and not d.startswith(".")]
        for name in files:
            path = Path(folder) / name
            if path.suffix.lower() in extensions or name in names or name.split(".")[0] in names:
                found.append(path)
    return found


def _priority(rel: str) -> tuple:
    """Read the README and build files first, then entry points, then the rest; tests last."""
    low = rel.lower()
    name = low.rsplit("/", 1)[-1]
    if name.startswith("readme"):
        rank = 0
    elif (name in ("requirements.txt", "pyproject.toml", "package.json", "setup.py", "dockerfile", "makefile")
          or name.endswith((".toml", ".cfg")) or low.startswith("docs/")):
        rank = 1
    elif re.match(r"(main|app|server|train|infer|inference|pipeline|model|api|cli|run)[\w-]*\.", name):
        rank = 2
    elif re.match(r"^(src|app|lib|core|backend|server|models?|pipelines?)/", low):
        rank = 3
    elif "test" in low:
        rank = 6
    elif name.endswith((".json", ".yaml", ".yml", ".txt")):
        rank = 5
    else:
        rank = 4
    return rank, low.count("/"), low


def _read(path: Path, limit: int) -> str:
    try:
        raw = path.read_bytes()[: limit * 4]
    except OSError:
        return ""
    if b"\x00" in raw[:2000]:  # binary
        return ""
    text = raw.decode("utf-8", errors="ignore")
    if path.suffix == ".ipynb":
        try:
            notebook = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            text = "\n\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", [])
                               if cell.get("cell_type") in ("code", "markdown"))
        except (json.JSONDecodeError, OSError):
            pass
    return text[:limit] + ("\n... [truncated]" if len(text) > limit else "")


def digest(root: Path, cfg: dict) -> tuple[str, dict]:
    """The file tree plus as many files as fit the character budget, most telling files first."""
    files = _candidate_files(root, cfg)
    ranked = sorted(((f.relative_to(root).as_posix(), f) for f in files), key=lambda item: _priority(item[0]))
    ranked = ranked[: cfg.get("max_files", 400)]
    tree = "\n".join(rel for rel, _ in ranked[:300])
    parts = [f"## File tree ({len(files)} relevant files)\n{tree}\n"]
    used, budget = len(parts[0]), cfg.get("digest_max_chars", 60000)
    included = 0
    for rel, file in ranked:
        body = _read(file, cfg.get("max_file_chars", 6000))
        if not body.strip():
            continue
        block = f"\n## {rel}\n{body}\n"
        if used + len(block) > budget:
            continue
        parts.append(block)
        used += len(block)
        included += 1
    return "".join(parts), {"files_total": len(files), "files_included": included, "chars": used}


async def analyze(source: str, cfg: dict, progress: Progress = _quiet, force: bool = False) -> dict:
    key = normalize(source)
    check_allowed(key)
    store = ReportStore(cfg)
    await progress("check", f"checking {key}")
    try:
        sha = await fingerprint(key, cfg)
    except Exception as e:  # offline, or no git: fall back to any cached report
        log.warning("fingerprint failed for %s: %s", key, e)
        sha = "unknown"
    cached, entry = store.get(key)
    if cached and not force and cfg.get("reuse_cached", True):
        fresh = (time.time() - entry["created"]) / 86400 <= cfg.get("max_age_days", 120)
        same = entry["sha"] == sha
        if fresh and (same or sha == "unknown" or cfg.get("reuse_if_newer_commit", True)):
            await progress("cached", f"reused saved report ({'same commit' if same else 'older commit'})")
            return {**cached, "from_cache": True, "stale": not same}

    path = await fetch(key, cfg, progress)
    text, stats = await asyncio.to_thread(digest, path, cfg)
    await progress("read", f"{stats['files_included']} of {stats['files_total']} files, {stats['chars']:,} chars")
    name = slug(key)
    chain = cfg.get("chain") or settings.report_chain
    temperature, max_tokens = cfg.get("temperature", 0.3), cfg.get("max_tokens", 4000)
    report = None
    if len(text) <= cfg.get("single_pass_max_chars", 120000):
        await progress("analyse", "reading the whole repository in one pass")
        prompt = cfg["prompt"].format(repo=name, material="Repository contents:\n" + text)
        try:
            report = await complete_json([{"role": "user", "content": prompt}], chain, temperature, max_tokens,
                                         retries=1)
        except LLMUnavailable as e:  # the big-context model is busy, or it's too much for the free tier
            log.warning("single-pass analysis failed (%s); switching to chunked analysis", str(e)[:200])
    if report is None:
        material = await _map_reduce(text, name, cfg, progress)
        await progress("analyse", "writing the repository report")
        prompt = cfg["prompt"].format(repo=name, material=material)
        report = await complete_json([{"role": "user", "content": prompt}], (cfg.get("map_chain") or []) + chain,
                                     temperature, max_tokens, retries=1)
    report.update({"source": key, "sha": sha, "analysed_at": time.time(), **stats})
    store.put(key, sha, report)
    await progress("done", report.get("summary", "")[:160])
    return {**report, "from_cache": False, "stale": False}


async def _map_reduce(text: str, name: str, cfg: dict, progress: Progress) -> str:
    """Too big for one call: summarise it in chunks that fit Groq's free tier, then report on the notes."""
    size = cfg.get("map_chunk_chars", 12000)
    chunks = [text[i:i + size] for i in range(0, len(text), size)]
    notes = []
    for i, chunk in enumerate(chunks, 1):
        await progress("summarise", f"part {i}/{len(chunks)}")
        prompt = cfg["map_prompt"].format(repo=name, files=chunk)
        try:
            notes.append(await complete([{"role": "user", "content": prompt}],
                                        cfg.get("map_chain") or settings.llm_chain, 0.2, 700, retries=2))
        except LLMUnavailable as e:
            log.warning("map step %d failed: %s", i, e)
    return "Summaries of the repository's parts:\n\n" + "\n\n".join(notes)
