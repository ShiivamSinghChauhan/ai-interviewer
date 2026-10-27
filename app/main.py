"""The web server: one page, a WebSocket per live session, and a handful of JSON endpoints.

    python -m app.main   ->  http://127.0.0.1:8000
"""
import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
import yaml
from fastapi import Body, FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.audio.models import SmartTurn, SpeakerGate, StreamingSTT
from app.audio.tts import TTS
from app.config import CONFIG_DIR, MODELS_DIR, ROOT, load, settings
from app.core.session import patience_labels
from app.core.store import sessions
from app.llm import decisions
from app.llm.providers import POOLS
from app.modes import conversation, phone
from app.modes.interview import prep, repos
from app.modes.interview import session as interview

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname).1s %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("main")

WEB = ROOT / "web"
MODES = {"interview": interview, "conversation": conversation, "phone": phone}
SESSIONS = {"interview": interview.InterviewSession, "conversation": conversation.ConversationSession,
            "phone": phone.PhoneSession}
EDITABLE = ("common", "interview", "conversation", "phone")
KEEP_JOBS = 50


class Models:
    """Loaded once at startup and shared by every session."""

    def __init__(self):
        self.stt = StreamingSTT(settings.stt_dir, settings.stt_threads)
        self.smart_turn = SmartTurn(settings.smart_turn_model)
        self.speaker = SpeakerGate(settings.speaker_model) if Path(settings.speaker_model).exists() else None
        self.tts = TTS()


class ResumeText(BaseModel):
    resume: str = ""


class RepoRequest(BaseModel):
    source: str
    force: bool = False


models: Models | None = None
live_slot = asyncio.Lock()  # one CPU's worth of speech models, so one live session at a time
jobs: dict[str, dict] = {}
_job_tasks: set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global models
    log.info("loading local models from %s ...", MODELS_DIR)
    models = await asyncio.to_thread(Models)
    log.info("ready: open http://%s:%d%s", settings.host, settings.port,
             " (public demo)" if settings.public_demo else "")
    yield


app = FastAPI(title="Practice Studio", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.middleware("http")
async def always_revalidate(request, call_next):
    # without this the browser keeps serving yesterday's style.css after an update
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/")
async def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/meta")
async def meta():
    tts = models.tts
    return {"providers": {name: len(pool.keys) for name, pool in POOLS.items()},
            "decisions": decisions.available(), "speaker_gate": models.speaker is not None,
            "engines": {"edge": True, "fish": tts.fish.available, "elevenlabs": tts.eleven.available,
                        "kokoro": tts.kokoro_available},
            "default_engine": settings.tts_engine, "patience": patience_labels(),
            "public": settings.public_demo, "max_session_min": settings.max_session_min, "busy": live_slot.locked()}


@app.get("/api/{mode}/options")
async def mode_options(mode: str):
    if mode not in MODES:
        raise HTTPException(404)
    return MODES[mode].options()


@app.get("/api/conversation/surprise")
async def surprise():
    return conversation.surprise()


@app.get("/api/voices/{engine}")
async def voices(engine: str, accent: str | None = None):
    if engine == "elevenlabs" and models.tts.eleven.available:
        return await models.tts.eleven.voices()
    if engine == "fish" and models.tts.fish.available:
        return await models.tts.fish.voices(accent)
    return []


@app.post("/api/interview/projects")
async def projects(body: ResumeText):
    return await prep.extract_projects(body.resume, load("interview"))


def _repo_reports() -> repos.ReportStore:
    return repos.ReportStore(load("interview")["repos"])


@app.get("/api/interview/repos")
async def repo_list():
    return _repo_reports().list()


@app.get("/api/interview/repos/report")
async def repo_report(source: str):
    report, _ = _repo_reports().get(repos.normalize(source))
    if not report:
        raise HTTPException(404)
    return report


@app.delete("/api/interview/repos")
async def repo_delete(source: str):
    if settings.public_demo:
        raise HTTPException(403, "Saved reports can't be deleted on the public demo.")
    return {"deleted": _repo_reports().delete(repos.normalize(source))}


@app.post("/api/interview/repos/analyze")
async def repo_analyze(body: RepoRequest):
    """Runs in the background; the page polls /api/interview/repos/job/{id}."""
    if settings.public_demo and any(job["status"] == "running" for job in jobs.values()):
        raise HTTPException(429, "Another repository is being analysed. Try again in a minute.")
    for old in [job_id for job_id, job in jobs.items() if job["status"] != "running"][:-KEEP_JOBS]:
        del jobs[old]
    job_id = uuid.uuid4().hex[:8]
    job = jobs[job_id] = {"status": "running", "log": []}

    async def progress(stage: str, detail: str) -> None:
        job["log"].append(f"{stage}: {detail}")

    async def work() -> None:
        try:
            report = await repos.analyze(body.source, load("interview")["repos"], progress, body.force)
            job.update(status="done", summary=report.get("summary", ""), from_cache=report.get("from_cache"))
        except Exception as e:
            job.update(status="failed", error=str(e)[:300])

    task = asyncio.create_task(work())
    _job_tasks.add(task)
    task.add_done_callback(_job_tasks.discard)
    return {"job": job_id}


@app.get("/api/interview/repos/job/{job_id}")
async def repo_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404)
    return jobs[job_id]


@app.get("/api/sessions")
async def session_list():
    return sessions.list()


@app.get("/api/sessions/{mode}/{session_id}")
async def session_get(mode: str, session_id: str):
    if not (record := sessions.get(mode, session_id)):
        raise HTTPException(404)
    return record


def _config_file(name: str) -> Path:
    if name not in EDITABLE:
        raise HTTPException(404)
    return CONFIG_DIR / f"{name}.yaml"


@app.get("/api/config/{name}", response_class=PlainTextResponse)
async def config_get(name: str):
    return _config_file(name).read_text(encoding="utf-8")


@app.put("/api/config/{name}")
async def config_put(name: str, body: str = Body(..., media_type="text/plain")):
    path = _config_file(name)
    if settings.public_demo:
        raise HTTPException(403, "Settings are read-only on the public demo.")
    try:
        yaml.safe_load(body)
    except yaml.YAMLError as e:
        raise HTTPException(400, f"Invalid YAML: {e}") from e
    path.with_suffix(".yaml.bak").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text(body, encoding="utf-8")
    return {"saved": True}


@app.websocket("/ws/{mode}")
async def live(ws: WebSocket, mode: str):
    await ws.accept()
    if mode not in SESSIONS:
        return await ws.close()
    if live_slot.locked():
        await ws.send_json({"type": "error", "text": "Another session is running right now (another tab, or "
                                                     "another visitor). Try again in a few minutes."})
        return await ws.close()
    async with live_slot:
        await SESSIONS[mode](ws, models).run()


if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning")
