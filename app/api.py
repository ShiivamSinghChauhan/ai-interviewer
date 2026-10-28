"""The REST API: set up an interview for someone, send them the link, collect the result.

    POST   /api/v1/sessions        {"agent": "interview", "setup": {...}, "webhook_url": "https://..."}
                                   -> {"id": "...", "join_url": "...", "status": "created", ...}
    GET    /api/v1/sessions/{id}   the status, then the result: every answer with its transcript, timing and
                                   scores, the delivery metrics and the agent's report
    GET    /api/v1/sessions        the newest sessions
    DELETE /api/v1/sessions/{id}

The person opens join_url in a browser and talks to the agent. When the session ends, the result is stored and
POSTed to webhook_url, signed with WEBHOOK_SECRET (header X-Signature: sha256=<hex HMAC-SHA256 of the body>).
Every call needs "Authorization: Bearer <token>" once API_TOKENS is set. Interactive docs live at /docs.
"""
import asyncio
import hashlib
import hmac
import json
import logging
import secrets
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, HttpUrl

from app.config import settings

log = logging.getLogger("api")

JOIN_TTL_S = 24 * 3600            # a link nobody opened within a day expires
KEEP = 500                        # sessions held in memory
PRIVATE_SETUP = ("resume", "jd")  # never sent to the person's browser
WEBHOOK_TRIES = 3


class Repo(BaseModel):
    project: str = ""
    source: str = Field(description="A public git URL (a local folder also works when self-hosted)")


class SessionCreate(BaseModel):
    agent: Literal["interview"] = "interview"
    setup: dict = Field(default_factory=dict, description="The agent's setup fields, the same as on its setup "
                        "screen. GET /api/{agent}/options lists the choices and the defaults.")
    repos: list[Repo] = Field(default_factory=list, description="Interview only: repositories to read first")
    overrides: dict = Field(default_factory=dict, description="config/<agent>.yaml keys for this session only")
    webhook_url: HttpUrl | None = Field(None, description="Receives the finished session as a signed POST")
    show_report: bool = Field(False, description="Show the agent's report to the person when they finish")
    metadata: dict = Field(default_factory=dict, description="Anything of yours (a candidate id...), returned as is")

    model_config = {"json_schema_extra": {"examples": [{
        "agent": "interview",
        "setup": {"candidate_name": "Priya", "role": "Backend Engineer", "level": "junior", "duration_min": 15,
                  "jd": "We build payment APIs with Python, FastAPI and Postgres..."},
        "webhook_url": "https://example.com/hooks/interviews",
        "metadata": {"candidate_id": "c-1042"},
    }]}}


@dataclass
class AgentSession:
    id: str
    agent: str
    request: dict  # setup, repos, overrides: exactly what the session's setup() reads
    webhook_url: str | None = None
    show_report: bool = False
    metadata: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    status: str = "created"  # created -> live -> completed | abandoned (the person left early)
    started_at: float | None = None
    ended_at: float | None = None
    result: dict | None = None
    webhook: dict | None = None  # how the last delivery went

    def current_status(self) -> str:
        if self.status == "created" and time.time() - self.created_at > JOIN_TTL_S:
            return "expired"
        return self.status

    def view(self, base_url: str) -> dict:
        return {**asdict(self), "status": self.current_status(), "join_url": f"{base_url}/?session={self.id}"}


class SessionBook:
    """Agent sessions by id: in memory, plus one JSON file each when a folder is given."""

    def __init__(self, folder: Path | None):
        self.folder = folder
        self.items: dict[str, AgentSession] = {}
        self._tasks: set[asyncio.Task] = set()
        if folder and folder.is_dir():
            files = sorted(folder.glob("*.json"), key=lambda f: f.stat().st_mtime)[-KEEP:]
            for file in files:
                try:
                    session = AgentSession(**json.loads(file.read_text(encoding="utf-8")))
                except (OSError, ValueError, TypeError):
                    continue
                self.items[session.id] = session

    def create(self, body: SessionCreate) -> AgentSession:
        request = {"setup": body.setup, "repos": [r.model_dump() for r in body.repos], "overrides": body.overrides}
        session = AgentSession(id=secrets.token_urlsafe(12), agent=body.agent, request=request,
                               webhook_url=str(body.webhook_url) if body.webhook_url else None,
                               show_report=body.show_report, metadata=body.metadata)
        self.items[session.id] = session
        for old in list(self.items)[:-KEEP]:
            del self.items[old]
        self._save(session)
        return session

    def get(self, session_id: str) -> AgentSession | None:
        return self.items.get(session_id)

    def newest(self, limit: int) -> list[AgentSession]:
        return sorted(self.items.values(), key=lambda s: s.created_at, reverse=True)[:limit]

    def delete(self, session_id: str) -> bool:
        if not self.items.pop(session_id, None):
            return False
        if self.folder:
            (self.folder / f"{session_id}.json").unlink(missing_ok=True)
        return True

    def started(self, session: AgentSession) -> None:
        session.status, session.started_at = "live", time.time()
        self._save(session)

    def finished(self, session: AgentSession, record: dict, completed: bool) -> None:
        """The live session is over: keep its record as the result and tell the webhook."""
        session.status = "completed" if completed else "abandoned"
        session.ended_at = time.time()
        session.result = {k: v for k, v in record.items() if k not in ("mode", "history", "summary")}
        self._save(session)
        if session.webhook_url:
            task = asyncio.create_task(deliver(session))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            task.add_done_callback(lambda _: self._save(session))

    def _save(self, session: AgentSession) -> None:
        if self.folder:
            self.folder.mkdir(parents=True, exist_ok=True)
            (self.folder / f"{session.id}.json").write_text(json.dumps(asdict(session), default=str),
                                                            encoding="utf-8")


def sign(body: bytes) -> str:
    return "sha256=" + hmac.new(settings.webhook_secret.encode(), body, hashlib.sha256).hexdigest()


async def deliver(session: AgentSession, pause_s: float = 2.0) -> None:
    """POST the finished session to its webhook, retrying with backoff."""
    body = json.dumps({**asdict(session), "status": session.status}, default=str).encode()
    headers = {"Content-Type": "application/json", "X-Session-Id": session.id}
    if settings.webhook_secret:
        headers["X-Signature"] = sign(body)
    error = ""
    async with httpx.AsyncClient(timeout=10) as client:
        for attempt in range(WEBHOOK_TRIES):
            try:
                response = await client.post(session.webhook_url, content=body, headers=headers)
                if response.status_code < 300:
                    session.webhook = {"delivered": True, "status": response.status_code, "attempts": attempt + 1}
                    return
                error = f"HTTP {response.status_code}"
            except httpx.HTTPError as e:
                error = type(e).__name__
            await asyncio.sleep(pause_s * 2 ** attempt)
    log.warning("webhook for %s failed: %s", session.id, error)
    session.webhook = {"delivered": False, "error": error, "attempts": WEBHOOK_TRIES}


book = SessionBook(None if settings.public_demo else settings.api_sessions_dir)
router = APIRouter(prefix="/api/v1", tags=["sessions"])
bearer = HTTPBearer(auto_error=False, description="One of the tokens in API_TOKENS")


def authorized(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]) -> None:
    if not settings.api_tokens:
        return
    token = credentials.credentials if credentials else ""
    if not any(secrets.compare_digest(token, known) for known in settings.api_tokens):
        raise HTTPException(401, "Missing or wrong API token", headers={"WWW-Authenticate": "Bearer"})


def base_url(request: Request) -> str:
    """Where people reach this server. Behind a proxy (Hugging Face, nginx) the forwarded headers say."""
    if settings.public_url:
        return settings.public_url.rstrip("/")
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or request.url.netloc
    return f"{proto}://{host}"


def _found(session_id: str) -> AgentSession:
    if not (session := book.get(session_id)):
        raise HTTPException(404, "No such session")
    return session


@router.post("/sessions", status_code=201, dependencies=[Depends(authorized)])
async def create_session(body: SessionCreate, request: Request) -> dict:
    """Set up a session and get the link to send to the person who will talk to the agent."""
    if body.webhook_url and settings.public_demo:
        raise HTTPException(400, "Webhooks are off on the public demo. Poll GET /api/v1/sessions/{id} instead.")
    return book.create(body).view(base_url(request))


@router.get("/sessions", dependencies=[Depends(authorized)])
async def list_sessions(request: Request, limit: int = 50) -> list[dict]:
    """The newest sessions first."""
    if settings.public_demo:
        raise HTTPException(403, "Listing is off on the public demo: fetch your own sessions by id.")
    return [session.view(base_url(request)) for session in book.newest(max(1, min(limit, KEEP)))]


@router.get("/sessions/{session_id}", dependencies=[Depends(authorized)])
async def get_session(session_id: str, request: Request) -> dict:
    """The status, and once it is "completed" or "abandoned", the result."""
    return _found(session_id).view(base_url(request))


@router.delete("/sessions/{session_id}", status_code=204, dependencies=[Depends(authorized)])
async def delete_session(session_id: str) -> Response:
    if not book.delete(session_id):
        raise HTTPException(404, "No such session")
    return Response(status_code=204)


@router.get("/sessions/{session_id}/join", include_in_schema=False)
async def join_info(session_id: str) -> dict:
    """For the person's browser: which agent to open. The id is the only key, so nothing private goes out."""
    session = _found(session_id)
    if (status := session.current_status()) != "created":
        raise HTTPException(410, f"This link can't be used any more (the session is {status}).")
    setup = {k: v for k, v in session.request.get("setup", {}).items() if k not in PRIVATE_SETUP}
    return {"agent": session.agent, "setup": setup}
