import asyncio
import hashlib
import hmac
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from app import api
from app.config import settings
from app.main import app

SETUP = {"candidate_name": "Priya", "role": "Backend Engineer", "jd": "Python, Postgres", "resume": "secret"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "book", api.SessionBook(tmp_path))
    monkeypatch.setattr(settings, "api_tokens", [])
    monkeypatch.setattr(settings, "public_demo", False)
    return TestClient(app)  # no `with`: the speech models are never loaded


def create(client, **extra):
    response = client.post("/api/v1/sessions", json={"agent": "interview", "setup": SETUP, **extra},
                           headers={"x-forwarded-proto": "https", "host": "demo.example"})
    assert response.status_code == 201, response.text
    return response.json()


def test_create_get_and_join(client):
    session = create(client, metadata={"candidate_id": "c-1"})
    assert session["status"] == "created"
    assert session["join_url"] == f"https://demo.example/?session={session['id']}"
    assert client.get(f"/api/v1/sessions/{session['id']}").json()["metadata"] == {"candidate_id": "c-1"}
    join = client.get(f"/api/v1/sessions/{session['id']}/join").json()
    assert join["agent"] == "interview"
    assert "jd" not in join["setup"] and "resume" not in join["setup"]  # the person's browser never sees them


def test_unknown_agent_and_session(client):
    assert client.post("/api/v1/sessions", json={"agent": "karaoke"}).status_code == 422
    assert client.get("/api/v1/sessions/nope").status_code == 404


def test_tokens(client, monkeypatch):
    monkeypatch.setattr(settings, "api_tokens", ["s3cret"])
    body = {"agent": "interview"}
    assert client.post("/api/v1/sessions", json=body).status_code == 401
    assert client.post("/api/v1/sessions", json=body, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.post("/api/v1/sessions", json=body, headers={"Authorization": "Bearer s3cret"}).status_code == 201


def test_public_demo_limits(client, monkeypatch):
    monkeypatch.setattr(settings, "public_demo", True)
    hook = {"agent": "interview", "webhook_url": "http://169.254.169.254/latest"}
    assert client.post("/api/v1/sessions", json=hook).status_code == 400  # no webhooks from a public server
    assert client.get("/api/v1/sessions").status_code == 403             # nor a list of other people's sessions


def test_finished_session_keeps_its_result_and_its_link_dies(client):
    session = api.book.get(create(client)["id"])
    record = {"mode": "interview", "history": [{"role": "user", "content": "[instructions]"}], "summary": "",
              "minutes": 9.5, "report_markdown": "Hire", "delivery": {"words_per_minute": 130}, "turns": []}
    api.book.finished(session, record, completed=True)
    view = client.get(f"/api/v1/sessions/{session.id}").json()
    assert view["status"] == "completed"
    assert view["result"] == {"minutes": 9.5, "report_markdown": "Hire", "delivery": {"words_per_minute": 130},
                              "turns": []}
    assert client.get(f"/api/v1/sessions/{session.id}/join").status_code == 410
    assert api.SessionBook(api.book.folder).get(session.id).status == "completed"  # it survives a restart


def test_unused_links_expire(client):
    session = api.book.get(create(client)["id"])
    session.created_at = time.time() - api.JOIN_TTL_S - 1
    assert client.get(f"/api/v1/sessions/{session.id}").json()["status"] == "expired"
    assert client.get(f"/api/v1/sessions/{session.id}/join").status_code == 410


def test_delete(client):
    session_id = create(client)["id"]
    assert client.delete(f"/api/v1/sessions/{session_id}").status_code == 204
    assert client.get(f"/api/v1/sessions/{session_id}").status_code == 404


def _webhook(monkeypatch, statuses: list[int]) -> list[httpx.Request]:
    seen = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(statuses[min(len(seen), len(statuses)) - 1])
    real = httpx.AsyncClient
    monkeypatch.setattr(api.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(answer), **kw))
    return seen


def test_webhook_is_signed(monkeypatch):
    monkeypatch.setattr(settings, "webhook_secret", "hush")
    seen = _webhook(monkeypatch, [200])
    session = api.AgentSession(id="abc", agent="interview", request={}, webhook_url="https://example.com/hook")
    asyncio.run(api.deliver(session, pause_s=0))
    body = seen[0].content
    expected = "sha256=" + hmac.new(b"hush", body, hashlib.sha256).hexdigest()
    assert seen[0].headers["X-Signature"] == expected
    assert json.loads(body)["id"] == "abc"
    assert session.webhook == {"delivered": True, "status": 200, "attempts": 1}


def test_webhook_retries_then_gives_up(monkeypatch):
    seen = _webhook(monkeypatch, [500, 502, 503])
    session = api.AgentSession(id="abc", agent="interview", request={}, webhook_url="https://example.com/hook")
    asyncio.run(api.deliver(session, pause_s=0))
    assert len(seen) == api.WEBHOOK_TRIES
    assert session.webhook == {"delivered": False, "error": "HTTP 503", "attempts": 3}
