import asyncio
import json
from types import SimpleNamespace

from app.core.session import BaseSession


def test_audio_is_still_read_while_begin_waits_for_the_prep():
    """An unread socket stops answering keepalive pings and gets closed, so "begin" must not block the reading."""
    async def scenario():
        prep_done, began, heard = asyncio.Event(), [], []
        script = [{"type": "websocket.receive", "text": json.dumps({"type": "begin"})},
                  {"type": "websocket.receive", "bytes": b"\x00\x00"}]

        async def receive():
            if script:
                return script.pop(0)
            prep_done.set()  # every frame was read while "begin" was still waiting
            await asyncio.sleep(0.05)
            return {"type": "websocket.disconnect"}

        async def begin():  # the interview prep: minutes in real life
            await prep_done.wait()
            began.append(True)

        session = BaseSession.__new__(BaseSession)
        session.ws, session.events = SimpleNamespace(receive=receive), asyncio.Queue()
        session.worker = SimpleNamespace(start=lambda: None, stop=lambda: None, push=heard.append)
        session.response_task, session.started_at, session.ended = None, None, False
        session._begin, session._tick = begin, lambda: asyncio.sleep(3600)
        await asyncio.wait_for(session.run(), 2)
        return heard, began

    assert asyncio.run(scenario()) == ([b"\x00\x00"], [True])
