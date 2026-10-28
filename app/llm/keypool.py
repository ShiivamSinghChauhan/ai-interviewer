"""Several free accounts per provider, used one after another.

The pool sticks to one key until it fails. Rate limits are tracked per model, because Groq meters every model
separately and a busy model shouldn't block the others. Dead keys (401/403) and empty wallets (402) bench the
whole key.
"""
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import httpx

log = logging.getLogger("keys")

KEY_ERRORS = {401, 402, 403, 429}  # the key is the problem, so the next key may work
DAILY_HINTS = ("per day", "daily", "rpd", "tpd", "quota", "exhausted", "insufficient")


class ProviderError(Exception):
    pass


@dataclass
class Key:
    value: str
    benched_until: float = 0.0
    model_benched: dict[str, float] = field(default_factory=dict)
    failures: int = 0

    def free_at(self, model: str | None) -> float:
        return max(self.benched_until, self.model_benched.get(model or "", 0.0))


class KeyPool:
    def __init__(self, provider: str, keys: list[str]):
        self.provider = provider
        self.keys = [Key(k) for k in keys]
        self.current = 0

    def __bool__(self) -> bool:
        return bool(self.keys)

    def candidates(self, model: str | None = None) -> list[Key]:
        """Usable keys, starting with the one that worked last."""
        now = time.monotonic()
        ordered = self.keys[self.current:] + self.keys[:self.current]
        return [key for key in ordered if key.free_at(model) <= now]

    def ok(self, key: Key) -> None:
        key.failures = 0
        self.current = self.keys.index(key)

    def fail(self, key: Key, status: int, body: str = "", headers=None, model: str | None = None) -> None:
        text = body.lower().replace("-", " ")  # OpenRouter says "free-models-per-day"
        per_model = model is not None
        if status in (401, 403):
            wait, per_model = 24 * 3600, False
        elif status == 402:
            wait, per_model = 3600, False
        elif any(hint in text for hint in DAILY_HINTS) and "per minute" not in text:
            wait = 3600
        elif status == 429:
            wait = _retry_after(headers) or min(60, 10 * (key.failures + 1))
        else:
            wait, per_model = 5, False
        key.failures += 1
        until = time.monotonic() + wait
        if per_model:
            key.model_benched[model] = until
        else:
            key.benched_until = until
            self.current = (self.keys.index(key) + 1) % len(self.keys)
        log.warning("%s key ...%s%s benched for %.0fs (HTTP %s)", self.provider, key.value[-4:],
                    f" [{model}]" if per_model else "", wait, status)


def _retry_after(headers) -> float | None:
    try:
        return float(headers["retry-after"])
    except (TypeError, KeyError, ValueError):
        return None


async def request(pool: KeyPool, send: Callable[[str], Awaitable[httpx.Response]],
                  model: str | None = None) -> httpx.Response:
    """Call `send(key)` with each usable key until one gets a 200."""
    errors = []
    for key in pool.candidates(model):
        try:
            response = await send(key.value)
        except (httpx.TransportError, httpx.TimeoutException) as e:
            raise ProviderError(f"{pool.provider}: {type(e).__name__}") from e
        if response.status_code == 200:
            pool.ok(key)
            return response
        errors.append(f"{response.status_code}: {response.text[:200]}")
        if response.status_code not in KEY_ERRORS:
            break
        pool.fail(key, response.status_code, response.text, response.headers, model)
    raise ProviderError(f"{pool.provider}: " + ("; ".join(errors) or "no usable key"))
