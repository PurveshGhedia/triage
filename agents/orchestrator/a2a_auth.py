"""Authenticating calls between Cloud Run services (the agents are private).

Cloud Run only accepts requests carrying a Google-signed ID token whose audience is the
target service's URL. `IdTokenAuth` attaches such a token to every request an httpx client
makes (agent-card fetch and the A2A call itself). On Cloud Run the token comes from the
metadata server (as the calling service's own service account); on a developer machine it
falls back to `gcloud auth print-identity-token`.
"""
from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import threading
import time
from collections.abc import Callable, Generator

import httpx

TOKEN_FALLBACK_LIFETIME_S = 50 * 60
REFRESH_MARGIN_S = 5 * 60


def _default_token_fn(audience: str) -> str:
    try:
        import google.auth.transport.requests
        import google.oauth2.id_token

        return google.oauth2.id_token.fetch_id_token(google.auth.transport.requests.Request(), audience)
    except Exception:
        out = subprocess.run(["gcloud", "auth", "print-identity-token"], capture_output=True, text=True, timeout=30)
        token = out.stdout.strip()
        if out.returncode != 0 or not token:
            raise RuntimeError(
                "could not obtain an ID token (no metadata server, and `gcloud auth print-identity-token` failed)"
            )
        return token


def _expiry(token: str, default_lifetime_s: float, now: float) -> float:
    """The token's own `exp` claim if it can be read, else now + a safe default."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return float(json.loads(base64.urlsafe_b64decode(payload))["exp"])
    except Exception:
        return now + default_lifetime_s


class IdTokenAuth(httpx.Auth):
    """httpx auth that adds `Authorization: Bearer <id token for audience>`, refreshing before expiry."""

    def __init__(self, audience: str, token_fn: Callable[[str], str] | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.audience = audience.rstrip("/")
        self._token_fn = token_fn or _default_token_fn
        self._clock = clock
        self._lock = threading.Lock()
        self._token: str | None = None
        self._expires_at = 0.0

    def token(self) -> str:
        with self._lock:
            now = self._clock()
            if self._token is None or now >= self._expires_at - REFRESH_MARGIN_S:
                self._token = self._token_fn(self.audience)
                self._expires_at = _expiry(self._token, TOKEN_FALLBACK_LIFETIME_S, now)
            return self._token

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        request.headers["Authorization"] = f"Bearer {self.token()}"
        yield request

    async def async_auth_flow(self, request: httpx.Request):
        token = await asyncio.to_thread(self.token)  # the metadata-server call blocks; keep the loop free
        request.headers["Authorization"] = f"Bearer {token}"
        yield request
