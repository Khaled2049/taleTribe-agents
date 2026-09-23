"""Per-IP throttling for the unauthenticated MCP OAuth endpoints.

The SDK-generated routes (/register, /authorize, /token, /revoke) are mounted at
the domain root and reachable without any credential — Cloud Run invoker access
is public so MCP clients can discover and complete the OAuth flow themselves.
Each of those routes writes Firestore documents, and /register writes one per
call with no prior authentication, so an unauthenticated loop over it is an
unbounded write amplifier.

FastAPI dependencies can't reach inside a mounted Starlette app, so the guard is
ASGI middleware wrapped around the mount rather than a Depends(...).

Discovery documents (/.well-known/*) are deliberately NOT throttled: they are
static, cacheable, touch no database, and a client that cannot read them cannot
begin the flow at all.

Same process-local caveat as rate_limit.PerUserRateLimiter: with N Cloud Run
instances the effective ceiling is N * the configured value. That is acceptable
for an abuse guard whose job is to turn "unbounded" into "bounded".
"""

from __future__ import annotations

from typing import Any, Optional

import structlog
from starlette.requests import Request
from starlette.responses import JSONResponse

from rate_limit import PerUserRateLimiter

logger = structlog.get_logger(__name__)

# A client registers once per installation, so this can be tight.
REGISTER_PATH = "/register"
# A healthy client hits /token on every access-token renewal, so this is looser.
OAUTH_PATHS = frozenset({"/authorize", "/token", "/revoke"})
MAX_OAUTH_BODY_BYTES = 16 * 1024


class _BodyTooLarge(Exception):
    pass


def _oauth_error(status_code: int, error: str, description: str, headers=None):
    return JSONResponse(
        {"error": error, "error_description": description},
        status_code=status_code,
        headers=headers,
    )


def _too_large_response() -> JSONResponse:
    return _oauth_error(413, "invalid_request", "Request body too large.")


def _declared_length(scope) -> Optional[int]:
    for name, value in scope.get("headers", []):
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return -1
    return None


def client_ip(request: Request, trusted_hops: int = 1) -> str:
    peer = request.client.host if request.client else "unknown"
    if trusted_hops <= 0:
        return peer
    entries = [
        entry.strip()
        for entry in request.headers.get("x-forwarded-for", "").split(",")
        if entry.strip()
    ]
    if len(entries) < trusted_hops:
        return peer
    return entries[-trusted_hops]


class OAuthThrottleMiddleware:
    """Token buckets keyed by client IP in front of a mounted MCP app."""

    def __init__(
        self,
        app: Any,
        *,
        register_per_minute: int,
        oauth_per_minute: int,
        register_total_per_minute: int = 0,
        trusted_proxy_hops: int = 1,
        max_body_bytes: int = MAX_OAUTH_BODY_BYTES,
    ) -> None:
        self._app = app
        self._max_body_bytes = max_body_bytes
        self._trusted_proxy_hops = trusted_proxy_hops
        self._register_limiter = PerUserRateLimiter(register_per_minute)
        self._register_total_limiter = PerUserRateLimiter(register_total_per_minute)
        self._oauth_limiter = PerUserRateLimiter(oauth_per_minute)

    def _limiter_for(self, path: str) -> Optional[PerUserRateLimiter]:
        normalized = "/" + path.strip("/")
        if normalized == REGISTER_PATH:
            return self._register_limiter
        if normalized in OAUTH_PATHS:
            return self._oauth_limiter
        return None

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        limiter = self._limiter_for(scope.get("path", ""))
        if limiter is None:
            await self._app(scope, receive, send)
            return

        ip = client_ip(Request(scope), self._trusted_proxy_hops)
        if not await limiter.allow(ip):
            await self._reject(scope, receive, send, "mcp_oauth_throttled", ip)
            return
        if (
            limiter is self._register_limiter
            and not await self._register_total_limiter.allow("*")
        ):
            await self._reject(
                scope, receive, send, "mcp_oauth_register_budget_exhausted", ip
            )
            return

        await self._call_with_body_limit(scope, receive, send)

    async def _reject(self, scope, receive, send, event: str, ip: str) -> None:
        logger.warning(
            event,
            path=scope.get("path"),
            method=scope.get("method"),
            client_ip=ip,
        )
        response = _oauth_error(
            429,
            "temporarily_unavailable",
            "Too many requests to this endpoint. Retry in a minute.",
            headers={"Retry-After": "60"},
        )
        await response(scope, receive, send)

    async def _call_with_body_limit(self, scope, receive, send) -> None:
        limit = self._max_body_bytes
        declared = _declared_length(scope)
        if declared is not None and not 0 <= declared <= limit:
            logger.warning(
                "mcp_oauth_body_rejected",
                path=scope.get("path"),
                declared_length=declared,
            )
            await _too_large_response()(scope, receive, send)
            return

        received = 0
        response_started = False

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise _BodyTooLarge()
            return message

        async def tracking_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self._app(scope, limited_receive, tracking_send)
        except _BodyTooLarge:
            logger.warning(
                "mcp_oauth_body_rejected",
                path=scope.get("path"),
                received_bytes=received,
            )
            if response_started:
                raise
            await _too_large_response()(scope, receive, send)
