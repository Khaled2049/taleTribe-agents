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


def client_ip(request: Request) -> str:
    """Best-effort client IP.

    Cloud Run terminates at one proxy hop, so the first X-Forwarded-For entry is
    the caller. The header is client-controlled and therefore spoofable — good
    enough to bound accidental hammering and casual abuse, not a security
    boundary.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class OAuthThrottleMiddleware:
    """Token buckets keyed by client IP in front of a mounted MCP app."""

    def __init__(
        self,
        app: Any,
        *,
        register_per_minute: int,
        oauth_per_minute: int,
    ) -> None:
        self._app = app
        self._register_limiter = PerUserRateLimiter(register_per_minute)
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

        ip = client_ip(Request(scope))
        if not await limiter.allow(ip):
            logger.warning(
                "mcp_oauth_throttled",
                path=scope.get("path"),
                method=scope.get("method"),
                client_ip=ip,
            )
            # OAuth clients parse the RFC 6749 error shape; the 429 and
            # Retry-After tell a well-behaved one when to come back.
            response = JSONResponse(
                {
                    "error": "temporarily_unavailable",
                    "error_description": (
                        "Too many requests to this endpoint. Retry in a minute."
                    ),
                },
                status_code=429,
                headers={"Retry-After": "60"},
            )
            await response(scope, receive, send)
            return

        await self._app(scope, receive, send)
