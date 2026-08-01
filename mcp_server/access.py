from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

ACCESS_COLLECTION = "mcpAccess"

STATUS_REQUESTED = "requested"
STATUS_GRANTED = "granted"
STATUS_REVOKED = "revoked"
MAX_CACHED_UIDS = 10_000
DENIAL_CACHE_TTL_SECONDS = 10


class AccessGate:
    """TTL-cached allowlist lookup. Synchronous; callers bridge with anyio."""

    def __init__(self, db: Any, *, enabled: bool, cache_ttl_seconds: int) -> None:
        self._db = db
        self._enabled = enabled
        self._ttl = max(0, cache_ttl_seconds)
        self._cache: OrderedDict[str, tuple[bool, float]] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def is_allowed(self, uid: str) -> bool:
        """True when `uid` may use the MCP server.

        With the allowlist disabled this is always True, which is how the
        feature goes generally available: flip one flag, delete nothing.
        """
        if not self._enabled:
            return True
        if not uid:
            return False

        cached = self._cached(uid)
        if cached is not None:
            return cached

        try:
            snap = self._db.collection(ACCESS_COLLECTION).document(uid).get()
            record = snap.to_dict() if snap.exists else None
            allowed = bool(record) and record.get("status") == STATUS_GRANTED
        except Exception as exc:
            # Fail closed. Note this cannot mass-disconnect an active trial:
            # anyone already in the cache keeps their live entry (checked
            # above) until its TTL lapses.
            logger.warning(
                "mcp_access_lookup_failed",
                uid=uid,
                error_type=type(exc).__name__,
            )
            return False

        self._remember(uid, allowed)
        return allowed

    def invalidate(self, uid: str) -> None:
        """Drop a cached decision (tests, and any future admin hook)."""
        with self._lock:
            self._cache.pop(uid, None)

    def reset(self) -> None:
        with self._lock:
            self._cache.clear()

    def _cached(self, uid: str) -> bool | None:
        if self._ttl <= 0:
            return None
        now = time.monotonic()
        with self._lock:
            entry = self._cache.get(uid)
            if entry is None:
                return None
            allowed, expires_at = entry
            if expires_at <= now:
                del self._cache[uid]
                return None
            self._cache.move_to_end(uid)
            return allowed

    def _remember(self, uid: str, allowed: bool) -> None:
        if self._ttl <= 0:
            return
        ttl = self._ttl if allowed else min(self._ttl, DENIAL_CACHE_TTL_SECONDS)
        with self._lock:
            while len(self._cache) >= MAX_CACHED_UIDS:
                self._cache.popitem(last=False)
            self._cache[uid] = (allowed, time.monotonic() + ttl)
