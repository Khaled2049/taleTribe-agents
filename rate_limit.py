"""In-memory per-user token bucket rate limiter for /agent/execute."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass


@dataclass
class _Bucket:
    tokens: float
    last_update: float


class PerUserRateLimiter:
    """Token bucket keyed by user_id. Limit applies per rolling minute."""

    def __init__(self, max_per_minute: int) -> None:
        self._max_per_minute = max(0, max_per_minute)
        self._buckets: dict[str, _Bucket] = {}
        self._lock = asyncio.Lock()

    @property
    def max_per_minute(self) -> int:
        return self._max_per_minute

    async def allow(self, user_id: str) -> bool:
        """Return True if the request is allowed, False if rate limited."""
        if self._max_per_minute <= 0:
            return True

        key = user_id or "anonymous"
        now = time.monotonic()
        refill_rate = self._max_per_minute / 60.0

        async with self._lock:
            self._prune_stale(now)
            bucket = self._buckets.get(key)
            if bucket is None:
                self._buckets[key] = _Bucket(
                    tokens=float(self._max_per_minute - 1),
                    last_update=now,
                )
                return True

            elapsed = now - bucket.last_update
            bucket.tokens = min(
                float(self._max_per_minute),
                bucket.tokens + elapsed * refill_rate,
            )
            bucket.last_update = now

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True

            return False

    def reset(self) -> None:
        """Clear all buckets (for tests)."""
        self._buckets.clear()

    def _prune_stale(self, now: float) -> None:
        if len(self._buckets) < 5000:
            return
        stale = [k for k, b in self._buckets.items() if now - b.last_update > 120.0]
        for key in stale:
            del self._buckets[key]
