"""Tests for per-user rate limiting."""

import os
import time
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("CREDIT_PROXY_URL", "http://localhost:8080")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")

from rate_limit import PerUserRateLimiter
from server import create_app


@pytest.mark.asyncio
async def test_allow_respects_limit():
    limiter = PerUserRateLimiter(2)
    assert await limiter.allow("user-a") is True
    assert await limiter.allow("user-a") is True
    assert await limiter.allow("user-a") is False


@pytest.mark.asyncio
async def test_allow_isolated_per_user():
    limiter = PerUserRateLimiter(1)
    assert await limiter.allow("user-a") is True
    assert await limiter.allow("user-b") is True
    assert await limiter.allow("user-a") is False


@pytest.mark.asyncio
async def test_allow_disabled_when_max_zero():
    limiter = PerUserRateLimiter(0)
    for _ in range(5):
        assert await limiter.allow("user-a") is True


# ---------------------------------------------------------------------------
# Bucket table capacity
#
# The OAuth endpoints key these buckets by client IP from the open internet, so
# the key space is unbounded and attacker-chosen. These tests pin the ceiling.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bucket_table_is_capped():
    """A flood of distinct keys must not grow the table without limit."""
    limiter = PerUserRateLimiter(60, max_tracked_keys=500)
    for i in range(5_000):
        await limiter.allow(f"198.51.100.{i}")
    assert limiter.tracked_keys == 500
    assert limiter.evictions == 4_500


@pytest.mark.asyncio
async def test_eviction_is_least_recently_used():
    limiter = PerUserRateLimiter(60, max_tracked_keys=3)
    for key in ("a", "b", "c"):
        await limiter.allow(key)

    await limiter.allow("a")  # touch "a" so "b" becomes the LRU
    await limiter.allow("d")  # forces one eviction

    assert limiter.tracked_keys == 3
    assert limiter.evictions == 1
    # "b" is gone: its budget is fresh again. "a" and "c" kept their history.
    limiter_state = {k: v.tokens for k, v in limiter._buckets.items()}
    assert set(limiter_state) == {"a", "c", "d"}


@pytest.mark.asyncio
async def test_eviction_fails_open_not_closed():
    """An evicted caller gets a fresh budget; it is never locked out.

    Fail-closed here would let anyone who can fill the table deny service to
    every legitimate caller — worse than the evasion it would prevent, since
    rotating IPs defeats a per-IP limiter at any table size.
    """
    limiter = PerUserRateLimiter(1, max_tracked_keys=2)
    assert await limiter.allow("victim") is True
    assert await limiter.allow("victim") is False  # budget spent

    await limiter.allow("flood-1")
    await limiter.allow("flood-2")  # evicts "victim"

    assert await limiter.allow("victim") is True  # readmitted, not refused


@pytest.mark.asyncio
async def test_limiting_still_works_at_capacity():
    """Eviction must not stop the limiter from limiting a repeat caller."""
    limiter = PerUserRateLimiter(2, max_tracked_keys=10)
    for i in range(50):
        await limiter.allow(f"filler-{i}")

    assert await limiter.allow("hammer") is True
    assert await limiter.allow("hammer") is True
    assert await limiter.allow("hammer") is False


@pytest.mark.asyncio
async def test_per_call_cost_does_not_grow_with_table_size():
    """Regression guard: eviction used to be an O(n) scan on every call.

    The old implementation rescanned the whole dict once past 5 000 entries and
    evicted only buckets idle 120s — so during a flood it freed nothing and
    charged every request for the scan (measured ~300x at 20k keys). Compares a
    full table against a near-empty one with a deliberately loose bound, so this
    catches a return to linear behaviour without being timing-flaky.
    """
    small = PerUserRateLimiter(60, max_tracked_keys=20_000)
    await small.allow("only-key")

    large = PerUserRateLimiter(60, max_tracked_keys=20_000)
    for i in range(20_000):
        await large.allow(f"key-{i}")
    assert large.tracked_keys == 20_000

    async def timed(limiter: PerUserRateLimiter, prefix: str) -> float:
        start = time.perf_counter()
        for i in range(300):
            await limiter.allow(f"{prefix}-{i}")
        return time.perf_counter() - start

    baseline = await timed(small, "probe")
    loaded = await timed(large, "probe")

    # O(1) puts these within noise of each other; O(n) at 20k keys was ~300x.
    assert loaded < max(baseline * 20, 0.05), (
        f"per-call cost scales with table size: "
        f"{loaded:.4f}s at capacity vs {baseline:.4f}s empty"
    )


def test_agent_execute_returns_429_when_rate_limited(monkeypatch):
    monkeypatch.setenv("MAX_REQUESTS_PER_MINUTE_PER_USER", "2")
    test_app = create_app()
    test_app.state.agent.execute_agent = AsyncMock(return_value={"ok": True})

    with TestClient(test_app) as client:
        payload = {
            "action": "brainstormPlot",
            "parameters": {"storyId": "s1"},
            "user_id": "rate-test-user",
        }
        assert client.post("/agent/execute", json=payload).status_code == 200
        assert client.post("/agent/execute", json=payload).status_code == 200
        response = client.post("/agent/execute", json=payload)

    assert response.status_code == 429
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "RATE_LIMITED"


def test_agent_execute_rejects_missing_user_id(monkeypatch):
    """user_id is required so anonymous traffic cannot share one rate-limit bucket."""
    monkeypatch.setenv("MAX_REQUESTS_PER_MINUTE_PER_USER", "1000")
    test_app = create_app()

    with TestClient(test_app) as client:
        response = client.post(
            "/agent/execute",
            json={"action": "brainstormPlot", "parameters": {"storyId": "s1"}},
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_agent_execute_rejects_empty_user_id(monkeypatch):
    monkeypatch.setenv("MAX_REQUESTS_PER_MINUTE_PER_USER", "1000")
    test_app = create_app()

    with TestClient(test_app) as client:
        response = client.post(
            "/agent/execute",
            json={
                "action": "brainstormPlot",
                "parameters": {"storyId": "s1"},
                "user_id": "",
            },
        )

    assert response.status_code == 422
