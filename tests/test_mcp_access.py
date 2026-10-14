"""Unit tests for the MCP rollout allowlist (mcp_server.access)."""

import os
from unittest.mock import patch

import pytest

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")
os.environ.setdefault("CREDIT_PROXY_URL", "http://localhost:8080")
os.environ.setdefault("USE_MOCK", "true")

from mcp.server.auth.provider import AccessToken  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402
from mcp.server.fastmcp.exceptions import ToolError  # noqa: E402

from mcp_server import story_data  # noqa: E402
from mcp_server.access import AccessGate  # noqa: E402
from mcp_server.tools import register_tools  # noqa: E402
from rate_limit import PerUserRateLimiter  # noqa: E402
from tests.mcp_fakes import FakeFirestoreClient, FakeStoryData  # noqa: E402

UID = "user-a"


def _db(status: str | None = None) -> FakeFirestoreClient:
    db = FakeFirestoreClient()
    db.seed("stories/story-a", {"userId": UID, "title": "A", "updatedAt": None})
    if status is not None:
        db.seed(f"mcpAccess/{UID}", {"status": status})
    return db


def _gate(db, *, enabled: bool = True, ttl: int = 60) -> AccessGate:
    return AccessGate(db, enabled=enabled, cache_ttl_seconds=ttl)


# ---------------------------------------------------------------------------
# AccessGate itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["requested", "revoked", "denied", "", "GRANTED"])
def test_only_exactly_granted_allows(status):
    """No fuzzy matching: an access gate that accepts near-misses isn't one."""
    assert _gate(_db(status)).is_allowed(UID) is False


def test_missing_record_denies():
    assert _gate(_db(None)).is_allowed(UID) is False


def test_empty_uid_denies():
    assert _gate(_db("granted")).is_allowed("") is False


def test_lookup_failure_denies():
    """Fail CLOSED. A gate that opens when Firestore is unreachable is not a gate."""

    class Broken:
        def collection(self, _name):
            raise RuntimeError("firestore down")

    assert _gate(Broken()).is_allowed(UID) is False


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_zero_ttl_disables_caching_entirely():
    db = _db("granted")
    gate = _gate(db, ttl=0)
    assert gate.is_allowed(UID) is True
    db.seed(f"mcpAccess/{UID}", {"status": "revoked"})
    assert gate.is_allowed(UID) is False


def test_denial_cache_expires_faster_than_grant_cache():
    """The "user tries → refused → owner grants" sequence must resolve within
    DENIAL_CACHE_TTL_SECONDS, not the full (grant) TTL."""
    from mcp_server import access as access_module

    db = _db(None)
    gate = _gate(db, ttl=60)
    base = 1_000.0
    with patch.object(access_module.time, "monotonic", return_value=base):
        assert gate.is_allowed(UID) is False
        db.seed(f"mcpAccess/{UID}", {"status": "granted"})
        # Still denied: the negative entry is live.
        assert gate.is_allowed(UID) is False
    just_past_denial_ttl = base + access_module.DENIAL_CACHE_TTL_SECONDS + 1
    with patch.object(
        access_module.time, "monotonic", return_value=just_past_denial_ttl
    ):
        # The denial has lapsed long before the 60s grant TTL would have.
        assert gate.is_allowed(UID) is True


# ---------------------------------------------------------------------------
# Enforcement in the tool layer
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _story_data_backend():
    """These tests exercise the access gate through a read tool, so the read
    backend has to answer; the gate is what they assert on, not the payload."""
    fake = FakeStoryData()
    fake.seed_story("story-a", UID, title="Story A")
    story_data.configure(fake)
    yield
    story_data.configure(None)


def _server(db, gate) -> FastMCP:
    mcp = FastMCP("test")
    register_tools(
        mcp,
        db=db,
        rate_limiter=PerUserRateLimiter(1000),
        write_rate_limiter=PerUserRateLimiter(1000),
        enable_writes=True,
        access_gate=gate,
    )
    return mcp


def _token(scopes=("stories:read", "stories:write")) -> AccessToken:
    return AccessToken(
        token="mcp_at_test", client_id="c1", scopes=list(scopes), subject=UID
    )


async def test_tools_refuse_a_user_not_on_the_allowlist():
    db = _db(None)
    mcp = _server(db, _gate(db))
    with patch("mcp_server.tools.get_access_token", return_value=_token()):
        for tool, args in (
            ("list_my_stories", {}),
            ("create_story", {"title": "x"}),
        ):
            with pytest.raises(ToolError, match="has not been enabled"):
                await mcp.call_tool(tool, args)


async def test_tools_allow_a_granted_user():
    db = _db("granted")
    mcp = _server(db, _gate(db))
    with patch("mcp_server.tools.get_access_token", return_value=_token()):
        result = await mcp.call_tool("create_story", {"title": "Allowed"})
    assert result is not None


async def test_revocation_disconnects_an_existing_token():
    """The whole reason the gate is re-checked per call rather than only at
    consent: the MCP token stays valid for 30 days, so a grant-time-only check
    would leave a revoked user connected until it expired."""
    db = _db("granted")
    gate = _gate(db)
    mcp = _server(db, gate)
    with patch("mcp_server.tools.get_access_token", return_value=_token()):
        await mcp.call_tool("list_my_stories", {})
        db.seed(f"mcpAccess/{UID}", {"status": "revoked"})
        gate.invalidate(UID)
        with pytest.raises(ToolError, match="has not been enabled"):
            await mcp.call_tool("list_my_stories", {})


async def test_no_gate_configured_means_no_restriction():
    """register_tools without a gate (the default) must keep working."""
    db = _db(None)
    mcp = _server(db, None)
    with patch("mcp_server.tools.get_access_token", return_value=_token()):
        await mcp.call_tool("list_my_stories", {})
