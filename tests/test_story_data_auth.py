"""How the story-data client proves it is the agents service."""

import base64
import json
import time
from types import SimpleNamespace

import httpx
import pytest

from mcp_server import story_data

URL = "https://story-data.example.run.app"


def _jwt(exp: float) -> str:
    enc = lambda b: base64.urlsafe_b64encode(b).rstrip(b"=").decode()  # noqa: E731
    return f"{enc(b'{}')}.{enc(json.dumps({'exp': exp}).encode())}.{enc(b'sig')}"


def _recording_client(**kwargs):
    seen: list[httpx.Headers] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers)
        return httpx.Response(200, json=[])

    client = story_data.StoryDataClient(URL, **kwargs)
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client, seen


async def test_production_sends_a_cached_google_id_token():
    fetched: list[str] = []
    token = _jwt(time.time() + 3600)

    async def fetch(audience: str) -> str:
        fetched.append(audience)
        return token

    client, seen = _recording_client(id_token_audience=URL, fetch_id_token=fetch)
    await client.list_stories("user-1")
    await client.list_stories("user-2")

    assert fetched == [URL], "one token, for story-data's own URL, reused"
    for headers in seen:
        assert headers["X-Service-Authorization"] == f"Bearer {token}"
        assert "X-Service-Token" not in headers
    assert [h["X-User-ID"] for h in seen] == ["user-1", "user-2"]


async def test_a_token_near_expiry_is_refetched():
    tokens = iter([_jwt(time.time() + 60), _jwt(time.time() + 3600)])

    async def fetch(_audience: str) -> str:
        return next(tokens)

    client, seen = _recording_client(id_token_audience=URL, fetch_id_token=fetch)
    await client.list_stories("u")
    await client.list_stories("u")
    first, second = (h["X-Service-Authorization"] for h in seen)
    assert first != second


async def test_no_identity_token_fails_closed():
    async def fetch(_audience: str) -> str:
        return ""

    client, seen = _recording_client(id_token_audience=URL, fetch_id_token=fetch)
    with pytest.raises(story_data.StoryDataError):
        await client.list_stories("u")
    assert seen == [], "nothing is sent without a credential"


async def test_dev_sends_the_shared_secret():
    client, seen = _recording_client(service_token="dev-service-token")
    await client.list_stories("u")
    assert seen[0]["X-Service-Token"] == "dev-service-token"
    assert "X-Service-Authorization" not in seen[0]


def test_from_settings_picks_the_credential_by_environment():
    prod = story_data.from_settings(
        SimpleNamespace(
            environment="production",
            story_data_url=f"{URL}/",
            story_data_service_token="",
        )
    )
    assert prod._audience == URL and prod._service_token == ""
    dev = story_data.from_settings(
        SimpleNamespace(
            environment="development",
            story_data_url="http://localhost:8084",
            story_data_service_token=" dev-service-token ",
        )
    )
    assert dev._audience == "" and dev._service_token == "dev-service-token"


def test_both_credentials_at_once_is_a_mistake():
    with pytest.raises(ValueError):
        story_data.StoryDataClient(URL, "tok", id_token_audience=URL)
