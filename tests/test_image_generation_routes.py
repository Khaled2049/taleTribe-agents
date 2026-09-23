import asyncio
import os

import httpx
import pytest

os.environ.setdefault("CREDIT_PROXY_URL", "http://localhost:8080")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")

from config import Settings  # noqa: E402
from server import create_app  # noqa: E402


class _SlowImageService:
    def __init__(self, gate: asyncio.Event, error: Exception | None = None):
        self._gate = gate
        self._error = error

    async def generate_image(self, prompt: str):
        await self._gate.wait()
        if self._error is not None:
            raise self._error
        return object(), 0.01

    def encode_image(self, image) -> str:
        return "aW1hZ2U="


@pytest.fixture
def image_app(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("ENABLE_MCP", "false")
    monkeypatch.setenv("ENABLE_LOCAL_IMAGE_GENERATION", "true")
    monkeypatch.setenv("IMAGE_GENERATION_MAX_CONCURRENT", "1")

    def build(*, insecure_local_auth: bool, service=None):
        monkeypatch.setenv(
            "ALLOW_INSECURE_LOCAL_AUTH", "true" if insecure_local_auth else "false"
        )
        built = create_app()
        assert built.state.image_generation_available
        if service is not None:
            import app.api.routes as routes

            monkeypatch.setattr(routes, "get_image_service", lambda: service)
        return built

    return build


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


def test_image_generation_is_off_by_default(monkeypatch):
    monkeypatch.delenv("ENABLE_LOCAL_IMAGE_GENERATION", raising=False)
    assert Settings(_env_file=None).enable_local_image_generation is False


async def test_image_routes_require_internal_auth(image_app):
    app = image_app(insecure_local_auth=False)
    async with _client(app) as client:
        generate = await client.post("/generate-cover", json={"prompt": "a castle"})
        health = await client.get("/image-health")
    assert generate.status_code == 401
    assert health.status_code == 401


async def test_concurrent_generation_is_capped(image_app):
    gate = asyncio.Event()
    app = image_app(insecure_local_auth=True, service=_SlowImageService(gate))
    async with _client(app) as client:
        first = asyncio.create_task(
            client.post("/generate-cover", json={"prompt": "a castle"})
        )
        for _ in range(50):
            await asyncio.sleep(0)
        second = await client.post("/generate-cover", json={"prompt": "a tower"})
        gate.set()
        first_response = await first
        third = await client.post("/generate-cover", json={"prompt": "a moat"})
    assert second.status_code == 429
    assert first_response.status_code == 200
    assert third.status_code == 200


async def test_generation_errors_do_not_leak_details(image_app):
    gate = asyncio.Event()
    gate.set()
    secret = "/models/private/path CUDA out of memory"
    app = image_app(
        insecure_local_auth=True,
        service=_SlowImageService(gate, error=RuntimeError(secret)),
    )
    async with _client(app) as client:
        resp = await client.post("/generate-cover", json={"prompt": "a castle"})
    assert resp.status_code == 500
    assert secret not in resp.text
