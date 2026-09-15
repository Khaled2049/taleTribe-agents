"""HTTP client for the story-data service, as used by the MCP tools.

Authenticates as a service and asserts the caller's uid: the MCP token record
holds the uid (verified from a Firebase ID token once, at consent) but not a
Firebase token that could be replayed, so there is nothing to forward. story-data
trusts `X-User-ID` only when `X-Service-Token` matches, and derives ownership
from the asserted uid exactly as it does for a browser caller.

Errors are normalised here rather than at each call site: 403 and 404 both become
NotFound, because story-data distinguishes "absent" from "not yours" and the MCP
tools deliberately do not — a caller must not be able to probe story ids for
existence. 409 becomes Conflict (a lost If-Match, or a position already taken)
and 4xx rejections become Rejected, which carries story-data's own message
because the limits it enforces are worth reporting to the caller verbatim.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 15.0


def _error_message(response: httpx.Response) -> str:
    """story-data answers a rejection with {"error": "..."}; fall back to the status."""
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict):
        message = body.get("error")
        if isinstance(message, str) and message.strip():
            return message.strip()
    return f"story-data rejected the request ({response.status_code})"


class StoryDataError(Exception):
    """story-data was unreachable, or answered with something unusable."""


class NotFound(StoryDataError):
    """Absent, or not visible to the asserted user. The two are not separated."""


class Conflict(StoryDataError):
    """The row moved: a stale If-Match, or a uniqueness clash on insert."""


class Rejected(StoryDataError):
    """story-data refused the input. `message` is safe to show the caller."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class StoryDataClient:
    """Thin async wrapper over the story-data endpoints the read tools need.

    Holds one long-lived AsyncClient. A client per call would pay a fresh TLS
    handshake every time, which is the same reason EmbeddingProvider keeps one
    (see brain/embedding_provider.py).
    """

    def __init__(
        self,
        base_url: str,
        service_token: str,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        if not base_url:
            raise ValueError("story-data base URL is required")
        self._base = base_url.rstrip("/")
        self._service_token = service_token
        self._client = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()

    def _headers(self, uid: str, revision: Optional[str] = None) -> dict[str, str]:
        headers = {"X-User-ID": uid}
        if self._service_token:
            headers["X-Service-Token"] = self._service_token
        if revision is not None:
            headers["If-Match"] = revision
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        uid: str,
        *,
        params: Optional[dict] = None,
        json: Any = None,
        revision: Optional[str] = None,
    ) -> Any:
        try:
            response = await self._client.request(
                method,
                f"{self._base}{path}",
                headers=self._headers(uid, revision),
                params=params or None,
                json=json,
            )
        except httpx.HTTPError as exc:
            raise StoryDataError(f"story-data request failed: {exc}") from exc

        if response.status_code in (403, 404):
            raise NotFound(path)
        if response.status_code in (409, 412, 428):
            raise Conflict(path)
        if response.status_code in (400, 422):
            raise Rejected(_error_message(response))
        if response.status_code >= 400:
            # The body can carry the caller's content; log the status only.
            logger.warning(
                "story_data_error", extra={"path": path, "status": response.status_code}
            )
            raise StoryDataError(f"story-data returned {response.status_code}")
        if response.status_code == 204 or not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise StoryDataError("story-data returned a non-JSON body") from exc

    async def _get(self, path: str, uid: str, **params: Any) -> Any:
        return await self._request("GET", path, uid, params=params)

    # ------------------------------------------------------------------
    # Reads. Every one is scoped to the asserted uid by story-data itself.
    # ------------------------------------------------------------------

    async def list_stories(self, uid: str) -> list[dict]:
        return await self._get("/v1/stories", uid)

    async def get_story(self, uid: str, story_id: str) -> dict:
        return await self._get(f"/v1/stories/{story_id}", uid)

    async def list_chapter_index(self, uid: str, story_id: str) -> list[dict]:
        """Chapter metadata in reading order, without the bodies.

        content=false matters: the default listing returns every chapter's full
        text, so building a table of contents would otherwise transfer the whole
        manuscript.
        """
        return await self._get(f"/v1/stories/{story_id}/chapters", uid, content="false")

    async def get_chapter(self, uid: str, story_id: str, chapter_id: str) -> dict:
        return await self._get(f"/v1/stories/{story_id}/chapters/{chapter_id}", uid)

    async def list_entities(self, uid: str, story_id: str, kind: str) -> list[dict]:
        return await self._get(f"/v1/stories/{story_id}/{kind}", uid)

    async def get_entity(
        self, uid: str, story_id: str, kind: str, entity_id: str
    ) -> dict:
        return await self._get(f"/v1/stories/{story_id}/{kind}/{entity_id}", uid)

    async def get_assistant_thread(
        self, uid: str, story_id: str, thread_id: str
    ) -> dict:
        return await self._get(
            f"/v1/stories/{story_id}/assistant-threads/{thread_id}", uid
        )

    async def list_assistant_messages(
        self,
        uid: str,
        story_id: str,
        thread_id: str,
        *,
        cursor: int = 0,
        limit: int = 20,
    ) -> dict:
        params: dict[str, Any] = {"limit": str(limit)}
        if cursor > 0:
            params["cursor"] = str(cursor)
        return await self._get(
            f"/v1/stories/{story_id}/assistant-threads/{thread_id}/messages",
            uid,
            **params,
        )

    async def get_my_profile(self, uid: str) -> dict:
        return await self._get("/v1/profiles/me", uid)

    # ------------------------------------------------------------------
    # Writes. Ownership is story-data's to enforce, as it is for the reads;
    # the tools re-check it first only so a denial is indistinguishable from
    # a missing story.
    # ------------------------------------------------------------------

    async def create_story(self, uid: str, payload: dict) -> dict:
        return await self._request("POST", "/v1/stories", uid, json=payload)

    async def create_chapter(self, uid: str, story_id: str, payload: dict) -> dict:
        return await self._request(
            "POST", f"/v1/stories/{story_id}/chapters", uid, json=payload
        )

    async def update_chapter(
        self, uid: str, story_id: str, chapter_id: str, payload: dict, revision: str
    ) -> dict:
        return await self._request(
            "PATCH",
            f"/v1/stories/{story_id}/chapters/{chapter_id}",
            uid,
            json=payload,
            revision=revision,
        )


_client: Optional[StoryDataClient] = None


def configure(client: Optional[StoryDataClient]) -> None:
    """Install the process-wide client (or None to clear it, as tests do)."""
    global _client
    _client = client


def client() -> StoryDataClient:
    if _client is None:
        raise StoryDataError(
            "story-data client is not configured; set STORY_DATA_URL for the MCP server"
        )
    return _client


async def aclose() -> None:
    """Release the installed client, if any. Safe to call when unconfigured."""
    global _client
    if _client is not None:
        await _client.close()
        _client = None
