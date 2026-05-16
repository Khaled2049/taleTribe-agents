"""LLM provider abstraction — all AI calls route through creditProxy."""
import os
import json
from abc import ABC, abstractmethod
from contextvars import ContextVar
from typing import Optional, Dict, Any, List
import logging
import anyio
import httpx

logger = logging.getLogger(__name__)

# Per-request BYOK config. Set in server.py before each agent call.
# ContextVar is async-safe: each asyncio Task sees its own copy.
_byok_config: ContextVar[Optional[Dict[str, str]]] = ContextVar("byok_config", default=None)
_firebase_token: ContextVar[Optional[str]] = ContextVar("firebase_token", default=None)


class LLMProviderError(Exception):
    """Base class for all typed LLM provider errors."""


class InsufficientCreditsError(LLMProviderError):
    """Platform credit balance exhausted."""


class ProviderAuthError(LLMProviderError):
    """AI provider rejected the API key."""


class ProviderNotFoundError(LLMProviderError):
    """AI provider or model not found."""

    def __init__(self, provider: str = "", model: str = "") -> None:
        self.provider = provider
        self.model = model
        label = f"{provider}/{model}" if provider and model else model or provider
        super().__init__(label or "unknown model")


class BackendUnavailableError(LLMProviderError):
    """creditProxy is unreachable or returned an unexpected error."""


class RateLimitedError(LLMProviderError):
    """AI provider rate limit reached."""


class LLMTimeoutError(LLMProviderError):
    """AI request timed out."""


def _classify_http_error(
    status: int, body: str, config: Optional[Dict[str, str]]
) -> LLMProviderError:
    """Map a creditProxy HTTP error to a typed exception."""
    body_lower = body.lower()
    if "insufficient credits" in body_lower:
        return InsufficientCreditsError(body)
    if status == 401 or "unauthorized" in body_lower:
        return ProviderAuthError(body)
    if status == 429 or "rate limit" in body_lower or "too many requests" in body_lower:
        return RateLimitedError(body)
    if status == 404:
        provider = (config.get("provider", "") if config else "") or ""
        model = (config.get("model", "") if config else "") or ""
        if model or provider:
            return ProviderNotFoundError(provider, model)
        return BackendUnavailableError(f"creditProxy error 404: {body}")
    return LLMProviderError(f"creditProxy error {status}: {body}")


def _gcp_id_token(audience: str) -> str:
    """Fetch a GCP OIDC identity token from the metadata server.

    Returns '' when not running on GCP (local dev / tests) so callers can
    skip adding the Authorization header without any special-casing.
    """
    try:
        resp = httpx.get(
            "http://metadata.google.internal/computeMetadata/v1/instance/"
            f"service-accounts/default/identity?audience={audience}",
            headers={"Metadata-Flavor": "Google"},
            timeout=2.0,
        )
        resp.raise_for_status()
        return resp.text
    except Exception:
        return ""  # metadata server unreachable — not on GCP


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate_content(self, prompt: str) -> str:
        pass

    async def generate_content_async(self, prompt: str) -> str:
        """Generate content in a worker thread for async call sites."""
        return await anyio.to_thread.run_sync(self.generate_content, prompt)

    @abstractmethod
    async def generate_structured_content(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Dict[str, Any],
    ) -> List[str]:
        pass


class CreditProxyProvider(LLMProvider):
    """Routes LLM calls through the creditProxy gateway.

    Platform requests use the gateway's default provider.
    BYOK requests read per-request config from the _byok_config ContextVar
    (set by server.py before each agent call — async-safe, no tool changes needed).
    """

    def __init__(self, base_url: str, platform_user_id: str = "platform"):
        self.base_url = base_url.rstrip("/")
        self.platform_user_id = platform_user_id

    def _build_payload(self, prompt: str) -> Dict[str, Any]:
        config = _byok_config.get()
        user_id = config.get("user_id", self.platform_user_id) if config else self.platform_user_id
        provider = config.get("provider", "") if config else ""
        api_key = config.get("api_key", "") if config else ""
        model = config.get("model", "") if config else ""
        if api_key:
            logger.info("[LLM] BYOK  provider=%s model=%s user=%s", provider, model, user_id)
        else:
            logger.info("[LLM] platform  user=%s proxy=%s", user_id, self.base_url)
        return {
            "user_id": user_id,
            "prompt": prompt,
            "byok_provider": provider,
            "byok_api_key": api_key,
            "byok_model": model,
            "max_output_tokens": 8192,
        }

    def generate_content(self, prompt: str) -> str:
        payload = self._build_payload(prompt)
        headers = {}
        if token := _gcp_id_token(self.base_url):
            headers["Authorization"] = f"Bearer {token}"
        if firebase_token := _firebase_token.get():
            headers["X-Firebase-Token"] = firebase_token
        try:
            resp = httpx.post(
                f"{self.base_url}/v1/generate",
                json=payload,
                headers=headers,
                timeout=300.0,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["response"]["output"]
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(str(e)) from e
        except httpx.RequestError as e:
            raise BackendUnavailableError(f"creditProxy unreachable: {e}") from e
        except httpx.HTTPStatusError as e:
            raise _classify_http_error(
                e.response.status_code, e.response.text, _byok_config.get()
            ) from e

    async def generate_structured_content(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Dict[str, Any],
    ) -> List[str]:
        schema_hint = json.dumps(response_schema, indent=2)
        combined = (
            f"{system_prompt}\n\n"
            f"{user_prompt}\n\n"
            f"IMPORTANT: Respond with ONLY a valid JSON array matching this schema:\n{schema_hint}\n"
            "Do not include any text before or after the JSON array."
        )
        try:
            text = await self.generate_content_async(combined)
            text = text.strip()
            if text.startswith("```"):
                text = text.replace("```json", "").replace("```", "")
            json_data = json.loads(text)
            if isinstance(json_data, list):
                return [str(item) for item in json_data]
            if isinstance(json_data, dict):
                for value in json_data.values():
                    if isinstance(value, list):
                        return [str(item) for item in value]
            return []
        except (json.JSONDecodeError, KeyError):
            logger.warning("CreditProxyProvider: failed to parse structured response")
            return []


def get_llm_provider(project_id: Optional[str] = None, location: str = "us-central1") -> LLMProvider:
    """Return a CreditProxyProvider. CREDIT_PROXY_URL must be set.

    Provider/model selection is configured entirely in creditProxy via LLM_PROVIDER:
      mock     — canned responses, no API key needed
      ollama   — local LLM via OLLAMA_BASE_URL
      gemini   — GEMINI_API_KEY (platform default)
      BYOK     — per-request key forwarded via byok_* fields in the payload
    """
    credit_proxy_url = os.getenv("CREDIT_PROXY_URL")
    if not credit_proxy_url:
        raise ValueError(
            "CREDIT_PROXY_URL must be set. "
            "Run creditProxy via docker compose and set CREDIT_PROXY_URL=http://localhost:8080."
        )
    logger.info("Using CreditProxy Provider (%s)", credit_proxy_url)
    return CreditProxyProvider(base_url=credit_proxy_url)
