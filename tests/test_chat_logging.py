import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agents.storyAgent.llm_provider import _byok_config, _log_retry_metadata
from agents.storyAgent.tools.chat_with_context import ChatWithContextTool


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_chat_logs_metadata_without_content_or_keys(caplog, failure):
    provider = AsyncMock()
    provider.generate_content_async.return_value = "  Safe reply  "
    if failure:
        provider.generate_content_async.side_effect = RuntimeError(
            "secret-key private-prose"
        )
    tool = ChatWithContextTool("test", llm_provider=provider)
    token = _byok_config.set({"api_key": "secret-key"})
    try:
        with caplog.at_level(logging.INFO):
            try:
                result = await tool.execute(
                    "private-story-id",
                    "private-question",
                    chat_history=[{"role": "user", "content": "private-history"}],
                    chapter_excerpts="private-prose",
                    context_override={
                        "story": {"title": "private-title"},
                        "characters": [],
                        "places": [],
                        "plots": [],
                        "chapters": [],
                    },
                )
                assert result["response"] == "Safe reply"
            except RuntimeError:
                assert failure
    finally:
        _byok_config.reset(token)
    for sensitive in [
        "private-story-id",
        "private-question",
        "private-history",
        "private-prose",
        "private-title",
        "secret-key",
    ]:
        assert sensitive not in caplog.text
    assert "context_chars=" in caplog.text
    assert "duration_ms=" in caplog.text
    assert "correlation_id=" in caplog.text
    prompt = provider.generate_content_async.call_args.args[0]
    assert "private-prose" in prompt
    assert "private-history" in prompt


def test_retry_logging_excludes_upstream_error_body(caplog):
    with caplog.at_level(logging.WARNING):
        _log_retry_metadata(
            SimpleNamespace(
                attempt_number=1,
                outcome=SimpleNamespace(
                    exception=lambda: RuntimeError("secret-key private-prose")
                ),
            )
        )
    assert "credit_proxy_retry attempt=1 error_type=RuntimeError" in caplog.text
    assert "secret-key" not in caplog.text
    assert "private-prose" not in caplog.text


@pytest.mark.parametrize(
    "error_name", ["LLMProviderError", "BillingCommitError", "RuntimeError"]
)
def test_server_failure_logging_excludes_exception_body(
    monkeypatch, capsys, error_name
):
    from fastapi.testclient import TestClient

    from agents.storyAgent import llm_provider
    from server import app

    error_type = getattr(llm_provider, error_name, RuntimeError)
    monkeypatch.setattr(
        app.state.agent,
        "execute_agent",
        AsyncMock(side_effect=error_type("secret-key private-prose")),
    )
    response = TestClient(app).post(
        "/agent/execute",
        json={
            "action": "chatWithContext",
            "parameters": {"storyId": "s1", "message": "hello"},
            "user_id": "logging-test-user",
        },
    )
    assert response.status_code == 500
    captured = capsys.readouterr()
    assert "secret-key" not in captured.out + captured.err + response.text
    assert "private-prose" not in captured.out + captured.err + response.text
