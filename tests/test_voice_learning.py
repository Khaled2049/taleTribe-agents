import pytest

from agents.storyAgent.llm_provider import _byok_config, _firebase_token, _usage_feature
from assistant.executors import ToolRuntime, update_story_voice_profile
from assistant.prose_voice import load_voice_context
from assistant.tools import ToolContext, UpdateStoryVoiceProfileArgs
from assistant.voice_learning import (
    VoiceLearningWorker,
    VoiceProposal,
    _passages,
    validate_proposal,
)
from mcp_server import story_data


def test_passive_proposal_needs_three_independent_passages_and_agent_target():
    notes = [
        {
            "id": "writer-1",
            "text": "Keep the dialogue spare.",
            "origin": "writer",
            "pinned": True,
        },
        {
            "id": "agent-1",
            "text": "Use concrete sensory details.",
            "origin": "agent",
            "pinned": False,
        },
    ]
    proposal = VoiceProposal(
        operation="add",
        text="Let emotional shifts follow actions rather than explanation.",
        reason="Repeated in both saved chapters",
        evidence_indices=[0, 1, 2],
    )
    assert validate_proposal(proposal, notes)
    assert not validate_proposal(
        proposal.model_copy(update={"evidence_indices": [0, 1]}), notes
    )
    assert not validate_proposal(
        proposal.model_copy(update={"text": "Keep the dialogue spare."}), notes
    )
    assert not validate_proposal(
        VoiceProposal(
            operation="retire",
            note_id="writer-1",
            reason="Repeated across both chapters",
            evidence_indices=[0, 1, 2],
        ),
        notes,
    )
    assert validate_proposal(
        VoiceProposal(
            operation="retire",
            note_id="agent-1",
            reason="Repeated across both chapters",
            evidence_indices=[0, 1, 2],
        ),
        notes,
    )


def test_passages_are_bounded_and_span_two_chapters():
    source = {
        "id": "a",
        "revision": 4,
        "content": "<p>" + "A quiet motion. " * 200 + "</p>",
    }
    other = {
        "id": "b",
        "revision": 2,
        "content": "<p>" + "B quick exchange. " * 100 + "</p>",
    }
    passages = _passages(source, other)
    assert len(passages) == 3
    assert {part.chapter_id for part in passages} == {"a", "b"}
    assert passages[0].end <= passages[1].start
    assert _passages(source, {**other, "content": "short"}) == []


class FakeClient:
    def __init__(self):
        self.saved = []

    async def get_voice_profile(self, uid, story_id):
        assert (uid, story_id) == ("writer", "story")
        return {
            "revision": 2,
            "pointOfView": "first",
            "references": [
                {"chapterId": "current", "sourceRevision": 3, "start": 0, "end": 40},
                {"chapterId": "stale", "sourceRevision": 1, "start": 0, "end": 40},
            ],
        }

    async def get_chapter(self, uid, story_id, chapter_id):
        return {
            "id": chapter_id,
            "revision": 4 if chapter_id == "stale" else 3,
            "content": "<p>Quiet voices carried through the room.</p>",
        }

    async def update_story_voice_note(self, uid, story_id, revision, operation):
        self.saved.append((uid, story_id, revision, operation))
        return {"applied": True, "changeId": "change-1", "revision": 3}


@pytest.mark.asyncio
async def test_explicit_tool_requires_current_story_scope(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(story_data, "client", lambda: client)
    args = UpdateStoryVoiceProfileArgs(
        note="Keep narration restrained.",
        reason="Writer requested this for the entire story.",
    )
    local = ToolRuntime(
        ctx=ToolContext(user_id="writer", story_id="story"),
        current_user_text="Make this scene restrained.",
        run_id="run-1",
    )
    result = await update_story_voice_profile(args, local)
    assert result.result["saved"] is False
    assert client.saved == []
    scoped = ToolRuntime(
        ctx=local.ctx,
        current_user_text="For this story, keep narration restrained.",
        run_id="run-1",
    )
    result = await update_story_voice_profile(args, scoped)
    assert result.result == {"saved": True, "changeId": "change-1", "revision": 3}
    assert client.saved[0][2] == 2
    assert client.saved[0][3]["source"] == "explicit"


@pytest.mark.asyncio
async def test_stale_reference_is_ignored(monkeypatch):
    context = await load_voice_context(FakeClient(), "writer", "story")
    assert context and "Quiet voices" in context["referenceExcerpts"]
    assert context["referenceExcerpts"].count("Quiet voices") == 1


@pytest.mark.asyncio
async def test_learning_pass_uses_saved_evidence_and_writer_billing_choice(monkeypatch):
    class LearningClient:
        def __init__(self):
            self.saved = []

        async def get_voice_profile(self, uid, story_id):
            assert (uid, story_id) == ("writer", "story")
            return {"revision": 2, "learnFromWriting": True, "notes": []}

        async def get_chapter(self, uid, story_id, chapter_id):
            return {
                "id": chapter_id,
                "revision": 4 if chapter_id == "a" else 2,
                "content": "<p>" + "A quiet motion. " * 200 + "</p>",
            }

        async def list_chapter_index(self, uid, story_id):
            return [{"id": "a"}, {"id": "b"}]

        async def update_story_voice_note(self, uid, story_id, revision, operation):
            self.saved.append((revision, operation))
            return {"applied": True}

    class Pool:
        async def fetchrow(self, *_args):
            return {"pass_count": 1}

    class Provider:
        async def generate_content_async(self, prompt, max_output_tokens):
            assert "A quiet motion" in prompt
            assert max_output_tokens == 400
            assert _usage_feature.get() == "voice_analysis"
            assert _byok_config.get()["model"] == "gemini-2.5-flash"
            assert _firebase_token.get() == "firebase-token"
            return '{"operation":"add","text":"Let emotional shifts follow actions rather than explanation.","reason":"Repeated in both saved chapters","evidence_indices":[0,1,2]}'

    client = LearningClient()
    monkeypatch.setattr(story_data, "client", lambda: client)
    worker = VoiceLearningWorker(Pool(), Provider())
    result = await worker._process(
        {
            "id": "job-1",
            "owner_id": "writer",
            "story_id": "story",
            "chapter_id": "a",
            "source_revision": 4,
        },
        {"provider": "gemini", "api_key": "writer-key", "model": "gemini-2.5-flash"},
        "firebase-token",
    )
    assert result == "updated"
    assert client.saved[0][0] == 2
    assert client.saved[0][1]["source"] == "passive"
    assert len(client.saved[0][1]["evidence"]) == 3
