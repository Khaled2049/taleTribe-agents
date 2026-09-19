import pytest

from assistant.history import HistoryLimits, prior_turns
from mcp_server import story_data
from tests.mcp_fakes import FakeStoryData

pytestmark = pytest.mark.asyncio


@pytest.fixture
def threaded():
    fake = FakeStoryData()
    fake.seed_story("story-1", "uid-1", title="Saltmarsh")
    fake.seed_thread("thread-1", "story-1")
    story_data.configure(fake)
    yield fake
    story_data.configure(None)


async def fetch(limits=None, thread_id="thread-1", uid="uid-1"):
    return await prior_turns(
        uid=uid,
        story_id="story-1",
        thread_id=thread_id,
        limits=limits or HistoryLimits(),
    )


async def test_completed_exchanges_are_replayed_in_order(threaded):
    threaded.seed_message("thread-1", "user", "Who is Mina?")
    threaded.seed_message("thread-1", "assistant", "The lighthouse keeper.")

    turns = await fetch()

    assert turns == [
        {"role": "user", "parts": [{"type": "text", "text": "Who is Mina?"}]},
        {
            "role": "assistant",
            "parts": [{"type": "text", "text": "The lighthouse keeper."}],
        },
    ]


async def test_trailing_user_message_is_dropped(threaded):
    threaded.seed_message("thread-1", "user", "Who is Mina?")
    threaded.seed_message("thread-1", "assistant", "The lighthouse keeper.")
    threaded.seed_message("thread-1", "user", "Make this better")

    turns = await fetch()

    assert [turn["role"] for turn in turns] == ["user", "assistant"]
    assert "Make this better" not in repr(turns)


async def test_incomplete_and_failed_messages_are_skipped(threaded):
    threaded.seed_message("thread-1", "user", "Who is Mina?")
    threaded.seed_message("thread-1", "assistant", "half an ans", status="failed")
    threaded.seed_message("thread-1", "user", "Who is Tam?")
    threaded.seed_message("thread-1", "assistant", "The ferryman.")

    turns = await fetch()

    assert "half an ans" not in repr(turns)
    assert turns[-1]["parts"][0]["text"] == "The ferryman."


async def test_only_the_last_window_is_fetched(threaded):
    for index in range(10):
        threaded.seed_message("thread-1", "user", f"question {index}")
        threaded.seed_message("thread-1", "assistant", f"answer {index}")

    turns = await fetch(HistoryLimits(max_messages=4, max_chars=4000))

    assert len(turns) == 4
    assert turns[0]["parts"][0]["text"] == "question 8"
    assert turns[-1]["parts"][0]["text"] == "answer 9"


async def test_character_budget_keeps_the_newest_turns(threaded):
    threaded.seed_message("thread-1", "user", "x" * 300)
    threaded.seed_message("thread-1", "assistant", "y" * 300)
    threaded.seed_message("thread-1", "user", "recent question")
    threaded.seed_message("thread-1", "assistant", "recent answer")

    turns = await fetch(HistoryLimits(max_messages=10, max_chars=100))

    assert [turn["parts"][0]["text"] for turn in turns] == [
        "recent question",
        "recent answer",
    ]


async def test_no_thread_id_means_no_history(threaded):
    threaded.seed_message("thread-1", "user", "Who is Mina?")
    threaded.seed_message("thread-1", "assistant", "The lighthouse keeper.")

    assert await fetch(thread_id=None) == []


async def test_history_can_be_disabled(threaded):
    threaded.seed_message("thread-1", "user", "Who is Mina?")
    threaded.seed_message("thread-1", "assistant", "The lighthouse keeper.")

    assert await fetch(HistoryLimits(max_messages=0)) == []


async def test_another_users_thread_yields_nothing(threaded):
    threaded.seed_message("thread-1", "user", "Who is Mina?")
    threaded.seed_message("thread-1", "assistant", "The lighthouse keeper.")

    assert await fetch(uid="uid-2") == []


async def test_a_broken_story_data_does_not_fail_the_run(threaded):
    class Broken:
        async def get_assistant_thread(self, *_args, **_kwargs):
            raise story_data.StoryDataError("down")

    story_data.configure(Broken())

    assert await fetch() == []


async def test_an_unconfigured_client_does_not_fail_the_run(threaded):
    story_data.configure(None)

    assert await fetch() == []
