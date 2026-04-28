"""Unit tests for MemoryRouter."""
from agents.storyAgent.brain.engine.router import MemoryRouter


def test_default_route_fetches_all():
    router = MemoryRouter()
    d = router.route("Write the next chapter", "")
    assert d.fetch_working is True
    assert d.fetch_procedural is True
    assert d.fetch_semantic is True
    assert d.fetch_episodic is True
    assert d.semantic_query == "Write the next chapter"


def test_brainstorm_skips_semantic_and_episodic():
    router = MemoryRouter()
    for hint in ["brainstorm", "brainstormCharacter", "brainstormPlot", "brainstormIdeas"]:
        d = router.route("Give me ideas", hint)
        assert d.fetch_semantic is False, f"Expected semantic=False for hint={hint}"
        assert d.fetch_episodic is False, f"Expected episodic=False for hint={hint}"
        assert d.fetch_working is True
        assert d.fetch_procedural is True


def test_episodic_query_uses_tail():
    router = MemoryRouter()
    long_msg = "A" * 500
    d = router.route(long_msg, "")
    assert len(d.episodic_query) == 200
    assert d.episodic_query == long_msg[-200:]


def test_empty_message():
    router = MemoryRouter()
    d = router.route("", "")
    assert d.semantic_query == ""
    assert d.episodic_query == ""
