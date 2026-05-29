"""Tests for the StoryContextBuilder Firestore-read cache."""

import os
from unittest.mock import patch

import pytest

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")

from agents.storyAgent import context_builder
from agents.storyAgent.context_builder import StoryContextBuilder, clear_context_cache


class _FakeFirestoreClient:
    """Minimal stand-in so StoryContextBuilder.__init__ doesn't hit GCP."""

    def __init__(self, project="test-project"):
        self.project = project


def _make_builder(project="test-project"):
    with patch.object(
        context_builder.firestore, "Client", return_value=_FakeFirestoreClient(project)
    ):
        return StoryContextBuilder(project_id=project)


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_context_cache()
    yield
    clear_context_cache()


def _fixed_context():
    return {
        "story": {"id": "s1", "title": "Test"},
        "characters": [],
        "places": [],
        "plots": [],
        "chapters": [],
    }


class TestContextCache:
    def test_second_call_within_ttl_skips_firestore(self, monkeypatch):
        monkeypatch.setenv("STORY_CONTEXT_CACHE_TTL_SECONDS", "30")
        builder = _make_builder()

        with patch.object(
            builder, "_fetch_story_context", return_value=_fixed_context()
        ) as fetch:
            builder.build_story_context("s1")
            builder.build_story_context("s1")

        assert fetch.call_count == 1

    def test_different_stories_are_cached_separately(self, monkeypatch):
        monkeypatch.setenv("STORY_CONTEXT_CACHE_TTL_SECONDS", "30")
        builder = _make_builder()

        with patch.object(
            builder, "_fetch_story_context", return_value=_fixed_context()
        ) as fetch:
            builder.build_story_context("s1")
            builder.build_story_context("s2")

        assert fetch.call_count == 2

    def test_ttl_zero_disables_cache(self, monkeypatch):
        monkeypatch.setenv("STORY_CONTEXT_CACHE_TTL_SECONDS", "0")
        builder = _make_builder()

        with patch.object(
            builder, "_fetch_story_context", return_value=_fixed_context()
        ) as fetch:
            builder.build_story_context("s1")
            builder.build_story_context("s1")

        assert fetch.call_count == 2

    def test_expired_entry_refetches(self, monkeypatch):
        monkeypatch.setenv("STORY_CONTEXT_CACHE_TTL_SECONDS", "30")
        builder = _make_builder()

        times = iter([100.0, 100.0, 200.0, 200.0])
        monkeypatch.setattr(context_builder.time, "monotonic", lambda: next(times))

        with patch.object(
            builder, "_fetch_story_context", return_value=_fixed_context()
        ) as fetch:
            builder.build_story_context("s1")  # store @100
            builder.build_story_context("s1")  # check @200 -> expired, refetch

        assert fetch.call_count == 2

    def test_expired_entries_are_purged_from_map(self, monkeypatch):
        monkeypatch.setenv("STORY_CONTEXT_CACHE_TTL_SECONDS", "30")
        builder = _make_builder()

        # s1 stored @100; s2 miss @200 expires s1 and purges it from the map.
        times = iter([100.0, 100.0, 200.0, 200.0])
        monkeypatch.setattr(context_builder.time, "monotonic", lambda: next(times))

        with patch.object(
            builder, "_fetch_story_context", return_value=_fixed_context()
        ):
            builder.build_story_context("s1")
            builder.build_story_context("s2")

        keys = {k[1] for k in context_builder._CONTEXT_CACHE}
        assert keys == {"s2"}

    def test_returned_copy_is_isolated_from_cache(self, monkeypatch):
        monkeypatch.setenv("STORY_CONTEXT_CACHE_TTL_SECONDS", "30")
        builder = _make_builder()

        with patch.object(
            builder, "_fetch_story_context", return_value=_fixed_context()
        ):
            first = builder.build_story_context("s1")
            first["chapters"].append({"id": "mutated"})
            second = builder.build_story_context("s1")

        assert second["chapters"] == []
