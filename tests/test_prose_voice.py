"""The saved story voice is reduced to a small, bounded prompt context."""

from assistant.prose_voice import voice_context


def test_voice_context_keeps_set_fields_and_pinned_notes():
    voice = voice_context(
        {
            "pointOfView": "first",
            "tense": "past",
            "notes": [
                {
                    "id": "note-1",
                    "text": "Keep the unease beneath the dialogue.",
                    "origin": "writer",
                    "pinned": True,
                }
            ],
        }
    )
    assert voice == {
        "pointOfView": "first person",
        "tense": "past",
        "notes": "Pinned: Keep the unease beneath the dialogue.",
    }


def test_an_unset_profile_adds_nothing():
    assert (
        voice_context(
            {"pointOfView": "unspecified", "tense": "unspecified", "notes": []}
        )
        is None
    )
