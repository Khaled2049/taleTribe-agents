"""Single source of truth for metadata-entity (character/place/plot) scalar fields.

Both the embedding text (``embedding_text.compose_entity_text``) and the full generation
prompt (``context_format.format_context_for_prompt``) iterate this schema, so the
field lists can't drift between them. Each entry is
``(field_name, prompt_label, prompt_char_cap)``; the cap is used only by the prompt
renderer — the embedding text is not capped.

Array-of-object fields (``character.relationships``, ``plot.events``) format
differently in each renderer and are handled by bespoke code there; only their names
live here (``ENTITY_ARRAY_FIELDS``) so the full embedded-field set is documented in
one place.

The field NAMES must stay in lockstep with the frontend's ``SIGNATURE_FIELDS`` in
``taleTribe-frontend/functions/src/entityFields.ts`` — that list decides when a
re-embed fires, so a mismatch means edits to a field either never re-embed (the AI
answers from a stale vector) or re-embed needlessly (burning indexing budget).

**Nothing enforces this.** A cross-repo test in the frontend used to parse this
module and diff the two lists, but it needed a credential to clone this private
repo in CI and was removed as not worth the upkeep. Treat it as a manual
convention: change one side, change the other in the same breath. Both failure
modes are silent.

The order here also defines field order in prompts/embeddings.

The MCP server consumes this module too: ``mcp_server/data.py`` picks a one-line
entity descriptor from these fields, guarded by
``test_descriptor_fields_exist_in_entity_schema``.
"""

from typing import Dict, List, Tuple

# (field, prompt label, prompt char cap)
ENTITY_FIELD_SCHEMA: Dict[str, List[Tuple[str, str, int]]] = {
    "character": [
        ("age", "Age", 40),
        ("soul", "Soul", 800),
        ("personality", "Personality", 800),
        ("voice", "Voice", 600),
        ("backstory", "Backstory", 1500),
        ("affiliations", "Affiliations", 600),
        ("notes", "Notes", 600),
    ],
    "place": [
        ("description", "Description", 900),
        ("atmosphere", "Atmosphere", 500),
        ("geography", "Geography", 600),
        ("history", "History", 600),
        ("significance", "Significance", 600),
        ("notes", "Notes", 600),
    ],
    "plot": [
        ("description", "Description", 1200),
    ],
}

# Array-of-object fields embedded per kind (rendered by bespoke code in each site).
ENTITY_ARRAY_FIELDS: Dict[str, List[str]] = {
    "character": ["relationships"],
    "place": [],
    "plot": ["events"],
}


def embedded_field_names(kind: str) -> List[str]:
    """Full ordered set of fields that contribute to a kind's embedding/prompt:
    ``name`` + scalar fields + array fields. Mirrors the frontend SIGNATURE_FIELDS."""
    scalars = [field for field, _label, _cap in ENTITY_FIELD_SCHEMA.get(kind, [])]
    return ["name"] + scalars + ENTITY_ARRAY_FIELDS.get(kind, [])
