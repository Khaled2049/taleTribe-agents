"""Procedural memory layer — style, tone, preferences, always injected."""

import logging
from typing import Any

import anyio
from google.cloud import firestore

from ..types import ProceduralMemoryState

logger = logging.getLogger(__name__)


class ProceduralMemoryLayer:
    def __init__(self, db: firestore.Client, user_id: str, context_id: str):
        self._db = db
        self._user_id = user_id
        self._context_id = context_id

    def _global_ref(self):
        return (
            self._db.collection("users")
            .document(self._user_id)
            .collection("procedural_memory")
            .document("global")
        )

    def _context_ref(self):
        return (
            self._db.collection("stories")
            .document(self._context_id)
            .collection("procedural_memory")
            .document("context")
        )

    async def read(self) -> ProceduralMemoryState:
        def _get():
            g = self._global_ref().get()
            c = self._context_ref().get()
            return (
                g.to_dict() if g.exists else {},
                c.to_dict() if c.exists else {},
            )

        global_data, context_data = await anyio.to_thread.run_sync(_get)
        # Merge: global is base, context overrides
        merged = {**global_data, **context_data}
        return ProceduralMemoryState(
            tone=merged.get("tone", ""),
            style=merged.get("style", ""),
            pov=merged.get("pov", ""),
            genre=merged.get("genre", ""),
            narrative_rules=merged.get("narrative_rules", []),
            preferences=merged.get("preferences", {}),
        )

    async def write_global(self, fields: dict[str, Any]) -> None:
        if not fields:
            return

        def _set():
            self._global_ref().set(fields, merge=True)

        await anyio.to_thread.run_sync(_set)

    async def write_context(self, fields: dict[str, Any]) -> None:
        if not fields:
            return

        def _set():
            self._context_ref().set(fields, merge=True)

        await anyio.to_thread.run_sync(_set)

    async def clear_context(self) -> None:
        def _delete():
            self._context_ref().delete()

        await anyio.to_thread.run_sync(_delete)
