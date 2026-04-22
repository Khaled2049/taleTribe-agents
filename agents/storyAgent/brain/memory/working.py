"""Working memory layer — current scene state, always injected."""
import logging
from typing import Any
from google.cloud import firestore
import anyio

from ..types import WorkingMemoryState

logger = logging.getLogger(__name__)


class WorkingMemoryLayer:
    def __init__(self, db: firestore.Client, context_id: str):
        self._db = db
        self._context_id = context_id

    def _doc_ref(self):
        return self._db.collection("stories").document(self._context_id).collection("working_memory").document("state")

    async def read(self) -> WorkingMemoryState:
        def _get():
            doc = self._doc_ref().get()
            return doc.to_dict() if doc.exists else {}

        data = await anyio.to_thread.run_sync(_get)
        return WorkingMemoryState(
            current_scene=data.get("current_scene", ""),
            active_characters=data.get("active_characters", []),
            recent_events=data.get("recent_events", []),
            mood=data.get("mood", ""),
        )

    async def write(self, state: WorkingMemoryState) -> None:
        data = {
            "current_scene": state.current_scene,
            "active_characters": state.active_characters,
            "recent_events": state.recent_events,
            "mood": state.mood,
        }
        def _set():
            self._doc_ref().set(data)
        await anyio.to_thread.run_sync(_set)

    async def patch(self, fields: dict[str, Any]) -> None:
        if not fields:
            return
        def _update():
            self._doc_ref().set(fields, merge=True)
        await anyio.to_thread.run_sync(_update)
