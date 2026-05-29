"""Data types for the brain cognitive memory system."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class WorkingMemoryState:
    current_scene: str = ""
    active_characters: list[str] = field(default_factory=list)
    recent_events: list[str] = field(default_factory=list)
    mood: str = ""


@dataclass
class ProceduralMemoryState:
    tone: str = ""
    style: str = ""
    pov: str = ""
    genre: str = ""
    narrative_rules: list[str] = field(default_factory=list)
    preferences: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryDocument:
    id: str
    text: str
    embedding: list[float]
    created_at: datetime
    data: dict[str, Any] = field(default_factory=dict)
    type: str = ""
    summary: str = ""


@dataclass
class BrainConfig:
    user_id: str
    context_id: str
    project_id: str
    semantic_top_k: int = 5
    episodic_top_k: int = 3
    embedding_model: str = "all-MiniLM-L6-v2"


@dataclass
class AssembledPrompt:
    text: str
    working_injected: bool = False
    procedural_injected: bool = False
    semantic_count: int = 0
    episodic_count: int = 0


@dataclass
class ReflectionInput:
    user_message: str
    assistant_response: str
    assembled_prompt: AssembledPrompt
