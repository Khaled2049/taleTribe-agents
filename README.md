# novelsync-agents

The AI engine behind NovelSync — a FastAPI service that turns story context into generated content.

Every AI feature in NovelSync flows through this service. It reads the writer's characters, places, and plot from Firestore, builds a rich prompt, and returns structured results for the editor to use. All LLM calls are credit-metered through creditProxy, with optional per-user BYOK to bypass platform limits.

## Key features

- **Story and chapter generation** — full drafts grounded in the writer's existing world
- **Brainstorming** — ideas for characters, plots, and story directions on demand
- **In-editor assistance** — next-line suggestions and prose enhancement at cursor position
- **Context-aware chat** — answers questions about the story using the full document as context
- **Brain memory system** — four-layer cognitive memory (working, procedural, semantic, episodic) for richer continuity across sessions
- **BYOK support** — per-request API key forwarding to Gemini, Claude, or OpenAI without touching platform credits
- **Provider-agnostic** — LLM provider is configured in creditProxy; the agents never hard-code a model

## Quick start

```bash
pip install -r requirements.txt
python server.py
```
