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
# Install Poetry (if not already installed)
pip install poetry

# Install dependencies (no image generation)
poetry install --with dev

# Install with local image generation (heavy — ~5 GB)
poetry install --with dev,image-gen

# Run the server
poetry run python server.py
```

## Operational notes

- **Rate limiting** is per-process. The `MAX_REQUESTS_PER_MINUTE_PER_USER` env var caps requests per user *per instance*. On horizontally-scaled deployments (Cloud Run with N instances), the effective ceiling is `N * MAX_REQUESTS_PER_MINUTE_PER_USER`. For a true global cap, back the limiter with Redis/Memorystore.
- **Production env vars**: `AGENT_SERVICE_URL` (OIDC audience) and `FIREBASE_FUNCTIONS_SERVICE_ACCOUNT` (or `ALLOWED_SERVICE_ACCOUNTS`) must be set when `ENVIRONMENT=production`. The app fails fast at startup otherwise.
