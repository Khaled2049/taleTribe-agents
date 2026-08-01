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
- **Remote MCP server** — owner-scoped story tools for Claude and other MCP harnesses at `/mcp`, secured by an embedded OAuth 2.1 authorization server (`mcp_server/`)

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

## MCP server

The service also hosts a remote MCP server (streamable HTTP) at `/mcp`, plus the OAuth 2.1 endpoints MCP clients discover at the domain root (`/.well-known/*`, `/authorize`, `/token`, `/register`, `/revoke`). Tools are owner-scoped: list stories, read chapters (paginated), and inspect characters/places/plots — all needing only the `stories:read` scope. Two write tools (`create_story`, `create_chapter`) additionally require the `stories:write` scope *and* `ENABLE_MCP_WRITES=true`; they are off by default and are never registered when the flag is off. Login is delegated to the frontend consent page (`MCP_CONSENT_URL`); tokens are opaque, hashed, and stored in Firestore.

Local run against the emulators:

```bash
# Firestore + Auth emulators (from novelsync-frontend): firebase emulators:start
poetry run python server.py
# MCP endpoint: http://localhost:8000/mcp — test with:
npx @modelcontextprotocol/inspector
```

Set `ENABLE_MCP=false` to run the service without it. Full connection and tool reference: `../story/wiki/docs/14-mcp-server.md`.

## Operational notes

- **Rate limiting** is per-process. The `MAX_REQUESTS_PER_MINUTE_PER_USER` env var caps requests per user *per instance*. On horizontally-scaled deployments (Cloud Run with N instances), the effective ceiling is `N * MAX_REQUESTS_PER_MINUTE_PER_USER`. For a true global cap, back the limiter with Redis/Memorystore. The MCP tools have their own limiter (`MCP_MAX_REQUESTS_PER_MINUTE_PER_USER`, default 60) with the same caveat. Each limiter's bucket table is a fixed-capacity LRU (20 000 keys, ~4 MB) so that IP-keyed instances can't be grown without bound by a flood of distinct source addresses; eviction is fail-open, and a climbing `PerUserRateLimiter.evictions` means the key space is outrunning the table.
- **Unauthenticated MCP OAuth endpoints** are throttled per client IP, since Cloud Run invoker access is public and `/register` writes a Firestore document with no credential required: `MCP_REGISTER_REQUESTS_PER_MINUTE_PER_IP` (default 5) and `MCP_OAUTH_REQUESTS_PER_MINUTE_PER_IP` (default 30, covering `/authorize`, `/token`, `/revoke` and the consent-handoff routes). Discovery documents are never throttled — a client that can't read them can't start the flow. Client registrations expire after 7 days unused; the window slides out to 90 days once a client is actually used, so an active connection is never collected.
- **Production env vars**: `AGENT_SERVICE_URL` (OIDC audience) and `FIREBASE_FUNCTIONS_SERVICE_ACCOUNT` (or `ALLOWED_SERVICE_ACCOUNTS`) must be set when `ENVIRONMENT=production`. With `ENABLE_MCP=true` (the default), `MCP_CONSENT_URL` is also required. The app fails fast at startup otherwise.
