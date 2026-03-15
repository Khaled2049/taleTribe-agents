# Architecture

`novelsync-agents` is a single FastAPI service that loads story-generation logic and, optionally, local image-generation routes.

## Request flow

1. `server.py` loads `.env`, sets local emulator defaults, and creates the FastAPI app.
2. `POST /agent/execute` validates the request against `agents/storyAgent/action_schemas.py`.
3. `StoryAgent` handles the action and delegates to the relevant tool implementation under `agents/storyAgent/tools/`.
4. Responses are wrapped in a stable envelope: `success`, `data`, and structured `error`.

## Main modules

- `server.py`: HTTP surface, error handling, CORS, app state
- `agents/storyAgent/agent.py`: action dispatch
- `agents/storyAgent/context_builder.py`: story context retrieval and formatting
- `agents/storyAgent/llm_provider.py`: model interaction
- `agents/storyAgent/tools/`: action-specific logic

## Optional image generation

`server.py` attempts to load routes from `image-generation/` when `ENABLE_LOCAL_IMAGE_GENERATION=true`.

- when imports succeed, the router is mounted and `/health` reports `image_generation: available`
- when disabled or missing dependencies, the service still starts and reports `image_generation: unavailable`

This keeps production lean while allowing local image generation with the full dependency set.

## Testing and validation

- Python tests live in `tests/`
- PR validation runs formatting checks, `ruff`, `pytest`, `terraform fmt -check`, and `terraform validate`
- Cloud Run deployment is handled by `.github/workflows/deploy.yml`
