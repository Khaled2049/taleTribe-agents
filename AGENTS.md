# Repository Guidelines — novelsync-agents

## Multi-Repo Context

This repo is one of three in the NovelSync project:

- **novelsync-agents** (this repo): Python FastAPI service hosting AI story agents, deployed to Google Cloud Run.
- **novelsync-frontend** (`../novelsync-frontend`): React/TypeScript frontend + Firebase Cloud Functions.
- **contracts** (`../contracts`): Solidity smart contracts (TippingPlatform) built with Foundry.

## Project Structure

- `server.py`: FastAPI application entry point. Exposes `POST /agent/execute` and `GET /health`.
- `agents/storyAgent/`: StoryAgent implementation — `agent.py`, `tools.py`, `action_schemas.py`, `context_builder.py`, `llm_provider.py`.
- `image-generation/`: Optional local image generation module (disabled in production via `ENABLE_LOCAL_IMAGE_GENERATION`).
- `terraform/`: Terraform configuration for GCP (Cloud Run, IAM, Artifact Registry).
- `.github/workflows/`: CI (`ci.yml`) and deploy (`deploy.yml`) GitHub Actions workflows.
- `scripts/`: Utility scripts.
- `tests/`: pytest test suite.
- `requirements.txt`: Full Python dependencies including ML libs (local dev).
- `requirements-prod.txt`: Slim dependencies for Cloud Run (no ML libs; keeps image ~500 MB).
- `Dockerfile`: Multi-stage build using `requirements-prod.txt`.
- `DEPLOYMENT.md`: Full GCP + Terraform + GitHub Actions deployment guide.

## Build, Test, and Development Commands

- `python server.py`: run the FastAPI server locally (default port 8000; set `PORT` env var to override).
- `pip install -r requirements.txt`: install all deps including ML libs for local dev.
- `pip install -r requirements-prod.txt`: install Cloud Run-only deps.
- `pytest`: run the test suite (config in `pytest.ini`).
- `docker build --platform=linux/amd64 -t <image> .`: build the production Docker image (always use `linux/amd64` to avoid arch mismatch on Apple Silicon).

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_CLOUD_PROJECT` | Yes | GCP project ID |
| `VERTEX_AI_LOCATION` | No (default `us-central1`) | Vertex AI region |
| `FIRESTORE_EMULATOR_HOST` | No | Set automatically to `localhost:8080` in non-production if unset |
| `ENABLE_LOCAL_IMAGE_GENERATION` | No (default `true`) | Set to `false` to disable image generation module |
| `PORT` | No (default `8000`) | HTTP server port |

Copy `.env.example` to `.env` for local setup; never commit `.env`.

## Coding Style & Naming Conventions

- Python; follow existing module structure and naming.
- Use Pydantic models (`BaseModel`) for all request/response schemas.
- New agent actions must be added to `action_schemas.py` and wired up in `agent.py`.
- Keep `requirements.txt` and `requirements-prod.txt` in sync (prod omits ML libs only).

## Testing Guidelines

- Run `pytest` from the repo root.
- All new agent actions should have corresponding tests in `tests/`.
- Minimum validation before opening a PR: `pytest` passing and a manual `curl` of `/health` and `/agent/execute`.

## Deployment

Deployments are fully automated via GitHub Actions on push to `main`:
1. CI runs lint and tests (`ci.yml`).
2. Deploy workflow builds the Docker image, pushes to Artifact Registry, and applies Terraform to update Cloud Run (`deploy.yml`).

See `DEPLOYMENT.md` for setup instructions, secret management, rollback procedures, and troubleshooting.

## Commit & Pull Request Guidelines

- Short, imperative, lowercase commit messages (e.g. `add brainstorm action`, `fix context builder timeout`).
- PRs should include: scope, rationale, linked task (if any), and verification steps (`pytest` output + endpoint test).

## Security & Configuration Tips

- API keys (Google AI Studio) are stored in GCP Secret Manager and injected at runtime — never in code or `.env` committed to the repo.
- GitHub Actions uses Workload Identity Federation (keyless OIDC auth); no service account JSON keys are stored anywhere.
- Never commit `.env` or `terraform.tfvars` with real secrets (both are in `.gitignore`).
- See `DEPLOYMENT.md` → Security section for full details.
