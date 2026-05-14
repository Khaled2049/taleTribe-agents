# novelsync-agents

FastAPI service for NovelSync story-generation workflows. The repo exposes a unified HTTP API for story actions, optional local image generation routes, and the infrastructure used to deploy the service to Google Cloud Run.

## Overview

This repo is the AI execution layer behind NovelSync. It takes story context, validates action payloads, dispatches them to the appropriate generation or editing tool, and returns structured results to the frontend and related services. The app is built as a single FastAPI service so the request surface stays simple while the implementation remains modular.

## Features

- Unified `POST /agent/execute` API for all story actions
- Story generation and chapter generation workflows
- Brainstorming for ideas, characters, and plots
- Context-aware chat assistance for in-story writing help
- Text enhancement actions for rewriting and expansion flows
- Next-line generation for editor assistance
- **Brain — four-layer cognitive memory system** (working, procedural, semantic, episodic) with open-source embeddings
- **All LLM calls route through creditProxy** — provider (Gemini, OpenAI, Anthropic, Ollama, mock) is configured in creditProxy; individual requests can carry BYOK credentials that bypass platform quota and route through the user's own key
- Strict Pydantic validation with stable error responses
- Optional local image-generation routes behind an environment flag
- Cloud Run deployment with Terraform and GitHub Actions
- Local test suite plus PR validation for formatting, linting, tests, and Terraform checks

## Quick start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python server.py
```

The service listens on `http://localhost:8000` by default. Start with [docs/quickstart.md](./docs/quickstart.md) for the full local setup.

## Docs

- [Docs index](./docs/README.md)
- [Quickstart](./docs/quickstart.md)
- [API](./docs/api.md)
- [Architecture](./docs/architecture.md)
- [Brain — cognitive memory system](./docs/brain.md)
- [Deployment](./docs/deployment.md)

## How it fits into NovelSync

- `novelsync-agents`: AI service and generation orchestration
- `../novelsync-frontend`: editor, UI flows, Firebase functions, and client integrations
- `../contracts`: tipping infrastructure used by the platform’s web3 features

## Repo layout

- `server.py`: FastAPI app entrypoint
- `agents/storyAgent/`: action schemas, orchestration, tools, and context building
- `image-generation/`: optional local image-generation module loaded when enabled
- `tests/`: pytest suite
- `terraform/`: Cloud Run and IAM infrastructure
- `.github/workflows/`: PR validation and deployment automation

## Core commands

- `python server.py`
- `pytest`
- `docker build --platform=linux/amd64 -t <image> .`

## Tech stack

- Python with FastAPI and Pydantic
- Firestore-backed story context access
- Optional local ML/image-generation dependencies for offline workflows
- Docker plus Cloud Run for production hosting
- Terraform and GitHub Actions for infrastructure and delivery
