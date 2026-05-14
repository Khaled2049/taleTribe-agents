# Quickstart

Use this to run the FastAPI service locally.

## Prerequisites

- Python 3.10 or 3.11
- `pip`
- Firebase emulators if you want local Firestore access

## Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

## Install dependencies

Local development:

```bash
pip install -r requirements.txt
```

Cloud Run-compatible dependency set:

```bash
pip install -r requirements-prod.txt
```

`requirements.txt` includes local image-generation dependencies (torch, diffusers, transformers, sentencepiece) and the brain memory system dependencies (`sentence-transformers`, `numpy`). `requirements-prod.txt` excludes the large image-generation ML packages but keeps `sentence-transformers` and `numpy` since the brain system runs in production.

## Start creditProxy

All LLM calls route through creditProxy. Start it before running the agent server:

```bash
cd ../creditProxy
cp .env.example .env
# Set LLM_PROVIDER and the relevant API key in .env (or leave LLM_PROVIDER=mock for testing)
docker compose up --build
```

Gateway will be available at `http://localhost:8080`.

## Configure environment

Create `.env` in the repo root:

```dotenv
GOOGLE_CLOUD_PROJECT=your-project-id
CREDIT_PROXY_URL=http://localhost:8080
VERTEX_AI_LOCATION=us-central1
ENABLE_LOCAL_IMAGE_GENERATION=false
PORT=8000
```

Notes:

- `GOOGLE_CLOUD_PROJECT` and `CREDIT_PROXY_URL` are required.
- `FIRESTORE_EMULATOR_HOST` defaults to `localhost:8080` in non-production — if you run the Firebase emulator, set it to a different port (e.g. `localhost:8085`) to avoid collision with creditProxy.
- Set `ENABLE_LOCAL_IMAGE_GENERATION=false` if you do not want the image-generation router loaded.

## Run the service

```bash
python server.py
```

Health check:

```bash
curl http://localhost:8000/health
```

## Run tests

```bash
pytest tests/ -v
```

The PR workflow also runs `black --check`, `isort --check-only`, `ruff check`, and Terraform validation.
