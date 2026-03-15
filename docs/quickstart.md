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

`requirements.txt` includes local image-generation dependencies. `requirements-prod.txt` excludes the large ML packages used only for local image generation.

## Configure environment

Create `.env` in the repo root:

```dotenv
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_AI_STUDIO_API_KEY=your-api-key
VERTEX_AI_LOCATION=us-central1
FIRESTORE_EMULATOR_HOST=localhost:8080
ENABLE_LOCAL_IMAGE_GENERATION=true
PORT=8000
```

Notes:

- `GOOGLE_CLOUD_PROJECT` is required.
- `server.py` defaults `FIRESTORE_EMULATOR_HOST` to `localhost:8080` outside production if you do not set it.
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
