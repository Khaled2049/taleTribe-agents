# Python Environment Setup Guide

This guide will help you set up a local Python environment for the novelsync-agents service.

## Prerequisites

- **Python 3.9+** (Python 3.10 or 3.11 recommended)
- **pip** (usually comes with Python)
- **Firebase CLI** (for running emulators): `npm install -g firebase-tools`
- **Google Cloud SDK** (optional, only needed if not using emulator)

## Step 1: Create Virtual Environment

From the repo root:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

## Step 2: Install Dependencies

With your virtual environment activated:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs all dependencies needed for the agent service and optional local image generation:

- Firebase/Google Cloud libraries
- FastAPI and Uvicorn (agent HTTP server)
- HTTP libraries (requests, httpx)
- Data validation (pydantic)
- Environment management (python-dotenv)
- ML/image generation libs (torch, diffusers — for local image gen only)

For Cloud Run / CI, use `requirements-prod.txt` instead, which omits the large ML libraries.

## Step 3: Configure Environment Variables

Create a `.env` file in the repo root:

```bash
# Required: Google Cloud Project ID
GOOGLE_CLOUD_PROJECT=your-project-id

# Required: Google AI Studio API Key (for AI generation)
GOOGLE_AI_STUDIO_API_KEY=your-api-key

# Optional: AI Model (defaults to gemini-2.5-flash)
GOOGLE_AI_STUDIO_MODEL=gemini-2.5-flash

# Optional: For local development with Firebase emulators
# (auto-set to localhost:8080 if not in production and unset)
FIRESTORE_EMULATOR_HOST=localhost:8080

# Optional: Port for agent server (defaults to 8000)
PORT=8000

# Optional: Disable local image generation (set false to skip large ML deps)
ENABLE_LOCAL_IMAGE_GENERATION=true
```

### Getting Your Google AI Studio API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **Create API Key**
4. Copy the key and add it to your `.env` file

## Step 4: Verify Installation

```bash
python -c "import fastapi, uvicorn, requests, httpx; print('All core dependencies installed successfully!')"
```

## Step 5: Start the Server

```bash
python server.py
```

The server starts on `http://localhost:8000`. Test it with:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status": "healthy", "project_id": "your-project-id", "services": {"agent": "available", "image_generation": "available"}}
```

## Troubleshooting

### Virtual Environment Not Activating

**Windows:** Use `venv\Scripts\activate`. If you get an execution policy error: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

**Linux/Mac:** Use `source venv/bin/activate` (not `./venv/bin/activate`).

### Import Errors

1. Confirm the virtual environment is active (`(venv)` prefix in prompt).
2. Re-run `pip install -r requirements.txt` from the repo root.

### Firestore Connection Issues

1. Start Firebase emulators from the frontend repo: `cd ../novelsync-frontend/functions && npm run emulator`
2. Ensure `FIRESTORE_EMULATOR_HOST=localhost:8080` is set in `.env`.
3. Restart `server.py`.

### Port Already in Use

Change `PORT` in `.env` or stop the conflicting process on port 8000.

## Deactivating the Virtual Environment

```bash
deactivate
```

## Next Steps

- [API Documentation](./api-documentation.md) — full endpoint reference
- [Deployment Guide](./deployment.md) — GCP + Terraform + GitHub Actions
