# Python Environment Setup Guide

This guide will help you set up a single Python environment for both bots and agents.

## Prerequisites

- **Python 3.9+** (Python 3.10 or 3.11 recommended)
- **pip** (usually comes with Python)
- **Firebase CLI** (for running emulators): `npm install -g firebase-tools`
- **Google Cloud SDK** (optional, only needed if not using emulator)

## Step 1: Create Virtual Environment

Navigate to the `python` directory and create a virtual environment:

### Windows
```bash
cd python
python -m venv venv
venv\Scripts\activate
```

### Linux/Mac
```bash
cd python
python3 -m venv venv
source venv/bin/activate
```

## Step 2: Install Dependencies

With your virtual environment activated, install all dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install all dependencies needed for both bots and agents:
- Firebase/Google Cloud libraries
- FastAPI and Uvicorn (for agent server)
- HTTP libraries (requests, httpx)
- Data validation (pydantic)
- Environment management (python-dotenv)

## Step 3: Configure Environment Variables

Create a `.env` file in the `python` directory:

```bash
# Required: Google Cloud Project ID
GOOGLE_CLOUD_PROJECT=your-project-id

# Required: Google AI Studio API Key (for AI generation)
GOOGLE_AI_STUDIO_API_KEY=your-api-key

# Optional: AI Model (defaults to gemini-2.0-flash-exp)
GOOGLE_AI_STUDIO_MODEL=gemini-2.0-flash-exp

# Optional: For local development with Firebase emulators
FIRESTORE_EMULATOR_HOST=localhost:8080
FIREBASE_AUTH_EMULATOR_HOST=localhost:9099

# Optional: Port for agent server (defaults to 8000)
PORT=8000

# Optional: Use Ollama for local AI (set to true to use local Ollama instead of Google AI)
USE_OLLAMA=false
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# Optional: Use mock mode (no AI calls, returns predefined responses)
USE_MOCK=false
```

### Getting Your Google AI Studio API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key and add it to your `.env` file

## Step 4: Verify Installation

Test that everything is installed correctly:

```bash
python -c "import fastapi, uvicorn, firestore, requests, httpx; print('All dependencies installed successfully!')"
```

## Step 5: Test the Setup

### Test Agent Server

```bash
python server.py
```

The server should start on `http://localhost:8000`. You can test it with:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "healthy", "project_id": "your-project-id"}
```

### Test Bot (if configured)

```bash
python -m bots.bots.bot
```

## Troubleshooting

### Virtual Environment Not Activating

**Windows:**
- Make sure you're using `venv\Scripts\activate` (not `venv/Scripts/activate`)
- If you get an execution policy error, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

**Linux/Mac:**
- Make sure you're using `source venv/bin/activate` (not `venv/bin/activate`)

### Import Errors

If you get import errors, make sure:
1. Your virtual environment is activated (you should see `(venv)` in your terminal prompt)
2. All dependencies are installed: `pip install -r requirements.txt`
3. You're running commands from the `python` directory

### Firestore Connection Issues

If you're using the emulator:
1. Make sure Firebase emulators are running: `cd ../functions && npm run emulator`
2. Set `FIRESTORE_EMULATOR_HOST=localhost:8080` in your `.env` file
3. Restart your Python application

### Port Already in Use

If port 8000 is already in use:
1. Change the `PORT` in your `.env` file
2. Or stop the process using port 8000

## Next Steps

- See [QUICKSTART.md](./QUICKSTART.md) for a quick start guide
- See [LOCAL_DEVELOPMENT.md](./LOCAL_DEVELOPMENT.md) for detailed local development instructions
- See [README.md](./readme.md) for project overview

## Deactivating Virtual Environment

When you're done working, you can deactivate the virtual environment:

```bash
deactivate
```

requirements.txt
# Unified Python Dependencies
# This file contains all dependencies for bots, agents, and image-generation

# Firebase and Google Cloud
google-cloud-firestore>=2.13.0

# HTTP and API
requests>=2.31.0
httpx>=0.25.0

# Web Framework (for agent server and image-generation API)
fastapi>=0.104.1
uvicorn[standard]>=0.24.0

# Data Validation
pydantic>=2.5.0
pydantic-settings>=2.1.0

# Image Generation (for image-generation service)
diffusers>=0.24.0
transformers>=4.35.0
torch>=2.1.0
accelerate>=0.25.0
pillow>=10.1.0

# PyTorch: Install CPU version by default, or CUDA version for GPU support
# For CUDA 12.x: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# For CUDA 11.x: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Environment Management
python-dotenv>=1.0.0

