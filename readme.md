# Python Project

This directory contains bots, agents, and image-generation services for the Novel Sync application.

## Quick Start

1. **Setup**: See [SETUP.md](./SETUP.md) for environment setup
2. **Quick Start**: See [QUICKSTART.md](./QUICKSTART.md) for getting started quickly
3. **Local Development**: See [LOCAL_DEVELOPMENT.md](./LOCAL_DEVELOPMENT.md) for detailed development guide

## Structure

- `server.py` - Unified FastAPI server (agents and image generation)
- `agents/` - Story agent implementation code
- `bots/` - Automated bot implementations
- `image-generation/` - Image generation service code (integrated into unified server)
- `requirements.txt` - Unified dependencies for all Python code

## Installation

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running

### Unified Server (Agents + Image Generation)

The unified server (`server.py`) includes both story agent endpoints and image generation endpoints in a single FastAPI application.

**Quick Start:**

1. Start the unified server:
   ```bash
   python server.py
   ```

2. The API will be available at `http://localhost:8000`
   - Interactive API Docs: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health
   - Agent Endpoints: http://localhost:8000/agent/execute
   - Image Generation: http://localhost:8000/generate-cover

3. Generate an image:
   ```bash
   curl -X POST "http://localhost:8000/generate-cover" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "A beautiful sunset over mountains"}'
   ```

**Note:** 
- Image generation endpoints are automatically available if dependencies are installed
- The first image generation request will download the Stable Diffusion model (~1.4GB), which may take a few minutes
- If image generation dependencies are not installed, the server will start without them (with a warning)

### Bot
```bash
python -m bots.bots.bot
```

For more details, see [image-generation/README.md](./image-generation/README.md) or [QUICKSTART.md](./QUICKSTART.md).