# NovelSync AI Agents Service

Python FastAPI service providing AI-powered story generation capabilities for NovelSync.

## Features

- **Story Generation** - Full story generation using Google AI Studio (Gemini models)
- **Brainstorming Agents** - Specialized agents for:
  - General idea brainstorming
  - Character development
  - Plot creation and structuring
- **Context-Aware Generation** - Integrates with Firestore to understand story context
- **Chapter Generation** - Generate individual chapters
- **Next Line Suggestions** - Context-aware next line recommendations
- **Image Generation** - Cover image generation support (optional)

## Technology Stack

- **Framework:** FastAPI + Uvicorn
- **AI Provider:** Google AI Studio (Gemini 2.0 Flash)
- **Database:** Google Cloud Firestore
- **Deployment:** Google Cloud Run (containerized)
- **CI/CD:** GitHub Actions

## Local Development

### Prerequisites

- Python 3.11+
- Google AI Studio API Key ([Get one here](https://ai.google.dev/))
- Access to Firestore (emulator or production)
- Docker (for containerized development)

### Option 1: Run with Docker (Recommended)

This is the recommended approach for local development as it matches the production environment.

```bash
# Build image
docker build -t novelsync-agents .

# Run container (connecting to host emulator)
docker run -p 8000:8000 \
  -e GOOGLE_CLOUD_PROJECT=novelsync-f82ec \
  -e GOOGLE_AI_STUDIO_API_KEY=your-key \
  -e FIRESTORE_EMULATOR_HOST=host.docker.internal:8080 \
  --add-host host.docker.internal:host-gateway \
  novelsync-agents
```

**Using Docker Compose (from novelsync-frontend repo):**

```bash
cd ../novelsync-frontend
docker-compose -f docker-compose.simple.yml up python-agent
```

### Option 2: Run Natively with Python

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Edit .env and add your GOOGLE_AI_STUDIO_API_KEY

# Run with Firestore emulator (default for local)
FIRESTORE_EMULATOR_HOST=localhost:8080 python server.py

# Or run with production Firestore
ENVIRONMENT=production python server.py
```

### Verify Service is Running

```bash
# Health check
curl http://localhost:8000/health
# Should return: {"status":"healthy"}

# Execute a simple agent task (requires Firestore emulator)
curl -X POST http://localhost:8000/agent/execute \
  -H "Content-Type: application/json" \
  -d '{
    "action": "brainstorm_ideas",
    "params": {
      "genre": "science fiction",
      "themes": ["AI", "space exploration"]
    }
  }'
```

## Environment Variables

### Required

- **`GOOGLE_CLOUD_PROJECT`** - GCP project ID (e.g., `novelsync-f82ec`)
- **`GOOGLE_AI_STUDIO_API_KEY`** - Your Google AI Studio API key

### Optional

- **`GOOGLE_AI_STUDIO_MODEL`** - AI model to use (default: `gemini-2.0-flash-exp`)
- **`PORT`** - Server port (default: `8000`)
- **`FIRESTORE_EMULATOR_HOST`** - Firestore emulator address (auto-set in development)
- **`ENVIRONMENT`** - Set to `production` to disable emulator mode
- **`ENABLE_LOCAL_IMAGE_GENERATION`** - Enable image generation (default: `true` for local, `false` in Docker)

## API Endpoints

### Health Check

```bash
GET /health
```

Returns service health status.

### Execute Agent

```bash
POST /agent/execute
```

Execute an agent action with specified parameters.

**Request Body:**
```json
{
  "action": "brainstorm_ideas" | "brainstorm_character" | "brainstorm_plot" | "generate_story" | "generate_chapter" | "generate_next_lines",
  "params": {
    // Action-specific parameters
  },
  "context": {
    // Optional story context
  }
}
```

**Response:**
```json
{
  "result": "Generated content...",
  "metadata": {
    // Additional metadata
  }
}
```

See [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for complete API reference.

## Project Structure

```
novelsync-agents/
├── agents/
│   └── storyAgent/
│       ├── agent.py                 # Main agent orchestration
│       ├── context_builder.py       # Story context from Firestore
│       ├── llm_provider.py          # LLM provider abstraction
│       ├── brainstorming.py         # General brainstorming
│       ├── character_brainstorming.py
│       ├── plot_brainstorming.py
│       ├── story_generation.py
│       ├── chapter_generation.py
│       └── next_line_generation.py
├── bots/                            # Bot implementations
├── image-generation/                # Image generation service
├── server.py                        # FastAPI server entry point
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Container configuration
└── cloudbuild.yaml                  # Cloud Build configuration
```

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest

# Run with coverage
pytest --cov=agents --cov-report=html
```

## Deployment

### Automatic Deployment (GitHub Actions)

The service automatically deploys to Cloud Run when you push to the `main` or `develop` branch.

**Required GitHub Secrets:**
- `GCP_SA_KEY` - Google Cloud service account key (JSON)
- `GOOGLE_AI_STUDIO_API_KEY` - AI Studio API key

### Manual Deployment

```bash
# Build and submit to Cloud Build
gcloud builds submit --tag gcr.io/novelsync-f82ec/story-agent

# Deploy to Cloud Run
gcloud run deploy story-agent \
  --image gcr.io/novelsync-f82ec/story-agent \
  --platform managed \
  --region us-central1 \
  --set-env-vars GOOGLE_CLOUD_PROJECT=novelsync-f82ec,GOOGLE_AI_STUDIO_API_KEY=your-key
```

## Integration with NovelSync

This service is called by Firebase Functions in the `novelsync-frontend` repository:

1. Frontend calls Firebase Function (e.g., `generateStory`)
2. Function calls this service: `POST /agent/execute`
3. Service processes request using Google AI Studio
4. Service reads/writes story context to Firestore
5. Service returns generated content to Function
6. Function returns result to Frontend

### Configuration in Functions

After deploying, update the Cloud Run URL in Firebase Functions:

```bash
# Get the service URL
gcloud run services describe story-agent --region us-central1 --format 'value(status.url)'

# Set in Firebase Functions config
firebase functions:config:set agent_service.url="https://story-agent-xxxxx.run.app"
```

Or set via Firebase Console → Functions → Configuration → Environment variables → `AGENT_SERVICE_URL`

## Architecture

```
┌────────────────┐
│    Frontend    │
│   (React)      │
└───────┬────────┘
        │
        ▼
┌────────────────┐
│    Firebase    │
│   Functions    │
└───────┬────────┘
        │ HTTP
        ▼
┌────────────────┐      ┌──────────────┐
│  Python Agent  │◄────►│   Firestore  │
│   (FastAPI)    │      │   (context)  │
└───────┬────────┘      └──────────────┘
        │
        ▼
┌────────────────┐
│  Google AI     │
│   Studio       │
│  (Gemini)      │
└────────────────┘
```

## Development Workflow

For complete local development setup with all NovelSync services, see the [LOCAL_DEVELOPMENT.md](../novelsync-frontend/LOCAL_DEVELOPMENT.md) in the `novelsync-frontend` repository.

### Quick Start with Full Stack

```bash
# From novelsync-frontend directory
cd ../novelsync-frontend

# Start all services (including this agent)
./scripts/dev.sh

# Or start just the Python agent
docker-compose -f docker-compose.simple.yml up python-agent
```

## Troubleshooting

### Can't connect to Firestore Emulator

**Symptom:** `FIRESTORE_EMULATOR_HOST` not connecting

**Solution:**
1. Verify emulator is running: `firebase emulators:list`
2. Check the host:
   - Native Python: Use `localhost:8080`
   - Docker: Use `host.docker.internal:8080`
3. Verify environment variable is set correctly

### Google AI Studio API Errors

**Symptom:** `401 Unauthorized` or `403 Forbidden`

**Solution:**
1. Verify API key is correct
2. Check API is enabled: https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com
3. Verify quota limits haven't been exceeded

### Container Build Fails

**Symptom:** Docker build errors

**Solution:**
1. Check Dockerfile syntax
2. Verify all dependencies in requirements.txt are valid
3. Check Python version compatibility (must be 3.11)

## Contributing

1. Create a feature branch
2. Make your changes
3. Test locally with Docker
4. Push to GitHub (will trigger CI/CD)
5. Verify deployment in Cloud Run console

## Resources

- [Google AI Studio Documentation](https://ai.google.dev/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Firestore Python SDK](https://firebase.google.com/docs/firestore/quickstart)

## License

See LICENSE file in the main NovelSync repository.

## Support

For issues or questions:
1. Check this README and troubleshooting section
2. Check [API_DOCUMENTATION.md](API_DOCUMENTATION.md)
3. Review Cloud Run logs in GCP Console
4. Open an issue in the GitHub repository
