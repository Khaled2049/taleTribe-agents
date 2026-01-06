# Story Agent Service

Python-based AI agent service for generating stories, chapters, and brainstorming ideas using Google's Gemini model via **Google AI Studio API**.

## Setup

### Prerequisites

- Python 3.9+
- Google Cloud Project (for Firestore access)
- Google AI Studio API Key - Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
- Firestore database access

**Note**: Vertex AI is **not required**. The service uses Google AI Studio REST API directly.

### Installation

**Note**: This project now uses a unified Python environment. See the main [SETUP.md](../../SETUP.md) guide for complete setup instructions.

1. Install dependencies (from the `python/` root directory):
```bash
cd python
python -m venv venv
# Activate venv (Windows: venv\Scripts\activate, Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
```

2. Set environment variables (create `.env` file in `python/` directory):
```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"  # Required for Firestore
export GOOGLE_AI_STUDIO_API_KEY="your-api-key"  # Required for AI generation
export GOOGLE_AI_STUDIO_MODEL="gemini-2.0-flash-exp"  # Optional, defaults to gemini-2.0-flash-exp
```

### Running the Service

#### Local Development

```bash
# From python/ directory with venv activated
python server.py
```

The service will start on `http://localhost:8000`

#### Production (Cloud Run)

1. Build and deploy:
```bash
gcloud run deploy story-agent \
  --source . \
  --platform managed \
  --region us-central1 \
  --set-env-vars GOOGLE_CLOUD_PROJECT=your-project-id,GOOGLE_AI_STUDIO_API_KEY=your-api-key,GOOGLE_AI_STUDIO_MODEL=gemini-2.0-flash-exp
```

## API Endpoints

### POST /agent/execute

Execute an agent action.

**Request:**
```json
{
  "action": "generateStory|generateChapter|brainstormIdeas|brainstormCharacter|brainstormPlot",
  "parameters": {
    "storyId": "story-id",
    ...
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": { ... }
}
```

### GET /health

Health check endpoint.

## Agent Actions

### generateStory
Generates a complete story based on Firestore context.

**Parameters:**
- `storyId` (required): Firestore story document ID
- `genre` (optional): Story genre
- `tone` (optional): Story tone
- `length` (optional): Story length (short/medium/long)

### generateChapter
Generates a single chapter with continuity.

**Parameters:**
- `storyId` (required): Firestore story document ID
- `chapterNumber` (required): Chapter number to generate
- `previousChapters` (optional): List of previous chapters for context

### brainstormIdeas
Generates brainstorming ideas.

**Parameters:**
- `storyId` (required): Firestore story document ID
- `type` (required): Type of idea (characters/plots/places/themes)
- `prompt` (optional): Additional requirements
- `count` (optional): Number of ideas (default: 5)

### brainstormCharacter
Generates detailed character ideas.

**Parameters:**
- `storyId` (required): Firestore story document ID
- `role` (optional): Character role
- `archetype` (optional): Character archetype

### brainstormPlot
Generates plot ideas.

**Parameters:**
- `storyId` (required): Firestore story document ID
- `plotType` (optional): Type of plot (conflict/twist/subplot/development)

## Architecture

- `agent.py`: Main agent class that orchestrates tools
- `tools.py`: Individual tools for story generation, chapter creation, and brainstorming
- `context_builder.py`: Fetches and formats story context from Firestore
- `server.py`: FastAPI HTTP server for the agent service

