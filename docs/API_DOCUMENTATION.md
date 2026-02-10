# API Documentation

Complete API documentation for all endpoints that the frontend can interact with.

## Base URL

- **Production**: `https://<your-service-url>.run.app` (Cloud Run)
- **Local Development**: `http://localhost:8000`

**Note**: The API is a unified FastAPI service that includes both story agent endpoints and image generation endpoints.

---

## Authentication

**Note**: Authentication is currently handled by the frontend/client. The API endpoints do not require authentication headers in the current implementation. Authentication and authorization may be handled at the application level or through Firebase on the client side.

---

## Endpoints

### 1. Execute Agent Action

Execute a story agent action (unified endpoint for all agent operations).

**Endpoint:** `POST /agent/execute`

**Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
  "action": "string (required, one of: generateStory, generateChapter, brainstormIdeas, brainstormCharacter, brainstormPlot, generateNextLines)",
  "parameters": {
    "storyId": "string (required for most actions)",
    "...": "additional parameters based on action"
  }
}
```

**Success Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "...": "action-specific response data"
  }
}
```

**Error Response (200 OK with error):**
```json
{
  "success": false,
  "error": "Error message describing what went wrong"
}
```

**Available Actions:**

#### 1.1. generateStory

Generate a complete story based on Firestore context.

**Parameters:**
- `storyId` (string, required): Firestore story document ID
- `genre` (string, optional): Story genre
- `tone` (string, optional): Story tone
- `length` (string, optional): Story length (short/medium/long)

**Example Request:**
```json
{
  "action": "generateStory",
  "parameters": {
    "storyId": "story-123",
    "genre": "fantasy",
    "tone": "epic",
    "length": "medium"
  }
}
```

**Example Response:**
```json
{
  "success": true,
  "data": {
    "storyId": "story-123",
    "content": "Generated story content...",
    "metadata": {
      "genre": "fantasy",
      "tone": "epic",
      "length": "medium"
    }
  }
}
```

---

#### 1.2. generateChapter

Generate a single chapter with continuity based on previous chapters.

**Parameters:**
- `storyId` (string, required): Firestore story document ID
- `chapterNumber` (number, required): Chapter number to generate
- `previousChapters` (array, optional): List of previous chapters for context
- `plotContext` (string, optional): Optional plot context

**Example Request:**
```json
{
  "action": "generateChapter",
  "parameters": {
    "storyId": "story-123",
    "chapterNumber": 2,
    "previousChapters": [
      {
        "chapterNumber": 1,
        "title": "Chapter 1",
        "content": "Previous chapter content..."
      }
    ]
  }
}
```

**Example Response:**
```json
{
  "success": true,
  "data": {
    "storyId": "story-123",
    "chapterNumber": 2,
    "title": "Chapter 2: The Journey Begins",
    "content": "Generated chapter content..."
  }
}
```

---

#### 1.3. brainstormIdeas

Generate brainstorming ideas synchronously (characters, plots, places, or themes).

**Parameters:**
- `storyId` (string, required): Firestore story document ID
- `type` (string, required): Type of idea - one of: `characters`, `plots`, `places`, `themes`
- `prompt` (string, optional): Additional requirements or specific prompt
- `count` (number, optional): Number of ideas to generate (default: 5)

**Example Request:**
```json
{
  "action": "brainstormIdeas",
  "parameters": {
    "storyId": "story-123",
    "type": "characters",
    "prompt": "mysterious and powerful",
    "count": 5
  }
}
```

**Example Response:**
```json
{
  "success": true,
  "data": {
    "storyId": "story-123",
    "type": "characters",
    "ideas": [
      {
        "text": "A mysterious sorcerer who guards ancient secrets"
      },
      {
        "text": "A powerful warrior with a hidden past"
      }
    ]
  }
}
```

---

#### 1.4. brainstormCharacter

Generate detailed character ideas synchronously.

**Parameters:**
- `storyId` (string, required): Firestore story document ID
- `role` (string, optional): Character role (e.g., protagonist, antagonist, supporting)
- `archetype` (string, optional): Character archetype (e.g., hero, villain, mentor)

**Example Request:**
```json
{
  "action": "brainstormCharacter",
  "parameters": {
    "storyId": "story-123",
    "role": "protagonist",
    "archetype": "hero"
  }
}
```

**Example Response:**
```json
{
  "success": true,
  "data": {
    "storyId": "story-123",
    "character": {
      "role": "protagonist",
      "archetype": "hero",
      "profile": "Detailed character profile description..."
    }
  }
}
```

---

#### 1.5. brainstormPlot

Generate plot ideas synchronously.

**Parameters:**
- `storyId` (string, required): Firestore story document ID
- `plotType` (string, optional): Type of plot element - one of: `conflict`, `twist`, `subplot`, `development` (default: `conflict`)

**Example Request:**
```json
{
  "action": "brainstormPlot",
  "parameters": {
    "storyId": "story-123",
    "plotType": "twist"
  }
}
```

**Example Response:**
```json
{
  "success": true,
  "data": {
    "storyId": "story-123",
    "plotType": "twist",
    "plot": "Detailed plot suggestion..."
  }
}
```

---

#### 1.6. generateNextLines

Generate 3 next line suggestions based on chapter content and cursor position.

**Parameters:**
- `storyId` (string, required): Firestore story document ID
- `content` (string, required): Current content of the chapter being edited
- `cursorPosition` (number, required): Character index where the new line should be inserted
- `chapterId` (string, optional): Chapter document ID for better context and validation

**Example Request:**
```json
{
  "action": "generateNextLines",
  "parameters": {
    "storyId": "story-123",
    "content": "The hero stood at the edge of the cliff, looking down at the valley below.",
    "cursorPosition": 65,
    "chapterId": "chapter-456"
  }
}
```

**Example Response:**
```json
{
  "success": true,
  "data": {
    "suggestions": [
      "The wind howled around him, carrying the scent of distant forests.",
      "Below, he could see the ancient ruins shimmering in the moonlight.",
      "A sense of foreboding filled his heart as he prepared to descend."
    ]
  }
}
```

---

### 2. Health Check

Check if the API is running and which services are available.

**Endpoint:** `GET /health`

**Headers:** None required

**Request Body:** None

**Success Response (200 OK):**
```json
{
  "status": "healthy",
  "project_id": "your-project-id",
  "services": {
    "agent": "available",
    "image_generation": "available"
  }
}
```

**Response Fields:**
- `status` (string): Overall service status ("healthy" or "unhealthy")
- `project_id` (string): Google Cloud Project ID
- `services` (object): Availability of individual services
  - `agent` (string): Agent service status ("available" or "unavailable")
  - `image_generation` (string): Image generation service status ("available" or "unavailable")

**Example Request:**
```bash
curl http://localhost:8000/health
```

**Example Response:**
```json
{
  "status": "healthy",
  "project_id": "my-project-id",
  "services": {
    "agent": "available",
    "image_generation": "available"
  }
}
```

**Notes:**
- The image generation service may be unavailable if dependencies are not installed or if `ENABLE_LOCAL_IMAGE_GENERATION` is set to `false`
- If image generation is unavailable, the server will still run but image generation endpoints will not be accessible

---

## Image Generation API

The image generation API is integrated into the unified FastAPI service. It provides endpoints for generating cover images using Stable Diffusion models.

**Base URL**: Same as main API (`http://localhost:8000` for local development)

**Note**: Image generation endpoints do not require authentication.

---

### 3. Generate Cover Image

Generate a cover image from a text prompt using Stable Diffusion.

**Endpoint:** `POST /generate-cover`

**Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
  "prompt": "string (required, 1-500 characters)"
}
```

**Success Response (200 OK):**
```json
{
  "image": "iVBORw0KGgoAAAANSUhEUgAA...",
  "prompt": "A beautiful sunset over mountains",
  "model": "stabilityai/sd-turbo",
  "generation_time": 1.23
}
```

**Response Fields:**
- `image` (string): Base64-encoded PNG image
- `prompt` (string): The prompt that was used to generate the image
- `model` (string): The model used for generation (e.g., "stabilityai/sd-turbo")
- `generation_time` (float): Time taken to generate the image in seconds

**Error Responses:**
- **422 Unprocessable Entity:**
```json
{
  "detail": [
    {
      "loc": ["body", "prompt"],
      "msg": "ensure this value has at least 1 characters",
      "type": "value_error.any_str.min_length"
    }
  ]
}
```

- **500 Internal Server Error:**
```json
{
  "detail": "Image generation failed: <error message>"
}
```

**Example Request:**
```bash
curl -X POST "http://localhost:8000/generate-cover" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A futuristic cityscape at night with neon lights"}'
```

**Example Response:**
```json
{
  "image": "iVBORw0KGgoAAAANSUhEUgAA...",
  "prompt": "A futuristic cityscape at night with neon lights",
  "model": "stabilityai/sd-turbo",
  "generation_time": 1.45
}
```

**Notes:**
- The first request may take longer as the model needs to be downloaded (~1.4GB)
- Generation time varies based on hardware (CPU: 5-30s, GPU: 0.5-2s)
- Images are returned as base64-encoded PNG format
- The default model is optimized for low-end machines and uses minimal inference steps

---

### 4. Image Generation Health Check

Check if the image generation API is running and the model is loaded.

**Endpoint:** `GET /image-health`

**Headers:** None required

**Request Body:** None

**Success Response (200 OK):**
```json
{
  "status": "healthy",
  "model": "stabilityai/sd-turbo",
  "device": "cpu",
  "model_loaded": true
}
```

**Response Fields:**
- `status` (string): Service status ("healthy", "model_not_loaded", or "unhealthy")
- `model` (string): The model name configured
- `device` (string): Device being used ("cpu" or "cuda")
- `model_loaded` (boolean): Whether the model is currently loaded in memory

**Example Request:**
```bash
curl http://localhost:8000/image-health
```

**Example Response:**
```json
{
  "status": "healthy",
  "model": "stabilityai/sd-turbo",
  "device": "cpu",
  "model_loaded": true
}
```

**Notes:**
- Use this endpoint to verify the service is ready before making generation requests
- If `model_loaded` is `false`, the first generation request will load the model (which may take time)

---

## Common Error Response Format

All endpoints may return the following error responses:

### 400 Bad Request
Invalid request parameters:
```json
{
  "error": "Error message describing what's wrong"
}
```

### 500 Internal Server Error
Server error:
```json
{
  "success": false,
  "error": "Error message"
}
```

For image generation endpoints, errors follow FastAPI's validation format:
```json
{
  "detail": "Error message"
}
```

---

## Notes

1. **Unified API**: All story agent operations are accessed through the unified `/agent/execute` endpoint with different `action` values.

2. **Synchronous Operations**: All agent actions (`generateStory`, `generateChapter`, `brainstormIdeas`, `brainstormCharacter`, `brainstormPlot`, `generateNextLines`) are synchronous and return results immediately. There is no job queue system in the current implementation.

3. **Story Context**: The agent automatically fetches story context (characters, places, plots, chapters) from Firestore based on the `storyId` parameter. No separate context fetching endpoint is needed.

4. **Firestore Integration**: All agent actions require a valid `storyId` that exists in Firestore. The agent reads and writes to Firestore automatically.

5. **Image Generation**: Image generation is optional and may not be available if:
   - Dependencies are not installed
   - `ENABLE_LOCAL_IMAGE_GENERATION` environment variable is set to `false`
   - The service is running in a production environment where image generation is disabled

6. **CORS**: All endpoints support CORS for cross-origin requests.

7. **FastAPI Documentation**: Interactive API documentation is available at:
   - Swagger UI: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

8. **Environment Variables**: Required environment variables:
   - `GOOGLE_CLOUD_PROJECT`: Google Cloud Project ID (required for Firestore)
   - `GOOGLE_AI_STUDIO_API_KEY`: Google AI Studio API key (required for agent)
   - `ENABLE_LOCAL_IMAGE_GENERATION`: Set to `true` or `false` to enable/disable image generation (default: `true`)

9. **Port Configuration**: The server uses port 8000 by default for local development. In Cloud Run, the port is set by the `PORT` environment variable (defaults to 8080).

---

## Migration Notes

If you're migrating from the old API structure:

- **Old**: `POST /generateStory` → **New**: `POST /agent/execute` with `action: "generateStory"`
- **Old**: `POST /generateChapter` → **New**: `POST /agent/execute` with `action: "generateChapter"`
- **Old**: `POST /brainstormIdeas` → **New**: `POST /agent/execute` with `action: "brainstormIdeas"`
- **Old**: `POST /brainstormCharacter` → **New**: `POST /agent/execute` with `action: "brainstormCharacter"`
- **Old**: `POST /brainstormPlot` → **New**: `POST /agent/execute` with `action: "brainstormPlot"`
- **New**: `POST /agent/execute` with `action: "generateNextLines"` (new feature)

All other endpoints (`/authenticate`, `/getData`, `/jobStatus`, `/storyJobs`, `/getStoryContext`, `/updateContext`) are no longer available in the current implementation.
