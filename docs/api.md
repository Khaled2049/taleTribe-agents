# API

Base URL:

- local: `http://localhost:8000`
- production: Cloud Run service URL

## Endpoints

### `GET /health`

Returns service health and whether the optional image router is available.

Example response:

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

### `POST /agent/execute`

Request body:

```json
{
  "action": "generateStory",
  "parameters": {
    "storyId": "story-123"
  },
  "user_id": "firebase-uid-123",
  "provider_config": {
    "provider": "claude",
    "api_key": "sk-ant-...",
    "model": "claude-sonnet-4-6"
  }
}
```

`user_id` and `provider_config` are optional. When `provider_config` is present and `CREDIT_PROXY_URL` is set, the agent forwards the BYOK credentials to creditProxy for that request; the platform's default API key is not used.

Standard request (no BYOK):

```json
{
  "action": "generateStory",
  "parameters": {
    "storyId": "story-123"
  }
}
```

Success response:

```json
{
  "success": true,
  "data": {}
}
```

Error response:

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid parameters",
    "details": []
  }
}
```

## Supported actions

### `generateStory`

- `storyId`
- `genre` optional
- `tone` optional
- `length` optional
- `generateFirstChapterOnly` optional, defaults to `true`
- `plotContext` optional

### `generateChapter`

- `storyId`
- `chapterNumber`
- `previousChapters` optional
- `plotContext` optional

### `brainstormIdeas`

- `storyId`
- `type`
- `prompt` optional
- `count` optional, `1-20`, default `5`

### `brainstormCharacter`

- `storyId`
- `role` optional
- `archetype` optional

### `brainstormPlot`

- `storyId`
- `plotType` optional, defaults to `conflict`

### `generateNextLines`

- `storyId`
- `content`
- `cursorPosition`
- `chapterId` optional

### `chatWithContext`

- `storyId`
- `message`
- `chatHistory` optional
- `userId` optional — used to scope procedural memory (tone, style preferences) to the user; defaults to `"anonymous"` when omitted

When `userId` is provided, the brain memory system augments the prompt with relevant facts, past events, and user style preferences retrieved from Firestore. After the response is sent, memory is updated in the background via reflection.

### `enhanceText`

- `storyId`
- `action`: `expand`, `dialogue`, or `rewrite`
- `selectedText`
- `chapterId` optional

### `enhanceWizardInput`

- `type`: `premise`, `character`, `place`, `conflict`, or `blueprint`
- `data`: object payload for the selected `type`
- `userId`
- response:
  - `premise|character|place|conflict` -> `{ "enhanced": "..." }`
  - `blueprint` -> `{ "blueprint": { ... } }`

### `generateStoryChoices`

Generates an opening scene with branching choices (first launch) or continuation choices (co-write).

- `storyId`
- `mode`: `opening`, `continuation`, or `ending`
- `currentContent` optional, HTML from the editor — empty string for opening, defaults to `""`
- `chapterId` optional
- `turnCount` optional, number of choices selected so far — used for arc-aware prompting, defaults to `0`
- `userId` optional — scopes procedural memory (tone, style preferences) to the user; enables brain memory augmentation when provided

Response for `mode: "opening"`:

```json
{
  "storyId": "story-123",
  "openingScene": "The rain had been falling for three days...",
  "choices": [
    { "label": "Elena discovers the hidden letter", "sceneText": "She found it tucked beneath the floorboard..." },
    { "label": "A stranger arrives at the inn", "sceneText": "The door swung open against the wind..." },
    { "label": "The market erupts in chaos", "sceneText": "First came the sound — a low crack..." }
  ]
}
```

Response for `mode: "continuation"`:

```json
{
  "storyId": "story-123",
  "choices": [
    { "label": "Confront Marcus directly", "sceneText": "..." },
    { "label": "Follow the shadow into the alley", "sceneText": "..." },
    { "label": "Return to the archive", "sceneText": "..." }
  ]
}
```

Response for `mode: "ending"`:

```json
{
  "storyId": "story-123",
  "choices": [
    { "label": "The story reaches its end", "sceneText": "...", "isFinal": true }
  ]
}
```

## Validation and error behavior

- unknown actions return `400 BAD_REQUEST`
- schema validation failures return `422 VALIDATION_ERROR`
- unexpected failures return `500 INTERNAL_ERROR`
- unknown request fields are rejected because the action schemas are strict

The validation source of truth is `agents/storyAgent/action_schemas.py`.
