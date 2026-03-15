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

### `enhanceText`

- `storyId`
- `action`: `expand`, `dialogue`, or `rewrite`
- `selectedText`
- `chapterId` optional

## Validation and error behavior

- unknown actions return `400 BAD_REQUEST`
- schema validation failures return `422 VALIDATION_ERROR`
- unexpected failures return `500 INTERNAL_ERROR`
- unknown request fields are rejected because the action schemas are strict

The validation source of truth is `agents/storyAgent/action_schemas.py`.
