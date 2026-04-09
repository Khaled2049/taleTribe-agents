# How to add a new StoryAgent tool

This guide walks through the full path from the HTTP API to a new tool implementation, lists the tools that exist today, and ends with how **`StoryContextBuilder`** fits in. It closes with a concrete walkthrough for the **`enhanceWizardInput`** action described in [`example.md`](../example.md) at the repo root.

---

## End-to-end request flow

1. **`POST /agent/execute`** in [`server.py`](../server.py) receives JSON: `action` (string) and `parameters` (object).
2. FastAPI parses the body into `AgentRequest`, where `action` is typed as **`ActionName`** from [`action_schemas.py`](../agents/storyAgent/action_schemas.py) (only known actions are accepted at the type level).
3. **`validate_action_parameters(action, parameters)`** runs the Pydantic schema for that action. It rejects unknown fields (`StrictModel` uses `extra="forbid"`) and returns a **normalized dict using camelCase aliases** (what `StoryAgent.execute_agent` expects).
4. **`StoryAgent.execute_agent`** dispatches on `action` and maps parameters into a dedicated async method, which delegates to a **tool class** in `agents/storyAgent/tools/`.

```mermaid
flowchart LR
  HTTP["POST /agent/execute"]
  VAL["validate_action_parameters"]
  AG["StoryAgent.execute_agent"]
  TL["Tool.execute"]

  HTTP --> VAL --> AG --> TL
```

---

## 1. [`server.py`](../server.py)

You normally **do not** change `server.py` when adding a tool. It already:

- Instantiates **`StoryAgent`** once at startup (`create_app`), using `GOOGLE_CLOUD_PROJECT` and optional `VERTEX_AI_LOCATION`.
- On each request, validates parameters and calls:

```python
validated_params = validate_action_parameters(request.action, request.parameters)
result = await app.state.agent.execute_agent(request.action, validated_params)
```

If validation fails, clients get **422** with `VALIDATION_ERROR`. Unknown actions fail validation because they are not in `ActionName` / `_ACTION_SCHEMAS`.

---

## 2. [`action_schemas.py`](../agents/storyAgent/action_schemas.py)

Every new action must be registered here.

### Steps

1. Add the action string to the **`ActionName`** `Literal` union (e.g. `"enhanceWizardInput"`).
2. Define a **`StrictModel`** subclass with:
   - **`Field(validation_alias=AliasChoices("camelCase", "snake_case"), serialization_alias="camelCase")`** for each parameter the API should accept in either style.
   - Appropriate types, defaults, and constraints (`ge`/`le`, nested `List`/`Dict`, inner `Literal` types, etc.).
3. Map the action name to the model in **`_ACTION_SCHEMAS`**.

`validate_action_parameters` returns **`model_dump(by_alias=True, exclude_none=True)`**, so downstream code should keep reading **camelCase** keys (as `StoryAgent._param` does).

---

## 3. Tool implementation (`agents/storyAgent/tools/`)

Concrete tools live as modules under **[`agents/storyAgent/tools/`](../agents/storyAgent/tools/)**. The package **[`__init__.py`](../agents/storyAgent/tools/__init__.py)** re-exports tool classes for `from agents.storyAgent.tools import ...`.

> **Note:** There is also a sibling file [`agents/storyAgent/tools.py`](../agents/storyAgent/tools.py). In this repo, Python resolves **`agents.storyAgent.tools` to the `tools/` package** (not that file). Treat **`tools/__init__.py`** as the canonical export list when adding imports.

### Typical tool pattern

Most tools follow the same shape as [`brainstorming.py`](../agents/storyAgent/tools/brainstorming.py) or [`enhance_text.py`](../agents/storyAgent/tools/enhance_text.py):

| Piece | Role |
|--------|------|
| **`__init__(self, project_id, location)`** | Stores ids; builds **`LLMProvider`** via `get_llm_provider` and often **`StoryContextBuilder(project_id)`**. |
| **`async def execute(self, ...)`** | Loads Firestore context if needed, builds prompts, calls the LLM, returns a **`Dict[str, Any]`** matching the product contract. |

Use the same **try/except ImportError** pattern as sibling modules if you need imports to work when the package is run from different working directories.

After adding a new module:

1. Export the class from **`tools/__init__.py`** (and `__all__`).
2. Optionally update **`tools.py`** for consistency with older docs (the package remains authoritative for imports).

---

## 4. [`agent.py`](../agents/storyAgent/agent.py)

Wire the new tool into **`StoryAgent`**:

1. **Import** the new tool class from `agents.storyAgent.tools` (see note above about the package).
2. In **`__init__`**, instantiate it: `self.my_tool = MyTool(self.project_id, self.location)`.
3. Add a **small async method** (optional but consistent) that forwards to `self.my_tool.execute(...)`.
4. In **`execute_agent`**, add an `if action == "yourAction":` branch that pulls parameters with **`_param(parameters, "camelKey", "snake_key", default)`** and `await`s your method.

`_param` checks **camelCase first**, then **snake_case**, matching how clients and tests send data.

---

## 5. Tests and manual checks

- Extend **[`tests/test_agent.py`](../tests/test_agent.py)** with dispatch tests: mock the new method, call `execute_agent`, assert arguments (include a **snake_case** case if you support it).
- Run **`pytest`** from the repo root.
- Manually hit **`GET /health`** and **`POST /agent/execute`** with a minimal valid body (see [`docs/api.md`](api.md)).

---

## Current tools (by action)

| Action | Tool class | Role (short) |
|--------|------------|----------------|
| `generateStory` | `StoryGenerationTool` | Generate story (optional first chapter only, genre/tone/length). |
| `generateChapter` | `ChapterGenerationTool` | Generate a numbered chapter with optional prior chapter context. |
| `brainstormIdeas` | `BrainstormingTool` | Ideas by type (e.g. characters, plots, places, themes). |
| `brainstormCharacter` | `CharacterBrainstormingTool` | Character-focused brainstorm. |
| `brainstormPlot` | `PlotBrainstormingTool` | Plot-focused brainstorm. |
| `generateNextLines` | `NextLineGenerationTool` | Next-line suggestions from chapter text + cursor position. |
| `chatWithContext` | `ChatWithContextTool` | Chat with RAG / story context. |
| `enhanceText` | `EnhanceTextTool` | Expand / dialogue / rewrite on a text selection. |

Module files: [`story_generation.py`](../agents/storyAgent/tools/story_generation.py), [`chapter_generation.py`](../agents/storyAgent/tools/chapter_generation.py), [`brainstorming.py`](../agents/storyAgent/tools/brainstorming.py), [`character_brainstorming.py`](../agents/storyAgent/tools/character_brainstorming.py), [`plot_brainstorming.py`](../agents/storyAgent/tools/plot_brainstorming.py), [`next_line_generation.py`](../agents/storyAgent/tools/next_line_generation.py), [`chat_with_context.py`](../agents/storyAgent/tools/chat_with_context.py), [`enhance_text.py`](../agents/storyAgent/tools/enhance_text.py).

---

## Context builder (last)

[`context_builder.py`](../agents/storyAgent/context_builder.py) defines **`StoryContextBuilder`**, used by tools that need the full NovelSync story world from Firestore.

### Responsibilities

1. **`build_story_context(story_id)`**  
   - Reads the `stories/{story_id}` document.  
   - Raises **`ValueError`** if the story does not exist.  
   - Loads subcollections: **`characters`**, **`places`**, **`plots`**, **`chapters`**.  
   - Sorts chapters by **`chapterNumber`**.  
   - Returns a dict: `story`, `characters`, `places`, `plots`, `chapters` (each document includes `id`).

2. **`format_context_for_prompt(context)`**  
   - Turns that dict into a single string with labeled sections (title, genre, tone, description, characters, places, plots, and a short summary of existing chapters—first five plus a count).

### When to use it

- Use **`build_story_context` + `format_context_for_prompt`** when the model should stay consistent with **saved** story data (most chapter/story/brainstorm flows).
- You may **skip** it for actions that only consume **request payload** data (e.g. a wizard step that does not yet have a `storyId`). The [`example.md`](../example.md) **`enhanceWizardInput`** spec uses **`userId`** and a **`data`** blob; you can still call Firestore later if you store drafts per user or link `userId` to a story.

---

## Example: adding `enhanceWizardInput` from [`example.md`](../example.md)

[`example.md`](../example.md) specifies:

- **Action name:** `enhanceWizardInput`
- **Request shape:** `{ "type": "<premise|character|place|conflict|blueprint>", "data": { ... }, "userId": "string" }`
- **Responses:** `{ "enhanced": "..." }` for the first four types, or `{ "blueprint": { ... } }` for `blueprint`.

Below is a **minimal integration outline** (not drop-in production code). Adapt prompts to match the exact copy and JSON rules in `example.md`.

### 1. `action_schemas.py`

```python
# Add to ActionName:
# "enhanceWizardInput",

class EnhanceWizardInputParams(StrictModel):
    user_id: str = Field(
        validation_alias=AliasChoices("userId", "user_id"),
        serialization_alias="userId",
    )
    type: Literal["premise", "character", "place", "conflict", "blueprint"]
    data: Dict[str, Any]

# In _ACTION_SCHEMAS:
# "enhanceWizardInput": EnhanceWizardInputParams,
```

Using a single **`data: Dict[str, Any]`** keeps the schema simple; you can tighten validation later (e.g. discriminated unions per `type`) once the frontend payload is stable.

### 2. New file `agents/storyAgent/tools/enhance_wizard_input.py`

- Class **`EnhanceWizardInputTool`** with `__init__(project_id, location)` → `get_llm_provider(...)`.
- **`async def execute(self, user_id: str, wizard_type: str, data: Dict[str, Any])`**:
  - Branch on `wizard_type`.
  - For `premise` / `character` / `place` / `conflict`, build a user prompt from `data`, call the LLM, return `{"enhanced": "..."}`.
  - For `blueprint`, send the full wizard snapshot, parse/validate model output, return `{"blueprint": {...}}`.
- Optionally use **`StoryContextBuilder`** only if you add a **`storyId`** later or load a draft document keyed by `user_id`.

### 3. `tools/__init__.py`

Export **`EnhanceWizardInputTool`** and add it to **`__all__`**.

### 4. `agent.py`

- Import and instantiate **`self.enhance_wizard_tool`**.
- Add **`async def enhance_wizard_input(self, user_id, wizard_type, data)`** delegating to the tool.
- In **`execute_agent`**:

```python
if action == "enhanceWizardInput":
    return await self.enhance_wizard_input(
        self._param(parameters, "userId", "user_id"),
        self._param(parameters, "type"),  # JSON key is "type"; avoid shadowing builtin if you name the param wizard_type in the method
        self._param(parameters, "data", default={}) or {},
    )
```

Name the Python parameter **`wizard_type`** (or similar) in your method signature so you do not overwrite the builtin **`type`**.

### 5. Tests

In **`tests/test_agent.py`**, mock **`enhance_wizard_input`**, call **`execute_agent("enhanceWizardInput", {...})`**, and assert the tool was awaited with the expected `userId`, type string, and `data`.

### 6. API contract

Document the new action in [`docs/api.md`](api.md) if you expose it to frontend or Cloud Functions teams.

---

## Checklist

- [ ] `ActionName` + params model + `_ACTION_SCHEMAS` in `action_schemas.py`
- [ ] New tool module under `agents/storyAgent/tools/`
- [ ] Export in `tools/__init__.py`
- [ ] `StoryAgent` import, instance, method, and `execute_agent` branch in `agent.py`
- [ ] `pytest` + manual `/agent/execute` call
- [ ] Decide whether **`StoryContextBuilder`** is required for this action or payload-only LLM calls are enough
