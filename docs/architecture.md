# Architecture

`novelsync-agents` is a single FastAPI service that loads story-generation logic and, optionally, local image-generation routes.

## Request flow

1. `server.py` loads `.env`, sets local emulator defaults, and creates the FastAPI app.
2. `POST /agent/execute` validates the request against `agents/storyAgent/action_schemas.py`.
3. `StoryAgent` handles the action and delegates to the relevant tool implementation under `agents/storyAgent/tools/`.
4. Brain-enabled tools call `Brain.assemble()` to augment the prompt with retrieved memory before calling the LLM.
5. Responses are wrapped in a stable envelope: `success`, `data`, and structured `error`.
6. After the response is sent, `Brain.reflect()` runs as a FastAPI `BackgroundTask` to update memory layers.

## Main modules

- `server.py`: HTTP surface, error handling, CORS, app state
- `agents/storyAgent/agent.py`: action dispatch; owns shared `SentenceTransformer`, Firestore client, and LLM provider
- `agents/storyAgent/context_builder.py`: story context retrieval and formatting (legacy, still used by non-brain tools)
- `agents/storyAgent/llm_provider.py`: model interaction
- `agents/storyAgent/tools/`: action-specific logic
- `agents/storyAgent/brain/`: four-layer cognitive memory system (see below)

## Brain — cognitive memory

`brain/` is a self-contained sub-package that gives the agent persistent, retrieval-augmented memory. It is currently integrated into:

1. `chatWithContext`
2. `generateStoryChoices`

### Four memory layers

| Layer          | What it stores                         | Firestore path                                                                   | Injection                              |
| -------------- | -------------------------------------- | -------------------------------------------------------------------------------- | -------------------------------------- |
| **Working**    | Current scene, active characters, mood | `stories/{id}/working_memory/state`                                              | Always                                 |
| **Procedural** | Tone, style, POV, narrative rules      | `users/{id}/procedural_memory/global` + `stories/{id}/procedural_memory/context` | Always                                 |
| **Semantic**   | Facts, lore, character details         | `stories/{id}/semantic_memory/{id}`                                              | Retrieved — top-5 by cosine similarity |
| **Episodic**   | Past event summaries                   | `stories/{id}/episodic_memory/{id}`                                              | Retrieved — top-3 by cosine similarity |

Working and procedural are small and always injected. Semantic and episodic are retrieved selectively using embedding similarity against the current user message.

### Embeddings and vector search

Embeddings use `sentence-transformers all-MiniLM-L6-v2` (384 dimensions, ~80 MB, open source, $0). Embedding arrays are stored as Firestore document fields. Vector search is performed in Python using numpy cosine similarity — no external vector database. At ~100–300 documents per story this takes ~5–15 ms per request.

### Reflection

After every LLM response, `Brain.reflect()` runs four concurrent background extractions (all via the existing LLM provider):

1. **Working** — extract current scene state and patch the working memory document
2. **Procedural** — detect style signals and update preferences if changed
3. **Semantic** — extract up to 5 new facts, embed, and store each as a Firestore document
4. **Episodic** — summarize the narrative event, embed, and store

Reflection runs via FastAPI `BackgroundTasks` so it never blocks the response to the user. Individual extraction failures are logged and swallowed.

### Sequence Diagram

sequenceDiagram
autonumber
participant C as Client
participant API as FastAPI route<br/>`server.execute_agent`
participant VAL as `validate_action_parameters`
participant SA as `StoryAgent.execute_agent`
participant CHAT as `StoryAgent.chat_with_context`
participant B as `Brain`
participant ASM as Brain Assembler<br/>`brain.assemble(...)`
participant TOOL as `ChatWithContextTool.execute`
participant CTX as `StoryContextBuilder`
participant LLM as LLM Provider
participant BG as FastAPI BackgroundTasks
participant REF as Brain Reflector<br/>`brain.reflect(...)`

    C->>API: POST `/agent/execute`<br/>{ action: "chatWithContext", parameters... }

    API->>VAL: Validate + normalize action params
    VAL-->>API: validated params (camelCase)

    API->>SA: `execute_agent(action, params, background_tasks)`
    SA->>CHAT: `chat_with_context(story_id, message, chat_history, user_id, background_tasks)`

    alt Embedder available (`self._embedder is not None`)
        CHAT->>B: `_make_brain(user_id, story_id)`
        B-->>CHAT: Brain instance (scoped to user+story)

        CHAT->>ASM: `assemble(message, action_hint="chatWithContext")`
        Note over ASM: Combines working/episodic/semantic/procedural<br/>memory into one assembled prompt context
        ASM-->>CHAT: `assembled` (includes `.text`)
        CHAT->>CHAT: `brain_context = assembled.text`
    else Embedder unavailable
        Note over CHAT: Skip brain assembly<br/>`brain_context = None`
    end

    CHAT->>TOOL: `chat_tool.execute(story_id, message, chat_history, brain_context)`

    TOOL->>CTX: `build_story_context(story_id)` (Firestore context payload)
    CTX-->>TOOL: story/chapters/characters/plots/places

    alt brain_context provided
        Note over TOOL: Use brain-assembled context text
    else no brain_context
        Note over TOOL: Build legacy context string from Firestore payload
    end

    TOOL->>LLM: `generate_content_async(full_prompt)`
    LLM-->>TOOL: assistant response text
    TOOL-->>CHAT: `{ response, contextUsed }`

    alt Brain exists AND background_tasks exists AND response non-empty
        CHAT->>BG: `add_task(brain.reflect, ReflectionInput(...))`
        Note over BG,REF: Runs after HTTP response is returned
        BG->>REF: `brain.reflect(ReflectionInput)`
        Note over REF: Writes/updates long-term memory artifacts
    else Any condition missing
        Note over CHAT: No reflection task scheduled
    end

    CHAT-->>SA: result
    SA-->>API: result
    API-->>C: `AgentResponse(success=true, data=result)`

### Brain package structure

```
agents/storyAgent/brain/
├── brain.py          # Brain class — public API
├── types.py          # Dataclasses: BrainConfig, AssembledPrompt, ReflectionInput, MemoryDocument, ...
├── engine/
│   ├── assembler.py  # PromptAssembler — builds layered prompt string
│   ├── reflector.py  # MemoryReflector — 4 concurrent background extractions
│   └── router.py     # MemoryRouter — decides which layers to query per action
└── memory/
    ├── working.py    # WorkingMemoryLayer
    ├── procedural.py # ProceduralMemoryLayer
    ├── semantic.py   # SemanticMemoryLayer + cosine similarity
    └── episodic.py   # EpisodicMemoryLayer
```

### Shared resources

`StoryAgent.__init__` loads these once per process and injects them into every `Brain` instance:

- `self._embedder` — `SentenceTransformer("all-MiniLM-L6-v2")`
- `self._db` — `firestore.Client`
- `self._llm_provider` — `CreditProxyProvider` (routes all LLM calls through creditProxy gateway)

If `sentence-transformers` is not installed, `_embedder` is `None` and brain memory retrieval is silently skipped. The tool falls back to the legacy Firestore context string.

## Optional image generation

`server.py` attempts to load routes from `image-generation/` when `ENABLE_LOCAL_IMAGE_GENERATION=true`.

- when imports succeed, the router is mounted and `/health` reports `image_generation: available`
- when disabled or missing dependencies, the service still starts and reports `image_generation: unavailable`

This keeps production lean while allowing local image generation with the full dependency set.

## Testing and validation

- Python tests live in `tests/`
- Brain-specific tests: `test_brain_assembler.py`, `test_brain_router.py`, `test_brain_cosine.py`, `test_brain_reflector.py`
- PR validation runs formatting checks, `ruff`, `pytest`, `terraform fmt -check`, and `terraform validate`
- Cloud Run deployment is handled by `.github/workflows/deploy.yml`
