# Brain — Cognitive Memory System

The brain gives the AI a persistent, structured memory across writing sessions. Without it, every request starts from scratch — the AI reads all your story data from the database, uses it once, and forgets everything. With it, the AI builds up a living understanding of your story over time, retrieves exactly what's relevant to the moment, and continuously improves its memory after every response.

This document explains how that works — in plain language first, then with technical detail for developers who want to extend it.

---

## The problem it solves

Imagine asking a human writing assistant to help with chapter 12. A good assistant remembers that Elena has a fear of fire established in chapter 3, that the villain's motivation shifted in chapter 8, and that you prefer a melancholic tone. A bad assistant reads your entire manuscript from page one on every question, pulls in things that have nothing to do with the current scene, and forgets everything you told them last session.

The old approach was the bad assistant. Every time you asked for help, the system:

1. Fetched everything — all characters, all plots, all chapters — from the database
2. Stuffed it all into the prompt
3. Got a response
4. Threw all of it away

As stories grow, this gets expensive and noisy. A 30-chapter novel dumps irrelevant early content into every request, filling the context window with things that don't matter to what's happening right now.

The brain fixes this with the same approach human memory uses: some things are always kept in mind, and other things are recalled only when they're relevant.

---

## The four memory layers

Think of these like four different kinds of memory a person uses when telling a story.

### Working memory — "what's happening right now"

This is the AI's short-term awareness of the current scene. It's small, always present, and constantly updated.

**What it holds:**
- The current scene description
- Which characters are actively present
- The last few things that happened
- The current emotional mood

**Human analogy:** A storyteller keeping track of where they are mid-sentence — who's in the room, what was just said, how the atmosphere feels.

**Example:**
```
Scene: Elena enters the abandoned library, searching for the letter
Active characters: Elena, The Archivist
Recent events:
  - Marcus revealed he was working for the Order
  - Elena escaped the burning tower
Mood: dread with a thread of desperate hope
```

Working memory is updated automatically after every response. If the AI just wrote a scene where Elena finally confronts Marcus, working memory will note that in "recent events" so the next scene knows about it without being told.

---

### Procedural memory — "how this story should be written"

This is the AI's understanding of your personal preferences and the rules of your specific story. It's also always injected — every request benefits from it.

There are two levels:

- **Global (per user):** Your general writing preferences across all stories — preferred POV, typical tone, stylistic habits you've developed.
- **Story-level:** Rules specific to this story — its genre, narrative constraints, anything you've decided should always be true.

**Human analogy:** A ghostwriter who has worked with you long enough to know you hate adverbs, always write in tight third-person, and prefer dialogue that feels terse and realistic — even when you don't remind them.

**Example:**
```
Tone: melancholic, restrained
Style: terse sentences, sparse dialogue, no adverbs
POV: third-person limited (Elena's perspective only)
Genre: gothic fantasy
Rules:
  - The villain's motivations are always sympathetic
  - Magic is never explained, only experienced
  - Never break the fourth wall
```

Procedural memory is updated when the AI detects consistent patterns in the writing. If you keep editing responses to remove florid language, eventually the AI will learn that and stop producing it.

---

### Semantic memory — "what is true in this world"

This is long-term factual knowledge about your story's world. Characters, locations, lore, rules of magic, backstories, relationships — anything that could be relevant at some point but doesn't need to be in every prompt.

Rather than injecting all of this every time (which would be enormous for a developed story), the system retrieves only the facts relevant to what you're currently writing.

**Human analogy:** A novelist's reference notes. They don't re-read the entire bible before writing a scene — they look up what's relevant. "Who is Mira's father again? What does the Obsidian Flame actually do?"

**How retrieval works:** When you ask "write the scene where Elena finds the letter," the system converts that request into a mathematical representation (an embedding) and compares it against all stored facts. The closest matches — the ones most semantically related to your request — are retrieved and injected. Unrelated facts (what the market in the eastern quarter looks like, how trade guilds work) stay in storage.

**Example — what gets retrieved for "Elena finds the letter":**
```
- Elena's mother was a former Archivist who died under suspicious circumstances
- The letter is written in Old Remnish, a language only four people can read
- The Archivist first appeared during the Winter Festival, warning Elena obliquely
- The Order controls the city's water supply and uses it to enforce compliance
- Elena has been unable to read Old Remnish since her accident at age twelve
```

These five facts were retrieved out of potentially hundreds because they're the ones actually relevant to this scene. The lore about trade guilds wasn't retrieved because it has nothing to do with the letter.

---

### Episodic memory — "what has happened in the story so far"

This stores summaries of significant narrative events — what happened, when, and why it mattered. Like semantic memory, it's retrieved selectively rather than dumped wholesale.

**Human analogy:** Remembering the plot of a book you read six months ago. You don't recall every sentence, but you remember the key turning points — the betrayal, the discovery, the confrontation.

**What gets stored:** After every AI response, a background process reads what was just written and generates a 1–2 sentence summary if a meaningful narrative event occurred. That summary is stored with an embedding so it can be found later.

**Example — what gets retrieved for "Elena confronts the Archivist":**
```
- Elena discovered that Marcus was an informant for the Order after she found the coded ledger
- The burning of the East Tower was not an accident — Elena narrowly escaped when the Order realized she had been there
- The Archivist warned Elena at their first meeting that the past has teeth
```

These past events give the AI narrative continuity — it knows what Elena has already been through and can write the confrontation with that weight behind it.

---

## How a request flows through the brain

When you send a message (say: "Write the scene where Elena finally opens the letter"), here is what happens:

```
1. Your message arrives at POST /agent/execute

2. Brain.assemble() runs — all four layers fetched in parallel:
   ├── Working memory → fetched (always)
   ├── Procedural memory → fetched (always)
   ├── Semantic memory → top 5 facts by similarity to your message
   └── Episodic memory → top 3 past events by similarity to your message

3. The assembled prompt looks like this:

   === WRITER STYLE & PREFERENCES ===
   Tone: melancholic, restrained
   Style: terse sentences, sparse dialogue
   POV: third-person limited
   Genre: gothic fantasy
   Rules: - The villain's motivations are always sympathetic
          - Magic is never explained, only experienced

   === CURRENT SCENE ===
   Scene: Elena enters the abandoned library
   Active characters: Elena, The Archivist
   Recent events:
   - Marcus revealed he was working for the Order
   - Elena escaped the burning tower
   Mood: dread with a thread of desperate hope

   === RELEVANT FACTS & LORE ===
   - Elena's mother was a former Archivist who died under suspicious circumstances
   - The letter is written in Old Remnish, a language only four people can read
   - The Archivist first appeared during the Winter Festival, warning Elena obliquely
   - Elena has been unable to read Old Remnish since her accident at age twelve
   - The Order controls the city's water supply

   === RELEVANT PAST EVENTS ===
   - Elena discovered Marcus was an informant after finding the coded ledger
   - The East Tower burning was not an accident — Elena narrowly escaped

   === CURRENT REQUEST ===
   Write the scene where Elena finally opens the letter

4. The LLM generates a response using all of this context.

5. The response is sent to you immediately.

6. In the background (without slowing you down), Brain.reflect() runs:
   ├── Working memory → updated: scene is now "Elena reads the letter", mood shifts
   ├── Procedural → checked for style signals, updated if patterns detected
   ├── Semantic → new facts extracted: "The letter reveals Elena's mother faked her death"
   └── Episodic → event summarized: "Elena read the letter and learned her mother is alive"
```

The next time you ask about the story, the brain already knows Elena read the letter, what she learned, and how the scene felt. You don't have to tell it.

---

## What the brain learns over time

The brain gets more useful the longer you use it. Here is what accumulates:

| After this happens | The brain stores this |
|---|---|
| AI writes a chapter | New facts about characters, events summarized in episodic memory |
| You write in a consistent style | Procedural memory updated with detected patterns |
| AI describes a new location | Location facts stored in semantic memory |
| A plot twist is revealed | Event summary stored in episodic memory |
| A character's backstory is mentioned | Character fact stored in semantic memory |

Over a full novel, the brain builds up a rich, searchable map of your story that the AI uses to stay consistent — without you having to manually manage any of it.

---

## Cost and infrastructure

The brain adds zero additional cost:

| Component | Technology | Cost |
|---|---|---|
| Embeddings | `sentence-transformers all-MiniLM-L6-v2` (open source, runs in the container) | $0 |
| Vector search | numpy cosine similarity in Python (no external service) | $0 |
| Storage | Firestore (already in use, free tier) | $0 |
| Async reflection | FastAPI `BackgroundTasks` (built into the framework) | $0 |
| LLM for extraction | Google AI Studio free tier (already in use) | free tier |

The embedding model is ~80 MB and is baked into the Docker image at build time so there is no download delay on startup. Per-story memory stays small enough (typically under 1 MB) that Firestore free tier easily covers it.

---

## Developer reference

### Public API

```python
from agents.storyAgent.brain import Brain, BrainConfig, ReflectionInput

brain = Brain(
    config=BrainConfig(user_id="u1", context_id="story-123", project_id="my-project"),
    llm_provider=llm_provider,
    embedder=embedder,   # SentenceTransformer instance, shared
    db=db_client,        # firestore.Client, shared
)

# Build layered prompt
assembled = await brain.assemble("Write the next scene", action_hint="chatWithContext")
# assembled.text → full prompt string with all memory sections

# Update memory in background
ri = ReflectionInput(
    user_message="Write the next scene",
    assistant_response=llm_response,
    assembled_prompt=assembled,
)
await brain.reflect(ri)  # never raises
```

### Wiring into a new tool

See [how-to-add-new-tool.md](./how-to-add-new-tool.md) — the "Optional: brain memory integration" section has a copy-paste pattern for adding brain context to any tool.

Key points:
- `StoryAgent` owns the shared `_embedder`, `_db`, and `_llm_provider` — instantiated once, injected into every `Brain`
- `_make_brain(user_id, context_id)` is the factory method on `StoryAgent`
- If `_embedder` is `None` (package not installed), brain retrieval is silently skipped and the tool falls back to the legacy context string
- `Brain.reflect()` must be called via `BackgroundTasks` so it never delays the response

### Firestore data model

```
users/{user_id}/
  procedural_memory/global          → {tone, style, pov, preferences}

stories/{context_id}/
  working_memory/state              → {current_scene, active_characters, recent_events, mood}
  procedural_memory/context         → {genre, narrative_rules, story_level_prefs}
  semantic_memory/{auto_id}         → {text, type, data, embedding:[float×384], created_at}
  episodic_memory/{auto_id}         → {text, summary, embedding:[float×384], created_at}
```

### Source files

| File | Responsibility |
|---|---|
| `brain/brain.py` | Public `Brain` class |
| `brain/types.py` | All dataclasses |
| `brain/engine/router.py` | Routing decisions per action (brainstorm skips semantic/episodic) |
| `brain/engine/assembler.py` | Builds the layered prompt string |
| `brain/engine/reflector.py` | Four concurrent background extractions |
| `brain/memory/working.py` | Working memory CRUD |
| `brain/memory/procedural.py` | Procedural memory CRUD (global + context merge) |
| `brain/memory/semantic.py` | Semantic memory store + cosine similarity retrieval |
| `brain/memory/episodic.py` | Episodic memory store + cosine similarity retrieval |

### Tests

```bash
pytest tests/test_brain_assembler.py   # prompt assembly, all sections, edge cases
pytest tests/test_brain_router.py      # routing decisions for different action hints
pytest tests/test_brain_cosine.py      # cosine similarity correctness
pytest tests/test_brain_reflector.py   # reflection pipeline with mocked LLM
```
