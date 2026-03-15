---
name: blog-writer
description: >
  Write engaging, well-structured blog posts that are fun to read and easy to understand —
  even when the topic is technical or complex. Use this skill whenever the user asks to
  write, draft, create, or improve a blog post, article, or written piece for an audience.
  Trigger when the user says things like "write me a blog", "create a post about X",
  "make this concept easy to understand", "explain X for a general audience", or
  "help me write an article". Also trigger for requests to "punch up" or "make more
  engaging" any existing written content. Never use emojis in any output.
---

# Blog Writer Skill

A skill for crafting blog posts that people actually want to read — posts that teach
something real, make hard ideas feel obvious, and leave the reader glad they showed up.

---

## Core Philosophy

Great blog writing is a conversation, not a lecture. The goal is always to make the
reader feel like the smartest person in the room by the time they finish — not to
show off how much the writer knows.

Three commitments guide every post:

1. **Clarity over cleverness.** If a sentence sounds impressive but takes two reads,
   cut it or rephrase it. The reader's time is the most valuable thing on the page.

2. **Earn every paragraph.** Each section should give the reader something they did not
   have before — a new idea, a surprising angle, or a cleaner way to think about something
   they already knew.

3. **Never condescend, never over-explain.** Treat the reader as intelligent but busy.
   They may not know the jargon, but they are perfectly capable of following the logic
   once it is laid out plainly.

---

## Step 1: Understand the Assignment

Before writing a single word, gather this information. Some of it will be obvious from
the user's request; ask only for what is genuinely missing.

| Question                                                   | Why it matters                                                                         |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| What is the topic?                                         | Defines scope.                                                                         |
| Who is the audience?                                       | Sets vocabulary, assumed knowledge, and tone.                                          |
| What is the one thing the reader should walk away knowing? | Prevents the post from trying to say everything.                                       |
| What is the target length?                                 | Typical ranges: short (400-700 words), standard (800-1,400 words), long-form (1,500+). |
| What is the tone?                                          | Conversational and warm? Authoritative and sharp? Dry and witty?                       |
| Are there any hard constraints?                            | Word count limits, no-go topics, required links or CTAs.                               |
| Should emojis be used?                                     | Default is NO. Never use emojis unless the user explicitly requests them.              |

If the user's request answers most of these, skip straight to writing.

---

## Step 2: Choose a Structure

Pick the structure that fits the topic and audience. Do not rigidly apply the same
template to every post.

### The Inverted Pyramid (best for news-adjacent or time-sensitive topics)

Lead with the most important point. Support it. Add context and nuance at the bottom.
Readers who bail early still got the main idea.

### The Problem-Solution Arc (best for how-to, technical, and tutorial posts)

```
Hook (the problem, made vivid)
  -> Why it is hard / why people fail
    -> The insight that changes the frame
      -> The solution, step by step
        -> What success looks like
```

### The Concept Explainer (best for "what is X?" and educational posts)

```
Relatable analogy or story to open
  -> The formal definition, once the reader has a mental model
    -> Why it matters
      -> Common misconceptions
        -> Where to go next
```

### The Opinion / Argument Piece (best for takes, predictions, and essays)

```
Bold claim up front
  -> Acknowledge the strongest counterargument
    -> Evidence for the claim
      -> Nuance and limits
        -> Call to action or open question
```

### The Listicle (use sparingly; best for reference material)

Works when the content is genuinely enumerable — tools, tips, reasons. Avoid when
the numbered structure is just a crutch to avoid real transitions.

---

## Step 3: Write the Hook

The first 2-3 sentences decide whether the reader continues. There are four reliable
hook types — pick the one that fits the post:

- **The vivid scene:** Drop the reader into a specific moment. "It is 2 AM. The deploy
  just failed. The logs say nothing useful."
- **The surprising fact:** Lead with something the reader did not expect to be true.
- **The direct question:** Ask something the reader genuinely wonders about. Make it
  specific, not generic.
- **The bold claim:** State the thesis with confidence. Works when the claim is
  genuinely counter-intuitive or provocative.

Avoid these weak openers:

- "In today's fast-paced world..."
- "Have you ever wondered..."
- "X is a very important topic..."
- Defining a term in the first sentence (define it once the reader has a reason to care)

---

## Step 4: Explain Hard Concepts Simply

This is the core skill of blog writing. Use these techniques when a concept is
technical, abstract, or unfamiliar to the target audience.

### The Analogy Bridge

Find something the reader already understands and map the new concept onto it.
The analogy does not have to be perfect — it just has to carry the reader far enough
to build their own mental model.

> Bad: "A hash function maps data of arbitrary size to fixed-size values."
> Better: "A hash function is like a meat grinder. You can put in a whole chicken or
> a single pea, and what comes out is always the same shape. You cannot reverse the
> grinder to get the original ingredients back."

The best analogies are:

- Drawn from everyday life (cooking, sports, commuting, carpentry)
- Slightly imperfect — and honest about the imperfection
- Specific rather than vague ("like sorting mail" not "like organizing things")

### The Progressive Reveal

Do not front-load all the complexity. Introduce the simple version of an idea first,
let the reader get comfortable, then add the nuance.

1. Lay out the core idea in one sentence.
2. Give an example that makes it concrete.
3. Now introduce the complication or exception.
4. Resolve the complication.

### The "Why Should I Care" Anchor

Before explaining how something works, explain why the reader's life is different
because of it. People learn details better when they are already motivated.

### Concrete Over Abstract

Every abstract claim should have a concrete example within 1-2 sentences.

> Abstract: "Caching improves performance."
> Concrete: "Without caching, every time a user loads your homepage, the server
> re-runs the same database query 50,000 times a day. With caching, it runs once
> and everyone shares the result."

### The One Idea Per Paragraph Rule

Each paragraph should carry exactly one idea. If a paragraph is doing two things,
split it. If it is doing nothing, cut it.

---

## Step 5: Maintain Voice and Flow

### Sentence Variety

Alternate short and long sentences. Short sentences punch. Longer sentences, when
they carry the reader through a chain of connected ideas, give the prose a sense of
momentum that a sequence of short staccato lines can never achieve. Then a short one.

### Transitions That Actually Transition

Avoid transitions that just announce a direction ("Now let's talk about...").
Use transitions that carry an idea forward.

> Weak: "Now let's talk about the downsides."
> Stronger: "That speed comes at a cost."

### Active Voice, Almost Always

Passive voice hides the actor and slows the sentence down. Use it only when the
actor is genuinely unknown or unimportant.

> Passive: "Mistakes were made in the rollout."
> Active: "The team made three mistakes in the rollout."

### Read It Out Loud (or imagine doing so)

If a sentence would sound unnatural spoken aloud, it needs to be rewritten. Blog
prose should feel like a smart friend explaining something, not like a whitepaper.

---

## Step 6: End with Purpose

The conclusion should do one of three things — not all three:

1. **Crystallize the insight.** Restate the main idea in a new way, now that the
   reader has the full context to appreciate it.
2. **Open a door.** Point toward the natural next question or next step, and trust
   the reader to walk through it.
3. **Issue a call to action.** Tell the reader what to do now if the post was
   instructional. Be specific ("try this in your own project this week") rather
   than generic ("I hope you found this useful").

Do not summarize every point from the post. The reader just read it. They remember.

---

## Formatting Rules

These defaults apply unless the user specifies otherwise.

| Element           | Guideline                                                                                         |
| ----------------- | ------------------------------------------------------------------------------------------------- |
| Emojis            | Never. Not in headers, not in bullets, not anywhere.                                              |
| Headers           | Use H2 for major sections, H3 for subsections. Avoid H4+.                                         |
| Bold              | Use to highlight key terms or critical sentences. Do not bold for decoration.                     |
| Italics           | Use for emphasis, titles, or the first use of a technical term.                                   |
| Bullet lists      | Only when the content is genuinely list-like (parallel, discrete items). Avoid bulleting prose.   |
| Code blocks       | Always for code, commands, and filenames. Never inline code for concepts.                         |
| Images / diagrams | Suggest where a diagram would help; describe what it should show.                                 |
| Links             | Suggest where links add value; leave placeholder text for the user to fill in.                    |
| Paragraph length  | 3-5 sentences in most cases. Break anything longer.                                               |
| Reading level     | Aim for clarity sufficient for a smart 16-year-old, unless the audience is explicitly specialist. |

---

## Tone Presets

Match the tone to the audience and platform. Ask the user if unclear.

**Conversational (default)**
Direct, warm, slightly informal. Contractions are fine. First person is fine.
Best for personal blogs, developer-focused posts, startup content.

**Authoritative**
Confident, precise, no hedging without cause. Fewer contractions.
Best for thought leadership, industry publications, company blogs.

**Dry and Witty**
Understated humor. Observations delivered straight-faced. Irony that does not need
to announce itself. Hard to fake — use only if the user's own voice has this quality.

**Educational**
Patient, structured, never condescending. Definitions come after motivation.
Best for tutorials, explainers, documentation-adjacent content.

---

## Quality Checklist

Before finalizing any post, verify:

- [ ] The hook would make a stranger want to keep reading.
- [ ] Every technical term is defined or analogized before being used heavily.
- [ ] No paragraph runs more than 5 sentences without a clear reason.
- [ ] No sentence uses "utilize" when "use" would work.
- [ ] No emojis anywhere in the output.
- [ ] The conclusion does not just repeat the intro.
- [ ] The title is specific enough to set accurate expectations.
- [ ] The post answers the question: "What does the reader now know or feel that they did not before?"

---

## Title Writing

The title is the post's first and most important sentence. Write 3-5 title options
and let the user choose, unless they have already provided one.

Strong title patterns:

- **The Specific How-To:** "How to Cut Your AWS Bill in Half Without Touching Your Architecture"
- **The Honest Explainer:** "What Kubernetes Actually Does (and Why It Took Me Two Years to Get It)"
- **The Counterintuitive Claim:** "Slower Tests Are Usually a Sign of Better Code"
- **The Before/After:** "We Rewrote Our API in Go. Here Is What We Got Wrong."
- **The Number (only when earned):** "Five Reasons Your Database Is Lying to You"

Avoid:

- Clickbait that the post does not deliver on
- Vague titles ("Thoughts on AI", "Some Notes on Distributed Systems")
- Titles that are just the topic with no angle ("React Hooks")

---

## Working With User-Provided Drafts

If the user provides a rough draft to improve:

1. Identify what is already working and preserve it.
2. Diagnose the primary problem (unclear structure, weak hook, jargon overload,
   passive prose, unfocused scope).
3. Fix the structure before fixing the sentences.
4. Do not rewrite the user's voice out of the post — amplify it.
5. Flag any sections where more information is needed to complete the rewrite.
