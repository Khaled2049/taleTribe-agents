"""Check which specialists the director chooses, against the real model.

Routing is the model's decision, so mocked tests cannot say whether it is any
good. This sends each labelled prompt to the configured provider as the
director's *first step only* -- one model call per prompt, no consults run --
and reports what it asked for. It is opt-in and never part of CI: it needs a
running creditProxy and spends real calls.

    python -m scripts.eval_specialist_routing --user-id <uid>

Run it before enabling ASSISTANT_SPECIALISTS_ENABLED and after changing the
director's prompt or a specialist's description.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Optional

from agents.storyAgent.llm_provider import (
    CreditProxyProvider,
    LLMProviderError,
    _byok_config,
)
from assistant.run import SYSTEM_RULES, _model_tools, _system_prompt

# A roster stands in for a real story: only the routing decision is measured.
ROSTER = """Story: Saltmarsh
Characters: Mina, Tobias, The Harbourmaster
Places: The Lamp Room, The Quay
Plot lines: The Wreck (events: The warning, The storm, The inquest)
Chapters: 1 The Lamp Room, 2 Low Water, 3 The Inquest"""

# (prompt, the specialists a good director would consult; empty means none)
CASES: list[tuple[str, set[str]]] = [
    ("How many chapters are there?", set()),
    ("Who is Tobias?", set()),
    ("Add an abandoned hospital as a location.", set()),
    ("Create a character named Elena.", set()),
    ("Thanks, that helps.", set()),
    ("The middle of my story feels boring.", {"story_architect"}),
    ("What should happen after the inquest?", {"story_architect"}),
    ("The stakes feel low in The Wreck plot line.", {"story_architect"}),
    ("Does The storm follow from The warning?", {"story_architect"}),
    ("Give Mina a stronger motivation.", {"character_editor"}),
    ("Is Tobias consistent between The storm and The inquest?", {"character_editor"}),
    ("Why does the Harbourmaster feel flat?", {"character_editor"}),
    ("Is my twist at the inquest predictable?", {"critic"}),
    ("Give me an honest editorial read of chapter 2.", {"critic"}),
    ("Why does my ending feel weak?", {"critic", "story_architect"}),
    ("Is the dialogue in chapter 1 cliched?", {"critic"}),
    ("Write the scene for The storm from Mina's point of view.", {"drafter"}),
    ("Draft the dialogue between Mina and Tobias at The inquest.", {"drafter"}),
    ("Rewrite chapter 1 in first person.", {"drafter"}),
    (
        "The villain is obvious and the last few events aren't exciting.",
        {"critic", "character_editor", "story_architect"},
    ),
]


class _Roster:
    pool = object()

    async def slim_context(self, story_id: str) -> str:
        return ROSTER


async def _first_step(
    provider: CreditProxyProvider, prompt: str, key: str
) -> list[str]:
    system = await _system_prompt(
        _Roster(),
        "eval",
        edits_enabled=False,
        entity_proposals_enabled=True,
        max_consults=2,
    )
    assert system.startswith(SYSTEM_RULES)
    tools = _model_tools(
        edits_enabled=False, entity_proposals_enabled=True, specialists_enabled=True
    )
    messages = [
        {"role": "system", "parts": [{"type": "text", "text": system}]},
        {"role": "user", "parts": [{"type": "text", "text": prompt}]},
    ]
    calls: dict[int, dict[str, str]] = {}
    async for event in provider.chat_stream(
        messages, tools, max_output_tokens=512, idempotency_key=key
    ):
        if event.get("type") == "tool_call_delta":
            fragment = event.get("tool_call") or {}
            call = calls.setdefault(
                int(fragment.get("index", 0)), {"name": "", "args": ""}
            )
            call["name"] = fragment.get("name") or call["name"]
            call["args"] += fragment.get("arguments_delta") or ""
        elif event.get("type") in ("done", "error"):
            break
    chosen: list[str] = []
    for call in calls.values():
        if call["name"] != "consult_specialist":
            chosen.append(f"({call['name']})")
            continue
        try:
            chosen.append(str(json.loads(call["args"] or "{}").get("specialist")))
        except ValueError:
            chosen.append("(unparseable consult)")
    return chosen


async def main(user_id: str, only: Optional[int], delay: float) -> int:
    provider = CreditProxyProvider(
        os.getenv("CREDIT_PROXY_URL", "http://localhost:8090")
    )
    _byok_config.set({"user_id": user_id, "provider": "", "api_key": "", "model": ""})
    cases = CASES if only is None else CASES[only : only + 1]
    hits = errors = 0
    try:
        for number, (prompt, expected) in enumerate(cases):
            if number:
                # Free-tier providers cap requests per minute; pace, don't burst.
                await asyncio.sleep(delay)
            try:
                chosen = await _first_step(
                    provider, prompt, f"routing-eval:{os.getpid()}:{number}"
                )
            except LLMProviderError as exc:
                errors += 1
                print(f"ERR  {prompt[:58]:58} -> {type(exc).__name__}")
                continue
            consulted = {name for name in chosen if not name.startswith("(")}
            # A subset of the expected set is a pass: fewer consults is cheaper,
            # and an unrelated specialist or a needless consult is the failure.
            ok = consulted <= expected and (bool(consulted) == bool(expected))
            hits += ok
            print(
                f"{'ok  ' if ok else 'MISS'} {prompt[:58]:58} -> "
                f"{', '.join(chosen) or '-'}"
            )
    finally:
        await provider.aclose()
    print(f"\n{hits}/{len(cases)} routed as expected, {errors} provider errors")
    return 0 if hits == len(cases) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", required=True, help="a local user with credits")
    parser.add_argument("--only", type=int, help="run a single case by index")
    parser.add_argument(
        "--delay", type=float, default=5.0, help="seconds to wait between calls"
    )
    arguments = parser.parse_args()
    raise SystemExit(
        asyncio.run(main(arguments.user_id, arguments.only, arguments.delay))
    )
