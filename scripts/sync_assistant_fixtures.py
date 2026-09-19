"""Vendor the canonical fixtures into the consumer repositories, or check drift.

The three services live in three separate git repositories, so there is no
shared package for a consumer to import and no single CI job that sees all of
them. A vendored copy plus a checked-in SHA-256 manifest is the cheapest honest
substitute: ``--check`` fails loudly when a copy has drifted from the canonical
set, which is the failure this arrangement exists to catch.

    python scripts/sync_assistant_fixtures.py            # copy into consumers
    python scripts/sync_assistant_fixtures.py --check    # verify, change nothing

Run from the agents repository. Consumer paths are resolved relative to the
TheTaleTribe workspace root (the parent of this repository's directory), and a
consumer that is not checked out is skipped rather than treated as a failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

AGENTS_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = AGENTS_ROOT.parent
FIXTURE_DIR = AGENTS_ROOT / "assistant" / "fixtures"
MANIFEST = FIXTURE_DIR / "MANIFEST.json"
SCHEMA = AGENTS_ROOT / "assistant" / "schema" / "v1.json"

# Only the frontend. creditProxy is deliberately absent: it owns the *model*
# contract (agents to provider), not the assistant protocol, and its fixtures
# live with that contract in pkg/contracts/testdata/chat. Copying assistant
# events there would imply Go parses them, which it never does.
CONSUMERS = {
    "taleTribe-frontend": {
        "fixtures": Path("packages/assistant-contracts/fixtures"),
        "schema": Path("packages/assistant-contracts/schema/v1.json"),
    },
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical() -> dict[str, str]:
    return {
        path.name: digest(path)
        for path in sorted(FIXTURE_DIR.glob("*.json"))
        if path.name != MANIFEST.name
    }


def write_manifest(entries: dict[str, str]) -> None:
    MANIFEST.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify vendored copies match the manifest; make no changes",
    )
    args = parser.parse_args()

    entries = canonical()
    if not entries:
        print("no fixtures found; run scripts/generate_assistant_fixtures.py first")
        return 1

    failures: list[str] = []

    if args.check:
        recorded = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
        if recorded != entries:
            failures.append(
                "assistant/fixtures/MANIFEST.json is stale; regenerate and re-sync"
            )
    else:
        write_manifest(entries)

    if not SCHEMA.exists():
        failures.append("assistant/schema/v1.json is missing; export it first")

    for repo, paths in CONSUMERS.items():
        target = WORKSPACE_ROOT / repo / paths["fixtures"]
        schema_target = WORKSPACE_ROOT / repo / paths["schema"]
        if not (WORKSPACE_ROOT / repo).is_dir():
            print(f"skip {repo}: not checked out")
            continue
        if args.check:
            for name, expected in entries.items():
                copy = target / name
                if not copy.exists():
                    failures.append(f"{repo}: missing {name}")
                elif digest(copy) != expected:
                    failures.append(f"{repo}: {name} differs from canonical")
            if not schema_target.exists():
                failures.append(f"{repo}: missing assistant schema v1.json")
            elif SCHEMA.exists() and digest(schema_target) != digest(SCHEMA):
                failures.append(
                    f"{repo}: assistant schema v1.json differs from canonical"
                )
        else:
            target.mkdir(parents=True, exist_ok=True)
            schema_target.parent.mkdir(parents=True, exist_ok=True)
            for name in entries:
                shutil.copyfile(FIXTURE_DIR / name, target / name)
            shutil.copyfile(MANIFEST, target / MANIFEST.name)
            shutil.copyfile(SCHEMA, schema_target)
            print(f"synced {len(entries)} fixtures and schema -> {repo}")

    if failures:
        for failure in failures:
            print(f"drift: {failure}", file=sys.stderr)
        return 1
    print("fixtures in sync" if args.check else "sync complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
