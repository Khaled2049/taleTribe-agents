from __future__ import annotations

import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

LOCK_PATH = Path(__file__).resolve().parent.parent / "poetry.lock"


def batches(lock_path: Path) -> list[list[str]]:
    packages = tomllib.loads(lock_path.read_text())["package"]
    result: list[list[str]] = []
    counts: dict[str, int] = {}
    for package in packages:
        index = counts.get(package["name"], 0)
        counts[package["name"]] = index + 1
        if index == len(result):
            result.append([])
        result[index].append(f"{package['name']}=={package['version']}")
    return result


def main() -> int:
    failed = False
    with tempfile.TemporaryDirectory() as tmp:
        for number, requirements in enumerate(batches(LOCK_PATH)):
            path = Path(tmp) / f"batch-{number}.txt"
            path.write_text("\n".join(requirements) + "\n")
            print(f"Auditing {len(requirements)} locked packages (batch {number})")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip_audit",
                    "--requirement",
                    str(path),
                    "--no-deps",
                    "--disable-pip",
                    "--progress-spinner",
                    "off",
                ],
                check=False,
            )
            failed = failed or completed.returncode != 0
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
