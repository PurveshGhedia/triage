"""Write deploy/requirements.txt from the packages installed in THIS environment, so the container
runs exactly the versions the tests ran against.

    python deploy/freeze_requirements.py

Required packages must be installed (exit 1 otherwise); optional ones are included only if present.
The Dockerfile imports the services at build time, so a missing dependency fails the build
immediately instead of failing on the day of the demo.
"""
from __future__ import annotations

import sys
from collections.abc import Callable
from importlib import metadata
from pathlib import Path

REQUIRED = ["google-adk", "a2a-sdk", "google-genai", "google-cloud-firestore", "pydantic", "networkx",
            "fastapi", "uvicorn", "httpx", "pillow"]
OPTIONAL = ["python-dotenv", "pyyaml", "requests", "anthropic", "numpy"]
EXTRAS = {"google-adk": "[a2a]"}          # google-adk needs its a2a extra for the agent-to-agent protocol


def pins(version_fn: Callable[[str], str] = metadata.version,
         required: list[str] = REQUIRED, optional: list[str] = OPTIONAL) -> tuple[list[str], list[str], list[str]]:
    """(requirement lines, missing required packages, skipped optional packages)."""
    lines, missing, skipped = [], [], []
    for name in required:
        try:
            lines.append(f"{name}{EXTRAS.get(name, '')}=={version_fn(name)}")
        except metadata.PackageNotFoundError:
            missing.append(name)
    for name in optional:
        try:
            lines.append(f"{name}{EXTRAS.get(name, '')}=={version_fn(name)}")
        except metadata.PackageNotFoundError:
            skipped.append(name)
    return lines, missing, skipped


def main() -> int:
    lines, missing, skipped = pins()
    if missing:
        print("Missing required packages: " + ", ".join(missing))
        print("Install them (pip install " + " ".join(missing) + ") and run this again.")
        return 1
    out = Path(__file__).parent / "requirements.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out} ({len(lines)} packages)")
    for line in lines:
        print("  " + line)
    if skipped:
        print("Not installed here, so not included: " + ", ".join(skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
