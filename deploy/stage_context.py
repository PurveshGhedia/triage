"""Assemble the Docker build context in deploy/_context from only what the services need.

    python -m deploy.stage_context
    python -m deploy.stage_context --disasters hurricane-harvey santa-rosa-wildfire

Copying selected folders (rather than uploading the repo) means the multi-gigabyte xBD data, run logs,
Firestore credentials and test files can never end up in the image. A size guard stops the build if
the context is unexpectedly large.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

CODE_DIRS = ("agents", "shared", "tools", "scripts", "dashboard")
DEMO_ITEMS = ("images", "pre", "labels")
DEMO_FILES = ("geo.json", "demo.json", "footprints.json")
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def stage(repo: Path, out: Path, disasters: list[str] | None = None, max_mb: int = 800) -> dict:
    """Build `out` from `repo`. Returns {"disasters": [...], "cache_entries": n, "size_mb": x}."""
    for name in ("Dockerfile", "requirements.txt"):
        if not (repo / "deploy" / name).is_file():
            raise FileNotFoundError(f"deploy/{name} is missing"
                                    + (" (run: python deploy/freeze_requirements.py)" if name == "requirements.txt" else ""))
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for name in CODE_DIRS:
        if (repo / name).is_dir():
            shutil.copytree(repo / name, out / name, ignore=IGNORE)
    shutil.copy2(repo / "deploy" / "Dockerfile", out / "Dockerfile")
    shutil.copy2(repo / "deploy" / "requirements.txt", out / "requirements.txt")

    demo_root = repo / "data" / "demo"
    available = sorted(p.name for p in demo_root.iterdir() if (p / "images").is_dir()) if demo_root.is_dir() else []
    chosen = disasters or available
    unknown = [d for d in chosen if d not in available]
    if unknown:
        raise ValueError(f"no demo data for: {', '.join(unknown)} (have: {', '.join(available) or 'none'})")
    for d in chosen:
        for item in DEMO_ITEMS:
            if (demo_root / d / item).is_dir():
                shutil.copytree(demo_root / d / item, out / "data" / "demo" / d / item, ignore=IGNORE)
        for fname in DEMO_FILES:
            if (demo_root / d / fname).is_file():
                (out / "data" / "demo" / d).mkdir(parents=True, exist_ok=True)
                shutil.copy2(demo_root / d / fname, out / "data" / "demo" / d / fname)

    cache_src = repo / "data" / "cache" / "vision"
    cache_entries = 0
    if cache_src.is_dir():
        (out / "data" / "cache" / "vision").mkdir(parents=True)
        for f in cache_src.glob("*.json"):
            shutil.copy2(f, out / "data" / "cache" / "vision" / f.name)
            cache_entries += 1

    size_mb = dir_size(out) / 1e6
    if size_mb > max_mb:
        biggest = sorted(((dir_size(p), p.name) for p in out.iterdir() if p.is_dir()), reverse=True)[:3]
        raise RuntimeError(f"build context is {size_mb:.0f} MB (limit {max_mb} MB); largest: "
                           + ", ".join(f"{n} {s / 1e6:.0f} MB" for s, n in biggest))
    return {"disasters": chosen, "cache_entries": cache_entries, "size_mb": round(size_mb, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("deploy/_context"))
    parser.add_argument("--disasters", nargs="*", default=None)
    parser.add_argument("--max-mb", type=int, default=800)
    args = parser.parse_args()
    try:
        info = stage(args.repo.resolve(), args.out.resolve(), args.disasters, args.max_mb)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        sys.exit(str(exc))
    print(f"Staged {args.out}: {info['size_mb']} MB; disasters: {', '.join(info['disasters'])}; "
          f"{info['cache_entries']} cached vision answers")


if __name__ == "__main__":
    main()
