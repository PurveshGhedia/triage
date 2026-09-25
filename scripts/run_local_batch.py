"""Run Agent 1 over a local directory of tile images and persist the results.

Usage:
    python scripts/run_local_batch.py --tiles-dir data/sample_tiles
    python scripts/run_local_batch.py --tiles-dir data/sample_tiles --dry-run
    python scripts/run_local_batch.py --tiles-dir data/sample_tiles --use-llm-selector

By default uses the round-robin placeholder selector (free, fully offline)
and writes each TileAssessment to Firestore. --dry-run prints results instead
of writing anywhere. --use-llm-selector switches to the real Claude-on-Vertex
selector (one LLM call per tool-selection decision — small but nonzero cost;
needs `gcloud auth application-default login` and .env configured).

Optional geolocation: pass --geo-file pointing at a JSON file shaped like
    {"tile_001": {"latitude": 12.3, "longitude": 45.6}, ...}
Tiles not listed there get no geolocation — fine for local testing now, but
Agent 2's adjacency graph will need it later.

IMPORTANT — run this as a module, from the repo root, not as a bare script:
    python -m scripts.run_local_batch --tiles-dir data/sample_tiles --dry-run
Running it directly (`python scripts/run_local_batch.py ...`) will fail with
ModuleNotFoundError, since Python then only puts scripts/ on the import path,
not the repo root where agents/tools/shared live.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agents.perception.agent import PerceptionAgent, round_robin_selector
from agents.perception.schemas import TileStatus
from tools.classifier.client import ClassifierClient
from tools.detector.client import DetectorClient
from tools.sam.client import SAMClient

TILE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def find_tiles(tiles_dir: Path) -> list[Path]:
    return sorted(p for p in tiles_dir.iterdir() if p.suffix.lower() in TILE_EXTENSIONS)


def load_geo(geo_file: Path | None) -> dict:
    if geo_file is None:
        return {}
    return json.loads(geo_file.read_text())


def build_agent(use_llm_selector: bool) -> PerceptionAgent:
    tools = [ClassifierClient(), DetectorClient(), SAMClient()]
    if use_llm_selector:
        from agents.perception.llm_selector import make_llm_tool_selector
        from shared.claude_vertex_client import ClaudeVertexClient

        selector = make_llm_tool_selector(ClaudeVertexClient())
        return PerceptionAgent(tools=tools, tool_selector=selector)
    return PerceptionAgent(tools=tools, tool_selector=round_robin_selector)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiles-dir", type=Path, required=True)
    parser.add_argument("--geo-file", type=Path, default=None)
    parser.add_argument("--use-llm-selector", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.tiles_dir.is_dir():
        sys.exit(f"Not a directory: {args.tiles_dir}")

    tiles = find_tiles(args.tiles_dir)
    if not tiles:
        sys.exit(
            f"No tile images found in {args.tiles_dir} (looked for {TILE_EXTENSIONS})")

    geo = load_geo(args.geo_file)
    agent = build_agent(args.use_llm_selector)

    store = None
    if not args.dry_run:
        from shared.firestore_client import TileAssessmentStore

        store = TileAssessmentStore()

    assessed, escalated = 0, 0
    for tile_path in tiles:
        tile_id = tile_path.stem
        location = geo.get(tile_id, {})
        assessment = agent.process_tile(
            tile_id=tile_id,
            tile_path=str(tile_path),
            latitude=location.get("latitude"),
            longitude=location.get("longitude"),
        )

        if assessment.status == TileStatus.ASSESSED:
            assessed += 1
        else:
            escalated += 1

        print(
            f"{tile_id}: {assessment.status.value}"
            f" (severity={assessment.damage_severity}, confidence={assessment.confidence},"
            f" tool_calls={len(assessment.tool_call_history)})"
        )

        if store is not None:
            store.save(assessment)

    print(
        f"\nDone: {assessed} assessed, {escalated} escalated, {len(tiles)} total.")
    if args.dry_run:
        print("(dry run — nothing written to Firestore)")


if __name__ == "__main__":
    main()
