"""One command: Agent 1 over a demo batch -> Agent 2 -> re-inspection rounds ->
re-plan, with every step written to an event log the dashboard can replay.

Usage (run as a module from the repo root):
    python -m scripts.run_e2e --disaster hurricane-harvey
    python -m scripts.run_e2e --disaster hurricane-harvey --use-llm-selector
    python -m scripts.run_e2e --disaster hurricane-harvey --offline     # no Firestore

Reads data/demo/<disaster>/ (made by scripts.select_demo_tiles): the post-disaster
images, geo.json and demo.json (which holds the recommended adjacency radius).
Each run gets its own batch id (<disaster>-<timestamp>) unless you pass --batch-id,
so repeated demo runs never mix versions. Events go to data/runs/<run_id>/events.jsonl.

--offline keeps tile records in memory (no Firestore/network); the zone ledger and
event log are still written to disk, so the dashboard works the same way.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from agents.orchestrator.events import EventLog
from agents.orchestrator.models import TileSpec
from agents.orchestrator.pipeline import RunConfig, new_run_id, run_pipeline
from agents.orchestrator.ports import InProcessPerception, InProcessPlanner
from shared.agent_factory import build_perception_agent, build_second_opinion_agent
from shared.local_zone_store import LocalZoneRankingStore

TILE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def load_specs(demo_dir: Path) -> list[TileSpec]:
    images = sorted(p for p in (demo_dir / "images").iterdir() if p.suffix.lower() in TILE_EXTENSIONS)
    geo_path = demo_dir / "geo.json"
    geo = json.loads(geo_path.read_text()) if geo_path.exists() else {}
    return [
        TileSpec(
            tile_id=p.stem,
            tile_path=str(p),
            latitude=geo.get(p.stem, {}).get("latitude"),
            longitude=geo.get(p.stem, {}).get("longitude"),
        )
        for p in images
    ]


def print_event(event: dict) -> None:
    t = event["type"]
    if t == "run_started":
        print(f"[run] {event['tile_count']} tiles, batch {event['batch_id']}, "
              f"radius {event['adjacency_radius_m']:.0f} m")
    elif t == "tile_finished":
        tile = event["tile"]
        sev = tile["damage_severity"] or "-"
        extra = f"  ({tile['failure_reason']})" if tile["failure_reason"] else ""
        print(f"  [agent 1] {tile['tile_id']}: {tile['status']} {sev} "
              f"conf={tile['confidence']}{extra}")
    elif t == "zones_ranked":
        print(f"[agent 2] version {event['version']}: {len(event['zones'])} zone(s), "
              f"{len(event['reinspection_requests'])} re-inspection request(s)")
        for z in event["zones"][:5]:
            print(f"    {z['zone_id']}: priority {z['priority_score']} "
                  f"(damage {z['damage_score']} + access {z['accessibility_penalty']}) {z['status']}")
    elif t == "reinspection_requested":
        print(f"[loop] round {event['round']}: re-inspecting "
              f"{[r['tile_id'] for r in event['requests']]}")
    elif t == "tile_reinspected":
        b, a = event["before"], event["after"]
        print(f"  [agent 1] {a['tile_id']}: {b['damage_severity'] if b else '?'} -> "
              f"{a['damage_severity']}")
    elif t == "reinspection_no_change":
        print(f"[loop] round {event['round']}: second opinion changed nothing; stopping")
    elif t == "reinspection_exhausted":
        print(f"[loop] cap reached; still unresolved: {event['unresolved']}")
    elif t == "run_finished":
        print(f"[run] done: versions {event['versions']}, {event['zone_count']} zone(s), "
              f"{len(event['escalated_tile_ids'])} escalated tile(s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disaster", required=True)
    parser.add_argument("--demo-dir", type=Path, default=None)
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--radius-m", type=float, default=None)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--per-tile-cap", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--use-llm-selector", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--events-dir", type=Path, default=Path("data/runs"))
    args = parser.parse_args()

    demo_dir = args.demo_dir or Path("data/demo") / args.disaster
    if not (demo_dir / "images").is_dir():
        sys.exit(f"No tiles at {demo_dir}/images. Run: python -m scripts.select_demo_tiles "
                 f"--disaster {args.disaster}")

    meta_path = demo_dir / "demo.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    radius = args.radius_m or meta.get("adjacency_radius_m") or 500.0

    specs = load_specs(demo_dir)
    if not specs:
        sys.exit(f"No tile images in {demo_dir}/images")

    run_id = new_run_id()
    batch_id = args.batch_id or f"{args.disaster}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    config = RunConfig(
        batch_id=batch_id, run_id=run_id, adjacency_radius_m=radius,
        max_rounds=args.rounds, per_tile_cap=args.per_tile_cap,
    )

    if args.offline:
        from shared.memory_store import InMemoryTileStore

        store = InMemoryTileStore(batch_id=batch_id)
    else:
        from shared.firestore_client import TileAssessmentStore

        store = TileAssessmentStore(batch_id=batch_id)

    events_path = args.events_dir / run_id / "events.jsonl"
    events = EventLog(run_id, events_path, listener=print_event)
    perception = InProcessPerception(
        agent=build_perception_agent(args.use_llm_selector),
        reinspection_agent=build_second_opinion_agent(args.use_llm_selector),
        max_workers=args.concurrency,
    )

    run_pipeline(
        specs, config, perception, InProcessPlanner(), store, LocalZoneRankingStore(), events
    )
    print(f"\nEvent log: {events_path}")


if __name__ == "__main__":
    main()
