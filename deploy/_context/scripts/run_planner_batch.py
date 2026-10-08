"""Run Agent 2 (Triage Planner) over ONE batch of tile assessments in
Firestore, append the resulting zone rankings to a persistent ledger, and
actually close the re-inspection loop (SYS-01): a dispatched re-inspection
re-runs Agent 1's tool-selection loop on that tile for real, and the updated
result is written back to Firestore immediately.

Usage:
    python -m scripts.run_planner_batch
    python -m scripts.run_planner_batch --batch-id guatemala-volcano
    python -m scripts.run_planner_batch              # auto: next version for the batch
    python -m scripts.run_planner_batch --expected-tiles 50 --wait-timeout-s 300
    python -m scripts.run_planner_batch --dry-run
    python -m scripts.run_planner_batch --use-bigquery
    python -m scripts.run_planner_batch --use-llm-selector

--batch-id picks which batch to plan (default: "default"). Only that batch's
tiles are loaded, so clustering never mixes batches (SYS-05).

By default, rankings are appended to a local JSON-Lines file
(data/zone_rankings.jsonl) via LocalZoneRankingStore. Pass --use-bigquery to
write to the real BigQuery ledger instead, once/if that's working on your
project.

Any re-inspection this run triggers reruns Agent 1 with the round-robin
placeholder selector by default (free, offline); pass --use-llm-selector to
use the real Gemini-directed selector instead (small Vertex AI cost).

Run as a module, from the repo root:
    python -m scripts.run_planner_batch
"""
from __future__ import annotations

import argparse

from agents.perception.agent import PerceptionAgent
from agents.perception.schemas import DEFAULT_BATCH_ID
from agents.planner.agent import TriagePlanner
from shared.agent_factory import build_perception_agent
from shared.batch_wait import wait_for_batch
from shared.firestore_client import TileAssessmentStore
from shared.live_reinspection_dispatcher import LiveReinspectionDispatcher
from shared.local_zone_store import LocalZoneRankingStore


def build_reinspection_agent(use_llm_selector: bool) -> PerceptionAgent:
    return build_perception_agent(use_llm_selector)


def _resolve_version(args) -> int:
    """Auto-increment per batch; refuse a manual version that would not be newer
    than what the ledger already holds (it would break version ordering)."""
    if args.use_bigquery:
        return args.version or 1
    ledger = LocalZoneRankingStore()
    latest = ledger.latest_version(args.batch_id)
    if args.version is None:
        return latest + 1
    if args.version <= latest:
        raise SystemExit(
            f"Refusing --version {args.version}: batch '{args.batch_id}' already has "
            f"version {latest} in the ledger. Omit --version to auto-increment."
        )
    return args.version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjacency-radius-m", type=float, default=500.0)
    parser.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
    parser.add_argument(
        "--version", type=int, default=None,
        help="Ledger version to write. Default: the batch's latest version + 1.",
    )
    parser.add_argument(
        "--expected-tiles", type=int, default=None,
        help="How many tiles this batch should contain. If fewer have reported, the "
             "planner waits up to --wait-timeout-s, then plans with what it has (SYS-04).",
    )
    parser.add_argument("--wait-timeout-s", type=float, default=0.0)
    parser.add_argument("--poll-interval-s", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--use-bigquery", action="store_true")
    parser.add_argument("--use-llm-selector", action="store_true")
    args = parser.parse_args()

    tile_store = TileAssessmentStore(batch_id=args.batch_id)
    readiness = wait_for_batch(
        lambda: tile_store.list_by_batch(args.batch_id),
        expected=args.expected_tiles,
        timeout_s=args.wait_timeout_s,
        poll_interval_s=args.poll_interval_s,
    )
    tiles = readiness.tiles
    if not tiles:
        print(
            f"No tile assessments found in Firestore for batch '{args.batch_id}'. "
            "Run Agent 1 first (with a matching --batch-id)."
        )
        return

    print(f"Batch '{args.batch_id}': loaded {len(tiles)} tile assessment(s) from Firestore.")
    if readiness.incomplete:
        # SYS-04: never block on a tile that isn't coming — plan with what reported.
        print(
            f"WARNING: planning with {len(tiles)} of {readiness.expected} expected tiles "
            f"({readiness.missing} never reported). Their zones are NOT in this ranking."
        )
    print()

    reinspection_agent = build_reinspection_agent(args.use_llm_selector)
    dispatcher = LiveReinspectionDispatcher(agent=reinspection_agent, tile_store=tile_store)
    planner = TriagePlanner(dispatcher=dispatcher, adjacency_radius_m=args.adjacency_radius_m)
    version = _resolve_version(args)
    print(f"Writing ledger version {version} for batch '{args.batch_id}'.\n")
    rankings = planner.plan(tiles, version=version)

    for ranking in rankings:
        extra = ""
        if ranking.ungeolocated_tile_ids:
            extra = f", no geolocation={ranking.ungeolocated_tile_ids}"
        print(
            f"{ranking.zone_id}: {ranking.status.value}"
            f" (priority={ranking.priority_score}, damage={ranking.damage_score},"
            f" access_penalty={ranking.accessibility_penalty}, tiles={ranking.tile_ids}{extra})"
        )

    if dispatcher.dispatched:
        print(f"\nDispatched {len(dispatcher.dispatched)} re-inspection request(s) "
              f"(each was actually re-run against Agent 1 -- see Firestore for updated results):")
        for tile_id, reason in dispatcher.dispatched:
            print(f"  - {tile_id}: {reason}")
        print("Re-scored tiles are saved to Firestore now. Run this command again to "
              "write the next ledger version that reflects the correction.")

    if args.dry_run:
        print(f"\n(dry run — {len(rankings)} zone ranking(s) not written anywhere)")
        return

    if args.use_bigquery:
        from shared.bigquery_client import ZoneRankingStore

        zone_store = ZoneRankingStore()
        zone_store.append_all(rankings)
        print(f"\nWrote {len(rankings)} zone ranking(s) to BigQuery.")
    else:
        from shared.local_zone_store import LocalZoneRankingStore

        zone_store = LocalZoneRankingStore()
        zone_store.append_all(rankings)
        print(f"\nAppended {len(rankings)} zone ranking(s) to {zone_store._path}.")


if __name__ == "__main__":
    main()
