"""Run Agent 2 (Triage Planner) over all tile assessments currently in
Firestore, and append the resulting zone rankings to a persistent ledger.

Usage:
    python -m scripts.run_planner_batch
    python -m scripts.run_planner_batch --version 2
    python -m scripts.run_planner_batch --dry-run
    python -m scripts.run_planner_batch --use-bigquery

By default, rankings are appended to a local JSON-Lines file
(data/zone_rankings.jsonl) via LocalZoneRankingStore — this satisfies the
same append-only requirement (PLN-05) as BigQuery would, without needing
BigQuery dataset creation to succeed (which is blocked on some accounts by
an org policy that rejects IAM policy writes even for the project owner).
Pass --use-bigquery to write to the real BigQuery ledger instead, once/if
that's working on your project.

--dry-run prints the rankings instead of writing anywhere. Re-inspection
dispatch uses LocalReinspectionDispatcher (prints requests instead of
publishing anywhere) since real Pub/Sub dispatch isn't wired up yet.

Run as a module, from the repo root:
    python -m scripts.run_planner_batch
"""
from __future__ import annotations

import argparse

from agents.planner.agent import TriagePlanner
from shared.firestore_client import TileAssessmentStore
from shared.reinspection_dispatcher import LocalReinspectionDispatcher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjacency-radius-m", type=float, default=500.0)
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--use-bigquery", action="store_true")
    args = parser.parse_args()

    tile_store = TileAssessmentStore()
    tiles = tile_store.list_all()
    if not tiles:
        print("No tile assessments found in Firestore. Run Agent 1 first.")
        return

    print(f"Loaded {len(tiles)} tile assessment(s) from Firestore.\n")

    dispatcher = LocalReinspectionDispatcher()
    planner = TriagePlanner(dispatcher=dispatcher, adjacency_radius_m=args.adjacency_radius_m)
    rankings = planner.plan(tiles, version=args.version)

    for ranking in rankings:
        print(
            f"{ranking.zone_id}: {ranking.status.value}"
            f" (priority={ranking.priority_score}, damage={ranking.damage_score},"
            f" access_penalty={ranking.accessibility_penalty}, tiles={ranking.tile_ids})"
        )

    if dispatcher.dispatched:
        print(f"\nDispatched {len(dispatcher.dispatched)} re-inspection request(s):")
        for tile_id, reason in dispatcher.dispatched:
            print(f"  - {tile_id}: {reason}")

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
