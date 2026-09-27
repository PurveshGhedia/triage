"""Unit test for the BigQuery row serialization helper. Only exercises the
pure `_to_bigquery_row` static method — never instantiates a real client, so
this needs no GCP credentials or network access to run.
"""
from __future__ import annotations

from agents.planner.schemas import ZoneRanking, ZoneStatus
from shared.bigquery_client import ZoneRankingStore


def test_to_bigquery_row_flattens_status_and_conflicting_pairs():
    ranking = ZoneRanking(
        zone_id="zone-tile_a",
        version=1,
        tile_ids=["tile_a", "tile_b"],
        damage_score=2.0,
        accessibility_penalty=3.0,
        priority_score=5.0,
        status=ZoneStatus.NEEDS_MANUAL_REVIEW,
        escalated_tile_ids=["tile_a"],
        conflicting_tile_pairs=[["tile_a", "tile_b"]],
    )

    row = ZoneRankingStore._to_bigquery_row(ranking)

    assert row["status"] == "needs_manual_review"
    assert isinstance(row["status"], str)
    assert row["conflicting_tile_pairs"] == ["tile_a|tile_b"]
    assert row["tile_ids"] == ["tile_a", "tile_b"]
    assert isinstance(row["created_at"], str)  # ISO string, not a raw datetime
