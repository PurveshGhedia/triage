"""PLN-05: a re-inspection updating a zone's score must produce a NEW version
in the ledger, with the prior version retained, not overwritten.
"""
from __future__ import annotations

from agents.planner.schemas import ZoneRanking, ZoneStatus
from shared.local_zone_store import LocalZoneRankingStore


def make_ranking(zone_id: str, version: int, priority: float) -> ZoneRanking:
    return ZoneRanking(
        zone_id=zone_id,
        version=version,
        tile_ids=["tile_a"],
        damage_score=priority,
        accessibility_penalty=0.0,
        priority_score=priority,
        status=ZoneStatus.RANKED,
    )


def test_appending_a_new_version_keeps_the_old_one(tmp_path):
    store = LocalZoneRankingStore(path=tmp_path / "zones.jsonl")

    store.append(make_ranking("zone-a", version=1, priority=2.0))
    store.append(make_ranking("zone-a", version=2, priority=3.5))  # re-inspection result

    all_rows = store.read_all()
    versions = {r.version for r in all_rows if r.zone_id == "zone-a"}

    assert versions == {1, 2}  # BOTH versions present — v1 was never overwritten
    assert len(all_rows) == 2


def test_append_all_writes_every_ranking(tmp_path):
    store = LocalZoneRankingStore(path=tmp_path / "zones.jsonl")

    store.append_all(
        [
            make_ranking("zone-a", version=1, priority=1.0),
            make_ranking("zone-b", version=1, priority=2.0),
        ]
    )

    assert len(store.read_all()) == 2


def test_read_all_on_nonexistent_file_returns_empty_list(tmp_path):
    store = LocalZoneRankingStore(path=tmp_path / "does_not_exist.jsonl")
    assert store.read_all() == []
