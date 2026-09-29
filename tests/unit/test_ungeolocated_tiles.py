"""PLN-04 (extended): tiles the planner cannot place in a zone are surfaced for
manual review, never silently dropped — whether Agent 1 escalated them or they
were assessed normally but simply have no geolocation.
"""
from __future__ import annotations

import pytest

from agents.perception.schemas import TileAssessment, TileStatus
from agents.planner.agent import TriagePlanner
from agents.planner.schemas import ZoneStatus
from shared.reinspection_dispatcher import LocalReinspectionDispatcher


@pytest.fixture(autouse=True)
def stub_scoring(monkeypatch):
    monkeypatch.setattr("agents.planner.agent.score_zone", lambda tiles: (2.0, 0.5, 2.5))
    monkeypatch.setattr("agents.planner.agent.find_conflicting_pairs", lambda graph, by_id: [])


def tile(tile_id, lat=None, lon=None, status=TileStatus.ASSESSED):
    return TileAssessment(
        tile_id=tile_id,
        latitude=lat,
        longitude=lon,
        status=status,
        damage_severity="major-damage" if status == TileStatus.ASSESSED else None,
        confidence=0.9 if status == TileStatus.ASSESSED else None,
    )


def plan(tiles):
    return TriagePlanner(dispatcher=LocalReinspectionDispatcher()).plan(tiles)


def by_zone_id(rankings):
    return {r.zone_id: r for r in rankings}


def test_assessed_but_ungeolocated_tile_is_surfaced_not_dropped():
    zones = by_zone_id(plan([tile("t1", 10.0, 10.0), tile("no_geo")]))

    ranking = zones["zone-ungeolocated-no_geo"]
    assert ranking.status == ZoneStatus.NEEDS_MANUAL_REVIEW
    assert ranking.tile_ids == ["no_geo"]
    assert ranking.ungeolocated_tile_ids == ["no_geo"]
    assert ranking.escalated_tile_ids == []  # Agent 1 did NOT escalate it
    assert ranking.priority_score == 0.0


def test_escalated_and_ungeolocated_tile_records_both_reasons():
    zones = by_zone_id(plan([tile("esc_no_geo", status=TileStatus.ESCALATED)]))

    ranking = zones["zone-ungeolocated-esc_no_geo"]
    assert ranking.status == ZoneStatus.NEEDS_MANUAL_REVIEW
    assert ranking.escalated_tile_ids == ["esc_no_geo"]
    assert ranking.ungeolocated_tile_ids == ["esc_no_geo"]


def test_tile_with_only_one_coordinate_counts_as_ungeolocated():
    zones = by_zone_id(plan([tile("half_lat", lat=10.0), tile("half_lon", lon=10.0)]))

    assert set(zones) == {"zone-ungeolocated-half_lat", "zone-ungeolocated-half_lon"}


def test_geolocated_tiles_still_cluster_normally_alongside():
    tiles = [tile("a", 10.0, 10.0), tile("b", 10.001, 10.0), tile("no_geo")]
    zones = by_zone_id(plan(tiles))

    clustered = zones["zone-a"]
    assert clustered.status == ZoneStatus.RANKED
    assert clustered.tile_ids == ["a", "b"]
    assert clustered.ungeolocated_tile_ids == []


def test_every_input_tile_appears_in_some_ranking():
    tiles = [
        tile("geo1", 10.0, 10.0),
        tile("geo2", 50.0, 50.0),
        tile("no_geo"),
        tile("esc_no_geo", status=TileStatus.ESCALATED),
    ]
    surfaced = {tid for r in plan(tiles) for tid in r.tile_ids}
    assert surfaced == {t.tile_id for t in tiles}


def test_ungeolocated_zones_sort_after_ranked_zones():
    rankings = plan([tile("no_geo"), tile("geo", 10.0, 10.0)])
    assert rankings[0].zone_id == "zone-geo"
    assert rankings[-1].zone_id == "zone-ungeolocated-no_geo"
