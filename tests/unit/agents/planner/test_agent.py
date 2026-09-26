"""Integration-level tests for TriagePlanner — PLN-04 (escalated tiles
surfaced, not dropped) plus confirming conflicts genuinely get dispatched
rather than silently averaged into a zone score.
"""
from __future__ import annotations

from agents.perception.schemas import TileAssessment, TileStatus, ToolCallRecord
from agents.planner.agent import TriagePlanner
from agents.planner.schemas import ZoneStatus
from shared.reinspection_dispatcher import LocalReinspectionDispatcher


def make_assessed(tile_id: str, lat: float, lon: float, label: str) -> TileAssessment:
    return TileAssessment(
        tile_id=tile_id,
        latitude=lat,
        longitude=lon,
        status=TileStatus.ASSESSED,
        damage_severity=label,
        confidence=0.9,
        tool_call_history=[ToolCallRecord(tool_name="classifier", confidence=0.9, label=label)],
    )


def make_escalated(tile_id: str, lat: float | None = None, lon: float | None = None) -> TileAssessment:
    return TileAssessment(
        tile_id=tile_id,
        latitude=lat,
        longitude=lon,
        status=TileStatus.ESCALATED,
        damage_severity=None,
        confidence=None,
        tool_call_history=[
            ToolCallRecord(tool_name="classifier", confidence=0.3, label="minor-damage"),
            ToolCallRecord(tool_name="detector", confidence=0.3, label="access-degraded"),
            ToolCallRecord(tool_name="sam", confidence=0.3, label="partial-damage"),
        ],
    )


def test_zone_containing_an_escalated_tile_is_marked_for_manual_review():
    """PLN-04: the escalated tile's zone must be flagged, not silently
    ranked as if everything in it were successfully assessed.
    """
    escalated = make_escalated("tile_escalated", lat=29.5000, lon=-95.1000)
    neighbor = make_assessed("tile_ok", lat=29.5002, lon=-95.1000, label="minor-damage")

    planner = TriagePlanner(dispatcher=LocalReinspectionDispatcher())
    rankings = planner.plan([escalated, neighbor])

    zone = next(r for r in rankings if "tile_escalated" in r.tile_ids)
    assert zone.status == ZoneStatus.NEEDS_MANUAL_REVIEW
    assert "tile_escalated" in zone.escalated_tile_ids


def test_escalated_tile_with_no_geolocation_is_still_surfaced():
    """PLN-04: a tile that never even got geolocated (so it can't join any
    zone) must still appear somewhere in the output — never dropped.
    """
    escalated = make_escalated("tile_lost", lat=None, lon=None)
    unrelated = make_assessed("tile_other", lat=29.5000, lon=-95.1000, label="no-damage")

    planner = TriagePlanner(dispatcher=LocalReinspectionDispatcher())
    rankings = planner.plan([escalated, unrelated])

    matching = [r for r in rankings if "tile_lost" in r.tile_ids]
    assert len(matching) == 1
    assert matching[0].status == ZoneStatus.NEEDS_MANUAL_REVIEW


def test_conflicting_tiles_are_dispatched_for_reinspection_not_averaged():
    """PLN-03 via the full planner: a sharp disagreement between adjacent
    tiles must trigger an actual dispatch call — not just a quieter number
    buried in an averaged zone score.
    """
    tile_a = make_assessed("tile_a", 29.5000, -95.1000, label="destroyed")
    tile_b = make_assessed("tile_b", 29.5004, -95.1000, label="no-damage")

    dispatcher = LocalReinspectionDispatcher()
    planner = TriagePlanner(dispatcher=dispatcher)
    rankings = planner.plan([tile_a, tile_b])

    dispatched_tile_ids = {tile_id for tile_id, _reason in dispatcher.dispatched}
    assert dispatched_tile_ids == {"tile_a", "tile_b"}

    zone = rankings[0]
    assert ["tile_a", "tile_b"] in zone.conflicting_tile_pairs or [
        "tile_b",
        "tile_a",
    ] in zone.conflicting_tile_pairs


def test_rankings_are_sorted_by_priority_score_descending():
    high_priority = make_assessed("tile_high", 29.5000, -95.1000, label="destroyed")
    low_priority_far_away = make_assessed("tile_low", 30.0000, -96.0000, label="no-damage")

    planner = TriagePlanner(dispatcher=LocalReinspectionDispatcher())
    rankings = planner.plan([high_priority, low_priority_far_away])

    assert rankings[0].priority_score >= rankings[-1].priority_score
