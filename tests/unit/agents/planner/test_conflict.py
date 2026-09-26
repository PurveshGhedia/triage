"""PLN-03: two adjacent tiles with sharply conflicting damage signals are
flagged for re-inspection rather than silently averaged.
"""
from __future__ import annotations

from agents.perception.schemas import TileAssessment, TileStatus, ToolCallRecord
from agents.planner.adjacency import build_adjacency_graph
from agents.planner.conflict import find_conflicting_pairs


def make_tile(tile_id: str, lat: float, lon: float, label: str) -> TileAssessment:
    return TileAssessment(
        tile_id=tile_id,
        latitude=lat,
        longitude=lon,
        status=TileStatus.ASSESSED,
        damage_severity=label,
        confidence=0.9,
        tool_call_history=[ToolCallRecord(tool_name="classifier", confidence=0.9, label=label)],
    )


def test_sharply_disagreeing_adjacent_tiles_are_flagged():
    tile_a = make_tile("tile_a", 29.5000, -95.1000, label="destroyed")
    tile_b = make_tile("tile_b", 29.5004, -95.1000, label="no-damage")  # adjacent, ~44m

    graph = build_adjacency_graph([tile_a, tile_b], adjacency_radius_m=500.0)
    conflicts = find_conflicting_pairs(graph, {"tile_a": tile_a, "tile_b": tile_b})

    assert ("tile_a", "tile_b") in conflicts or ("tile_b", "tile_a") in conflicts


def test_adjacent_tiles_with_similar_damage_are_not_flagged():
    tile_a = make_tile("tile_a", 29.5000, -95.1000, label="major-damage")
    tile_b = make_tile("tile_b", 29.5004, -95.1000, label="destroyed")  # adjacent, close severity

    graph = build_adjacency_graph([tile_a, tile_b], adjacency_radius_m=500.0)
    conflicts = find_conflicting_pairs(graph, {"tile_a": tile_a, "tile_b": tile_b})

    assert conflicts == []


def test_non_adjacent_disagreeing_tiles_are_never_compared():
    """Tiles far enough apart to have no graph edge shouldn't be flagged at
    all, however different their damage — they're not each other's neighbors.
    """
    tile_a = make_tile("tile_a", 29.5000, -95.1000, label="destroyed")
    tile_b = make_tile("tile_b", 29.6000, -95.1000, label="no-damage")  # ~11km away

    graph = build_adjacency_graph([tile_a, tile_b], adjacency_radius_m=500.0)
    conflicts = find_conflicting_pairs(graph, {"tile_a": tile_a, "tile_b": tile_b})

    assert conflicts == []
