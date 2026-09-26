"""PLN-01: geographically adjacent tiles are clustered into the same zone,
never grouped merely because their damage readings look similar.
"""
from __future__ import annotations

from agents.perception.schemas import TileAssessment, TileStatus, ToolCallRecord
from agents.planner.adjacency import build_adjacency_graph, cluster_into_zones


def make_tile(tile_id: str, lat: float | None, lon: float | None, label: str) -> TileAssessment:
    return TileAssessment(
        tile_id=tile_id,
        latitude=lat,
        longitude=lon,
        status=TileStatus.ASSESSED,
        damage_severity=label,
        confidence=0.9,
        tool_call_history=[ToolCallRecord(tool_name="classifier", confidence=0.9, label=label)],
    )


def test_close_tiles_with_different_damage_are_still_clustered_together():
    """Two tiles ~50m apart, but with opposite damage readings, should still
    end up in the same zone — adjacency is purely spatial.
    """
    tile_a = make_tile("tile_a", 29.5000, -95.1000, label="destroyed")
    tile_b = make_tile("tile_b", 29.5004, -95.1000, label="no-damage")  # ~44m north

    graph = build_adjacency_graph([tile_a, tile_b], adjacency_radius_m=500.0)
    zones = cluster_into_zones(graph)

    assert zones == [["tile_a", "tile_b"]]


def test_far_apart_tiles_with_identical_damage_are_not_clustered():
    """Two tiles with the same damage label but ~11km apart must NOT be
    grouped just because their scores match — that's the exact failure mode
    PLN-01 guards against.
    """
    tile_a = make_tile("tile_a", 29.5000, -95.1000, label="destroyed")
    tile_b = make_tile("tile_b", 29.6000, -95.1000, label="destroyed")  # ~11km north

    graph = build_adjacency_graph([tile_a, tile_b], adjacency_radius_m=500.0)
    zones = cluster_into_zones(graph)

    assert sorted(zones) == [["tile_a"], ["tile_b"]]


def test_tiles_without_geolocation_are_excluded_from_the_graph():
    tile_a = make_tile("tile_a", 29.5000, -95.1000, label="destroyed")
    tile_b = make_tile("tile_b", None, None, label="destroyed")

    graph = build_adjacency_graph([tile_a, tile_b], adjacency_radius_m=500.0)

    assert "tile_b" not in graph.nodes
    assert "tile_a" in graph.nodes
