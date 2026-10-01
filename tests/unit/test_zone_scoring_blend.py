"""Zone scores blend the worst tile with the average tile, so large zones no
longer saturate at the maximum and the ranking separates zones again.
"""
from __future__ import annotations

import pytest

from agents.perception.schemas import TileAssessment, TileStatus, ToolCallRecord
from agents.planner.scoring import MAX_WEIGHT, score_zone, tile_damage_score


def tile(tile_id, damage_label=None, access_label=None):
    history = []
    if damage_label:
        history.append(ToolCallRecord(tool_name="classifier", confidence=0.9, label=damage_label))
    if access_label:
        history.append(ToolCallRecord(tool_name="detector", confidence=0.9, label=access_label))
    return TileAssessment(tile_id=tile_id, status=TileStatus.ASSESSED, tool_call_history=history)


def test_single_tile_zone_scores_exactly_as_before():
    d, a, p = score_zone([tile("t", "major-damage", "access-degraded")])
    assert (d, a, p) == (2.0, 1.5, 3.5)


def test_empty_zone_scores_zero():
    assert score_zone([]) == (0.0, 0.0, 0.0)


def test_one_catastrophic_tile_no_longer_saturates_a_mostly_fine_zone():
    one_bad = [tile("a", "destroyed", "access-blocked")] + [
        tile(f"ok{i}", "no-damage", "access-clear") for i in range(7)
    ]
    d, a, p = score_zone(one_bad)
    assert d < 3.0 and a < 3.0 and p < 6.0
    assert d > 0.0 and a > 0.0  # but the bad tile is still visible


def test_zone_where_everything_is_hit_outranks_zone_with_a_single_bad_tile():
    widespread = [tile(f"w{i}", "destroyed", "access-blocked") for i in range(4)]
    isolated = [tile("x", "destroyed", "access-blocked")] + [
        tile(f"y{i}", "no-damage", "access-clear") for i in range(3)
    ]
    assert score_zone(widespread)[2] > score_zone(isolated)[2]


def test_saturation_is_gone_distinct_zones_get_distinct_scores():
    zones = {
        "light": [tile(f"l{i}", "minor-damage", "access-clear") for i in range(4)],
        "mixed": [tile("m0", "destroyed", "access-clear")] + [tile(f"m{i}", "minor-damage", "access-clear") for i in range(1, 4)],
        "severe": [tile(f"s{i}", "destroyed", "access-degraded") for i in range(4)],
    }
    priorities = {name: score_zone(t)[2] for name, t in zones.items()}
    assert len(set(priorities.values())) == 3
    assert priorities["light"] < priorities["mixed"] < priorities["severe"]


def test_accessibility_still_overrides_raw_damage_in_multi_tile_zones():
    """PLN-02: similar damage, but one zone is cut off -> it ranks higher."""
    cut_off = [tile(f"c{i}", "major-damage", "access-blocked") for i in range(3)]
    reachable = [tile(f"r{i}", "major-damage", "access-clear") for i in range(3)]
    assert score_zone(cut_off)[2] > score_zone(reachable)[2]
    # and blocked access can beat a worse-hit but reachable zone
    worse_but_reachable = [tile(f"w{i}", "destroyed", "access-clear") for i in range(3)]
    assert score_zone(cut_off)[2] > score_zone(worse_but_reachable)[2] - 0.01


def test_blend_formula_matches_the_documented_weights():
    zone = [tile("a", "destroyed"), tile("b", "no-damage")]  # max 3.0, mean 1.5
    d, _, _ = score_zone(zone)
    assert d == pytest.approx(MAX_WEIGHT * 3.0 + (1 - MAX_WEIGHT) * 1.5)


def test_scores_never_exceed_the_per_tile_maximum():
    zone = [tile(f"t{i}", "destroyed", "access-blocked") for i in range(10)]
    d, a, p = score_zone(zone)
    assert (d, a, p) == (3.0, 3.0, 6.0)  # uniform worst-case zone still reaches the ceiling
