"""PLN-02: a moderate-damage zone bounded by blocked access ranks above an
isolated zone with similar damage but intact access.
"""
from __future__ import annotations

from agents.perception.schemas import TileAssessment, TileStatus, ToolCallRecord
from agents.planner.scoring import score_zone, tile_access_penalty, tile_damage_score


def test_damage_score_reads_only_classifier_and_sam_calls_not_detector():
    """A tile accepted early on a detector call alone (no classifier/SAM call
    at all) has no genuine damage reading yet — score should be 0.0, not
    whatever the detector's access label happens to map to.
    """
    tile = TileAssessment(
        tile_id="tile_a",
        status=TileStatus.ASSESSED,
        damage_severity="access-clear",  # this is an ACCESS label, not damage
        confidence=0.9,
        tool_call_history=[
            ToolCallRecord(tool_name="detector", confidence=0.9, label="access-clear")
        ],
    )
    assert tile_damage_score(tile) == 0.0


def test_access_penalty_reads_only_detector_calls():
    tile = TileAssessment(
        tile_id="tile_a",
        status=TileStatus.ASSESSED,
        damage_severity="destroyed",
        confidence=0.9,
        tool_call_history=[
            ToolCallRecord(tool_name="classifier", confidence=0.9, label="destroyed")
        ],
    )
    assert tile_access_penalty(tile) == 0.0  # no detector call happened


def test_blocked_access_zone_outranks_clear_access_zone_at_equal_damage():
    """Both zones have identical major-damage classifier readings; only
    access differs. The blocked-access zone must score higher overall.
    """
    zone_blocked = [
        TileAssessment(
            tile_id="tile_blocked",
            status=TileStatus.ASSESSED,
            damage_severity="major-damage",
            confidence=0.9,
            tool_call_history=[
                ToolCallRecord(tool_name="classifier", confidence=0.9, label="major-damage"),
                ToolCallRecord(tool_name="detector", confidence=0.85, label="access-blocked"),
            ],
        )
    ]
    zone_clear = [
        TileAssessment(
            tile_id="tile_clear",
            status=TileStatus.ASSESSED,
            damage_severity="major-damage",
            confidence=0.9,
            tool_call_history=[
                ToolCallRecord(tool_name="classifier", confidence=0.9, label="major-damage"),
                ToolCallRecord(tool_name="detector", confidence=0.85, label="access-clear"),
            ],
        )
    ]

    damage_blocked, penalty_blocked, priority_blocked = score_zone(zone_blocked)
    damage_clear, penalty_clear, priority_clear = score_zone(zone_clear)

    assert damage_blocked == damage_clear  # same raw damage
    assert penalty_blocked > penalty_clear
    assert priority_blocked > priority_clear
