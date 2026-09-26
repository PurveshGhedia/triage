"""Zone scoring — combines damage severity with road/building accessibility
(PLN-02).

IMPORTANT: a tile's top-level `damage_severity` field may have come from
*any* of the three Agent 1 tools (classifier, detector, or SAM) — whichever
one happened to clear the confidence threshold first. Only classifier/SAM
labels are actually damage readings; detector labels describe road/building
access, not damage. So scoring pulls damage and accessibility signals
separately from each tile's full `tool_call_history`, rather than trusting
the single top-level field.
"""
from __future__ import annotations

from agents.perception.schemas import TileAssessment

DAMAGE_LABEL_SCORES: dict[str, float] = {
    # classifier vocabulary
    "no-damage": 0.0,
    "minor-damage": 1.0,
    "major-damage": 2.0,
    "destroyed": 3.0,
    # SAM vocabulary (coarser 3-level scale)
    "partial-damage": 1.5,
    "severe-damage": 3.0,
}

ACCESS_LABEL_PENALTIES: dict[str, float] = {
    "access-clear": 0.0,
    "access-degraded": 1.5,
    "access-blocked": 3.0,
}

DAMAGE_TOOL_NAMES = {"classifier", "sam"}
ACCESS_TOOL_NAME = "detector"


def tile_damage_score(tile: TileAssessment) -> float:
    """Highest-confidence damage signal from classifier/SAM calls in this
    tile's history. 0.0 if the tile never got a damage-specific reading
    (e.g. it was accepted early on a detector call alone).
    """
    damage_calls = [r for r in tile.tool_call_history if r.tool_name in DAMAGE_TOOL_NAMES]
    if not damage_calls:
        return 0.0
    best = max(damage_calls, key=lambda r: r.confidence)
    return DAMAGE_LABEL_SCORES.get(best.label, 0.0)


def tile_access_penalty(tile: TileAssessment) -> float:
    """Highest-confidence accessibility signal from detector calls in this
    tile's history. 0.0 if the detector was never called for this tile.
    """
    access_calls = [r for r in tile.tool_call_history if r.tool_name == ACCESS_TOOL_NAME]
    if not access_calls:
        return 0.0
    best = max(access_calls, key=lambda r: r.confidence)
    return ACCESS_LABEL_PENALTIES.get(best.label, 0.0)


def score_zone(tiles: list[TileAssessment]) -> tuple[float, float, float]:
    """Returns (damage_score, accessibility_penalty, priority_score) for a
    zone. Accessibility can override raw damage (PLN-02): both components sit
    on the same 0-3 scale, so a zone with moderate damage but blocked access
    can out-rank a zone with similar damage and intact access.
    """
    if not tiles:
        return 0.0, 0.0, 0.0

    damage_score = max(tile_damage_score(t) for t in tiles)
    accessibility_penalty = max(tile_access_penalty(t) for t in tiles)
    priority_score = damage_score + accessibility_penalty
    return damage_score, accessibility_penalty, priority_score
