"""Conflict detection between adjacent tiles (PLN-03) — flags disagreements
for a targeted re-inspection dispatch rather than silently averaging scores.
"""
from __future__ import annotations

import networkx as nx

from agents.perception.schemas import TileAssessment
from agents.planner.scoring import tile_damage_score

CONFLICT_THRESHOLD = 2.0  # damage-score points apart, on the 0-3 scale


def find_conflicting_pairs(
    graph: nx.Graph, tiles_by_id: dict[str, TileAssessment]
) -> list[tuple[str, str]]:
    """Returns (tile_id_a, tile_id_b) pairs that are geographically adjacent
    but disagree sharply on damage — e.g. one reads "destroyed" while its
    immediate neighbor reads "no-damage". These are surfaced for a targeted
    re-inspection rather than averaged into a single zone score.
    """
    conflicts = []
    for tile_a_id, tile_b_id in graph.edges():
        tile_a, tile_b = tiles_by_id[tile_a_id], tiles_by_id[tile_b_id]
        score_a, score_b = tile_damage_score(tile_a), tile_damage_score(tile_b)
        if abs(score_a - score_b) >= CONFLICT_THRESHOLD:
            conflicts.append((tile_a_id, tile_b_id))
    return conflicts
