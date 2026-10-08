"""Agent 2 — Triage Planner: builds a geolocation-based adjacency graph,
clusters tiles into zones, scores zones by damage + accessibility (PLN-02),
flags conflicting adjacent tiles for re-inspection (PLN-03) rather than
averaging them, and surfaces tiles it cannot place in a zone (escalated by
Agent 1, or simply missing geolocation) for manual review (PLN-04) instead of
silently dropping them.

A planning run covers exactly one batch (SYS-05 / E2E-02): handing the planner
tiles from more than one batch raises MixedBatchError.
"""
from __future__ import annotations

from agents.perception.schemas import TileAssessment, TileStatus
from agents.planner.adjacency import build_adjacency_graph, cluster_into_zones
from agents.planner.conflict import find_conflicting_pairs
from agents.planner.schemas import ZoneRanking, ZoneStatus
from agents.planner.scoring import score_zone
from shared.batching import single_batch_id
from shared.reinspection_dispatcher import ReinspectionDispatcher


class TriagePlanner:
    def __init__(
        self,
        dispatcher: ReinspectionDispatcher,
        adjacency_radius_m: float = 500.0,
    ) -> None:
        self.dispatcher = dispatcher
        self.adjacency_radius_m = adjacency_radius_m

    def plan(self, tiles: list[TileAssessment], version: int = 1) -> list[ZoneRanking]:
        batch_id = single_batch_id(tiles)  # SYS-05: refuse to cluster across batches
        tiles_by_id = {t.tile_id: t for t in tiles}
        graph = build_adjacency_graph(tiles, self.adjacency_radius_m)
        zones = cluster_into_zones(graph)
        escalated_ids = {t.tile_id for t in tiles if t.status == TileStatus.ESCALATED}

        rankings: list[ZoneRanking] = []
        for zone_tile_ids in zones:
            zone_tiles = [tiles_by_id[tid] for tid in zone_tile_ids]
            damage_score, accessibility_penalty, priority_score = score_zone(zone_tiles)
            zone_escalated = [tid for tid in zone_tile_ids if tid in escalated_ids]

            # Conflicts are only meaningful between tiles that are actually
            # adjacent — use the subgraph induced by this zone's tiles rather
            # than re-scanning every tile pair in the batch.
            zone_subgraph = graph.subgraph(zone_tile_ids)
            conflicts = find_conflicting_pairs(zone_subgraph, tiles_by_id)
            for tile_a_id, tile_b_id in conflicts:
                self.dispatcher.dispatch(
                    tile_a_id, reason=f"Conflicting damage signal with adjacent tile {tile_b_id}"
                )
                self.dispatcher.dispatch(
                    tile_b_id, reason=f"Conflicting damage signal with adjacent tile {tile_a_id}"
                )

            status = ZoneStatus.NEEDS_MANUAL_REVIEW if zone_escalated else ZoneStatus.RANKED

            rankings.append(
                ZoneRanking(
                    zone_id=f"zone-{min(zone_tile_ids)}",
                    batch_id=batch_id,
                    version=version,
                    tile_ids=zone_tile_ids,
                    damage_score=damage_score,
                    accessibility_penalty=accessibility_penalty,
                    priority_score=priority_score,
                    status=status,
                    escalated_tile_ids=zone_escalated,
                    conflicting_tile_pairs=[list(pair) for pair in conflicts],
                )
            )

        # Tiles with no geolocation never entered the adjacency graph, so they are
        # in no zone above. PLN-04 says escalated tiles must be surfaced; the same
        # logic applies to assessed-but-ungeolocated tiles — dropping them would
        # hide a real assessment from the responder. Each gets its own zone entry
        # flagged for manual review, with the reason traceable via
        # ungeolocated_tile_ids (and escalated_tile_ids when Agent 1 escalated it).
        zoned_tile_ids = {tid for zone in zones for tid in zone}
        ungeolocated_ids = {
            t.tile_id
            for t in tiles
            if (t.latitude is None or t.longitude is None) and t.tile_id not in zoned_tile_ids
        }
        for tile_id in sorted(ungeolocated_ids):
            rankings.append(
                ZoneRanking(
                    zone_id=f"zone-ungeolocated-{tile_id}",
                    batch_id=batch_id,
                    version=version,
                    tile_ids=[tile_id],
                    damage_score=0.0,
                    accessibility_penalty=0.0,
                    priority_score=0.0,
                    status=ZoneStatus.NEEDS_MANUAL_REVIEW,
                    escalated_tile_ids=[tile_id] if tile_id in escalated_ids else [],
                    ungeolocated_tile_ids=[tile_id],
                )
            )

        rankings.sort(key=lambda r: r.priority_score, reverse=True)
        return rankings
