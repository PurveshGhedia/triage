"""Builds a tile-adjacency graph purely from geolocation, and clusters tiles
into zones via connected components. Deliberately ignores damage score when
deciding adjacency (PLN-01) — two tiles are adjacent because they are close
in space, never because their damage readings look similar.
"""
from __future__ import annotations

import math

import networkx as nx

from agents.perception.schemas import TileAssessment

EARTH_RADIUS_METERS = 6_371_000


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_METERS * math.asin(math.sqrt(a))


def build_adjacency_graph(
    tiles: list[TileAssessment], adjacency_radius_m: float = 500.0
) -> nx.Graph:
    """One node per geolocated tile; an edge between any two tiles within
    `adjacency_radius_m` of each other. Tiles with no geolocation are excluded
    entirely — they can't be placed in space, so they can't be clustered.

    NOTE: this is an O(n^2) pairwise-distance scan, fine for course-scale
    batches. If tile counts grow large, swap in a spatial index (e.g. a
    KD-tree or grid) rather than rewriting the clustering logic.
    """
    graph = nx.Graph()
    geolocated = [t for t in tiles if t.latitude is not None and t.longitude is not None]

    for tile in geolocated:
        graph.add_node(tile.tile_id)

    for i, tile_a in enumerate(geolocated):
        for tile_b in geolocated[i + 1 :]:
            distance = haversine_distance_m(
                tile_a.latitude, tile_a.longitude, tile_b.latitude, tile_b.longitude
            )
            if distance <= adjacency_radius_m:
                graph.add_edge(tile_a.tile_id, tile_b.tile_id, distance_m=distance)

    return graph


def cluster_into_zones(graph: nx.Graph) -> list[list[str]]:
    """Connected components of the adjacency graph = zones. Each returned list
    is a set of tile_ids that are geographically clustered together.
    """
    return [sorted(component) for component in nx.connected_components(graph)]
