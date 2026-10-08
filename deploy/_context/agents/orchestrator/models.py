"""Plain data models passed between the orchestrator and the two agents.

They are pydantic models (JSON-serialisable) on purpose: the same objects cross
a function call today and an A2A/HTTP boundary later without changing shape.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from agents.planner.schemas import ZoneRanking


class TileSpec(BaseModel):
    """One tile the orchestrator asks Agent 1 to assess."""

    tile_id: str
    tile_path: str
    latitude: float | None = None
    longitude: float | None = None


class ReinspectionRequest(BaseModel):
    tile_id: str
    reason: str


class PlanOutcome(BaseModel):
    """What Agent 2 hands back: the ranking, plus tiles it wants re-checked."""

    rankings: list[ZoneRanking]
    reinspection_requests: list[ReinspectionRequest] = Field(default_factory=list)


ARRIVAL_ORDERS = ("sweep", "sorted", "shuffle")


def order_for_arrival(specs: list[TileSpec], mode: str = "sweep", seed: int = 7) -> list[TileSpec]:
    """The order tiles 'arrive' in a simulated feed.

    sweep:   north to south, then west to east, like a satellite pass over the area.
    sorted:  by tile id.
    shuffle: a fixed pseudo-random order (same every run, so demos are repeatable).
    Tiles without coordinates go last in 'sweep'.
    """
    if mode == "sorted":
        return sorted(specs, key=lambda s: s.tile_id)
    if mode == "shuffle":
        import random

        ordered = sorted(specs, key=lambda s: s.tile_id)
        random.Random(seed).shuffle(ordered)
        return ordered
    if mode == "sweep":
        located = [s for s in specs if s.latitude is not None and s.longitude is not None]
        unlocated = [s for s in specs if s not in located]
        located.sort(key=lambda s: (-s.latitude, s.longitude, s.tile_id))
        return located + sorted(unlocated, key=lambda s: s.tile_id)
    raise ValueError(f"unknown arrival order {mode!r}; choose from {ARRIVAL_ORDERS}")
