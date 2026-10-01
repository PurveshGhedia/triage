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
