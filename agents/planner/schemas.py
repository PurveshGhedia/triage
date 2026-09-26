"""Agent 2 (Triage Planner) data schemas."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class ZoneStatus(str, Enum):
    RANKED = "ranked"
    NEEDS_MANUAL_REVIEW = "needs_manual_review"


class ZoneRanking(BaseModel):
    """One versioned entry in the append-only zone-ranking ledger (BigQuery).

    A re-inspection updating a zone's score produces a NEW ZoneRanking with an
    incremented `version` — never mutates a prior one (PLN-05).
    """

    zone_id: str
    version: int
    tile_ids: list[str]
    damage_score: float
    accessibility_penalty: float
    priority_score: float
    status: ZoneStatus
    escalated_tile_ids: list[str] = Field(default_factory=list)
    conflicting_tile_pairs: list[list[str]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
