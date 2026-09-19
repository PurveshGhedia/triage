"""Agent 1 (Perception) data schemas — the structured record emitted per tile.

Covers PER-04 (structured output completeness): every assessment must carry
damage severity, confidence, and the full tool-call history, not just a label.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from shared.tool_interface import ToolResult


class TileStatus(str, Enum):
    ASSESSED = "assessed"
    ESCALATED = "escalated"


class ToolCallRecord(BaseModel):
    """One entry in a tile's tool-call history."""

    tool_name: str
    confidence: float
    label: str
    called_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_tool_result(cls, result: ToolResult) -> "ToolCallRecord":
        return cls(tool_name=result.tool_name, confidence=result.confidence, label=result.label)


class TileAssessment(BaseModel):
    """Final Agent 1 output for a single tile — written to Firestore."""

    tile_id: str
    latitude: float | None = None
    longitude: float | None = None
    status: TileStatus
    damage_severity: str | None = None  # None when escalated without a usable result
    confidence: float | None = None
    tool_call_history: list[ToolCallRecord] = Field(default_factory=list)
