"""Agent 1 (Perception) data schemas — the structured record emitted per tile.

Covers PER-04 (structured output completeness): every assessment must carry
damage severity, confidence, and the full tool-call history, not just a label.
`selection_reasoning` additionally captures *why* each tool was chosen, for
traceability (SYS-02). `tile_path` is stored so a later re-inspection (SYS-01)
can find and reprocess the same image — without it, a dispatched re-inspection
has nothing to actually re-run against. `batch_id` scopes a tile to one disaster
batch so Agent 2 never clusters across batches (SYS-05, E2E-02).
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from shared.tool_interface import ToolResult

# Used when no batch is specified, and for records saved before batch_id existed.
DEFAULT_BATCH_ID = "default"


class TileStatus(str, Enum):
    ASSESSED = "assessed"
    ESCALATED = "escalated"


class ToolCallRecord(BaseModel):
    """One entry in a tile's tool-call history."""

    tool_name: str
    confidence: float
    label: str
    selection_reasoning: str | None = None
    called_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_tool_result(
        cls, result: ToolResult, selection_reasoning: str | None = None
    ) -> "ToolCallRecord":
        return cls(
            tool_name=result.tool_name,
            confidence=result.confidence,
            label=result.label,
            selection_reasoning=selection_reasoning,
        )


class TileAssessment(BaseModel):
    """Final Agent 1 output for a single tile — written to Firestore."""

    tile_id: str
    batch_id: str = DEFAULT_BATCH_ID  # older Firestore docs lack this and load as "default"
    tile_path: str | None = None  # needed so SYS-01 re-inspection can reprocess this tile
    latitude: float | None = None
    longitude: float | None = None
    status: TileStatus
    damage_severity: str | None = None  # None when escalated without a usable result
    confidence: float | None = None
    tool_call_history: list[ToolCallRecord] = Field(default_factory=list)
