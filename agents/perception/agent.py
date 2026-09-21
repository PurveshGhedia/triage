"""Agent 1 — Perception: per-tile tool-selection loop with confidence-driven retry.

The `tool_selector` is where the real agentic decision happens (see project
design principles: LLM-directed, not rule-based). `round_robin_selector` below
is a placeholder — swap in `agents.perception.llm_selector.make_llm_tool_selector`
via `tool_selector=` for the real design; the loop itself doesn't change.

A selector returns (next_tool, reasoning): `reasoning` is optional free text
explaining the pick, captured on the resulting ToolCallRecord for audit trails
(SYS-02). Returns (None, None) once no untried tool remains.
"""
from __future__ import annotations

from collections.abc import Callable

from agents.perception.schemas import TileAssessment, TileStatus, ToolCallRecord
from shared.tool_interface import ToolClient, ToolResult

CONFIDENCE_THRESHOLD = 0.75

ToolSelector = Callable[
    [list[ToolCallRecord], list[ToolClient]], "tuple[ToolClient | None, str | None]"
]


def round_robin_selector(
    history: list[ToolCallRecord], available_tools: list[ToolClient]
) -> tuple[ToolClient | None, str | None]:
    """Placeholder selector: try each untried tool once, in a fixed order.

    Guarantees PER-01 (never retries the same tool) but is NOT the final
    design — replace with an LLM-directed selector (see llm_selector.py)
    before this counts as agentic tool selection.
    """
    tried = {record.tool_name for record in history}
    for tool in available_tools:
        if tool.name not in tried:
            return tool, None
    return None, None  # all tools exhausted


class PerceptionAgent:
    def __init__(
        self,
        tools: list[ToolClient],
        tool_selector: ToolSelector = round_robin_selector,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        if not tools:
            raise ValueError("PerceptionAgent requires at least one tool")
        self.tools = tools
        self.tool_selector = tool_selector
        self.confidence_threshold = confidence_threshold

    def process_tile(
        self,
        tile_id: str,
        tile_path: str,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> TileAssessment:
        history: list[ToolCallRecord] = []
        best_result: ToolResult | None = None

        while True:
            tool, reasoning = self.tool_selector(history, self.tools)
            if tool is None:
                # PER-03: retries exhausted without reaching threshold -> escalate
                # rather than emitting a final low-confidence assessment.
                return TileAssessment(
                    tile_id=tile_id,
                    latitude=latitude,
                    longitude=longitude,
                    status=TileStatus.ESCALATED,
                    damage_severity=None,
                    confidence=None,
                    tool_call_history=history,
                )

            result = tool.run(tile_path)
            history.append(ToolCallRecord.from_tool_result(result, selection_reasoning=reasoning))

            if best_result is None or result.confidence > best_result.confidence:
                best_result = result

            if result.confidence >= self.confidence_threshold:
                # PER-02: accept immediately, no unnecessary extra tool calls.
                break

        return TileAssessment(
            tile_id=tile_id,
            latitude=latitude,
            longitude=longitude,
            status=TileStatus.ASSESSED,
            damage_severity=best_result.label,
            confidence=best_result.confidence,
            tool_call_history=history,
        )
