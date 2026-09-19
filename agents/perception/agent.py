"""Agent 1 — Perception: per-tile tool-selection loop with confidence-driven retry.

NOTE ON AGENTIC DESIGN (see project design principles): the `tool_selector`
passed into PerceptionAgent is where the LLM should reason over tool-call
history to pick the next tool (Option A — LLM-directed, not a rule-based
workflow). The default `round_robin_selector` below is a placeholder so the
loop and its tests can run locally before the Vertex AI LLM call is wired in.
Swap it for a real selector via `tool_selector=` once Claude/Gemini
tool-selection is implemented — the loop itself doesn't change. Do not treat
the placeholder as the final design; it is intentionally not "agentic" yet.
"""
from __future__ import annotations

from collections.abc import Callable

from agents.perception.schemas import TileAssessment, TileStatus, ToolCallRecord
from shared.tool_interface import ToolClient, ToolResult

CONFIDENCE_THRESHOLD = 0.75

ToolSelector = Callable[[list[ToolCallRecord], list[ToolClient]], "ToolClient | None"]


def round_robin_selector(
    history: list[ToolCallRecord], available_tools: list[ToolClient]
) -> ToolClient | None:
    """Placeholder selector: try each untried tool once, in a fixed order.

    Guarantees PER-01 (never retries the same tool) but is NOT the final
    design — replace with an LLM call reasoning over `history` before this
    counts as agentic tool selection.
    """
    tried = {record.tool_name for record in history}
    for tool in available_tools:
        if tool.name not in tried:
            return tool
    return None  # all tools exhausted


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
            tool = self.tool_selector(history, self.tools)
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
            history.append(ToolCallRecord.from_tool_result(result))

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
