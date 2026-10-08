"""Agent 1 — Perception: per-tile tool-selection loop with confidence-driven retry.

The `tool_selector` is where the real agentic decision happens (see project
design principles: LLM-directed, not rule-based). `round_robin_selector` below
is a placeholder — swap in `agents.perception.llm_selector.make_llm_tool_selector`
via `tool_selector=` for the real design; the loop itself doesn't change.

A selector returns (next_tool, reasoning): `reasoning` is optional free text
explaining the pick, captured on the resulting ToolCallRecord for audit trails
(SYS-02). Returns (None, None) once no untried tool remains.

Damage vs access tools: only tools with `provides_damage` (classifier, SAM) can
satisfy the confidence threshold or set a tile's `damage_severity`. Auxiliary
tools (the detector, whose labels describe road access) never end the loop. With
`run_auxiliary_tools=True`, each auxiliary tool is run once after a damage result
is accepted, so every assessed tile carries an accessibility signal for Agent 2.

Selector faults: if `tool_selector` itself raises (e.g. a Gemini 429 or timeout),
that one decision falls back to round-robin instead of failing the tile.

Fault handling (FLT-01 / FLT-02) — a tile must never crash its job:
  * `tile_validator` (optional) is a pre-flight readability check. If it raises,
    the tile is escalated with a `failure_reason` and no tool is called.
    Off by default so unit tests with fake paths keep working; the scripts
    switch it on with shared.tile_validation.validate_tile_image.
  * A tool that raises is retried up to `max_tool_attempts` times with
    exponential backoff. If it still fails, it is dropped from this tile's
    candidate list and the selector picks an alternate. NotImplementedError
    (e.g. real mode with no endpoint deployed) is permanent, so it is not retried.
  * Every failed attempt is recorded in `tool_errors` and logged, but never put
    in `tool_call_history`, which Agent 2 reads for scoring.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable

from agents.perception.schemas import (
    DEFAULT_BATCH_ID,
    TileAssessment,
    TileStatus,
    ToolCallRecord,
    ToolErrorRecord,
)
from shared.tool_interface import ToolClient, ToolResult

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.75
MAX_TOOL_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 0.5
MAX_ERROR_MESSAGE_CHARS = 500

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
        tile_validator: Callable[[str], None] | None = None,
        max_tool_attempts: int = MAX_TOOL_ATTEMPTS,
        backoff_base_s: float = BACKOFF_BASE_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
        run_auxiliary_tools: bool = False,
    ) -> None:
        if not tools:
            raise ValueError("PerceptionAgent requires at least one tool")
        if max_tool_attempts < 1:
            raise ValueError("max_tool_attempts must be at least 1")
        self.tools = tools
        self.tool_selector = tool_selector
        self.confidence_threshold = confidence_threshold
        self.tile_validator = tile_validator
        self.max_tool_attempts = max_tool_attempts
        self.backoff_base_s = backoff_base_s
        self._sleep = sleep
        self.run_auxiliary_tools = run_auxiliary_tools

    def process_tile(
        self,
        tile_id: str,
        tile_path: str,
        latitude: float | None = None,
        longitude: float | None = None,
        batch_id: str = DEFAULT_BATCH_ID,
    ) -> TileAssessment:
        history: list[ToolCallRecord] = []
        tool_errors: list[ToolErrorRecord] = []
        failed_tools: set[str] = set()
        best_result: ToolResult | None = None  # best DAMAGE result only

        def escalate(failure_reason: str | None) -> TileAssessment:
            return TileAssessment(
                tile_id=tile_id,
                batch_id=batch_id,
                tile_path=tile_path,
                latitude=latitude,
                longitude=longitude,
                status=TileStatus.ESCALATED,
                damage_severity=None,
                confidence=None,
                tool_call_history=history,
                tool_errors=tool_errors,
                failure_reason=failure_reason,
            )

        # FLT-01: an unreadable image is escalated with a reason, never a crash.
        if self.tile_validator is not None:
            try:
                self.tile_validator(tile_path)
            except Exception as exc:
                logger.warning("Tile %s failed validation: %s", tile_id, exc)
                return escalate(f"unreadable tile image: {exc}")

        while True:
            # Tools that failed for good on this tile are hidden from the selector,
            # otherwise it would keep offering a tool that has no result in `history`.
            usable_tools = [t for t in self.tools if t.name not in failed_tools]
            tool, reasoning = self._select_tool(history, usable_tools, tile_id)
            if tool is None:
                # PER-03: retries exhausted without reaching threshold -> escalate
                # rather than emitting a final low-confidence assessment.
                if tool_errors and best_result is None:
                    prefix = "all tools failed" if not history else "no damage tool produced a result"
                    return escalate(f"{prefix}: " + _summarise_failures(tool_errors))
                return escalate(None)

            if tool.name in failed_tools:
                # A selector that re-offers a tool we already gave up on would spin
                # forever; escalate instead of hanging the job.
                logger.error("Selector re-offered failed tool %s for tile %s", tool.name, tile_id)
                return escalate(
                    f"selector re-offered failed tool '{tool.name}': "
                    + _summarise_failures(tool_errors)
                )

            result = self._run_tool_with_retry(tool, tile_id, tile_path, tool_errors)
            if result is None:
                failed_tools.add(tool.name)  # FLT-02: fall back to an alternate tool
                continue

            history.append(ToolCallRecord.from_tool_result(result, selection_reasoning=reasoning))

            if not _provides_damage(tool):
                continue  # auxiliary signal (e.g. road access): never ends the loop

            if best_result is None or result.confidence > best_result.confidence:
                best_result = result

            if result.confidence >= self.confidence_threshold:
                # PER-02: accept immediately, no unnecessary extra damage-tool calls.
                break

        if self.run_auxiliary_tools:
            self._run_remaining_auxiliary_tools(
                history, failed_tools, tile_id, tile_path, tool_errors
            )

        return TileAssessment(
            tile_id=tile_id,
            batch_id=batch_id,
            tile_path=tile_path,
            latitude=latitude,
            longitude=longitude,
            status=TileStatus.ASSESSED,
            damage_severity=best_result.label,
            confidence=best_result.confidence,
            tool_call_history=history,
            tool_errors=tool_errors,
        )

    def _select_tool(
        self,
        history: list[ToolCallRecord],
        usable_tools: list[ToolClient],
        tile_id: str,
    ) -> tuple[ToolClient | None, str | None]:
        """Ask the selector for the next tool; if the selector itself blows up
        (LLM timeout, quota, auth), degrade to round-robin for this decision.
        """
        try:
            return self.tool_selector(history, usable_tools)
        except Exception as exc:
            logger.warning(
                "Tool selector failed for tile %s (%s: %s); falling back to round-robin",
                tile_id, type(exc).__name__, str(exc)[:200],
            )
            tool, _ = round_robin_selector(history, usable_tools)
            return tool, f"fallback (selector error: {type(exc).__name__})"

    def _run_remaining_auxiliary_tools(
        self,
        history: list[ToolCallRecord],
        failed_tools: set[str],
        tile_id: str,
        tile_path: str,
        tool_errors: list[ToolErrorRecord],
    ) -> None:
        """Run each not-yet-run auxiliary tool once. Failures are audited in
        tool_errors but never fail the tile: the access signal is a bonus.
        """
        tried = {r.tool_name for r in history} | failed_tools
        for tool in self.tools:
            if _provides_damage(tool) or tool.name in tried:
                continue
            result = self._run_tool_with_retry(tool, tile_id, tile_path, tool_errors)
            if result is None:
                continue
            history.append(
                ToolCallRecord.from_tool_result(
                    result, selection_reasoning="auxiliary signal: run once after damage assessment"
                )
            )

    def _run_tool_with_retry(
        self,
        tool: ToolClient,
        tile_id: str,
        tile_path: str,
        tool_errors: list[ToolErrorRecord],
    ) -> ToolResult | None:
        """Run `tool`, retrying transient failures with exponential backoff.
        Returns None if the tool never produced a result. Every failed attempt is
        appended to `tool_errors` and logged.
        """
        for attempt in range(1, self.max_tool_attempts + 1):
            try:
                return tool.run(tile_path)
            except NotImplementedError as exc:
                self._record_failure(tool, tile_id, attempt, exc, tool_errors)
                return None  # permanent: retrying cannot help
            except Exception as exc:
                self._record_failure(tool, tile_id, attempt, exc, tool_errors)
                if attempt < self.max_tool_attempts:
                    self._sleep(self.backoff_base_s * 2 ** (attempt - 1))
        return None

    @staticmethod
    def _record_failure(
        tool: ToolClient,
        tile_id: str,
        attempt: int,
        exc: Exception,
        tool_errors: list[ToolErrorRecord],
    ) -> None:
        message = str(exc)[:MAX_ERROR_MESSAGE_CHARS]
        logger.warning(
            "Tool %s failed on tile %s (attempt %d): %s: %s",
            tool.name, tile_id, attempt, type(exc).__name__, message,
        )
        tool_errors.append(
            ToolErrorRecord(
                tool_name=tool.name,
                attempt=attempt,
                error_type=type(exc).__name__,
                message=message,
            )
        )


def _provides_damage(tool: ToolClient) -> bool:
    # getattr keeps duck-typed test doubles (no attribute) behaving as damage tools.
    return getattr(tool, "provides_damage", True)


def _summarise_failures(tool_errors: list[ToolErrorRecord]) -> str:
    tools = sorted({e.tool_name for e in tool_errors})
    return f"{len(tool_errors)} failed attempt(s) across {', '.join(tools)}"
