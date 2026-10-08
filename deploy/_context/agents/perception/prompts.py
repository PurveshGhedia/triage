"""Prompts and response parsing for Agent 1's LLM-directed tool selection."""
from __future__ import annotations

import json

from agents.perception.schemas import ToolCallRecord
from shared.tool_interface import ToolClient

SYSTEM_PROMPT = (
    "You are the tool-selection reasoner for a disaster-damage perception agent. "
    "Given the tools already tried on this satellite tile and their results, choose "
    "which untried tool to try next to best assess damage. Respond with ONLY a JSON "
    'object: {"next_tool": "<tool_name>", "reasoning": "<one short sentence>"}. '
    "No markdown, no extra text."
)


def build_user_prompt(history: list[ToolCallRecord], remaining_tools: list[ToolClient]) -> str:
    history_lines = [
        f"- {r.tool_name}: confidence={r.confidence}, label={r.label}" for r in history
    ] or ["(no tools tried yet)"]
    tool_lines = [f"- {t.name}: {getattr(t, 'description', 'no description')}" for t in remaining_tools]
    return (
        "Tool-call history so far:\n"
        + "\n".join(history_lines)
        + "\n\nUntried tools available:\n"
        + "\n".join(tool_lines)
        + "\n\nWhich untried tool should be called next, and why?"
    )


def parse_tool_selection(
    raw_response: str, remaining_tools: list[ToolClient]
) -> tuple[str | None, str | None]:
    """Returns (tool_name, reasoning). On any parsing/validation failure,
    returns (None, raw_response) so the caller can decide how to fall back.
    """
    cleaned = raw_response.strip()
    for fence in ("```json", "```"):
        if cleaned.startswith(fence):
            cleaned = cleaned[len(fence) :]
    cleaned = cleaned.removesuffix("```").strip()

    try:
        parsed = json.loads(cleaned)
        tool_name = parsed.get("next_tool")
        reasoning = parsed.get("reasoning")
    except (json.JSONDecodeError, AttributeError):
        return None, raw_response

    valid_names = {t.name for t in remaining_tools}
    if tool_name not in valid_names:
        return None, raw_response
    return tool_name, reasoning
