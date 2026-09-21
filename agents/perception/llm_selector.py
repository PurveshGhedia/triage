"""LLM-directed tool selector — the actual agentic design (see design
principles: the LLM must reason over tool-call history to direct next
actions, not a rule-based workflow). Pass the output of `make_llm_tool_selector`
as `tool_selector=` into PerceptionAgent to replace round_robin_selector.

The LLM always chooses among untried tools only — it decides ORDER/STRATEGY,
not whether to skip a tool outright — so PER-01 (never repeat) and PER-03
(exhaust all before escalating) hold regardless of what the model says. If the
model's output is malformed or names an already-tried/unknown tool, we don't
trust it blindly: we fall back to the first genuinely untried tool and log a
warning, rather than crashing the pipeline or silently repeating a tool.
"""
from __future__ import annotations

import logging

from agents.perception.prompts import SYSTEM_PROMPT, build_user_prompt, parse_tool_selection
from agents.perception.schemas import ToolCallRecord
from shared.reasoning_client import ReasoningClient
from shared.tool_interface import ToolClient

logger = logging.getLogger(__name__)


def make_llm_tool_selector(reasoning_client: ReasoningClient):
    def llm_tool_selector(
        history: list[ToolCallRecord], available_tools: list[ToolClient]
    ) -> tuple[ToolClient | None, str | None]:
        tried = {r.tool_name for r in history}
        remaining = [t for t in available_tools if t.name not in tried]
        if not remaining:
            return None, None

        user_prompt = build_user_prompt(history, remaining)
        raw_response = reasoning_client.complete(SYSTEM_PROMPT, user_prompt)
        tool_name, reasoning = parse_tool_selection(raw_response, remaining)

        if tool_name is None:
            logger.warning(
                "LLM tool selection returned invalid output (%r); falling back to %s",
                reasoning,
                remaining[0].name,
            )
            return remaining[0], f"fallback (invalid LLM output): {reasoning}"

        selected = next(t for t in remaining if t.name == tool_name)
        return selected, reasoning

    return llm_tool_selector
