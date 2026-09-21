"""Unit tests for the LLM-directed tool selector — entirely offline, using a
scripted fake ReasoningClient so these never hit a real API (no cost, fully
deterministic, and they run without any GCP/Anthropic credentials configured).
"""
from __future__ import annotations

import json

from agents.perception.llm_selector import make_llm_tool_selector
from agents.perception.schemas import ToolCallRecord
from shared.reasoning_client import ReasoningClient
from shared.tool_interface import ToolClient, ToolResult


class FakeTool(ToolClient):
    description = "fake tool for testing"

    def __init__(self, name: str) -> None:
        self.name = name

    def run(self, tile_path: str) -> ToolResult:
        return ToolResult(tool_name=self.name, confidence=0.5, label="x")


class ScriptedReasoningClient(ReasoningClient):
    """Returns pre-scripted responses in order, one per call."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self._responses.pop(0)


def test_llm_selector_picks_the_tool_named_in_the_response():
    client = ScriptedReasoningClient(
        [json.dumps({"next_tool": "sam", "reasoning": "classifier and detector both struggled with occlusion"})]
    )
    selector = make_llm_tool_selector(client)
    tools = [FakeTool("classifier"), FakeTool("detector"), FakeTool("sam")]
    history = [
        ToolCallRecord(tool_name="classifier", confidence=0.3, label="x"),
        ToolCallRecord(tool_name="detector", confidence=0.4, label="x"),
    ]

    tool, reasoning = selector(history, tools)

    assert tool.name == "sam"
    assert "occlusion" in reasoning


def test_llm_selector_rejects_an_already_tried_tool_even_if_the_model_asks():
    # Model incorrectly names a tool already tried -> validation rejects it,
    # selector falls back to a genuinely untried tool rather than trusting the
    # LLM blindly (this is what keeps PER-01 true regardless of model output).
    client = ScriptedReasoningClient([json.dumps({"next_tool": "classifier", "reasoning": "..."})])
    selector = make_llm_tool_selector(client)
    tools = [FakeTool("classifier"), FakeTool("detector")]
    history = [ToolCallRecord(tool_name="classifier", confidence=0.3, label="x")]

    tool, reasoning = selector(history, tools)

    assert tool.name == "detector"
    assert "fallback" in reasoning


def test_llm_selector_handles_malformed_json_gracefully():
    client = ScriptedReasoningClient(["not valid json at all"])
    selector = make_llm_tool_selector(client)
    tools = [FakeTool("classifier"), FakeTool("detector")]

    tool, reasoning = selector([], tools)

    assert tool.name == "classifier"  # falls back to the first untried tool
    assert "fallback" in reasoning


def test_llm_selector_returns_none_when_all_tools_exhausted():
    client = ScriptedReasoningClient([])  # should never be called
    selector = make_llm_tool_selector(client)
    tools = [FakeTool("classifier")]
    history = [ToolCallRecord(tool_name="classifier", confidence=0.3, label="x")]

    tool, reasoning = selector(history, tools)

    assert tool is None
    assert reasoning is None
