"""Unit tests for Agent 1's tool-selection loop — maps to PER-01 through PER-04
in TRIAGE_Test_Cases.docx. PER-05 (tile-level fault isolation across parallel
Cloud Run Jobs) is an infra-level concern, not unit-testable here.
"""
from __future__ import annotations

from agents.perception.agent import PerceptionAgent, round_robin_selector
from agents.perception.schemas import TileStatus, ToolCallRecord
from shared.tool_interface import ToolClient, ToolResult


class FakeTool(ToolClient):
    """Deterministic fake tool — no hashing, no I/O, just a fixed confidence."""

    def __init__(self, name: str, confidence: float, label: str = "major-damage") -> None:
        self.name = name
        self._confidence = confidence
        self._label = label
        self.call_count = 0

    def run(self, tile_path: str) -> ToolResult:
        self.call_count += 1
        return ToolResult(tool_name=self.name, confidence=self._confidence, label=self._label)


def test_early_acceptance_on_high_confidence():
    """PER-02: first tool call returns high confidence -> emit immediately, no extra calls."""
    classifier = FakeTool("classifier", confidence=0.9)
    detector = FakeTool("detector", confidence=0.5)
    agent = PerceptionAgent(tools=[classifier, detector])

    assessment = agent.process_tile("tile-1", "fake/path.png")

    assert assessment.status == TileStatus.ASSESSED
    assert assessment.confidence == 0.9
    assert classifier.call_count == 1
    assert detector.call_count == 0


def test_retry_selects_a_different_tool():
    """PER-01: low confidence on tool 1 -> retry uses tool 2, never tool 1 again."""
    classifier = FakeTool("classifier", confidence=0.3)
    detector = FakeTool("detector", confidence=0.9)
    sam = FakeTool("sam", confidence=0.2)
    agent = PerceptionAgent(tools=[classifier, detector, sam])

    assessment = agent.process_tile("tile-2", "fake/path.png")

    assert assessment.status == TileStatus.ASSESSED
    assert [r.tool_name for r in assessment.tool_call_history] == ["classifier", "detector"]
    assert sam.call_count == 0  # never needed once detector cleared the threshold


def test_retry_exhaustion_triggers_escalation():
    """PER-03: all tools stay below threshold -> escalate, don't emit a low-confidence result."""
    tools = [FakeTool(name, confidence=0.3) for name in ("classifier", "detector", "sam")]
    agent = PerceptionAgent(tools=tools)

    assessment = agent.process_tile("tile-3", "fake/path.png")

    assert assessment.status == TileStatus.ESCALATED
    assert assessment.damage_severity is None
    assert len(assessment.tool_call_history) == 3


def test_structured_output_completeness():
    """PER-04: emitted record includes severity, confidence, and full tool-call history."""
    classifier = FakeTool("classifier", confidence=0.8, label="destroyed")
    agent = PerceptionAgent(tools=[classifier])

    assessment = agent.process_tile("tile-4", "fake/path.png", latitude=12.3, longitude=45.6)

    assert assessment.damage_severity == "destroyed"
    assert assessment.confidence == 0.8
    assert len(assessment.tool_call_history) == 1
    assert assessment.tool_call_history[0].tool_name == "classifier"
    assert assessment.latitude == 12.3
    assert assessment.longitude == 45.6


def test_round_robin_selector_never_repeats_a_tried_tool():
    history = [ToolCallRecord(tool_name="classifier", confidence=0.3, label="minor-damage")]
    tools = [FakeTool("classifier", 0.9), FakeTool("detector", 0.9)]

    selected, reasoning = round_robin_selector(history, tools)

    assert selected is not None
    assert selected.name == "detector"
    assert reasoning is None  # round-robin has no reasoning to give — LLM selector does
