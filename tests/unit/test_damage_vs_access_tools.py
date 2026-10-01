"""A detector (road-access) result must never satisfy the damage confidence
threshold, become a tile's damage_severity, or hide a missing damage reading.
"""
from __future__ import annotations

import pytest

from agents.perception.agent import PerceptionAgent
from agents.perception.schemas import TileStatus
from shared.agent_factory import build_perception_agent
from shared.tool_interface import ToolClient, ToolResult
from tools.classifier.client import DAMAGE_LABELS as CLASSIFIER_LABELS
from tools.detector.client import DetectorClient
from tools.sam.client import DAMAGE_LABELS as SAM_LABELS


class Fixed(ToolClient):
    def __init__(self, name, confidence, label, provides_damage=True):
        self.name = name
        self.provides_damage = provides_damage
        self._r = (confidence, label)
        self.calls = 0

    def run(self, tile_path):
        self.calls += 1
        return ToolResult(self.name, *self._r)


def agent(tools, **kw):
    return PerceptionAgent(tools=tools, sleep=lambda s: None, **kw)


def test_detector_has_provides_damage_false_and_others_true():
    assert DetectorClient.provides_damage is False
    assert ToolClient.provides_damage is True


def test_confident_detector_alone_cannot_accept_a_tile():
    classifier = Fixed("classifier", 0.30, "minor-damage")
    detector = Fixed("detector", 0.99, "access-clear", provides_damage=False)
    sam = Fixed("sam", 0.20, "partial-damage")

    result = agent([classifier, detector, sam]).process_tile("t1", "/x/t1.png")

    assert result.status == TileStatus.ESCALATED  # no damage tool reached 0.75
    assert sam.calls == 1  # loop kept going past the confident detector


def test_detector_label_never_becomes_damage_severity():
    classifier = Fixed("classifier", 0.60, "minor-damage")
    detector = Fixed("detector", 0.99, "access-blocked", provides_damage=False)
    sam = Fixed("sam", 0.80, "severe-damage")

    result = agent([classifier, detector, sam]).process_tile("t1", "/x/t1.png")

    assert result.status == TileStatus.ASSESSED
    assert result.damage_severity == "severe-damage"
    assert result.confidence == 0.80


def test_best_damage_result_ignores_a_higher_confidence_detector():
    classifier = Fixed("classifier", 0.80, "major-damage")
    detector = Fixed("detector", 0.99, "access-clear", provides_damage=False)

    result = agent([detector, classifier]).process_tile("t1", "/x/t1.png")

    assert result.damage_severity == "major-damage"
    assert result.confidence == 0.80


def test_auxiliary_tool_runs_once_after_acceptance_when_enabled():
    classifier = Fixed("classifier", 0.90, "destroyed")
    detector = Fixed("detector", 0.50, "access-blocked", provides_damage=False)

    result = agent([classifier, detector], run_auxiliary_tools=True).process_tile("t1", "/x/t1.png")

    assert [r.tool_name for r in result.tool_call_history] == ["classifier", "detector"]
    assert detector.calls == 1
    assert "auxiliary" in result.tool_call_history[1].selection_reasoning


def test_auxiliary_tool_off_by_default_keeps_early_acceptance_minimal():
    classifier = Fixed("classifier", 0.90, "destroyed")
    detector = Fixed("detector", 0.50, "access-blocked", provides_damage=False)

    result = agent([classifier, detector]).process_tile("t1", "/x/t1.png")

    assert [r.tool_name for r in result.tool_call_history] == ["classifier"]
    assert detector.calls == 0


def test_auxiliary_tool_not_rerun_if_selector_already_ran_it():
    classifier = Fixed("classifier", 0.30, "minor-damage")
    detector = Fixed("detector", 0.60, "access-clear", provides_damage=False)
    sam = Fixed("sam", 0.90, "severe-damage")

    result = agent([classifier, detector, sam], run_auxiliary_tools=True).process_tile(
        "t1", "/x/t1.png"
    )

    assert detector.calls == 1
    assert [r.tool_name for r in result.tool_call_history].count("detector") == 1


def test_failing_auxiliary_tool_does_not_fail_the_tile():
    class DownDetector(Fixed):
        def run(self, tile_path):
            raise ConnectionError("down")

    classifier = Fixed("classifier", 0.90, "destroyed")
    detector = DownDetector("detector", 0, "x", provides_damage=False)

    result = agent([classifier, detector], run_auxiliary_tools=True).process_tile("t1", "/x/t1.png")

    assert result.status == TileStatus.ASSESSED
    assert result.damage_severity == "destroyed"
    assert {e.tool_name for e in result.tool_errors} == {"detector"}


def test_all_damage_tools_failing_is_explained_even_if_detector_worked():
    class Down(Fixed):
        def run(self, tile_path):
            raise ConnectionError("down")

    detector = Fixed("detector", 0.95, "access-clear", provides_damage=False)
    tools = [Down("classifier", 0, "x"), detector, Down("sam", 0, "x")]

    result = agent(tools).process_tile("t1", "/x/t1.png")

    assert result.status == TileStatus.ESCALATED
    assert "no damage tool produced a result" in result.failure_reason


def test_real_mock_tools_never_leak_access_labels_into_damage_severity():
    """The exact bug seen on real data (severity=access-clear), over many paths."""
    factory_agent = build_perception_agent()
    factory_agent.tile_validator = None  # fake paths
    damage_vocab = set(CLASSIFIER_LABELS) | set(SAM_LABELS)
    assessed = escalated = 0
    for i in range(200):
        r = factory_agent.process_tile(f"t{i}", f"/fake/tile_{i}.png")
        if r.status == TileStatus.ASSESSED:
            assessed += 1
            assert r.damage_severity in damage_vocab
            assert any(c.tool_name == "detector" for c in r.tool_call_history)
        else:
            escalated += 1
            assert r.damage_severity is None
    assert assessed > 0 and escalated > 0


def test_factory_agent_is_configured_for_production_use():
    a = build_perception_agent()
    assert a.run_auxiliary_tools is True
    assert a.tile_validator is not None
    assert [t.name for t in a.tools] == ["classifier", "detector", "sam"]


# --- selector faults --------------------------------------------------------


def test_selector_exception_falls_back_to_round_robin_instead_of_crashing():
    def exploding_selector(history, tools):
        raise RuntimeError("429 quota exceeded")

    classifier = Fixed("classifier", 0.90, "destroyed")

    result = agent([classifier], tool_selector=exploding_selector).process_tile("t1", "/x/t1.png")

    assert result.status == TileStatus.ASSESSED
    assert "fallback (selector error: RuntimeError)" in result.tool_call_history[0].selection_reasoning


def test_selector_that_always_fails_still_walks_every_tool_then_escalates():
    def exploding_selector(history, tools):
        raise TimeoutError("llm timeout")

    tools = [Fixed("classifier", 0.2, "a"), Fixed("sam", 0.3, "b")]

    result = agent(tools, tool_selector=exploding_selector).process_tile("t1", "/x/t1.png")

    assert result.status == TileStatus.ESCALATED
    assert [r.tool_name for r in result.tool_call_history] == ["classifier", "sam"]
