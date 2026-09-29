"""SYS-01: a dispatched re-inspection must actually re-enter Agent 1's
tool-selection loop and persist an updated assessment — not just log that a
request happened.
"""
from __future__ import annotations

from agents.perception.agent import PerceptionAgent
from agents.perception.schemas import TileStatus
from shared.live_reinspection_dispatcher import LiveReinspectionDispatcher
from shared.tool_interface import ToolClient, ToolResult


class FakeTool(ToolClient):
    def __init__(self, name: str, confidence: float, label: str = "major-damage") -> None:
        self.name = name
        self._confidence = confidence
        self._label = label
        self.call_count = 0

    def run(self, tile_path: str) -> ToolResult:
        self.call_count += 1
        return ToolResult(tool_name=self.name, confidence=self._confidence, label=self._label)


class FakeTileStore:
    """Minimal dict-backed stand-in for TileAssessmentStore -- dispatch() only
    needs .get(tile_id) -> dict | None and .save(assessment) -> None.
    """

    def __init__(self, initial: dict[str, dict]) -> None:
        self._data = dict(initial)
        self.saved: list = []

    def get(self, tile_id: str) -> dict | None:
        return self._data.get(tile_id)

    def save(self, assessment) -> None:
        self.saved.append(assessment)
        self._data[assessment.tile_id] = assessment.model_dump(mode="python")


def test_dispatch_reruns_agent_1_and_saves_the_updated_result():
    # Original assessment had low confidence; the re-inspection tool run
    # returns a high-confidence result, simulating a genuinely improved read.
    store = FakeTileStore(
        {
            "tile_a": {
                "tile_id": "tile_a",
                "tile_path": "fake/tile_a.png",
                "latitude": 29.5,
                "longitude": -95.1,
                "status": "assessed",
                "damage_severity": "minor-damage",
                "confidence": 0.4,
                "tool_call_history": [],
            }
        }
    )
    high_confidence_tool = FakeTool("classifier", confidence=0.95, label="destroyed")
    agent = PerceptionAgent(tools=[high_confidence_tool])
    dispatcher = LiveReinspectionDispatcher(agent=agent, tile_store=store)

    dispatcher.dispatch("tile_a", reason="conflicting signal with neighbor")

    assert dispatcher.dispatched == [("tile_a", "conflicting signal with neighbor")]
    assert high_confidence_tool.call_count == 1  # Agent 1 genuinely ran again
    assert len(store.saved) == 1
    updated = store.saved[0]
    assert updated.tile_id == "tile_a"
    assert updated.status == TileStatus.ASSESSED
    assert updated.damage_severity == "destroyed"  # the corrected reading
    assert updated.confidence == 0.95
    # geolocation carried over from the original record
    assert updated.latitude == 29.5
    assert updated.longitude == -95.1


def test_dispatch_handles_missing_tile_gracefully():
    store = FakeTileStore({})
    agent = PerceptionAgent(tools=[FakeTool("classifier", confidence=0.9)])
    dispatcher = LiveReinspectionDispatcher(agent=agent, tile_store=store)

    dispatcher.dispatch("tile_missing", reason="test")

    assert dispatcher.dispatched == [("tile_missing", "test")]
    assert store.saved == []  # nothing to re-run, nothing saved, no crash


def test_dispatch_handles_missing_tile_path_gracefully():
    store = FakeTileStore(
        {
            "tile_no_path": {
                "tile_id": "tile_no_path",
                "tile_path": None,
                "status": "assessed",
                "tool_call_history": [],
            }
        }
    )
    tool = FakeTool("classifier", confidence=0.9)
    agent = PerceptionAgent(tools=[tool])
    dispatcher = LiveReinspectionDispatcher(agent=agent, tile_store=store)

    dispatcher.dispatch("tile_no_path", reason="test")

    assert tool.call_count == 0  # never even tried -- nothing to reprocess
    assert store.saved == []
