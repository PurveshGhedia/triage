"""SYS-05 / E2E-02: no cross-batch data leakage, at the data layer.

Agent 2 only ever plans over one batch; tiles from another batch — even ones
geographically adjacent — must never appear in its zones. Scoring and conflict
detection are stubbed out: these tests are about batch scoping, not scoring.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agents.perception.agent import PerceptionAgent
from agents.perception.schemas import DEFAULT_BATCH_ID, TileAssessment, TileStatus
from agents.planner.agent import TriagePlanner
from shared.batching import MixedBatchError, filter_by_batch, single_batch_id
from shared.reinspection_dispatcher import LocalReinspectionDispatcher


@pytest.fixture(autouse=True)
def stub_scoring(monkeypatch):
    monkeypatch.setattr("agents.planner.agent.score_zone", lambda tiles: (1.0, 0.0, 1.0))
    monkeypatch.setattr("agents.planner.agent.find_conflicting_pairs", lambda graph, by_id: [])


def make_tile(tile_id, batch_id, lat, lon, status=TileStatus.ASSESSED):
    return TileAssessment(
        tile_id=tile_id,
        batch_id=batch_id,
        latitude=lat,
        longitude=lon,
        status=status,
        damage_severity="major-damage" if status == TileStatus.ASSESSED else None,
        confidence=0.9 if status == TileStatus.ASSESSED else None,
    )


def planner():
    return TriagePlanner(dispatcher=LocalReinspectionDispatcher())


# --- schema / helpers ------------------------------------------------------


def test_legacy_record_without_batch_id_loads_as_default():
    legacy = {"tile_id": "old_tile", "status": "assessed", "damage_severity": "minor-damage"}
    assert TileAssessment.model_validate(legacy).batch_id == DEFAULT_BATCH_ID


def test_filter_by_batch_keeps_only_matching_tiles():
    tiles = [
        make_tile("a1", "A", 1.0, 1.0),
        make_tile("b1", "B", 1.0, 1.0),
        make_tile("a2", "A", 1.0, 1.001),
    ]
    assert [t.tile_id for t in filter_by_batch(tiles, "A")] == ["a1", "a2"]
    assert [t.tile_id for t in filter_by_batch(tiles, "B")] == ["b1"]
    assert filter_by_batch(tiles, "nope") == []


def test_single_batch_id_defaults_when_empty_and_rejects_mixed():
    assert single_batch_id([]) == DEFAULT_BATCH_ID
    assert single_batch_id([make_tile("a1", "A", 1.0, 1.0)]) == "A"
    with pytest.raises(MixedBatchError):
        single_batch_id([make_tile("a1", "A", 1.0, 1.0), make_tile("b1", "B", 1.0, 1.0)])


# --- planner ---------------------------------------------------------------


def test_planner_refuses_tiles_from_multiple_batches():
    tiles = [make_tile("a1", "A", 10.0, 10.0), make_tile("b1", "B", 10.0, 10.0)]
    with pytest.raises(MixedBatchError):
        planner().plan(tiles)


def test_adjacent_tiles_from_other_batch_never_join_a_zone():
    # b1 sits ~11 m from a1 — well inside the 500 m adjacency radius.
    all_tiles = [
        make_tile("a1", "A", 10.0000, 10.0000),
        make_tile("a2", "A", 10.0010, 10.0000),
        make_tile("b1", "B", 10.0001, 10.0000),
    ]
    rankings_a = planner().plan(filter_by_batch(all_tiles, "A"))
    rankings_b = planner().plan(filter_by_batch(all_tiles, "B"))

    tiles_in_a = {tid for r in rankings_a for tid in r.tile_ids}
    tiles_in_b = {tid for r in rankings_b for tid in r.tile_ids}
    assert tiles_in_a == {"a1", "a2"}
    assert tiles_in_b == {"b1"}
    assert all(r.batch_id == "A" for r in rankings_a)
    assert all(r.batch_id == "B" for r in rankings_b)


def test_rankings_default_to_default_batch():
    rankings = planner().plan([make_tile("t1", DEFAULT_BATCH_ID, 5.0, 5.0)])
    assert [r.batch_id for r in rankings] == [DEFAULT_BATCH_ID]


# --- perception agent ------------------------------------------------------


class FakeTool:
    def __init__(self, name, confidence):
        self.name = name
        self._confidence = confidence

    def run(self, tile_path):
        return SimpleNamespace(tool_name=self.name, confidence=self._confidence, label="destroyed")


def test_agent_stamps_batch_id_on_assessed_and_escalated_tiles():
    assessed = PerceptionAgent(tools=[FakeTool("classifier", 0.95)]).process_tile(
        "t1", "/x/t1.png", batch_id="A"
    )
    escalated = PerceptionAgent(tools=[FakeTool("classifier", 0.1)]).process_tile(
        "t2", "/x/t2.png", batch_id="A"
    )
    assert assessed.status == TileStatus.ASSESSED and assessed.batch_id == "A"
    assert escalated.status == TileStatus.ESCALATED and escalated.batch_id == "A"


def test_agent_batch_id_defaults_when_omitted():
    result = PerceptionAgent(tools=[FakeTool("classifier", 0.95)]).process_tile("t1", "/x/t1.png")
    assert result.batch_id == DEFAULT_BATCH_ID
