"""Orchestrator: Agent 1 -> Agent 2 -> bounded re-inspection -> re-plan, with an
event log. Uses real PerceptionAgent / TriagePlanner / scoring with scripted tools;
only the conflict rule is pinned (to the documented >=2.0-apart rule) so the tests
do not depend on how conflict.py is written internally.
"""
from __future__ import annotations

import threading

import pytest

from agents.orchestrator.events import EventLog, read_events
from agents.orchestrator.models import TileSpec
from agents.orchestrator.pipeline import RunConfig, run_pipeline
from agents.orchestrator.ports import InProcessPerception, InProcessPlanner
from agents.perception.agent import PerceptionAgent
from agents.perception.schemas import TileAssessment, TileStatus
from agents.planner.scoring import tile_damage_score
from shared.local_zone_store import LocalZoneRankingStore
from shared.memory_store import InMemoryTileStore
from shared.tool_interface import ToolClient, ToolResult


@pytest.fixture(autouse=True)
def pinned_conflict_rule(monkeypatch):
    def rule(graph, tiles_by_id):
        return [
            tuple(sorted((a, b)))
            for a, b in graph.edges
            if abs(tile_damage_score(tiles_by_id[a]) - tile_damage_score(tiles_by_id[b])) >= 2.0
        ]

    monkeypatch.setattr("agents.planner.agent.find_conflicting_pairs", rule)


class MapTool(ToolClient):
    """Classifier stand-in: label/confidence chosen by the tile's file name."""

    name = "classifier"

    def __init__(self, labels: dict[str, tuple[str, float]], default=("no-damage", 0.9), boom=()):
        self.labels, self.default, self.boom = labels, default, set(boom)

    def run(self, tile_path):
        stem = tile_path.rsplit("/", 1)[-1].removesuffix(".png")
        if stem in self.boom:
            raise RuntimeError("segfault in tool")
        label, conf = self.labels.get(stem, self.default)
        return ToolResult(self.name, conf, label)


def agent(labels, **kw):
    return PerceptionAgent(tools=[MapTool(labels, **kw)], sleep=lambda s: None)


def line_specs(n=4, spacing_deg=0.001):
    """n tiles in a row ~111 m apart: one zone at the default 500 m radius."""
    return [
        TileSpec(tile_id=f"t{i}", tile_path=f"/x/t{i}.png", latitude=10.0 + i * spacing_deg, longitude=20.0)
        for i in range(n)
    ]


def run(specs, first, second=None, *, tmp_path, store=None, **cfg):
    store = store if store is not None else InMemoryTileStore()
    ledger = LocalZoneRankingStore(tmp_path / "ledger.jsonl")
    events = EventLog("run1", tmp_path / "events.jsonl")
    perception = InProcessPerception(first, second or first, max_workers=3)
    config = RunConfig(batch_id=cfg.pop("batch_id", "A"), run_id="run1", **cfg)
    result = run_pipeline(specs, config, perception, InProcessPlanner(), store, ledger, events)
    return result, store, ledger, events


# --- happy path -----------------------------------------------------------------


def test_clean_batch_runs_end_to_end_with_one_ledger_version(tmp_path):
    # neighbours differ by at most one severity tier, so nothing counts as a conflict
    first = agent({"t0": ("no-damage", 0.9), "t1": ("minor-damage", 0.9)},
                  default=("minor-damage", 0.9))
    result, store, ledger, events = run(line_specs(), first, tmp_path=tmp_path)

    assert result.versions == [1]
    assert result.rounds_used == 0
    assert len(store.list_by_batch("A")) == 4
    assert [r.zone_id for r in ledger.read_latest("A")] == ["zone-t0"]
    types = [e["type"] for e in events.events]
    assert types[0] == "run_started" and types[-1] == "run_finished"
    assert types.count("tile_finished") == 4
    assert "reinspection_requested" not in types


# --- re-inspection loop ----------------------------------------------------------


def two_conflicting():
    first = agent({"t0": ("minor-damage", 0.9), "t1": ("destroyed", 0.9)})
    second = agent({"t0": ("minor-damage", 0.9), "t1": ("major-damage", 0.9)})  # opinion differs
    return first, second


def test_conflict_triggers_reinspection_and_a_second_ledger_version(tmp_path):
    first, second = two_conflicting()
    result, store, ledger, events = run(line_specs(2), first, second, tmp_path=tmp_path)

    assert result.versions == [1, 2]
    assert result.reinspected_tile_ids == ["t0", "t1"] or "t1" in result.reinspected_tile_ids
    v1 = [r for r in ledger.read_all() if r.version == 1][0]
    v2 = ledger.read_latest("A")[0]
    assert v1.conflicting_tile_pairs and v1.damage_score == 3.0
    assert v2.version == 2 and v2.conflicting_tile_pairs == [] and v2.damage_score == 2.0
    assert len(ledger.read_all()) == 2  # v1 retained, append-only (PLN-05)
    types = [e["type"] for e in events.events]
    assert "reinspection_requested" in types and "tile_reinspected" in types


def test_reinspection_event_records_before_and_after(tmp_path):
    first, second = two_conflicting()
    *_, events = run(line_specs(2), first, second, tmp_path=tmp_path)

    changed = [e for e in events.events if e["type"] == "tile_reinspected" and e["after"]["tile_id"] == "t1"][0]
    assert changed["before"]["damage_severity"] == "destroyed"
    assert changed["after"]["damage_severity"] == "major-damage"


class FlipTool(ToolClient):
    """Gives a different answer on every call, so each re-inspection 'changes' the tile."""

    name = "classifier"

    def __init__(self):
        self.calls = {}

    def run(self, tile_path):
        stem = tile_path.rsplit("/", 1)[-1].removesuffix(".png")
        n = self.calls[stem] = self.calls.get(stem, 0) + 1
        if stem == "t0":
            return ToolResult(self.name, 0.9, "no-damage")
        # t1 stays in conflict with t0 (>= 2.0 apart) but its confidence keeps changing
        return ToolResult(self.name, round(0.76 + 0.01 * n, 2), "destroyed")


def test_persistent_conflict_with_changing_results_stops_at_max_rounds(tmp_path):
    flipping = PerceptionAgent(tools=[FlipTool()], sleep=lambda s: None)
    result, *_, events = run(line_specs(2), flipping, tmp_path=tmp_path, max_rounds=2, per_tile_cap=5)

    assert result.versions == [1, 2, 3]
    assert result.rounds_used == 2
    assert "reinspection_exhausted" in [e["type"] for e in events.events]


def test_second_opinion_that_changes_nothing_stops_the_loop_without_a_new_version(tmp_path):
    same = agent({"t0": ("no-damage", 0.9), "t1": ("destroyed", 0.9)})  # opinion never differs
    result, _, ledger, events = run(line_specs(2), same, tmp_path=tmp_path, max_rounds=5, per_tile_cap=5)

    types = [e["type"] for e in events.events]
    assert result.versions == [1]            # no identical duplicate version written
    assert len(ledger.read_all()) == 1
    assert "reinspection_no_change" in types and "reinspection_exhausted" not in types


def test_per_tile_cap_stops_redispatch_before_max_rounds(tmp_path):
    flipping = PerceptionAgent(tools=[FlipTool()], sleep=lambda s: None)
    result, *_ = run(line_specs(2), flipping, tmp_path=tmp_path, max_rounds=5, per_tile_cap=1)

    assert result.versions == [1, 2]  # each tile re-inspected once, then no eligible tiles


def test_zero_rounds_never_reinspects(tmp_path):
    same = agent({"t0": ("no-damage", 0.9), "t1": ("destroyed", 0.9)})
    result, *_ = run(line_specs(2), same, tmp_path=tmp_path, max_rounds=0)
    assert result.versions == [1] and result.reinspected_tile_ids == []


# --- faults ------------------------------------------------------------------------


def test_crashing_tile_is_isolated_and_planning_still_completes(tmp_path):
    # the tool raising an *unexpected* error is retried/fallen-back inside the agent; to force
    # a crash that escapes the agent we make process_tile itself raise.
    class ExplodingAgent(PerceptionAgent):
        def process_tile(self, tile_id, tile_path, *a, **kw):
            if tile_id == "t1":
                raise MemoryError("worker died")
            return super().process_tile(tile_id, tile_path, *a, **kw)

    first = ExplodingAgent(tools=[MapTool({})], sleep=lambda s: None)
    result, store, ledger, events = run(line_specs(3), first, tmp_path=tmp_path)

    by_id = {a.tile_id: a for a in store.list_by_batch("A")}
    assert by_id["t1"].status == TileStatus.ESCALATED
    assert "agent error: MemoryError" in by_id["t1"].failure_reason
    assert by_id["t0"].status == TileStatus.ASSESSED and by_id["t2"].status == TileStatus.ASSESSED
    zone = ledger.read_latest("A")[0]
    assert zone.status.value == "needs_manual_review" and zone.escalated_tile_ids == ["t1"]
    assert result.escalated_tile_ids == ["t1"]


def test_tile_the_agent_never_returned_is_added_as_escalated(tmp_path):
    class DropsOne:
        def assess(self, batch_id, specs, reinspection=False, on_result=None):
            kept = [s for s in specs if s.tile_id != "t2"]
            out = [
                TileAssessment(tile_id=s.tile_id, batch_id=batch_id, status=TileStatus.ASSESSED,
                               damage_severity="no-damage", latitude=s.latitude, longitude=s.longitude)
                for s in kept
            ]
            return out  # note: no per-tile callbacks at all (like a remote transport)

    store = InMemoryTileStore()
    ledger = LocalZoneRankingStore(tmp_path / "ledger.jsonl")
    events = EventLog("r", tmp_path / "e.jsonl")
    result = run_pipeline(line_specs(3), RunConfig(batch_id="A", run_id="r"),
                          DropsOne(), InProcessPlanner(), store, ledger, events)

    ids = {a.tile_id for a in store.list_by_batch("A")}
    assert ids == {"t0", "t1", "t2"}  # nothing silently lost
    assert result.escalated_tile_ids == ["t2"]
    assert "tile_missing" in [e["type"] for e in events.events]


# --- isolation & events ----------------------------------------------------------


def test_other_batches_tiles_never_enter_the_plan(tmp_path):
    store = InMemoryTileStore()
    store.save(TileAssessment(tile_id="intruder", batch_id="B", status=TileStatus.ASSESSED,
                              damage_severity="destroyed", latitude=10.0005, longitude=20.0))
    result, _, ledger, _ = run(line_specs(2), agent({}), tmp_path=tmp_path, store=store)

    planned = {tid for r in ledger.read_all() for tid in r.tile_ids}
    assert "intruder" not in planned
    assert ledger.read_latest("B") == []


def test_event_log_is_ordered_complete_and_replayable(tmp_path):
    first, second = two_conflicting()
    *_, events = run(line_specs(2), first, second, tmp_path=tmp_path)

    on_disk = read_events(tmp_path / "events.jsonl")
    assert [e["seq"] for e in on_disk] == list(range(1, len(on_disk) + 1))
    assert on_disk == [dict(e) for e in events.events]
    assert {e["run_id"] for e in on_disk} == {"run1"}


def test_event_log_is_thread_safe(tmp_path):
    log = EventLog("r", tmp_path / "e.jsonl")
    threads = [threading.Thread(target=lambda: [log.emit("x") for _ in range(100)]) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    seqs = [e["seq"] for e in read_events(tmp_path / "e.jsonl")]
    assert sorted(seqs) == list(range(1, 801))


def test_listener_output_from_many_threads_is_never_interleaved():
    lines = []
    log = EventLog("r", listener=lambda e: lines.append(e["seq"]))
    threads = [threading.Thread(target=lambda: [log.emit("x") for _ in range(50)]) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert lines == sorted(lines) == list(range(1, 301))  # delivered in seq order


def test_listener_receives_every_event(tmp_path):
    seen = []
    log = EventLog("r", listener=seen.append)
    log.emit("a")
    log.emit("b", k=1)
    assert [e["type"] for e in seen] == ["a", "b"] and seen[1]["k"] == 1


def test_in_memory_store_matches_the_firestore_store_contract():
    store = InMemoryTileStore(batch_id="A")
    a = TileAssessment(tile_id="t", batch_id="A", status=TileStatus.ASSESSED)
    b = TileAssessment(tile_id="t", batch_id="B", status=TileStatus.ESCALATED)
    store.save(a); store.save(b)
    assert store.get("t")["status"] == "assessed" and store.get("t", "B")["status"] == "escalated"
    assert [x.batch_id for x in store.list_by_batch("A")] == ["A"]
    assert store.list_batches() == ["A", "B"]
