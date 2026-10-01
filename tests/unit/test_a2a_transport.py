"""Agent 1 and Agent 2 over the real A2A protocol (Google ADK), on localhost.

These start genuine A2A servers and call them with ADK's RemoteA2aAgent, so they
exercise the same code path as a deployed run. Skipped if ADK is not installed.
"""
from __future__ import annotations

import pytest

pytest.importorskip("google.adk")
pytest.importorskip("a2a")

# ADK flags its A2A support as experimental on every call; that is noise in test output.
pytestmark = pytest.mark.filterwarnings(r"ignore:\[EXPERIMENTAL\]")

from agents.orchestrator.a2a_client import (  # noqa: E402
    A2APerception,
    A2APlanner,
    ResilientPerception,
    ResilientPlanner,
)
from agents.orchestrator.a2a_services import split_public_url  # noqa: E402
from agents.orchestrator.events import EventLog  # noqa: E402
from agents.orchestrator.local_services import free_port, start_local_services  # noqa: E402
from agents.orchestrator.models import TileSpec  # noqa: E402
from agents.orchestrator.pipeline import RunConfig, run_pipeline  # noqa: E402
from agents.orchestrator.ports import InProcessPerception, InProcessPlanner  # noqa: E402
from agents.perception.agent import PerceptionAgent  # noqa: E402
from agents.perception.schemas import TileAssessment, TileStatus  # noqa: E402
from agents.planner.scoring import tile_damage_score  # noqa: E402
from shared.local_zone_store import LocalZoneRankingStore  # noqa: E402
from shared.memory_store import InMemoryTileStore  # noqa: E402
from shared.tool_interface import ToolClient, ToolResult  # noqa: E402


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
    name = "classifier"

    def __init__(self, labels):
        self.labels = labels

    def run(self, tile_path):
        stem = tile_path.rsplit("/", 1)[-1].removesuffix(".png")
        label, conf = self.labels.get(stem, ("no-damage", 0.9))
        return ToolResult(self.name, conf, label)


def perception(first_labels, second_labels=None):
    first = PerceptionAgent(tools=[MapTool(first_labels)], sleep=lambda s: None)
    second = PerceptionAgent(tools=[MapTool(second_labels or first_labels)], sleep=lambda s: None)
    return InProcessPerception(first, second, max_workers=3)


def specs(n=4):
    return [
        TileSpec(tile_id=f"t{i}", tile_path=f"/x/t{i}.png", latitude=10.0 + i * 0.001, longitude=20.0)
        for i in range(n)
    ]


FIRST = {"t0": ("minor-damage", 0.9), "t1": ("destroyed", 0.9)}   # conflicts with t0
SECOND = {"t0": ("minor-damage", 0.9), "t1": ("major-damage", 0.9)}  # second opinion resolves it


@pytest.fixture(scope="module")
def services():
    svc = start_local_services(perception(FIRST, SECOND))
    yield svc
    svc.stop()


def run_with(perc, plan, tmp_path, tag):
    store = InMemoryTileStore()
    ledger = LocalZoneRankingStore(tmp_path / f"{tag}.jsonl")
    events = EventLog(tag, tmp_path / f"{tag}-events.jsonl")
    result = run_pipeline(specs(), RunConfig(batch_id="A", run_id=tag), perc, plan, store, ledger, events)
    return result, store, ledger, events


# --- the real protocol -----------------------------------------------------------


def test_perception_over_a2a_returns_the_same_assessments_as_in_process(services):
    local = perception(FIRST).assess("A", specs())
    remote = A2APerception(services.perception_url).assess("A", specs())

    key = lambda a: (a.tile_id, a.status, a.damage_severity, a.confidence, a.batch_id)
    assert sorted(map(key, remote)) == sorted(map(key, local))
    assert all(isinstance(a, TileAssessment) for a in remote)


def test_perception_over_a2a_reports_every_tile_through_the_callback(services):
    seen = []
    A2APerception(services.perception_url).assess("A", specs(), on_result=seen.append)
    assert sorted(a.tile_id for a in seen) == ["t0", "t1", "t2", "t3"]


def test_planner_over_a2a_matches_the_in_process_planner(services):
    tiles = perception(FIRST).assess("A", specs())
    local = InProcessPlanner().plan("A", tiles, 1, 500.0)
    remote = A2APlanner(services.planner_url).plan("A", tiles, 1, 500.0)

    strip = lambda r: r.model_dump(exclude={"created_at"})
    assert [strip(r) for r in remote.rankings] == [strip(r) for r in local.rankings]
    assert [r.tile_id for r in remote.reinspection_requests] == [r.tile_id for r in local.reinspection_requests]


def test_reinspection_flag_selects_the_second_opinion_agent_over_a2a(services):
    first = {a.tile_id: a for a in A2APerception(services.perception_url).assess("A", specs())}
    second = {a.tile_id: a for a in A2APerception(services.perception_url).assess("A", specs(), reinspection=True)}
    assert first["t1"].damage_severity == "destroyed"
    assert second["t1"].damage_severity == "major-damage"


def test_full_pipeline_over_a2a_gives_the_same_result_as_in_process(services, tmp_path):
    local = run_with(perception(FIRST, SECOND), InProcessPlanner(), tmp_path, "local")
    remote = run_with(A2APerception(services.perception_url), A2APlanner(services.planner_url), tmp_path, "remote")

    assert remote[0].versions == local[0].versions == [1, 2]
    strip = lambda r: r.model_dump(exclude={"created_at"})
    assert [strip(r) for r in remote[2].read_all()] == [strip(r) for r in local[2].read_all()]
    assert "reinspection_requested" in [e["type"] for e in remote[3].events]


# --- failure handling ---------------------------------------------------------------


def test_unreachable_agents_fall_back_to_in_process_and_say_so(tmp_path):
    dead = f"http://127.0.0.1:{free_port()}"  # nothing listens here
    fallbacks = []
    perc = ResilientPerception(A2APerception(dead), perception(FIRST, SECOND),
                               lambda who, exc: fallbacks.append(who))
    plan = ResilientPlanner(A2APlanner(dead), InProcessPlanner(), lambda who, exc: fallbacks.append(who))

    result, store, ledger, _ = run_with(perc, plan, tmp_path, "dead")

    assert result.versions == [1, 2]            # the run still completed correctly
    assert len(store.list_by_batch("A")) == 4
    assert "perception" in fallbacks and "planner" in fallbacks


def test_unreachable_agent_error_names_the_real_cause():
    dead = f"http://127.0.0.1:{free_port()}"
    with pytest.raises(RuntimeError, match=r"perception_agent returned no text: .*(resolve|initialize)"):
        A2APerception(dead).assess("A", specs())


def test_adk_runner_app_name_warning_is_silenced():
    import logging

    import agents.orchestrator.a2a_services  # noqa: F401  (import installs the filter)

    assert logging.getLogger("google_adk.google.adk.runners").level >= logging.ERROR


def test_error_returned_by_a_remote_agent_is_raised_then_falls_back():
    class Broken:
        def assess(self, *a, **k):
            raise ValueError("model exploded")

    svc = start_local_services(Broken())
    try:
        with pytest.raises(RuntimeError, match="ValueError: model exploded"):
            A2APerception(svc.perception_url).assess("A", specs())

        seen = []
        fallback = ResilientPerception(A2APerception(svc.perception_url), perception(FIRST),
                                       lambda who, exc: seen.append(str(exc)))
        out = fallback.assess("A", specs())
        assert len(out) == 4 and "model exploded" in seen[0]
    finally:
        svc.stop()


# --- public URL for Cloud Run -------------------------------------------------------


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://agent1-abc.a.run.app", ("https", "agent1-abc.a.run.app", 443)),
        ("https://agent1-abc.a.run.app/", ("https", "agent1-abc.a.run.app", 443)),
        ("http://localhost:8101", ("http", "localhost", 8101)),
        ("http://example.com", ("http", "example.com", 80)),
    ],
)
def test_split_public_url(url, expected):
    assert split_public_url(url) == expected


def test_split_public_url_rejects_garbage():
    with pytest.raises(ValueError):
        split_public_url("not a url")
