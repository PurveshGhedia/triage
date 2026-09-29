"""SYS-01 x SYS-05: a re-inspected tile must stay in the batch it came from."""
from __future__ import annotations

from types import SimpleNamespace

from agents.perception.agent import PerceptionAgent
from agents.perception.schemas import DEFAULT_BATCH_ID, TileStatus
from shared.live_reinspection_dispatcher import LiveReinspectionDispatcher


class FakeTool:
    name = "classifier"

    def run(self, tile_path):
        return SimpleNamespace(tool_name=self.name, confidence=0.95, label="destroyed")


class DictStore:
    """Just the get()/save() shape the dispatcher needs."""

    def __init__(self, docs):
        self.docs = dict(docs)
        self.saved = []

    def get(self, tile_id):
        return self.docs.get(tile_id)

    def save(self, assessment):
        self.saved.append(assessment)


def dispatcher_for(store):
    agent = PerceptionAgent(tools=[FakeTool()])
    return LiveReinspectionDispatcher(agent=agent, tile_store=store)


def test_reinspected_tile_keeps_its_batch_id():
    store = DictStore({"t1": {"tile_id": "t1", "tile_path": "/x/t1.png", "batch_id": "guatemala"}})
    dispatcher_for(store).dispatch("t1", reason="conflict")

    assert len(store.saved) == 1
    assert store.saved[0].batch_id == "guatemala"
    assert store.saved[0].status == TileStatus.ASSESSED


def test_legacy_doc_without_batch_id_reinspects_into_default_batch():
    store = DictStore({"t1": {"tile_id": "t1", "tile_path": "/x/t1.png"}})
    dispatcher_for(store).dispatch("t1", reason="conflict")

    assert store.saved[0].batch_id == DEFAULT_BATCH_ID


def test_reinspection_preserves_geolocation_alongside_batch():
    doc = {
        "tile_id": "t1",
        "tile_path": "/x/t1.png",
        "batch_id": "A",
        "latitude": 14.5,
        "longitude": -90.8,
    }
    store = DictStore({"t1": doc})
    dispatcher_for(store).dispatch("t1", reason="conflict")

    saved = store.saved[0]
    assert (saved.batch_id, saved.latitude, saved.longitude) == ("A", 14.5, -90.8)


def test_missing_doc_or_missing_tile_path_is_skipped_without_saving():
    store = DictStore({"no_path": {"tile_id": "no_path", "batch_id": "A"}})
    d = dispatcher_for(store)
    d.dispatch("ghost", reason="x")
    d.dispatch("no_path", reason="x")

    assert store.saved == []
    assert [t for t, _ in d.dispatched] == ["ghost", "no_path"]
