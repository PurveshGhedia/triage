"""Per-batch Firestore layout (SYS-05), migration, and per-batch ledger versions
(PLN-05 / SYS-03). Uses an in-memory fake Firestore: no GCP access needed.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agents.perception.schemas import TileAssessment, TileStatus
from agents.planner.schemas import ZoneRanking, ZoneStatus
from scripts.migrate_tiles_to_batches import migrate
from shared.firestore_client import TileAssessmentStore
from shared.local_zone_store import LocalZoneRankingStore


class FakeSnapshot:
    def __init__(self, path, data):
        self.id = path[-1]
        self._data = data
        self.exists = data is not None

    def to_dict(self):
        return dict(self._data) if self._data is not None else None


class FakeDocRef:
    def __init__(self, db, path):
        self.db, self.path = db, path

    def set(self, data, merge=False):
        if merge and self.path in self.db.data:
            data = {**self.db.data[self.path], **data}
        self.db.data[self.path] = dict(data)

    def get(self):
        return FakeSnapshot(self.path, self.db.data.get(self.path))

    def delete(self):
        self.db.data.pop(self.path, None)

    def collection(self, name):
        return FakeCollRef(self.db, self.path + (name,))


class FakeCollRef:
    def __init__(self, db, path):
        self.db, self.path = db, path

    def document(self, doc_id):
        return FakeDocRef(self.db, self.path + (doc_id,))

    def stream(self):
        n = len(self.path)
        return [
            FakeSnapshot(p, d)
            for p, d in sorted(self.db.data.items())
            if len(p) == n + 1 and p[:n] == self.path
        ]


class FakeFirestore:
    def __init__(self):
        self.data = {}

    def collection(self, name):
        return FakeCollRef(self, (name,))


def tile(tile_id, batch_id="default", status=TileStatus.ASSESSED):
    return TileAssessment(
        tile_id=tile_id, batch_id=batch_id, status=status,
        damage_severity="destroyed" if status == TileStatus.ASSESSED else None,
    )


@pytest.fixture
def db():
    return FakeFirestore()


# --- store ------------------------------------------------------------------


def test_two_batches_can_reuse_a_tile_id_without_overwriting(db):
    store = TileAssessmentStore(client=db)
    store.save(tile("t1", "A"))
    store.save(tile("t1", "B", status=TileStatus.ESCALATED))

    assert store.get("t1", "A")["status"] == "assessed"
    assert store.get("t1", "B")["status"] == "escalated"


def test_list_by_batch_returns_only_that_batch(db):
    store = TileAssessmentStore(client=db)
    for t in (tile("a1", "A"), tile("a2", "A"), tile("b1", "B")):
        store.save(t)

    assert sorted(t.tile_id for t in store.list_by_batch("A")) == ["a1", "a2"]
    assert [t.tile_id for t in store.list_by_batch("B")] == ["b1"]
    assert store.list_by_batch("nope") == []


def test_default_batch_is_used_by_get_delete_and_list_all(db):
    store = TileAssessmentStore(client=db, batch_id="A")
    store.save(tile("a1", "A"))
    store.save(tile("b1", "B"))

    assert store.get("a1") is not None          # dispatcher-style call: tile_id only
    assert store.get("b1") is None              # other batch is invisible by default
    assert [t.tile_id for t in store.list_all()] == ["a1"]
    store.delete("a1")
    assert store.get("a1") is None
    assert store.get("b1", "B") is not None     # untouched


def test_list_batches_includes_every_written_batch(db):
    store = TileAssessmentStore(client=db)
    store.save(tile("a1", "A"))
    store.save(tile("z1", "zeta"))
    assert store.list_batches() == ["A", "zeta"]


def test_roundtrip_preserves_the_assessment(db):
    store = TileAssessmentStore(client=db)
    original = tile("a1", "A")
    store.save(original)
    assert store.list_by_batch("A")[0] == original


# --- migration ---------------------------------------------------------------


def seed_legacy(db):
    """Seed the old flat collection under whatever name the project's config uses
    (never hard-code it: FIRESTORE_COLLECTION_TILES differs between setups)."""
    legacy = TileAssessmentStore(client=db)._legacy_collection
    db.data[(legacy, "old1")] = {"tile_id": "old1", "status": "assessed", "damage_severity": "x"}
    db.data[(legacy, "old2")] = {"tile_id": "old2", "status": "assessed", "batch_id": "A"}
    db.data[(legacy, "junk")] = {"nonsense": True}
    return legacy


def test_migration_moves_legacy_docs_into_their_batches(db):
    legacy = seed_legacy(db)
    store = TileAssessmentStore(client=db)

    migrated, skipped = migrate(store)

    assert sorted(migrated) == ["old1", "old2"]
    assert skipped == ["junk"]
    assert [t.tile_id for t in store.list_by_batch("default")] == ["old1"]  # no batch_id -> default
    assert [t.tile_id for t in store.list_by_batch("A")] == ["old2"]
    assert (legacy, "old1") in db.data  # old docs kept unless asked


def test_migration_dry_run_writes_nothing(db):
    legacy = seed_legacy(db)
    store = TileAssessmentStore(client=db)
    migrate(store, dry_run=True, delete_old=True)
    assert store.list_batches() == []
    assert (legacy, "old1") in db.data


def test_migration_delete_old_removes_only_migrated_docs(db):
    legacy = seed_legacy(db)
    migrate(TileAssessmentStore(client=db), delete_old=True)
    assert (legacy, "old1") not in db.data and (legacy, "old2") not in db.data
    assert (legacy, "junk") in db.data


# --- ledger versions ---------------------------------------------------------


def ranking(zone_id, version, priority, batch_id="default"):
    return ZoneRanking(
        zone_id=zone_id, batch_id=batch_id, version=version, tile_ids=[zone_id],
        damage_score=priority, accessibility_penalty=0.0, priority_score=priority,
        status=ZoneStatus.RANKED,
    )


def test_next_version_starts_at_one_and_increments_per_batch(tmp_path):
    ledger = LocalZoneRankingStore(tmp_path / "ledger.jsonl")
    assert ledger.next_version("A") == 1

    ledger.append_all([ranking("z1", 1, 3.0, "A"), ranking("z2", 1, 1.0, "A")])
    ledger.append_all([ranking("z1", 2, 2.0, "A")])
    ledger.append_all([ranking("y1", 1, 5.0, "B")])

    assert ledger.next_version("A") == 3
    assert ledger.next_version("B") == 2


def test_read_latest_returns_only_newest_version_sorted_by_priority(tmp_path):
    ledger = LocalZoneRankingStore(tmp_path / "ledger.jsonl")
    ledger.append_all([ranking("z1", 1, 9.0), ranking("z2", 1, 1.0)])
    ledger.append_all([ranking("z1", 2, 2.0), ranking("z2", 2, 4.0)])

    latest = ledger.read_latest()

    assert [(r.zone_id, r.version) for r in latest] == [("z2", 2), ("z1", 2)]
    assert len(ledger.read_all()) == 4  # history retained (append-only)


def test_read_latest_is_batch_scoped_and_empty_when_unknown(tmp_path):
    ledger = LocalZoneRankingStore(tmp_path / "ledger.jsonl")
    ledger.append_all([ranking("a", 3, 1.0, "A"), ranking("b", 1, 1.0, "B")])
    assert [r.zone_id for r in ledger.read_latest("B")] == ["b"]
    assert ledger.read_latest("none") == []


def test_legacy_ledger_rows_without_batch_id_count_as_default(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text(
        '{"zone_id":"z","version":2,"tile_ids":["t"],"damage_score":1.0,'
        '"accessibility_penalty":0.0,"priority_score":1.0,"status":"ranked"}\n'
    )
    assert LocalZoneRankingStore(path).next_version() == 3


# --- planner script version resolution ---------------------------------------


def test_script_auto_increments_and_rejects_stale_manual_versions(tmp_path, monkeypatch):
    import scripts.run_planner_batch as script

    ledger_path = tmp_path / "ledger.jsonl"
    LocalZoneRankingStore(ledger_path).append_all([ranking("z", 2, 1.0, "A")])
    monkeypatch.setattr(script, "LocalZoneRankingStore", lambda: LocalZoneRankingStore(ledger_path))

    args = lambda **kw: SimpleNamespace(batch_id="A", version=None, use_bigquery=False, **kw)

    assert script._resolve_version(args()) == 3                       # auto
    assert script._resolve_version(SimpleNamespace(batch_id="A", version=7, use_bigquery=False)) == 7
    with pytest.raises(SystemExit):
        script._resolve_version(SimpleNamespace(batch_id="A", version=2, use_bigquery=False))
    assert script._resolve_version(SimpleNamespace(batch_id="new", version=None, use_bigquery=False)) == 1
