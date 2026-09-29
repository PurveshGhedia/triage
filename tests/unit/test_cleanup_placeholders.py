"""scripts.cleanup_placeholders: deletes only the listed docs, dry-run touches nothing."""
from __future__ import annotations

from scripts.cleanup_placeholders import DEFAULT_PLACEHOLDER_IDS, cleanup


class FakeStore:
    def __init__(self, ids):
        self.docs = {i: {"tile_id": i} for i in ids}
        self.deleted: list[str] = []

    def get(self, tile_id):
        return self.docs.get(tile_id)

    def delete(self, tile_id):
        self.docs.pop(tile_id, None)
        self.deleted.append(tile_id)


def test_default_ids_are_the_four_placeholders():
    assert DEFAULT_PLACEHOLDER_IDS == ["tile_001", "tile_002", "tile_003", "tile_004"]


def test_deletes_placeholders_and_leaves_real_tiles_alone():
    store = FakeStore(DEFAULT_PLACEHOLDER_IDS + ["guatemala-volcano_00000000_post_disaster"])
    removed, missing = cleanup(store, DEFAULT_PLACEHOLDER_IDS, dry_run=False)

    assert removed == DEFAULT_PLACEHOLDER_IDS
    assert missing == []
    assert list(store.docs) == ["guatemala-volcano_00000000_post_disaster"]


def test_dry_run_reports_but_deletes_nothing():
    store = FakeStore(DEFAULT_PLACEHOLDER_IDS)
    removed, _ = cleanup(store, DEFAULT_PLACEHOLDER_IDS, dry_run=True)

    assert removed == DEFAULT_PLACEHOLDER_IDS
    assert store.deleted == []
    assert len(store.docs) == 4


def test_missing_ids_are_reported_and_rerun_is_safe():
    store = FakeStore(["tile_001"])
    removed, missing = cleanup(store, DEFAULT_PLACEHOLDER_IDS, dry_run=False)
    assert removed == ["tile_001"]
    assert missing == ["tile_002", "tile_003", "tile_004"]

    removed_again, missing_again = cleanup(store, DEFAULT_PLACEHOLDER_IDS, dry_run=False)
    assert removed_again == []
    assert len(missing_again) == 4
