"""Delete leftover placeholder tile documents from Firestore.

Early development left hand-made docs (tile_001 .. tile_004) in the tiles
collection, mixed in with real xBD tiles. This removes them.

Usage (run as a module, from the repo root):
    python -m scripts.cleanup_placeholders --dry-run     # show what would go
    python -m scripts.cleanup_placeholders               # actually delete
    python -m scripts.cleanup_placeholders --ids tile_001 tile_002

Only the listed ids are ever touched; anything not present is reported and
skipped, so re-running is safe.
"""
from __future__ import annotations

import argparse

DEFAULT_PLACEHOLDER_IDS = ["tile_001", "tile_002", "tile_003", "tile_004"]


def cleanup(store, tile_ids: list[str], dry_run: bool) -> tuple[list[str], list[str]]:
    """Returns (removed_or_would_remove, not_found). `store` needs get() and delete()."""
    removed: list[str] = []
    missing: list[str] = []
    for tile_id in tile_ids:
        if store.get(tile_id) is None:
            missing.append(tile_id)
            continue
        if not dry_run:
            store.delete(tile_id)
        removed.append(tile_id)
    return removed, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", nargs="+", default=DEFAULT_PLACEHOLDER_IDS)
    parser.add_argument("--batch-id", default="default")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from shared.firestore_client import TileAssessmentStore

    removed, missing = cleanup(TileAssessmentStore(batch_id=args.batch_id), args.ids, args.dry_run)

    verb = "Would delete" if args.dry_run else "Deleted"
    print(f"{verb} {len(removed)} document(s): {removed}")
    if missing:
        print(f"Not found (already gone): {missing}")
    if args.dry_run:
        print("(dry run — nothing was deleted)")


if __name__ == "__main__":
    main()
