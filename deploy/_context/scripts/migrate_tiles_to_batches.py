"""One-off: copy tile docs from the old flat Firestore collection into the
per-batch layout (batches/<batch_id>/tiles/<tile_id>).

Docs saved before batch_id existed land in the "default" batch. Nothing is
deleted unless you pass --delete-old, so the safe order is:

    python -m scripts.migrate_tiles_to_batches --dry-run
    python -m scripts.migrate_tiles_to_batches
    python -m scripts.migrate_tiles_to_batches --delete-old     # after checking

(Re-running is harmless: copies overwrite themselves with identical content.)
"""
from __future__ import annotations

import argparse

from agents.perception.schemas import TileAssessment


def migrate(store, dry_run: bool = False, delete_old: bool = False) -> tuple[list[str], list[str]]:
    """Returns (migrated_ids, skipped_ids). `store` needs iter_legacy_docs(),
    save() and delete_legacy() — i.e. a TileAssessmentStore or a fake.
    """
    migrated: list[str] = []
    skipped: list[str] = []
    for doc_id, data in list(store.iter_legacy_docs()):
        try:
            assessment = TileAssessment.model_validate(data)
        except Exception:
            skipped.append(doc_id)  # unparseable doc: leave it alone
            continue
        if not dry_run:
            store.save(assessment)
            if delete_old:
                store.delete_legacy(doc_id)
        migrated.append(doc_id)
    return migrated, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delete-old", action="store_true")
    args = parser.parse_args()

    from shared.firestore_client import TileAssessmentStore

    migrated, skipped = migrate(TileAssessmentStore(), args.dry_run, args.delete_old)
    verb = "Would migrate" if args.dry_run else "Migrated"
    print(f"{verb} {len(migrated)} doc(s): {migrated}")
    if skipped:
        print(f"Skipped (could not parse): {skipped}")
    if args.delete_old and not args.dry_run:
        print("Deleted the migrated docs from the old flat collection.")


if __name__ == "__main__":
    main()
