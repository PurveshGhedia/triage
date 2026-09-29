"""Batch scoping helpers (SYS-05 / E2E-02: no cross-batch data leakage).

A "batch" is one disaster event's worth of tiles. Every tile record and every
zone ranking carries a `batch_id`; Agent 2 only ever plans over a single batch.

These are pure functions (no GCP client), so they are unit-testable offline.
"""
from __future__ import annotations

from collections.abc import Iterable

from agents.perception.schemas import DEFAULT_BATCH_ID, TileAssessment


class MixedBatchError(ValueError):
    """Raised when Agent 2 is handed tiles from more than one batch."""


def filter_by_batch(tiles: Iterable[TileAssessment], batch_id: str) -> list[TileAssessment]:
    """Keep only tiles belonging to `batch_id`."""
    return [t for t in tiles if t.batch_id == batch_id]


def single_batch_id(tiles: Iterable[TileAssessment]) -> str:
    """The one batch_id shared by all `tiles` (DEFAULT_BATCH_ID if empty).

    Raises MixedBatchError if the tiles span more than one batch — clustering
    them together would leak tiles from batch A into batch B's zones.
    """
    batch_ids = {t.batch_id for t in tiles}
    if not batch_ids:
        return DEFAULT_BATCH_ID
    if len(batch_ids) > 1:
        raise MixedBatchError(
            f"Planner received tiles from multiple batches: {sorted(batch_ids)}. "
            "Filter to a single batch before planning."
        )
    return next(iter(batch_ids))
