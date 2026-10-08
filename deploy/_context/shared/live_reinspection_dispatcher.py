"""SYS-01: closes the re-inspection loop for real.

LocalReinspectionDispatcher (see reinspection_dispatcher.py) only records that
a request happened — it never actually re-runs anything. This dispatcher does
the real thing: on dispatch, it fetches the tile's stored assessment (for its
tile_path, geolocation and batch), re-enters Agent 1's tool-selection loop on
that same tile, and persists the updated result.

DESIGN NOTE on when the correction becomes visible: this dispatch happens
*during* Agent 2's plan() call, after that call has already read its input
tiles and started scoring zones. Rather than mutate a zone's score mid-
computation (fragile, hard to reason about), the corrected tile is written
back to the tile store immediately, and the CURRENT ranking pass keeps using
the data it started with. The correction shows up in the *next* planner run
(--version N+1) — which is exactly what the append-only versioned ledger
(PLN-05) is for: run the planner again, and the new version reflects the
re-inspected data.

BATCH NOTE (SYS-05): the re-inspected tile keeps the batch_id it was stored
with. Without this the fresh record would fall back to the default batch and
silently drop out of its own batch on the next planner run.

tile_store only needs .get(tile_id) -> dict | None and .save(assessment) ->
None — TileAssessmentStore (Firestore) satisfies this today; any object with
that shape works, including a plain dict-backed fake for tests.
"""
from __future__ import annotations

import logging

from agents.perception.agent import PerceptionAgent
from agents.perception.schemas import DEFAULT_BATCH_ID
from shared.reinspection_dispatcher import ReinspectionDispatcher

logger = logging.getLogger(__name__)


class LiveReinspectionDispatcher(ReinspectionDispatcher):
    def __init__(self, agent: PerceptionAgent, tile_store) -> None:
        self._agent = agent
        self._tile_store = tile_store
        self.dispatched: list[tuple[str, str]] = []

    def dispatch(self, tile_id: str, reason: str) -> None:
        self.dispatched.append((tile_id, reason))

        original = self._tile_store.get(tile_id)
        if original is None:
            logger.warning("Cannot re-inspect %s: no stored assessment found", tile_id)
            return

        tile_path = original.get("tile_path")
        if not tile_path:
            logger.warning(
                "Cannot re-inspect %s: no tile_path stored (assessed before this "
                "field existed?) — re-run Agent 1 on it once to backfill.",
                tile_id,
            )
            return

        updated = self._agent.process_tile(
            tile_id=tile_id,
            tile_path=tile_path,
            latitude=original.get("latitude"),
            longitude=original.get("longitude"),
            # Docs saved before batch_id existed have no such key -> default batch.
            batch_id=original.get("batch_id", DEFAULT_BATCH_ID),
        )
        self._tile_store.save(updated)
        logger.info("Re-inspected %s: now %s", tile_id, updated.status.value)
