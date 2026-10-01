"""In-memory stand-in for TileAssessmentStore: same methods the pipeline uses,
no Firestore. Used for tests and for `scripts.run_e2e --offline` (a demo that
must not depend on the network)."""
from __future__ import annotations

import threading

from agents.perception.schemas import DEFAULT_BATCH_ID, TileAssessment


class InMemoryTileStore:
    def __init__(self, batch_id: str = DEFAULT_BATCH_ID) -> None:
        self.batch_id = batch_id
        self._docs: dict[tuple[str, str], TileAssessment] = {}
        self._lock = threading.Lock()

    def save(self, assessment: TileAssessment) -> None:
        with self._lock:
            self._docs[(assessment.batch_id, assessment.tile_id)] = assessment.model_copy(deep=True)

    def get(self, tile_id: str, batch_id: str | None = None) -> dict | None:
        doc = self._docs.get((batch_id or self.batch_id, tile_id))
        return doc.model_dump(mode="python") if doc else None

    def list_by_batch(self, batch_id: str) -> list[TileAssessment]:
        with self._lock:
            return [a.model_copy(deep=True) for (b, _), a in sorted(self._docs.items()) if b == batch_id]

    def list_all(self) -> list[TileAssessment]:
        return self.list_by_batch(self.batch_id)

    def list_batches(self) -> list[str]:
        return sorted({b for b, _ in self._docs})
