"""Firestore repository for Agent 1 output.

Layout (batch-scoped, SYS-05): one subcollection of tiles per batch.

    batches/{batch_id}                       <- small metadata doc
    batches/{batch_id}/tiles/{tile_id}       <- one TileAssessment

Two batches can therefore reuse a tile_id without overwriting each other, and
reading a batch can never return another batch's tiles. The old flat collection
(`FIRESTORE_COLLECTION_TILES`) is only read by scripts.migrate_tiles_to_batches.

A store has a default `batch_id` used by get()/delete()/list_all(), so callers
that only know a tile_id (e.g. the live re-inspection dispatcher) keep working
when the store is built for the batch being processed. save() always writes to
the batch recorded on the assessment itself.

Requires the Firestore API enabled and a Native-mode database on the project.
Authenticates via Application Default Credentials (`gcloud auth application-default
login`) — no separate auth step.
"""
from __future__ import annotations

from google.cloud import firestore

from agents.perception.schemas import DEFAULT_BATCH_ID, TileAssessment
from shared.batching import filter_by_batch
from shared.config import FIRESTORE_COLLECTION_TILES, GCP_PROJECT_ID

BATCHES_COLLECTION = "batches"
TILES_SUBCOLLECTION = "tiles"


class TileAssessmentStore:
    def __init__(
        self,
        project_id: str | None = None,
        collection: str | None = None,
        batch_id: str = DEFAULT_BATCH_ID,
        client=None,
    ) -> None:
        self._client = client or firestore.Client(project=project_id or GCP_PROJECT_ID)
        self._legacy_collection = collection or FIRESTORE_COLLECTION_TILES
        self.batch_id = batch_id

    # -- paths ---------------------------------------------------------------

    def _tiles_ref(self, batch_id: str):
        return (
            self._client.collection(BATCHES_COLLECTION)
            .document(batch_id)
            .collection(TILES_SUBCOLLECTION)
        )

    # -- writes --------------------------------------------------------------

    def save(self, assessment: TileAssessment) -> None:
        batch_ref = self._client.collection(BATCHES_COLLECTION).document(assessment.batch_id)
        # Subcollection parents are not listable unless the parent doc exists.
        batch_ref.set({"batch_id": assessment.batch_id}, merge=True)
        self._tiles_ref(assessment.batch_id).document(assessment.tile_id).set(
            self._to_firestore_dict(assessment)
        )

    def delete(self, tile_id: str, batch_id: str | None = None) -> None:
        self._tiles_ref(batch_id or self.batch_id).document(tile_id).delete()

    # -- reads ---------------------------------------------------------------

    def get(self, tile_id: str, batch_id: str | None = None) -> dict | None:
        doc = self._tiles_ref(batch_id or self.batch_id).document(tile_id).get()
        return doc.to_dict() if doc.exists else None

    def list_by_batch(self, batch_id: str) -> list[TileAssessment]:
        """Every tile in one batch — Agent 2's input. Reads only that batch's
        subcollection, so other batches are never touched (SYS-05).
        """
        docs = self._tiles_ref(batch_id).stream()
        tiles = [TileAssessment.model_validate(doc.to_dict()) for doc in docs]
        # Belt and braces: a doc whose body disagrees with its path is dropped.
        return filter_by_batch(tiles, batch_id)

    def list_all(self) -> list[TileAssessment]:
        """Every tile in this store's default batch."""
        return self.list_by_batch(self.batch_id)

    def list_batches(self) -> list[str]:
        """Ids of all batches that have been written (for the dashboard)."""
        return sorted(doc.id for doc in self._client.collection(BATCHES_COLLECTION).stream())

    # -- migration support ---------------------------------------------------

    def iter_legacy_docs(self):
        """(doc_id, dict) pairs from the old flat collection."""
        for doc in self._client.collection(self._legacy_collection).stream():
            yield doc.id, doc.to_dict()

    def delete_legacy(self, doc_id: str) -> None:
        self._client.collection(self._legacy_collection).document(doc_id).delete()

    @staticmethod
    def _to_firestore_dict(assessment: TileAssessment) -> dict:
        """Pure serialization helper — no client involved, so this is unit-
        testable without any GCP credentials.
        """
        data = assessment.model_dump(mode="python")
        data["status"] = assessment.status.value  # store the plain string, not the Enum
        return data
