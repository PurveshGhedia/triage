"""Firestore repository for Agent 1 output.

Requires the Firestore API enabled and a Native-mode database created on the
GCP project (one-time manual setup — not yet handled by Terraform):

    gcloud services enable firestore.googleapis.com
    gcloud firestore databases create --location=<region> --type=firestore-native

Authenticates via the same Application Default Credentials already set up for
Vertex AI (`gcloud auth application-default login`) — no separate auth step.
"""
from __future__ import annotations

from google.cloud import firestore

from agents.perception.schemas import TileAssessment
from shared.config import FIRESTORE_COLLECTION_TILES, GCP_PROJECT_ID


class TileAssessmentStore:
    def __init__(self, project_id: str | None = None, collection: str | None = None) -> None:
        self._client = firestore.Client(project=project_id or GCP_PROJECT_ID)
        self._collection = collection or FIRESTORE_COLLECTION_TILES

    def save(self, assessment: TileAssessment) -> None:
        doc_ref = self._client.collection(self._collection).document(assessment.tile_id)
        doc_ref.set(self._to_firestore_dict(assessment))

    def get(self, tile_id: str) -> dict | None:
        doc = self._client.collection(self._collection).document(tile_id).get()
        return doc.to_dict() if doc.exists else None

    @staticmethod
    def _to_firestore_dict(assessment: TileAssessment) -> dict:
        """Pure serialization helper — no client involved, so this is unit-
        testable without any GCP credentials.
        """
        data = assessment.model_dump(mode="python")
        data["status"] = assessment.status.value  # store the plain string, not the Enum
        return data
