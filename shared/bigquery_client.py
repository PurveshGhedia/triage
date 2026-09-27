"""BigQuery repository for Agent 2's append-only zone-ranking ledger (PLN-05).

Every call to `append()`/`append_all()` INSERTS new rows — nothing here ever
updates or overwrites a prior row. A re-inspection that changes a zone's
score produces a new ZoneRanking with an incremented `version`; the old row
stays exactly as it was, so the full ranking history stays auditable (SYS-02).

Requires the BigQuery API enabled on the same billing-linked project already
set up for Firestore/Vertex AI:
    gcloud services enable bigquery.googleapis.com

The TABLE is created automatically on first use if it doesn't already exist.
The DATASET is not auto-created: on some projects (including some personal
Free Trial projects with a domain-restricted-sharing org policy), the API
call to create a new dataset fails with a 403 IAM error even for the
project owner, since dataset creation implicitly sets an IAM policy. Create
the dataset once, manually, before first use:
    bq mk --dataset --location=<region> <project-id>:<dataset-name>
or via the Console: BigQuery -> your project -> (⋮) -> Create dataset.
"""
from __future__ import annotations

from google.api_core.exceptions import NotFound
from google.cloud import bigquery

from agents.planner.schemas import ZoneRanking
from shared.config import BIGQUERY_DATASET, BIGQUERY_TABLE_ZONES, GCP_PROJECT_ID

_SCHEMA = [
    bigquery.SchemaField("zone_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("version", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("tile_ids", "STRING", mode="REPEATED"),
    bigquery.SchemaField("damage_score", "FLOAT", mode="REQUIRED"),
    bigquery.SchemaField("accessibility_penalty", "FLOAT", mode="REQUIRED"),
    bigquery.SchemaField("priority_score", "FLOAT", mode="REQUIRED"),
    bigquery.SchemaField("status", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("escalated_tile_ids", "STRING", mode="REPEATED"),
    # Each conflicting pair stored as "tile_a|tile_b" — BigQuery has no tuple
    # type, and a nested REPEATED RECORD is more ceremony than this needs.
    bigquery.SchemaField("conflicting_tile_pairs", "STRING", mode="REPEATED"),
    bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
]


class ZoneRankingStore:
    def __init__(
        self,
        project_id: str | None = None,
        dataset: str | None = None,
        table: str | None = None,
    ) -> None:
        self._project_id = project_id or GCP_PROJECT_ID
        if not self._project_id:
            raise ValueError(
                "GCP_PROJECT_ID is not set. Check that .env exists at the repo root."
            )
        self._dataset_id = dataset or BIGQUERY_DATASET
        self._table_id = table or BIGQUERY_TABLE_ZONES
        self._client = bigquery.Client(project=self._project_id)
        self._table_ref = self._ensure_table_exists()

    def _ensure_table_exists(self) -> bigquery.TableReference:
        dataset_ref = bigquery.DatasetReference(
            self._project_id, self._dataset_id)

        try:
            self._client.get_dataset(dataset_ref)
        except NotFound as exc:
            raise RuntimeError(
                f"BigQuery dataset '{self._dataset_id}' doesn't exist in project "
                f"'{self._project_id}'. Some projects can't create datasets via the "
                f"API (a 403 IAM error even for the owner) — create it manually once:\n"
                f"    bq mk --dataset --location=<region> {self._project_id}:{self._dataset_id}\n"
                f"or via the Console: BigQuery -> your project -> (⋮) -> Create dataset."
            ) from exc

        table_ref = dataset_ref.table(self._table_id)
        table = bigquery.Table(table_ref, schema=_SCHEMA)
        self._client.create_table(table, exists_ok=True)
        return table_ref

    def append(self, ranking: ZoneRanking) -> None:
        self.append_all([ranking])

    def append_all(self, rankings: list[ZoneRanking]) -> None:
        rows = [self._to_bigquery_row(r) for r in rankings]
        errors = self._client.insert_rows_json(self._table_ref, rows)
        if errors:
            raise RuntimeError(f"BigQuery insert failed: {errors}")

    @staticmethod
    def _to_bigquery_row(ranking: ZoneRanking) -> dict:
        """Pure serialization helper — no client involved, so this is unit-
        testable without any GCP credentials.
        """
        return {
            "zone_id": ranking.zone_id,
            "version": ranking.version,
            "tile_ids": ranking.tile_ids,
            "damage_score": ranking.damage_score,
            "accessibility_penalty": ranking.accessibility_penalty,
            "priority_score": ranking.priority_score,
            "status": ranking.status.value,
            "escalated_tile_ids": ranking.escalated_tile_ids,
            "conflicting_tile_pairs": [
                "|".join(pair) for pair in ranking.conflicting_tile_pairs
            ],
            "created_at": ranking.created_at.isoformat(),
        }
