"""Unit test for the Firestore serialization helper. Only exercises the pure
`_to_firestore_dict` static method — never instantiates a real client, so
this needs no GCP credentials or network access to run.
"""
from __future__ import annotations

from agents.perception.schemas import TileAssessment, TileStatus, ToolCallRecord
from shared.firestore_client import TileAssessmentStore


def test_to_firestore_dict_stores_plain_status_string():
    assessment = TileAssessment(
        tile_id="tile-1",
        status=TileStatus.ASSESSED,
        damage_severity="destroyed",
        confidence=0.9,
        tool_call_history=[
            ToolCallRecord(tool_name="classifier", confidence=0.9, label="destroyed")
        ],
    )

    data = TileAssessmentStore._to_firestore_dict(assessment)

    assert data["status"] == "assessed"
    assert isinstance(data["status"], str)
    assert data["tile_id"] == "tile-1"
    assert data["tool_call_history"][0]["tool_name"] == "classifier"
