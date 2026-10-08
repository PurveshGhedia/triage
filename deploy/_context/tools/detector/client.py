"""Object detector tool client (roads/bridges/buildings — DOTA/xView-based).

Used both as a retry tool for damage assessment and, downstream, to inform
Agent 2's road-accessibility scoring. Same mock/real split as the classifier.

`provides_damage = False`: its labels describe road/building ACCESS, not damage.
Agent 1 therefore never lets a detector result satisfy the confidence threshold
or become a tile's `damage_severity`; it is an auxiliary access signal.
"""
from __future__ import annotations

import hashlib
import os

from shared.tool_interface import ToolClient, ToolResult

ACCESS_LABELS = ["access-clear", "access-degraded", "access-blocked"]


class DetectorClient(ToolClient):
    name = "detector"
    provides_damage = False
    description = (
        "Object detector for roads, bridges, and buildings — best when access/"
        "infrastructure state matters, not just overall damage severity."
    )

    def __init__(self, use_mock: bool | None = None) -> None:
        self.use_mock = (
            use_mock
            if use_mock is not None
            else os.getenv("USE_MOCK_TOOLS", "true").lower() == "true"
        )

    def run(self, tile_path: str) -> ToolResult:
        if self.use_mock:
            return self._mock_run(tile_path)
        raise NotImplementedError(
            "Real detector inference requires a deployed Vertex AI Endpoint "
            "(DETECTOR_ENDPOINT_ID). Provision via infra/terraform first."
        )

    def _mock_run(self, tile_path: str) -> ToolResult:
        digest = int(hashlib.sha256((tile_path + "detector").encode()).hexdigest(), 16)
        confidence = 0.4 + (digest % 60) / 100
        label = ACCESS_LABELS[digest % len(ACCESS_LABELS)]
        return ToolResult(
            tool_name=self.name,
            confidence=round(confidence, 2),
            label=label,
            raw_output={"mock": True, "tile_path": tile_path},
        )
