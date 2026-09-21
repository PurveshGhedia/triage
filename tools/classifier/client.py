"""Damage classifier tool client.

MOCK MODE (default, controlled by USE_MOCK_TOOLS in .env): returns a
deterministic-but-varied confidence for local dev/testing, so repeated runs on
the same fixture are reproducible. Real mode will call the classifier's Vertex
AI Endpoint once CLASSIFIER_ENDPOINT_ID is set — not implemented yet, since
that requires Terraform provisioning first.
"""
from __future__ import annotations

import hashlib
import os

from shared.tool_interface import ToolClient, ToolResult

DAMAGE_LABELS = ["no-damage", "minor-damage", "major-damage", "destroyed"]


class ClassifierClient(ToolClient):
    name = "classifier"
    description = "General-purpose damage severity classifier (no-damage/minor/major/destroyed)."

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
            "Real classifier inference requires a deployed Vertex AI Endpoint "
            "(CLASSIFIER_ENDPOINT_ID). Provision via infra/terraform first."
        )

    def _mock_run(self, tile_path: str) -> ToolResult:
        digest = int(hashlib.sha256(tile_path.encode()).hexdigest(), 16)
        confidence = 0.4 + (digest % 60) / 100  # spread across 0.40-0.99
        label = DAMAGE_LABELS[digest % len(DAMAGE_LABELS)]
        return ToolResult(
            tool_name=self.name,
            confidence=round(confidence, 2),
            label=label,
            raw_output={"mock": True, "tile_path": tile_path},
        )
