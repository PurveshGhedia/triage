"""SAM (Segment Anything) zero-shot segmentation fallback tool client.

Used as the last-resort tool when classifier/detector confidence stays low —
zero-shot to avoid GPU fine-tuning cost within the budget constraint. Same
mock/real split as the other two tools.
"""
from __future__ import annotations

import hashlib
import os

from shared.tool_interface import ToolClient, ToolResult

DAMAGE_LABELS = ["no-damage", "partial-damage", "severe-damage"]


class SAMClient(ToolClient):
    name = "sam"

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
            "Real SAM inference requires a deployed Vertex AI Endpoint "
            "(SAM_ENDPOINT_ID). Provision via infra/terraform first."
        )

    def _mock_run(self, tile_path: str) -> ToolResult:
        digest = int(hashlib.sha256((tile_path + "sam").encode()).hexdigest(), 16)
        confidence = 0.4 + (digest % 60) / 100
        label = DAMAGE_LABELS[digest % len(DAMAGE_LABELS)]
        return ToolResult(
            tool_name=self.name,
            confidence=round(confidence, 2),
            label=label,
            raw_output={"mock": True, "tile_path": tile_path},
        )
