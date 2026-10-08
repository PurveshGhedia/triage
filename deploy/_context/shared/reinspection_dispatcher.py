"""Abstract interface for dispatching a targeted re-inspection request back
to Agent 1 for a specific tile (PLN-03, SYS-01). The real implementation will
publish to Pub/Sub once that's provisioned; `LocalReinspectionDispatcher`
below lets Agent 2's logic and tests run without any GCP messaging infra.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class ReinspectionDispatcher(ABC):
    @abstractmethod
    def dispatch(self, tile_id: str, reason: str) -> None:
        """Request that Agent 1 re-run its tool-selection loop on this tile."""
        raise NotImplementedError


class LocalReinspectionDispatcher(ReinspectionDispatcher):
    """Records dispatched requests in memory — no real messaging. Useful for
    local dev/tests before Pub/Sub is provisioned via Terraform.
    """

    def __init__(self) -> None:
        self.dispatched: list[tuple[str, str]] = []

    def dispatch(self, tile_id: str, reason: str) -> None:
        self.dispatched.append((tile_id, reason))
