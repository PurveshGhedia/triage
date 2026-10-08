"""The two seams the orchestrator talks through.

`PerceptionPort` is Agent 1, `PlannerPort` is Agent 2. The in-process adapters
below call the existing, tested agent classes directly. An ADK/A2A adapter
implements the same two methods by sending the same pydantic models over A2A,
so the orchestrator (pipeline.py) does not change when the transport does — and
if A2A misbehaves on demo day, the in-process adapters are the fallback.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from agents.orchestrator.models import PlanOutcome, ReinspectionRequest, TileSpec
from agents.perception.agent import PerceptionAgent
from agents.perception.schemas import TileAssessment, TileStatus
from agents.planner.agent import TriagePlanner
from shared.reinspection_dispatcher import LocalReinspectionDispatcher

logger = logging.getLogger(__name__)

ResultCallback = Callable[[TileAssessment], None]


class PerceptionPort(Protocol):
    def assess(
        self,
        batch_id: str,
        specs: list[TileSpec],
        reinspection: bool = False,
        on_result: ResultCallback | None = None,
    ) -> list[TileAssessment]: ...


class PlannerPort(Protocol):
    def plan(
        self,
        batch_id: str,
        tiles: list[TileAssessment],
        version: int,
        adjacency_radius_m: float,
    ) -> PlanOutcome: ...


class InProcessPerception:
    """Agent 1 in this process: tiles run concurrently, and one tile blowing up
    can never take down the others (PER-05) — it just becomes an escalated tile.
    """

    def __init__(
        self,
        agent: PerceptionAgent,
        reinspection_agent: PerceptionAgent | None = None,
        max_workers: int = 4,
    ) -> None:
        self._agent = agent
        self._reinspection_agent = reinspection_agent or agent
        self._max_workers = max_workers

    def assess(
        self,
        batch_id: str,
        specs: list[TileSpec],
        reinspection: bool = False,
        on_result: ResultCallback | None = None,
    ) -> list[TileAssessment]:
        agent = self._reinspection_agent if reinspection else self._agent

        def one(spec: TileSpec) -> TileAssessment:
            try:
                result = agent.process_tile(
                    tile_id=spec.tile_id,
                    tile_path=spec.tile_path,
                    latitude=spec.latitude,
                    longitude=spec.longitude,
                    batch_id=batch_id,
                )
            except Exception as exc:  # last line of defence: never lose a tile
                logger.exception("Agent 1 crashed on tile %s", spec.tile_id)
                result = TileAssessment(
                    tile_id=spec.tile_id,
                    batch_id=batch_id,
                    tile_path=spec.tile_path,
                    latitude=spec.latitude,
                    longitude=spec.longitude,
                    status=TileStatus.ESCALATED,
                    failure_reason=f"agent error: {type(exc).__name__}: {str(exc)[:200]}",
                )
            if on_result:
                on_result(result)
            return result

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            return list(pool.map(one, specs))


class InProcessPlanner:
    """Agent 2 in this process. Re-inspection requests are *collected* and handed
    back to the orchestrator, which routes them to Agent 1 — so the loop between
    the agents is visible and capped instead of hidden inside plan().
    """

    def plan(
        self,
        batch_id: str,
        tiles: list[TileAssessment],
        version: int,
        adjacency_radius_m: float,
    ) -> PlanOutcome:
        dispatcher = LocalReinspectionDispatcher()
        planner = TriagePlanner(dispatcher=dispatcher, adjacency_radius_m=adjacency_radius_m)
        rankings = planner.plan(tiles, version=version)

        requests: dict[str, ReinspectionRequest] = {}
        for tile_id, reason in dispatcher.dispatched:
            requests.setdefault(tile_id, ReinspectionRequest(tile_id=tile_id, reason=reason))
        return PlanOutcome(rankings=rankings, reinspection_requests=list(requests.values()))
