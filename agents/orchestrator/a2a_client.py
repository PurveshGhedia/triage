"""Orchestrator-side A2A adapters: implement PerceptionPort / PlannerPort by calling
the remote agents through ADK's RemoteA2aAgent, plus a wrapper that falls back to
the in-process agents if the remote call fails for any reason.
"""
from __future__ import annotations

import asyncio
import json
import uuid
import warnings
from collections.abc import Callable

warnings.filterwarnings("ignore", message=r"\[EXPERIMENTAL\]")

from google.adk.agents.remote_a2a_agent import RemoteA2aAgent  # noqa: E402
from google.adk.runners import Runner  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.genai import types  # noqa: E402

from agents.orchestrator.a2a_services import (  # noqa: E402
    AGENT_CARD_PATH,
    PERCEPTION_AGENT_NAME,
    PLANNER_AGENT_NAME,
)
from agents.orchestrator.models import PlanOutcome, TileSpec  # noqa: E402
from agents.orchestrator.ports import PerceptionPort, PlannerPort, ResultCallback  # noqa: E402
from agents.perception.schemas import TileAssessment  # noqa: E402

APP_NAME = "triage_orchestrator"
DEFAULT_TIMEOUT_S = 300.0


def card_url(base_url: str) -> str:
    return base_url.rstrip("/") + AGENT_CARD_PATH


class RemoteAgentCaller:
    """Send one JSON request to a remote A2A agent and return its JSON reply."""

    def __init__(self, base_url: str, agent_name: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self.base_url = base_url
        self.agent_name = agent_name
        self.timeout_s = timeout_s

    def call(self, payload: dict) -> dict:
        return asyncio.run(self._call(payload))

    async def _call(self, payload: dict) -> dict:
        remote = RemoteA2aAgent(
            name=self.agent_name,
            agent_card=card_url(self.base_url),
            description=f"remote {self.agent_name}",
        )
        runner = Runner(agent=remote, app_name=APP_NAME, session_service=InMemorySessionService())
        session_id = uuid.uuid4().hex
        await runner.session_service.create_session(
            app_name=APP_NAME, user_id="orchestrator", session_id=session_id
        )
        message = types.Content(role="user", parts=[types.Part(text=json.dumps(payload))])
        texts: list[str] = []
        errors: list[str] = []

        async def consume() -> None:
            async for event in runner.run_async(
                user_id="orchestrator", session_id=session_id, new_message=message
            ):
                if event.error_message:
                    errors.append(event.error_message)
                if event.content and event.content.parts:
                    texts.extend(p.text for p in event.content.parts if p.text)

        await asyncio.wait_for(consume(), timeout=self.timeout_s)

        if not texts:
            # ADK reports connection / agent-card failures as an error event with no text.
            detail = f": {errors[-1]}" if errors else ""
            raise RuntimeError(f"{self.agent_name} returned no text{detail}")
        # ADK's remote agent can surface the same reply more than once; the last is final.
        data = json.loads(texts[-1])
        if "error" in data:
            raise RuntimeError(f"{self.agent_name} error: {data['error']}")
        return data


class A2APerception:
    def __init__(self, base_url: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self._caller = RemoteAgentCaller(base_url, PERCEPTION_AGENT_NAME, timeout_s)

    def assess(
        self,
        batch_id: str,
        specs: list[TileSpec],
        reinspection: bool = False,
        on_result: ResultCallback | None = None,
    ) -> list[TileAssessment]:
        data = self._caller.call(
            {
                "batch_id": batch_id,
                "specs": [s.model_dump() for s in specs],
                "reinspection": reinspection,
            }
        )
        results = [TileAssessment.model_validate(a) for a in data["assessments"]]
        if on_result:  # a remote call reports everything at once, at the end
            for a in results:
                on_result(a)
        return results


class A2APlanner:
    def __init__(self, base_url: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self._caller = RemoteAgentCaller(base_url, PLANNER_AGENT_NAME, timeout_s)

    def plan(
        self,
        batch_id: str,
        tiles: list[TileAssessment],
        version: int,
        adjacency_radius_m: float,
    ) -> PlanOutcome:
        data = self._caller.call(
            {
                "batch_id": batch_id,
                "tiles": [t.model_dump(mode="json") for t in tiles],
                "version": version,
                "adjacency_radius_m": adjacency_radius_m,
            }
        )
        return PlanOutcome.model_validate(data["outcome"])


FallbackCallback = Callable[[str, Exception], None]


class ResilientPerception:
    """Try the A2A agent first; on any failure use the in-process agent instead."""

    def __init__(self, primary: PerceptionPort, fallback: PerceptionPort,
                 on_fallback: FallbackCallback | None = None) -> None:
        self._primary, self._fallback, self._on_fallback = primary, fallback, on_fallback

    def assess(self, batch_id, specs, reinspection=False, on_result=None):
        try:
            return self._primary.assess(batch_id, specs, reinspection, on_result)
        except Exception as exc:
            if self._on_fallback:
                self._on_fallback("perception", exc)
            return self._fallback.assess(batch_id, specs, reinspection, on_result)


class ResilientPlanner:
    def __init__(self, primary: PlannerPort, fallback: PlannerPort,
                 on_fallback: FallbackCallback | None = None) -> None:
        self._primary, self._fallback, self._on_fallback = primary, fallback, on_fallback

    def plan(self, batch_id, tiles, version, adjacency_radius_m):
        try:
            return self._primary.plan(batch_id, tiles, version, adjacency_radius_m)
        except Exception as exc:
            if self._on_fallback:
                self._on_fallback("planner", exc)
            return self._fallback.plan(batch_id, tiles, version, adjacency_radius_m)
