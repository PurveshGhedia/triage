"""Agent 1 and Agent 2 exposed as A2A services (Google ADK + A2A protocol).

Each service is a thin, deterministic ADK agent (no LLM in the wrapper): it reads
one JSON request from the incoming A2A message, calls the existing, tested agent
code, and returns one JSON response. Gemini still does the reasoning *inside*
Agent 1's tool selector; the wrapper only moves data across the A2A boundary.

NOTE: ADK documents its A2A support as experimental, so every request is wrapped
to return {"error": ...} instead of raising, and the client side (a2a_client.py)
falls back to in-process calls if a service misbehaves.
"""
from __future__ import annotations

import asyncio
import json
import logging
import warnings
from collections.abc import AsyncGenerator
from typing import Any
from urllib.parse import urlparse

warnings.filterwarnings("ignore", message=r"\[EXPERIMENTAL\]")
# ADK's Runner logs an "App name mismatch" warning on every call to an A2A agent; it is
# harmless here (the app name is just a label) and would clutter a live demo terminal.
logging.getLogger("google_adk.google.adk.runners").setLevel(logging.ERROR)

from google.adk.a2a.utils.agent_to_a2a import to_a2a  # noqa: E402
from google.adk.agents import BaseAgent  # noqa: E402
from google.adk.agents.invocation_context import InvocationContext  # noqa: E402
from google.adk.events import Event  # noqa: E402
from google.genai import types  # noqa: E402

from agents.orchestrator.models import TileSpec  # noqa: E402
from agents.orchestrator.ports import InProcessPerception, InProcessPlanner  # noqa: E402
from agents.perception.schemas import TileAssessment  # noqa: E402

PERCEPTION_AGENT_NAME = "perception_agent"
PLANNER_AGENT_NAME = "planner_agent"
AGENT_CARD_PATH = "/.well-known/agent-card.json"


def _request_text(ctx: InvocationContext) -> str:
    parts = ctx.user_content.parts if ctx.user_content and ctx.user_content.parts else []
    return "".join(p.text for p in parts if p.text)


def _reply(author: str, payload: dict) -> Event:
    return Event(
        author=author,
        content=types.Content(role="model", parts=[types.Part(text=json.dumps(payload))]),
    )


class PerceptionServiceAgent(BaseAgent):
    """Agent 1 over A2A: request {batch_id, specs[], reinspection} -> {assessments[]}."""

    perception: Any = None

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        try:
            request = json.loads(_request_text(ctx))
            specs = [TileSpec.model_validate(s) for s in request["specs"]]
            results = await asyncio.to_thread(
                self.perception.assess,
                request["batch_id"],
                specs,
                bool(request.get("reinspection", False)),
            )
            payload = {"assessments": [a.model_dump(mode="json") for a in results]}
        except Exception as exc:  # return the failure to the caller instead of crashing the task
            payload = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        yield _reply(self.name, payload)


class PlannerServiceAgent(BaseAgent):
    """Agent 2 over A2A: request {batch_id, tiles[], version, adjacency_radius_m} -> {outcome}."""

    planner: Any = None

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        try:
            request = json.loads(_request_text(ctx))
            tiles = [TileAssessment.model_validate(t) for t in request["tiles"]]
            outcome = await asyncio.to_thread(
                self.planner.plan,
                request["batch_id"],
                tiles,
                int(request["version"]),
                float(request["adjacency_radius_m"]),
            )
            payload = {"outcome": json.loads(outcome.model_dump_json())}
        except Exception as exc:
            payload = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        yield _reply(self.name, payload)


def split_public_url(url: str) -> tuple[str, str, int]:
    """'https://x.run.app' -> ('https', 'x.run.app', 443). The agent card advertises this
    address, and the client uses the card's address for every call — so behind Cloud Run it
    must be the public URL, not the container's own host/port."""
    parsed = urlparse(url)
    protocol = parsed.scheme or "http"
    if not parsed.hostname:
        raise ValueError(f"not a valid URL: {url!r}")
    port = parsed.port or (443 if protocol == "https" else 80)
    return protocol, parsed.hostname, port


def build_perception_app(
    perception: InProcessPerception, host: str, port: int, protocol: str = "http"
):
    agent = PerceptionServiceAgent(
        name=PERCEPTION_AGENT_NAME,
        description=(
            "Agent 1 (Perception): assesses satellite tile images for disaster damage, "
            "choosing among classifier, detector and SAM tools by confidence; escalates "
            "tiles it cannot assess."
        ),
        perception=perception,
    )
    return to_a2a(agent, host=host, port=port, protocol=protocol)


def build_planner_app(
    planner: InProcessPlanner, host: str, port: int, protocol: str = "http"
):
    agent = PlannerServiceAgent(
        name=PLANNER_AGENT_NAME,
        description=(
            "Agent 2 (Triage Planner): clusters assessed tiles into zones by geography, "
            "scores them on damage plus accessibility, and requests re-inspection of "
            "conflicting tiles."
        ),
        planner=planner,
    )
    return to_a2a(agent, host=host, port=port, protocol=protocol)
