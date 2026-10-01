"""One place that builds a production-configured Agent 1, so the batch script,
the planner's re-inspection loop and (later) the orchestrator all get the same
settings: tile validation on, auxiliary (access) tools run on every assessed tile.
"""
from __future__ import annotations

from agents.perception.agent import PerceptionAgent, round_robin_selector
from shared.tile_validation import validate_tile_image
from tools.classifier.client import ClassifierClient
from tools.detector.client import DetectorClient
from tools.sam.client import SAMClient


def build_perception_agent(
    use_llm_selector: bool = False, tool_order: tuple[str, ...] | None = None
) -> PerceptionAgent:
    tools = [ClassifierClient(), DetectorClient(), SAMClient()]
    if tool_order:
        rank = {name: i for i, name in enumerate(tool_order)}
        tools.sort(key=lambda t: rank.get(t.name, len(rank)))
    selector = round_robin_selector
    if use_llm_selector:
        from agents.perception.llm_selector import make_llm_tool_selector
        from shared.gemini_vertex_client import GeminiVertexClient

        selector = make_llm_tool_selector(GeminiVertexClient())
    return PerceptionAgent(
        tools=tools,
        tool_selector=selector,
        tile_validator=validate_tile_image,
        run_auxiliary_tools=True,
    )


SECOND_OPINION_ORDER = ("sam", "detector", "classifier")


def build_second_opinion_agent(use_llm_selector: bool = False) -> PerceptionAgent:
    """Agent 1 configured for re-inspection: it consults the tools in a different
    order (SAM first), so a tile whose reading conflicts with its neighbours gets a
    genuinely independent second opinion instead of a repeat of the same call.
    """
    return build_perception_agent(use_llm_selector, tool_order=SECOND_OPINION_ORDER)
