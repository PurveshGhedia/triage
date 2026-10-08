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


TOOLS_MODES = ("mock", "vision")


def build_tools(tools_mode: str = "mock", vision_variant: str | None = None) -> list:
    """The three perception tools. 'mock': deterministic stand-ins that ignore the image.
    'vision': Gemini looks at the real before/after imagery (see tools/vision). The tool
    names stay classifier / detector / sam because Agent 2's scoring is keyed on them."""
    if tools_mode == "mock":
        return [ClassifierClient(), DetectorClient(), SAMClient()]
    if tools_mode == "vision":
        from tools.vision.client import GeminiAccessTool, GeminiDamageTool

        return [GeminiDamageTool("classifier", variant=vision_variant), GeminiAccessTool("detector"),
                GeminiDamageTool("sam", second_pass=True, variant=vision_variant)]
    raise ValueError(f"tools must be one of {TOOLS_MODES}, got {tools_mode!r}")


def build_perception_agent(
    use_llm_selector: bool = False,
    tool_order: tuple[str, ...] | None = None,
    tools_mode: str = "mock",
    vision_variant: str | None = None,
) -> PerceptionAgent:
    tools = build_tools(tools_mode, vision_variant)
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


def build_second_opinion_agent(use_llm_selector: bool = False, tools_mode: str = "mock",
                               vision_variant: str | None = None) -> PerceptionAgent:
    """Agent 1 configured for re-inspection: it consults the tools in a different
    order (SAM first), so a tile whose reading conflicts with its neighbours gets a
    genuinely independent second opinion instead of a repeat of the same call.
    """
    return build_perception_agent(use_llm_selector, tool_order=SECOND_OPINION_ORDER,
                                  tools_mode=tools_mode, vision_variant=vision_variant)


def build_in_process_perception(tools: str = "mock", use_llm_selector: bool = False,
                                vision_variant: str | None = None, concurrency: int = 4):
    """Agent 1 plus its second-opinion twin, ready to serve a batch (used by the runner and the A2A service)."""
    from agents.orchestrator.ports import InProcessPerception

    return InProcessPerception(
        agent=build_perception_agent(use_llm_selector, tools_mode=tools, vision_variant=vision_variant),
        reinspection_agent=build_second_opinion_agent(use_llm_selector, tools_mode=tools,
                                                      vision_variant=vision_variant),
        max_workers=concurrency,
    )
