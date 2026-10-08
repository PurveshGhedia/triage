"""Print every prompt the agents send to Gemini, exactly as sent.

    python -m scripts.show_prompts                 # to the screen
    python -m scripts.show_prompts > prompts.md    # to a file you can put in the report

Agent 1 uses Gemini twice: (1) the vision tools look at the before/after images; (2) optionally, a tool
selector chooses which tool to try next. Agent 2 (the planner) does not use an LLM: it is deterministic code.
"""
from __future__ import annotations

from tools.vision.prompts import (
    ACCESS_PROMPT,
    DAMAGE_FRACTIONS_PROMPT,
    DAMAGE_PROMPT,
    PROMPT_VERSION,
    SECOND_PASS_FRACTIONS_PROMPT,
    SECOND_PASS_PROMPT,
)

VISION = [
    ("Damage assessment: variant 'classes' (tool name: classifier)", DAMAGE_PROMPT),
    ("Damage assessment: variant 'fractions' (tool name: classifier)", DAMAGE_FRACTIONS_PROMPT),
    ("Second opinion: variant 'classes' (tool name: sam)", SECOND_PASS_PROMPT),
    ("Second opinion: variant 'fractions' (tool name: sam)", SECOND_PASS_FRACTIONS_PROMPT),
    ("Road access (tool name: detector)", ACCESS_PROMPT),
]


def selector_section() -> str:
    """The tool-selector prompts, if that module is importable in this checkout."""
    try:
        from agents.perception.prompts import SYSTEM_PROMPT, build_user_prompt
        from agents.perception.schemas import ToolCallRecord
        from tools.classifier.client import ClassifierClient
        from tools.detector.client import DetectorClient
        from tools.sam.client import SAMClient
    except ImportError as exc:
        return f"(tool-selector prompts not available here: {exc})"
    history = [ToolCallRecord(tool_name="classifier", confidence=0.55, label="minor-damage")]
    remaining = [DetectorClient(), SAMClient()]
    return (f"### System prompt\n\n{SYSTEM_PROMPT}\n\n### Example user prompt "
            f"(after a low-confidence classifier reading; detector and SAM still untried)\n\n"
            f"{build_user_prompt(history, remaining)}")


def render() -> str:
    parts = [
        "# Prompts sent by the agents",
        "",
        "Images are attached to the vision prompts: the BEFORE image first, then the AFTER image, "
        "downscaled to at most 768 px. Replies are forced into a JSON schema (an enumerated label or four "
        f"damage shares, a confidence, and an evidence sentence). Prompt version: {PROMPT_VERSION}.",
        "",
        "## Agent 1, vision tools (Gemini 2.5 Flash on Vertex AI)",
    ]
    for title, text in VISION:
        parts += ["", f"### {title}", "", text]
    parts += ["", "## Agent 1, tool selector (optional: Gemini picks which tool to try next)", "", selector_section(),
              "", "## Agent 2, planner", "",
              "No LLM and no prompt. It clusters tiles into zones with a distance graph, scores each zone from "
              "the tools' damage and road-access readings, detects neighbouring tiles that disagree, and asks "
              "Agent 1 to re-inspect them. Everything it does is deterministic and unit-tested."]
    return "\n".join(parts)


def main() -> None:
    print(render())


if __name__ == "__main__":
    main()
