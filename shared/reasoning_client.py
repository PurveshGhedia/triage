"""Abstract reasoning-layer interface — the model-agnostic LLM backend used by
both agents (Agent 1's tool selection, Agent 2's planning/scoring later).
Swapping between Claude and Gemini, or between Vertex AI and a direct API key,
means implementing this interface once — nothing in the agent loops changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class ReasoningClient(ABC):
    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return the model's raw text response for a single-turn completion."""
        raise NotImplementedError
