"""Shared abstract interface all Agent 1 tools (classifier, detector, SAM) implement.

Keeping this interface tool-agnostic lets Agent 1's tool-selection loop call any
tool through the same `.run()` signature, whether the tool is a local mock, a
Vertex AI Endpoint, or something swapped in later.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """Normalized output from any Agent 1 tool."""

    tool_name: str
    confidence: float  # 0.0-1.0
    label: str  # e.g. damage severity class, or detected object class
    raw_output: dict[str, Any] = field(default_factory=dict)


class ToolClient(ABC):
    """Base class for classifier / detector / SAM clients."""

    name: str

    @abstractmethod
    def run(self, tile_path: str) -> ToolResult:
        """Run this tool on a tile image and return a normalized result."""
        raise NotImplementedError
