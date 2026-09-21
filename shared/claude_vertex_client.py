"""Claude via Vertex AI Model Garden — the production reasoning backend.

Requires `gcloud auth application-default login` locally (not an API key in
.env) since Vertex AI authenticates via Google's Application Default
Credentials. Project/region/model come from .env — see .env.example.
"""
from __future__ import annotations

import os

from anthropic import AnthropicVertex

from shared.reasoning_client import ReasoningClient


class ClaudeVertexClient(ReasoningClient):
    def __init__(
        self,
        project_id: str | None = None,
        region: str | None = None,
        model: str | None = None,
    ) -> None:
        self._project_id = project_id or os.environ["ANTHROPIC_VERTEX_PROJECT_ID"]
        self._region = region or os.getenv("VERTEX_AI_LOCATION", "us-central1")
        self._model = model or os.getenv("CLAUDE_MODEL_ID", "claude-sonnet-4-6")
        self._client = AnthropicVertex(project_id=self._project_id, region=self._region)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=300,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")
