"""Claude via Vertex AI Model Garden — the production reasoning backend.

NOTE: as of the $300 Free Trial credit research, Claude on Vertex AI is a
third-party "partner model" (Model-as-a-Service), which the free trial
credit explicitly does NOT cover. Use GeminiVertexClient instead while on
the trial — this class is kept for later (once on a paid billing account)
and to demonstrate genuine model-agnosticism per the architecture doc.

Requires `gcloud auth application-default login` locally (not an API key in
.env) since Vertex AI authenticates via Google's Application Default
Credentials. Project/region/model come from shared/config.py.
"""
from __future__ import annotations

from anthropic import AnthropicVertex

from shared.config import ANTHROPIC_VERTEX_PROJECT_ID, CLAUDE_MODEL_ID, VERTEX_AI_LOCATION
from shared.reasoning_client import ReasoningClient


class ClaudeVertexClient(ReasoningClient):
    def __init__(
        self,
        project_id: str | None = None,
        region: str | None = None,
        model: str | None = None,
    ) -> None:
        self._project_id = project_id or ANTHROPIC_VERTEX_PROJECT_ID
        if not self._project_id:
            raise ValueError(
                "ANTHROPIC_VERTEX_PROJECT_ID (or GCP_PROJECT_ID) is not set. "
                "Check that .env exists at the repo root."
            )
        self._region = region or VERTEX_AI_LOCATION
        self._model = model or CLAUDE_MODEL_ID
        self._client = AnthropicVertex(project_id=self._project_id, region=self._region)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=300,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")
