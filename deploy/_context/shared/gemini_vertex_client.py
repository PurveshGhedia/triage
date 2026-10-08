"""Gemini via Vertex AI — the credit-covered reasoning backend.

Unlike Claude on Vertex (a third-party "partner model" offered as
Model-as-a-Service), Gemini is Google's own first-party model, so it IS
covered by the $300 Free Trial credit. Same ReasoningClient interface as
ClaudeVertexClient, so nothing in agents/perception/llm_selector.py or
agent.py needs to change to use this instead — only which class gets
instantiated.

Requires the same `gcloud auth application-default login` ADC already set up
for Vertex AI / Firestore, and GCP_PROJECT_ID / VERTEX_AI_LOCATION in .env.
"""
from __future__ import annotations

from google import genai
from google.genai import types

from shared.config import GCP_PROJECT_ID, GEMINI_MODEL_ID, VERTEX_AI_LOCATION
from shared.reasoning_client import ReasoningClient


class GeminiVertexClient(ReasoningClient):
    def __init__(
        self,
        project_id: str | None = None,
        location: str | None = None,
        model: str | None = None,
    ) -> None:
        self._project_id = project_id or GCP_PROJECT_ID
        if not self._project_id:
            raise ValueError(
                "GCP_PROJECT_ID is not set. Check that .env exists at the repo "
                "root and contains GCP_PROJECT_ID=<your-project-id>."
            )
        self._location = location or VERTEX_AI_LOCATION
        self._model = model or GEMINI_MODEL_ID
        self._client = genai.Client(
            vertexai=True, project=self._project_id, location=self._location
        )

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.models.generate_content(
            model=self._model,
            contents=user_prompt,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
        )
        return response.text or ""
