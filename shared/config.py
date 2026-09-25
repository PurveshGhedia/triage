"""Centralized environment/config loading — reads .env once via python-dotenv
and exposes the settings the rest of the app needs, so nothing else calls
os.getenv directly for these values.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_REGION = os.getenv("GCP_REGION", "us-central1")

VERTEX_AI_LOCATION = os.getenv("VERTEX_AI_LOCATION", "us-central1")
ANTHROPIC_VERTEX_PROJECT_ID = os.getenv("ANTHROPIC_VERTEX_PROJECT_ID", GCP_PROJECT_ID)
CLAUDE_MODEL_ID = os.getenv("CLAUDE_MODEL_ID", "claude-sonnet-4-6")

FIRESTORE_COLLECTION_TILES = os.getenv("FIRESTORE_COLLECTION_TILES", "tile_assessments")

USE_MOCK_TOOLS = os.getenv("USE_MOCK_TOOLS", "true").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
