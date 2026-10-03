"""Gemini-vision perception tools.

These are real model calls on the real xBD imagery (Gemini on Vertex AI, paid by the
GCP credits; no endpoint to host). They are NOT trained xBD models: Gemini is shown the
BEFORE and AFTER images and asked for an xBD damage level or a road-access level, with
a short statement of what it sees. Its self-reported confidence is only roughly
calibrated, so measure accuracy against xBD ground truth (scripts.evaluate_run).

Reliability for demos:
  * answers are cached on disk, keyed by the image bytes + model + prompt version, so a
    repeat run is instant and identical (warm the cache once before presenting);
  * any failure (quota, timeout, unparseable answer) simply raises, and Agent 1's existing
    retry / fallback / escalation logic handles it.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from shared.tool_interface import ToolClient, ToolResult
from tools.vision.prompts import (
    ACCESS_LABELS,
    ACCESS_PROMPT,
    DAMAGE_LABELS,
    DAMAGE_PROMPT,
    PROMPT_VERSION,
    SECOND_PASS_PROMPT,
)

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_CACHE_DIR = Path("data/cache/vision")
MAX_SIDE_PX = 768

Image = tuple[bytes, str]  # (bytes, mime type)
GenerateFn = Callable[[list[Image], str, type[BaseModel]], str]


class DamageReading(BaseModel):
    label: Literal["no-damage", "minor-damage", "major-damage", "destroyed"]
    confidence: float
    evidence: str


class AccessReading(BaseModel):
    label: Literal["access-clear", "access-degraded", "access-blocked"]
    confidence: float
    evidence: str


# --- images ----------------------------------------------------------------------------


def pre_path_for(post_path: str | Path) -> Path | None:
    """The BEFORE image for a post-disaster tile, if there is one next to it.

    Supports the demo layout (.../images/X_post_disaster.png -> .../pre/X_pre_disaster.png)
    and xBD's own layout (same folder, _post_ -> _pre_)."""
    p = Path(post_path)
    pre_name = p.name.replace("_post_disaster", "_pre_disaster")
    if pre_name == p.name:
        return None
    for candidate in (p.parent.parent / "pre" / pre_name, p.parent / pre_name):
        if candidate.is_file():
            return candidate
    return None


def load_image(path: str | Path, max_side: int = MAX_SIDE_PX) -> Image:
    """Image bytes ready to send, shrunk to `max_side` px if Pillow is available (a full
    1024 px PNG costs far more tokens and time for no gain in this task)."""
    raw = Path(path).read_bytes()
    try:
        from PIL import Image as PILImage
    except ImportError:
        return raw, "image/png" if raw[:4] == b"\x89PNG" else "image/jpeg"
    with PILImage.open(io.BytesIO(raw)) as im:
        im = im.convert("RGB")
        if max(im.size) > max_side:
            scale = max_side / max(im.size)
            im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))))
        out = io.BytesIO()
        im.save(out, format="JPEG", quality=88)
    return out.getvalue(), "image/jpeg"


# --- parsing ---------------------------------------------------------------------------

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_reading(text: str, allowed_labels: list[str]) -> tuple[str, float, str]:
    """(label, confidence in [0,1], evidence) from the model's JSON answer.
    Raises ValueError for anything unusable, so the caller's retry logic takes over."""
    if not text or not text.strip():
        raise ValueError("empty response from vision model")
    try:
        data = json.loads(_FENCE.sub("", text.strip()))
    except json.JSONDecodeError as exc:
        raise ValueError(f"vision model did not return JSON: {text[:120]!r}") from exc
    if not isinstance(data, dict):
        raise ValueError("vision model JSON was not an object")
    label = data.get("label")
    if label not in allowed_labels:
        raise ValueError(f"unexpected label {label!r}; expected one of {allowed_labels}")
    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = min(1.0, max(0.0, confidence))
    return label, round(confidence, 2), str(data.get("evidence") or "").strip()[:400]


# --- the real model call -----------------------------------------------------------------


def make_vertex_generate_fn(
    project: str | None = None, location: str | None = None, model: str | None = None
) -> GenerateFn:
    """A generate function backed by Gemini on Vertex AI (Application Default Credentials)."""
    from google import genai
    from google.genai import types

    project = project or os.environ.get("GCP_PROJECT_ID")
    location = location or os.environ.get("VERTEX_AI_LOCATION", "us-central1")
    model = model or os.environ.get("VISION_MODEL", DEFAULT_MODEL)
    client = genai.Client(vertexai=True, project=project, location=location)

    def generate(images: list[Image], prompt: str, schema: type[BaseModel]) -> str:
        parts = [types.Part.from_bytes(data=b, mime_type=m) for b, m in images] + [prompt]
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.0,
            max_output_tokens=600,
            thinking_config=types.ThinkingConfig(thinking_budget=0),  # fast, and no tokens lost to thinking
        )
        response = client.models.generate_content(model=model, contents=parts, config=config)
        return response.text or ""

    return generate


# --- tools -------------------------------------------------------------------------------


class _VisionTool(ToolClient):
    labels: list[str]
    prompt: str
    schema: type[BaseModel]

    def __init__(
        self,
        generate_fn: GenerateFn | None = None,
        cache_dir: str | Path | None = None,
        model: str | None = None,
        use_pre: bool = True,
    ) -> None:
        self._generate_fn = generate_fn
        self.model = model or os.environ.get("VISION_MODEL", DEFAULT_MODEL)
        self.cache_dir = Path(cache_dir or os.environ.get("TRIAGE_VISION_CACHE", DEFAULT_CACHE_DIR))
        self.use_pre = use_pre

    def _generate(self, images: list[Image]) -> str:
        if self._generate_fn is None:  # built lazily so importing this module needs no GCP setup
            self._generate_fn = make_vertex_generate_fn(model=self.model)
        return self._generate_fn(images, self.prompt, self.schema)

    def _cache_key(self, images: list[Image]) -> str:
        h = hashlib.sha256()
        for data, mime in images:
            h.update(hashlib.sha256(data).digest())
            h.update(mime.encode())
        h.update(f"|{self.name}|{self.model}|{PROMPT_VERSION}|{self.prompt}".encode())
        return h.hexdigest()

    def run(self, tile_path: str) -> ToolResult:
        images: list[Image] = []
        pre = pre_path_for(tile_path) if self.use_pre else None
        if pre is not None:
            images.append(load_image(pre))
        images.append(load_image(tile_path))

        cache_file = self.cache_dir / f"{self.name}-{self._cache_key(images)}.json"
        cached = False
        if cache_file.is_file():
            try:
                label, confidence, evidence = parse_reading(cache_file.read_text(), self.labels)
                cached = True
            except ValueError:
                cache_file.unlink(missing_ok=True)  # corrupt cache entry: ask the model again
        if not cached:
            text = self._generate(images)
            label, confidence, evidence = parse_reading(text, self.labels)  # raises -> agent retries
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps({"label": label, "confidence": confidence, "evidence": evidence}))

        return ToolResult(
            tool_name=self.name,
            confidence=confidence,
            label=label,
            raw_output={"evidence": evidence, "model": self.model, "cached": cached,
                        "images": len(images), "has_before": pre is not None},
        )


class GeminiDamageTool(_VisionTool):
    """Damage level on the xBD four-class scale, from the before/after pair."""

    labels = DAMAGE_LABELS
    schema = DamageReading
    provides_damage = True

    def __init__(self, name: str = "classifier", second_pass: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = name
        self.prompt = SECOND_PASS_PROMPT if second_pass else DAMAGE_PROMPT
        self.description = (
            "Gemini building-by-building second review of the before/after pair (xBD damage scale)."
            if second_pass else
            "Gemini damage assessment of the before/after pair on the xBD scale "
            "(no-damage / minor-damage / major-damage / destroyed)."
        )


class GeminiAccessTool(_VisionTool):
    """Road accessibility for emergency vehicles. Not a damage reading."""

    labels = ACCESS_LABELS
    schema = AccessReading
    provides_damage = False
    prompt = ACCESS_PROMPT

    def __init__(self, name: str = "detector", **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = name
        self.description = "Gemini road-access check: are roads and bridges passable for emergency vehicles?"
