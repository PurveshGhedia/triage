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
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from shared.tool_interface import ToolClient, ToolResult
from tools.vision.prompts import (
    ACCESS_LABELS,
    ACCESS_PROMPT,
    DAMAGE_FRACTIONS_PROMPT,
    DAMAGE_LABELS,
    DAMAGE_PROMPT,
    PROMPT_VERSION,
    SECOND_PASS_FRACTIONS_PROMPT,
    SECOND_PASS_PROMPT,
)

VARIANTS = ("classes", "fractions")

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_CACHE_DIR = Path("data/cache/vision")
MAX_SIDE_PX = 768

Image = tuple[bytes, str]  # (bytes, mime type)
# A generate function returns the model's text, or (text, usage) where usage is
# {"prompt_tokens": int, "output_tokens": int} (what the call was billed for).
GenerateFn = Callable[[list[Image], str, type[BaseModel]], "str | tuple[str, dict]"]


class DamageReading(BaseModel):
    label: Literal["no-damage", "minor-damage", "major-damage", "destroyed"]
    confidence: float
    evidence: str


class FractionReading(BaseModel):
    no_damage: float
    minor_damage: float
    major_damage: float
    destroyed: float
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


def parse_fractions(text: str) -> dict:
    """Shares of buildings per damage level -> {label, score, confidence, evidence, fractions}.

    The score is the expected damage on the 0-3 scale (the same scale as xBD's per-tile mean); the
    label is the nearest class to it. Shares are re-normalised to sum to 1 if they are close; wildly
    wrong ones raise ValueError so the agent retries."""
    if not text or not text.strip():
        raise ValueError("empty response from vision model")
    try:
        data = json.loads(_FENCE.sub("", text.strip()))
    except json.JSONDecodeError as exc:
        raise ValueError(f"vision model did not return JSON: {text[:120]!r}") from exc
    if not isinstance(data, dict):
        raise ValueError("vision model JSON was not an object")
    keys = ["no_damage", "minor_damage", "major_damage", "destroyed"]
    try:
        shares = [float(data[k]) for k in keys]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"missing or non-numeric damage shares in {text[:120]!r}") from exc
    if any(x < 0 or x > 1.0001 for x in shares):
        raise ValueError(f"damage shares must each be between 0 and 1, got {shares}")
    total = sum(shares)
    if not 0.5 <= total <= 1.5:
        raise ValueError(f"damage shares should add up to about 1, got {total:.2f}")
    shares = [x / total for x in shares]
    score = round(sum(i * x for i, x in enumerate(shares)), 3)
    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    return {
        "label": DAMAGE_LABELS[min(3, int(score + 0.5))],
        "score": score,
        "confidence": round(min(1.0, max(0.0, confidence)), 2),
        "evidence": str(data.get("evidence") or "").strip()[:400],
        "fractions": [round(x, 3) for x in shares],
    }


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

    def generate(images: list[Image], prompt: str, schema: type[BaseModel]) -> tuple[str, dict]:
        parts = [types.Part.from_bytes(data=b, mime_type=m) for b, m in images] + [prompt]
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.0,
            max_output_tokens=600,
            thinking_config=types.ThinkingConfig(thinking_budget=0),  # fast, and no tokens lost to thinking
        )
        response = client.models.generate_content(model=model, contents=parts, config=config)
        meta = getattr(response, "usage_metadata", None)
        usage = {
            "prompt_tokens": int(getattr(meta, "prompt_token_count", 0) or 0),
            # thinking tokens (if any) are billed as output tokens
            "output_tokens": int(getattr(meta, "candidates_token_count", 0) or 0)
                             + int(getattr(meta, "thoughts_token_count", 0) or 0),
        }
        return response.text or "", usage

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

    def _generate(self, images: list[Image]) -> tuple[str, dict | None, float]:
        """(text, usage or None, seconds the model call took)."""
        if self._generate_fn is None:  # built lazily so importing this module needs no GCP setup
            self._generate_fn = make_vertex_generate_fn(model=self.model)
        started = time.perf_counter()
        out = self._generate_fn(images, self.prompt, self.schema)
        seconds = round(time.perf_counter() - started, 3)
        text, usage = out if isinstance(out, tuple) else (out, None)
        return text, usage, seconds

    def _cache_key(self, images: list[Image]) -> str:
        h = hashlib.sha256()
        for data, mime in images:
            h.update(hashlib.sha256(data).digest())
            h.update(mime.encode())
        h.update(f"|{self.name}|{self.model}|{PROMPT_VERSION}|{self.prompt}".encode())
        return h.hexdigest()

    def _parse(self, text: str) -> dict:
        label, confidence, evidence = parse_reading(text, self.labels)
        return {"label": label, "confidence": confidence, "evidence": evidence, "score": None}

    def _check_cached(self, data: dict) -> dict:
        """Validate a cache entry; raises ValueError if it is unusable."""
        if data.get("label") not in self.labels:
            raise ValueError("bad cached label")
        data["confidence"] = float(data["confidence"])
        if data.get("score") is not None:
            data["score"] = float(data["score"])
        return data

    def run(self, tile_path: str) -> ToolResult:
        images: list[Image] = []
        pre = pre_path_for(tile_path) if self.use_pre else None
        if pre is not None:
            images.append(load_image(pre))
        images.append(load_image(tile_path))

        cache_file = self.cache_dir / f"{self.name}-{self._cache_key(images)}.json"
        cached = False
        reading: dict = {}
        if cache_file.is_file():
            try:
                reading = self._check_cached(json.loads(cache_file.read_text()))
                cached = True
            except (ValueError, KeyError, TypeError):
                cache_file.unlink(missing_ok=True)  # corrupt cache entry: ask the model again
        if not cached:
            text, usage, seconds = self._generate(images)
            reading = self._parse(text)  # raises -> agent retries
            reading["usage"], reading["latency_s"] = usage, seconds
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(reading))

        return ToolResult(
            tool_name=self.name,
            confidence=reading["confidence"],
            label=reading["label"],
            raw_output={"evidence": reading.get("evidence", ""), "score": reading.get("score"),
                        "fractions": reading.get("fractions"), "model": self.model, "cached": cached,
                        "usage": reading.get("usage"), "latency_s": reading.get("latency_s"),
                        "images": len(images), "has_before": pre is not None},
        )


class GeminiDamageTool(_VisionTool):
    """Damage level on the xBD four-class scale, from the before/after pair.

    variant 'classes' (default): one level for the tile. variant 'fractions': the share of
    buildings at each level, from which a continuous 0-3 score is computed (matching xBD's
    per-tile mean). Choose with the argument or the TRIAGE_VISION_VARIANT environment variable."""

    labels = DAMAGE_LABELS
    schema = DamageReading
    provides_damage = True

    def __init__(self, name: str = "classifier", second_pass: bool = False,
                 variant: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = name
        self.variant = variant or os.environ.get("TRIAGE_VISION_VARIANT", "classes")
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {self.variant!r}")
        if self.variant == "fractions":
            self.schema = FractionReading
            self.prompt = SECOND_PASS_FRACTIONS_PROMPT if second_pass else DAMAGE_FRACTIONS_PROMPT
        else:
            self.prompt = SECOND_PASS_PROMPT if second_pass else DAMAGE_PROMPT
        self.description = (
            "Gemini building-by-building second review of the before/after pair (xBD damage scale)."
            if second_pass else
            "Gemini damage assessment of the before/after pair on the xBD scale "
            "(no-damage / minor-damage / major-damage / destroyed)."
        )

    def _parse(self, text: str) -> dict:
        if self.variant == "fractions":
            return parse_fractions(text)
        return super()._parse(text)


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
