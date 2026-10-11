"""Structured JSON audit lines on stdout, for Cloud Logging (SYS-02 / SEC-03 / SEC-02).

Cloud Run ships every stdout line to Cloud Logging, and a line that is a JSON object becomes a
structured entry: `severity` sets the level, `message` the text, the rest is searchable
(e.g. jsonPayload.tile_id="..."). Nothing here runs unless TRIAGE_JSON_LOGS=1, so local runs and
tests are unchanged.

What is logged is a WHITELIST built from the run's events, never the raw event:
  * per tool call:   run, tile, tool, label, confidence, score, cached, latency, tokens, phase, time
  * per failed call: run, tile, tool, attempt, error type, redacted+truncated message, time
  * per escalation / missing tile / fallback: severity WARNING or ERROR
Model output (selection reasoning, "sees:" evidence), image paths, service URLs and any field not
listed below are NOT written to the log (the full detail stays in data/runs/<run_id>/events.jsonl).
Free text that is logged is truncated and scrubbed of key/token-shaped strings.
"""
from __future__ import annotations

import json
import os
import re
import sys

MAX_TEXT = 200

_REDACTIONS = [
    (re.compile(r"AIza[0-9A-Za-z_\-]{20,}"), "[REDACTED_API_KEY]"),
    (re.compile(r"ya29\.[0-9A-Za-z_.\-]+"), "[REDACTED_TOKEN]"),
    (re.compile(r"eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]*"), "[REDACTED_JWT]"),
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/\-]+=*"), "Bearer [REDACTED]"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(-----END [A-Z ]*PRIVATE KEY-----|$)", re.S),
     "[REDACTED_PRIVATE_KEY]"),
    (re.compile(r"(?i)(api[_-]?key|secret|password|token)(\s*[=:]\s*)\S+"), r"\1\2[REDACTED]"),
]

# Scalar event fields that may be logged. Anything else in an event payload is dropped.
SCALAR_FIELDS = (
    "batch_id", "tile_id", "version", "round", "wave", "waves", "tile_count", "zone_count",
    "rounds_used", "transport", "agent", "disaster", "tools", "max_rounds", "wave_size",
)
# List fields are logged as a count only.
COUNTED_FIELDS = (
    "tile_ids", "zones", "reinspection_requests", "requests", "unresolved",
    "escalated_tile_ids", "reinspected_tile_ids", "versions", "tiles",
)
_WARNING_TYPES = {"warning", "transport_fallback", "reinspection_exhausted"}
_ERROR_TYPES = {"tile_missing"}


def json_logs_enabled() -> bool:
    return os.environ.get("TRIAGE_JSON_LOGS", "").strip().lower() in {"1", "true", "yes"}


def redact(text: object, limit: int = MAX_TEXT) -> str:
    s = str(text)
    for pattern, repl in _REDACTIONS:
        s = pattern.sub(repl, s)
    return s if len(s) <= limit else s[:limit] + "..."


def _scalar(value):
    if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return redact(value)
    return None  # lists / dicts / objects are never logged by value


def _base(event: dict, name: str, severity: str, message: str) -> dict:
    return {
        "severity": severity,
        "message": message,
        "event": name,
        "run_id": event.get("run_id"),
        "seq": event.get("seq"),
        "time": event.get("ts"),
    }


def _tool_lines(event: dict, tile: dict, phase: str) -> list[dict]:
    tid = tile.get("tile_id")
    out = []
    for i, c in enumerate(tile.get("tool_calls") or []):
        line = _base(event, "tool_call", "INFO", f"tool_call {c.get('tool')} tile={tid}")
        line.update(
            tile_id=tid, phase=phase, call_index=i, tool=_scalar(c.get("tool")),
            label=_scalar(c.get("label")), confidence=_scalar(c.get("confidence")),
            score=_scalar(c.get("score")), cached=_scalar(c.get("cached")),
            latency_s=_scalar(c.get("latency_s")),
            prompt_tokens=_scalar(c.get("prompt_tokens")),
            output_tokens=_scalar(c.get("output_tokens")),
        )
        out.append(line)
    for e in tile.get("errors") or []:
        line = _base(event, "tool_error", "WARNING", f"tool_error {e.get('tool')} tile={tid}")
        line.update(
            tile_id=tid, phase=phase, tool=_scalar(e.get("tool")), attempt=_scalar(e.get("attempt")),
            error_type=_scalar(e.get("error_type")), error_message=redact(e.get("message", "")),
            failed_at=_scalar(e.get("failed_at")),
        )
        out.append(line)
    if tile.get("status") == "escalated":
        line = _base(event, "tile_escalated", "WARNING", f"tile_escalated tile={tid}")
        line.update(
            tile_id=tid, phase=phase, tool_error_count=_scalar(tile.get("tool_errors")),
            failure_reason=redact(tile.get("failure_reason") or "tools exhausted without reaching the confidence threshold"),
        )
        out.append(line)
    return out


def build_lines(event: dict) -> list[dict]:
    """Whitelisted log lines (dicts) for one event. Pure function, no I/O."""
    etype = event.get("type", "unknown")
    severity = "ERROR" if etype in _ERROR_TYPES else "WARNING" if etype in _WARNING_TYPES else "INFO"
    tile = event.get("tile") or event.get("after") or {}
    tid = tile.get("tile_id") or event.get("tile_id")
    head = _base(event, etype, severity, f"{etype}" + (f" tile={tid}" if tid else ""))
    for key in SCALAR_FIELDS:
        if key in event:
            head[key] = _scalar(event[key])
    for key in COUNTED_FIELDS:
        if isinstance(event.get(key), list):
            head[f"{key}_count"] = len(event[key])
    if etype in ("warning", "transport_fallback"):
        head["detail"] = redact(event.get("message") or event.get("error") or "")
    if tile:
        head.update(tile_id=tid, tile_status=tile.get("status"), damage_severity=_scalar(tile.get("damage_severity")),
                    confidence=_scalar(tile.get("confidence")), tool_error_count=_scalar(tile.get("tool_errors")))
    lines = [head]
    if etype == "tile_finished":
        lines += _tool_lines(event, tile, "first_pass")
    elif etype == "tile_reinspected":
        lines += _tool_lines(event, tile, "reinspection")
    return lines


def write_audit_lines(event: dict, stream=None) -> None:
    out = stream if stream is not None else sys.stdout
    for line in build_lines(event):
        out.write(json.dumps(line, default=str, separators=(",", ":")) + "\n")
    out.flush()
