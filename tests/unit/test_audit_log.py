"""Structured audit logging for Cloud Logging (SYS-02, SEC-03, SEC-02 at runtime)."""
import json

import pytest

from agents.orchestrator import audit_log
from agents.orchestrator.events import EventLog

KEY = "AIza" + "SyA1234567890abcdefghijklmnopqrstuv"
TOKEN = "ya29.A0ARrdaM-secret-token-value"


def call(tool="classifier", conf=0.82):
    return {"tool": tool, "label": "major-damage", "confidence": conf, "score": 2.1,
            "reasoning": "SECRET-REASONING", "evidence": "SECRET-EVIDENCE", "latency_s": 3.0,
            "prompt_tokens": 1590, "output_tokens": 96, "cached": False}


def summary(tile_id="t1", status="assessed", calls=None, errors=None, failure_reason=None):
    return {"tile_id": tile_id, "status": status, "damage_severity": "major-damage", "confidence": 0.82,
            "latitude": 1.0, "longitude": 2.0, "failure_reason": failure_reason,
            "tool_errors": len(errors or []), "errors": errors or [], "tool_calls": calls or []}


def lines(capsys):
    out = capsys.readouterr().out
    return [json.loads(x) for x in out.splitlines() if x.strip()]


def test_off_by_default(capsys, monkeypatch):
    monkeypatch.delenv("TRIAGE_JSON_LOGS", raising=False)
    EventLog("r").emit("tile_finished", batch_id="b", tile=summary(calls=[call()]))
    assert capsys.readouterr().out == ""


def test_env_switch_turns_it_on(capsys, monkeypatch):
    monkeypatch.setenv("TRIAGE_JSON_LOGS", "1")
    EventLog("r").emit("run_started", batch_id="b", tile_count=3)
    assert lines(capsys)[0]["event"] == "run_started"


def test_every_line_is_json_with_severity_and_run_id(capsys):
    log = EventLog("run-9", json_logs=True)
    log.emit("tile_finished", batch_id="b", tile=summary(calls=[call("classifier"), call("sam", 0.9)]))
    got = lines(capsys)
    assert got and all(l["severity"] and l["run_id"] == "run-9" and l["time"] for l in got)


def test_one_line_per_tool_call_with_audit_fields(capsys):
    EventLog("r", json_logs=True).emit(
        "tile_finished", batch_id="b", tile=summary("tile_7", calls=[call("classifier", 0.6), call("sam", 0.9)]))
    calls = [l for l in lines(capsys) if l["event"] == "tool_call"]
    assert [(c["tile_id"], c["tool"], c["confidence"]) for c in calls] == [
        ("tile_7", "classifier", 0.6), ("tile_7", "sam", 0.9)]
    assert all(c["time"] and c["phase"] == "first_pass" for c in calls)


def test_sec03_retried_and_escalated_calls_all_appear(capsys):
    """SEC-03: successful, retried and escalated attempts each leave a line with tile, tool, time."""
    errors = [{"tool": "sam", "attempt": 1, "error_type": "ServiceUnavailable",
               "message": "endpoint restarting", "failed_at": "2026-10-11T12:00:00+00:00"}]
    EventLog("r", json_logs=True).emit(
        "tile_finished", batch_id="b",
        tile=summary("t9", status="escalated", calls=[call("classifier", 0.4)], errors=errors))
    got = lines(capsys)
    kinds = {l["event"] for l in got}
    assert {"tool_call", "tool_error", "tile_escalated"} <= kinds
    err = next(l for l in got if l["event"] == "tool_error")
    assert (err["tile_id"], err["tool"], err["attempt"], err["severity"]) == ("t9", "sam", 1, "WARNING")
    assert next(l for l in got if l["event"] == "tile_escalated")["severity"] == "WARNING"


def test_reinspection_calls_are_tagged_with_their_phase(capsys):
    EventLog("r", json_logs=True).emit(
        "tile_reinspected", batch_id="b", round=1, before=summary(calls=[call()]),
        after=summary(calls=[call(), call("sam", 0.9)]))
    assert {l["phase"] for l in lines(capsys) if l["event"] == "tool_call"} == {"reinspection"}


def test_missing_tile_is_error_and_fallback_is_warning(capsys):
    log = EventLog("r", json_logs=True)
    log.emit("tile_missing", batch_id="b", tile_id="gone")
    log.emit("transport_fallback", agent="perception", error="boom")
    sev = {l["event"]: l["severity"] for l in lines(capsys)}
    assert sev == {"tile_missing": "ERROR", "transport_fallback": "WARNING"}


def test_model_text_urls_and_unknown_fields_never_reach_the_log(capsys):
    """SEC-02 at runtime: reasoning/evidence, service URLs and stray fields are dropped."""
    log = EventLog("r", json_logs=True)
    log.emit("tile_finished", batch_id="b", tile=summary(calls=[call()]), api_key="sk-LEAK")
    log.emit("transport_selected", transport="a2a", agent1_url="https://agent1.example/secret-path")
    out = json.dumps(lines(capsys))
    for forbidden in ("SECRET-REASONING", "SECRET-EVIDENCE", "sk-LEAK", "secret-path"):
        assert forbidden not in out


def test_key_and_token_shaped_strings_are_redacted(capsys):
    errors = [{"tool": "sam", "attempt": 1, "error_type": "PermissionDenied",
               "message": f"401 key={KEY} token {TOKEN} Authorization: Bearer abc.def.ghi", "failed_at": "t"}]
    log = EventLog("r", json_logs=True)
    log.emit("tile_finished", batch_id="b", tile=summary(errors=errors))
    log.emit("warning", message=f"oops {TOKEN}")
    out = json.dumps(lines(capsys))
    for secret in (KEY, TOKEN, "abc.def.ghi"):
        assert secret not in out
    assert "REDACTED" in out


def test_long_text_is_truncated():
    assert len(audit_log.redact("x" * 5000)) < audit_log.MAX_TEXT + 10


def test_counts_replace_big_lists(capsys):
    EventLog("r", json_logs=True).emit("run_started", batch_id="b", tile_ids=["a", "b", "c"],
                                        tiles=[{"tile_id": "a", "latitude": 1}])
    line = lines(capsys)[0]
    assert line["tile_ids_count"] == 3 and "tile_ids" not in line and "tiles" not in line


def test_a_logging_failure_never_breaks_the_run(monkeypatch):
    def boom(_event):
        raise RuntimeError("stdout closed")
    monkeypatch.setattr(audit_log, "write_audit_lines", boom)
    event = EventLog("r", json_logs=True).emit("run_started", batch_id="b")
    assert event["type"] == "run_started"


def test_tile_summary_carries_failed_attempts_for_the_audit_trail():
    """The pipeline's tile_summary must expose tool_errors, or retries never reach the log."""
    from agents.orchestrator.pipeline import tile_summary
    from agents.perception.schemas import TileAssessment, TileStatus, ToolErrorRecord

    a = TileAssessment(tile_id="t1", status=TileStatus.ESCALATED, tool_errors=[
        ToolErrorRecord(tool_name="sam", attempt=1, error_type="Unavailable", message=f"down {KEY}")])
    errs = tile_summary(a)["errors"]
    assert errs[0]["tool"] == "sam" and errs[0]["attempt"] == 1 and errs[0]["failed_at"]
    assert KEY not in errs[0]["message"]
