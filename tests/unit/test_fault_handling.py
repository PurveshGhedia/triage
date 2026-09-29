"""FLT-01 (corrupted tile), FLT-02 (tool endpoint unavailable), and the fault
paths' interaction with Agent 2 (SYS-04 / PLN-04).
"""
from __future__ import annotations

import struct
import zlib

import pytest

from agents.perception.agent import PerceptionAgent
from agents.perception.schemas import TileStatus
from agents.planner.agent import TriagePlanner
from agents.planner.schemas import ZoneStatus
from shared.reinspection_dispatcher import LocalReinspectionDispatcher
from shared.tile_validation import CorruptTileError, validate_tile_image
from shared.tool_interface import ToolClient, ToolResult


# --- helpers ---------------------------------------------------------------


def make_png(width: int = 2, height: int = 2) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    row = b"\x00" + b"\x80\x80\x80" * width
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(row * height))
        + chunk(b"IEND", b"")
    )


class ScriptedTool(ToolClient):
    """Fails `fail_times` times (or forever if None) with `error`, then succeeds."""

    def __init__(self, name, confidence=0.95, label="destroyed", fail_times=0, error=None):
        self.name = name
        self._confidence = confidence
        self._label = label
        self._fail_times = fail_times
        self._error = error or ConnectionError("endpoint unavailable")
        self.calls = 0

    def run(self, tile_path):
        self.calls += 1
        if self._fail_times is None or self.calls <= self._fail_times:
            raise self._error
        return ToolResult(self.name, self._confidence, self._label)


def agent_for(tools, sleeps=None, **kwargs):
    sleeps = sleeps if sleeps is not None else []
    return PerceptionAgent(tools=tools, sleep=sleeps.append, **kwargs)


# --- FLT-01: corrupted tile image -----------------------------------------


def test_valid_png_passes_validation(tmp_path):
    good = tmp_path / "good.png"
    good.write_bytes(make_png())
    validate_tile_image(good)  # no exception


@pytest.mark.parametrize(
    "content, fragment",
    [
        (b"", "empty"),
        (b"this is not an image at all", "not a recognised image"),
        (make_png()[:-10], "truncated"),
        (make_png()[:20], "truncated"),
    ],
    ids=["empty", "garbage", "png-missing-end", "png-cut-early"],
)
def test_bad_files_are_rejected_with_a_reason(tmp_path, content, fragment):
    bad = tmp_path / "bad.png"
    bad.write_bytes(content)
    with pytest.raises(CorruptTileError, match=fragment):
        validate_tile_image(bad)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(CorruptTileError, match="not found"):
        validate_tile_image(tmp_path / "ghost.png")


def test_jpeg_structural_checks(tmp_path):
    ok = tmp_path / "ok.jpg"
    ok.write_bytes(b"\xff\xd8\xff\xe0" + b"0" * 20 + b"\xff\xd9")
    validate_tile_image(ok, decode=False)

    cut = tmp_path / "cut.jpg"
    cut.write_bytes(b"\xff\xd8\xff\xe0" + b"0" * 20)
    with pytest.raises(CorruptTileError, match="truncated"):
        validate_tile_image(cut, decode=False)


def test_corrupt_tile_is_escalated_gracefully_and_no_tool_runs(tmp_path):
    truncated = tmp_path / "t1.png"
    truncated.write_bytes(make_png()[:-10])
    tool = ScriptedTool("classifier")

    assessment = agent_for([tool], tile_validator=validate_tile_image).process_tile(
        "t1", str(truncated), batch_id="A"
    )

    assert assessment.status == TileStatus.ESCALATED
    assert assessment.batch_id == "A"
    assert "unreadable tile image" in assessment.failure_reason
    assert assessment.tool_call_history == []
    assert tool.calls == 0


def test_valid_tile_still_assessed_with_validator_enabled(tmp_path):
    good = tmp_path / "t1.png"
    good.write_bytes(make_png())

    assessment = agent_for(
        [ScriptedTool("classifier")], tile_validator=validate_tile_image
    ).process_tile("t1", str(good))

    assert assessment.status == TileStatus.ASSESSED
    assert assessment.failure_reason is None


def test_validator_is_off_by_default_so_fake_paths_still_work():
    assessment = agent_for([ScriptedTool("classifier")]).process_tile("t1", "/does/not/exist.png")
    assert assessment.status == TileStatus.ASSESSED


# --- FLT-02: tool endpoint unavailable -------------------------------------


def test_transient_failure_is_retried_with_exponential_backoff():
    sleeps: list[float] = []
    flaky = ScriptedTool("sam", fail_times=2)

    assessment = agent_for([flaky], sleeps).process_tile("t1", "/x/t1.png")

    assert assessment.status == TileStatus.ASSESSED
    assert flaky.calls == 3
    assert sleeps == [0.5, 1.0]
    assert [e.attempt for e in assessment.tool_errors] == [1, 2]
    assert len(assessment.tool_call_history) == 1  # only the real result


def test_persistently_failing_tool_falls_back_to_alternate_tool():
    sleeps: list[float] = []
    down = ScriptedTool("sam", fail_times=None)
    backup = ScriptedTool("classifier", label="minor-damage")

    assessment = agent_for([down, backup], sleeps).process_tile("t1", "/x/t1.png")

    assert assessment.status == TileStatus.ASSESSED
    assert assessment.damage_severity == "minor-damage"
    assert down.calls == 3  # tried max attempts, then dropped
    assert [r.tool_name for r in assessment.tool_call_history] == ["classifier"]
    assert {e.tool_name for e in assessment.tool_errors} == {"sam"}
    assert len(sleeps) == 2  # backoff between attempts, not after the last one


def test_not_implemented_is_permanent_and_not_retried():
    sleeps: list[float] = []
    no_endpoint = ScriptedTool("sam", fail_times=None, error=NotImplementedError("no endpoint"))
    backup = ScriptedTool("classifier")

    assessment = agent_for([no_endpoint, backup], sleeps).process_tile("t1", "/x/t1.png")

    assert no_endpoint.calls == 1
    assert sleeps == []
    assert assessment.status == TileStatus.ASSESSED
    assert assessment.tool_errors[0].error_type == "NotImplementedError"


def test_all_tools_failing_escalates_instead_of_crashing():
    tools = [ScriptedTool("sam", fail_times=None), ScriptedTool("classifier", fail_times=None)]

    assessment = agent_for(tools).process_tile("t1", "/x/t1.png")

    assert assessment.status == TileStatus.ESCALATED
    assert assessment.tool_call_history == []
    assert "all tools failed" in assessment.failure_reason
    assert len(assessment.tool_errors) == 6  # 3 attempts x 2 tools, all audited


def test_failed_tool_never_appears_in_history_used_for_scoring():
    down = ScriptedTool("sam", fail_times=None)
    backup = ScriptedTool("classifier")

    assessment = agent_for([down, backup]).process_tile("t1", "/x/t1.png")

    assert all(r.tool_name != "sam" for r in assessment.tool_call_history)


def test_low_confidence_exhaustion_is_ordinary_escalation_without_failure_reason():
    weak = ScriptedTool("classifier", confidence=0.2)

    assessment = agent_for([weak]).process_tile("t1", "/x/t1.png")

    assert assessment.status == TileStatus.ESCALATED
    assert assessment.failure_reason is None
    assert len(assessment.tool_call_history) == 1


def test_selector_that_reoffers_a_failed_tool_escalates_instead_of_hanging():
    down = ScriptedTool("sam", fail_times=None)

    def stubborn_selector(history, available_tools):
        return down, "always picks sam"  # ignores the candidate list it was given

    assessment = agent_for([down], tool_selector=stubborn_selector).process_tile("t1", "/x/t1.png")

    assert assessment.status == TileStatus.ESCALATED
    assert "re-offered failed tool" in assessment.failure_reason
    assert down.calls == 3  # gave up after the retry budget, never looped


def test_error_messages_are_truncated():
    huge = ScriptedTool("sam", fail_times=None, error=RuntimeError("x" * 5000))

    assessment = agent_for([huge], max_tool_attempts=1).process_tile("t1", "/x/t1.png")

    assert len(assessment.tool_errors[0].message) == 500


def test_max_tool_attempts_must_be_positive():
    with pytest.raises(ValueError):
        PerceptionAgent(tools=[ScriptedTool("sam")], max_tool_attempts=0)


# --- SYS-04 / PLN-04: a permanently failed tile doesn't block planning -----


def test_planner_completes_when_one_tile_failed_permanently(monkeypatch):
    monkeypatch.setattr("agents.planner.agent.score_zone", lambda tiles: (1.0, 0.0, 1.0))
    monkeypatch.setattr("agents.planner.agent.find_conflicting_pairs", lambda g, b: [])

    healthy = agent_for([ScriptedTool("classifier")])
    broken = agent_for([ScriptedTool("classifier", fail_times=None)])
    tiles = [
        healthy.process_tile("good_1", "/x/1.png", latitude=10.0, longitude=10.0),
        healthy.process_tile("good_2", "/x/2.png", latitude=20.0, longitude=20.0),
        broken.process_tile("bad", "/x/3.png", latitude=30.0, longitude=30.0),
    ]

    rankings = TriagePlanner(dispatcher=LocalReinspectionDispatcher()).plan(tiles)

    by_zone = {r.zone_id: r for r in rankings}
    assert set(by_zone) == {"zone-good_1", "zone-good_2", "zone-bad"}
    assert by_zone["zone-bad"].status == ZoneStatus.NEEDS_MANUAL_REVIEW
    assert by_zone["zone-bad"].escalated_tile_ids == ["bad"]
    assert by_zone["zone-good_1"].status == ZoneStatus.RANKED


def test_planner_plans_a_partial_batch_without_waiting(monkeypatch):
    monkeypatch.setattr("agents.planner.agent.score_zone", lambda tiles: (1.0, 0.0, 1.0))
    monkeypatch.setattr("agents.planner.agent.find_conflicting_pairs", lambda g, b: [])
    healthy = agent_for([ScriptedTool("classifier")])
    only_four_of_five = [
        healthy.process_tile(f"t{i}", f"/x/{i}.png", latitude=float(i * 10), longitude=0.0)
        for i in range(4)
    ]

    rankings = TriagePlanner(dispatcher=LocalReinspectionDispatcher()).plan(only_four_of_five)

    assert len(rankings) == 4
