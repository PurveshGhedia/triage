"""Cost/time reporting from recorded token usage and latency, and the prompt printer."""
from __future__ import annotations

import sys
import types

import pytest

from scripts.cost_report import DEFAULT_PRICE_IN, DEFAULT_PRICE_OUT, project, summarise
from scripts.show_prompts import render


def call(tool="classifier", cached=False, p=1000, o=100, lat=4.0):
    return {"tool": tool, "label": "x", "confidence": 0.9, "cached": cached, "prompt_tokens": p,
            "output_tokens": o, "latency_s": lat}


def tile_event(tile_id, calls):
    return {"type": "tile_finished", "tile": {"tile_id": tile_id, "tool_calls": calls}}


def run_events(tiles, wall_s=60, extra=()):
    return [{"type": "run_started", "ts": "2026-10-04T10:00:00+00:00", "tile_count": len(tiles)},
            *[tile_event(t, c) for t, c in tiles.items()], *extra,
            {"type": "run_finished", "ts": f"2026-10-04T10:{wall_s // 60:02d}:{wall_s % 60:02d}+00:00"}]


def test_cost_is_tokens_times_the_per_million_price():
    s = summarise(run_events({"a": [call(p=1_000_000, o=0)]}), price_in=0.30, price_out=2.50)
    assert s["cost_usd"] == 0.30
    s = summarise(run_events({"a": [call(p=0, o=1_000_000)]}), price_in=0.30, price_out=2.50)
    assert s["cost_usd"] == 2.50
    assert (DEFAULT_PRICE_IN, DEFAULT_PRICE_OUT) == (0.30, 2.50)


def test_per_tile_figures_and_latency_statistics():
    tiles = {f"t{i}": [call(lat=2.0 + i), call(tool="detector", lat=1.0)] for i in range(4)}
    s = summarise(run_events(tiles, wall_s=40), price_in=1.0, price_out=1.0)
    assert s["tiles"] == 4 and s["vision_calls"] == 8 and s["calls_per_tile"] == 2.0
    assert s["cost_usd"] == pytest.approx(8 * 1100 / 1e6, abs=1e-5) and s["cost_per_tile_usd"] == pytest.approx(2 * 1100 / 1e6, abs=1e-5)
    assert s["wall_seconds"] == 40 and s["wall_seconds_per_tile"] == 10.0
    assert s["sequential_seconds_per_tile"] == pytest.approx((2 + 3 + 4 + 5 + 4 * 1.0) / 4)
    # call latencies: 2, 3, 4, 5 and four 1s -> sorted 1 1 1 1 2 3 4 5: median (1+2)/2, mean 18/8
    assert s["latency_max_s"] == 5.0 and s["latency_median_s"] == 1.5 and s["latency_mean_s"] == 2.25


def test_cached_calls_are_free_now_and_reported_separately():
    s = summarise(run_events({"a": [call(cached=True, p=1_000_000, o=0)], "b": [call(cached=False, p=1_000_000, o=0)]}), 0.30, 2.50)
    assert s["billed_calls"] == 1 and s["replayed_calls"] == 1
    assert s["cost_usd"] == 0.30 and s["replayed_originally_cost_usd"] == 0.30


def test_reinspection_calls_are_counted_too():
    extra = [{"type": "tile_reinspected", "after": {"tile_id": "a", "tool_calls": [call(tool="sam", p=2_000_000, o=0)]}}]
    s = summarise(run_events({"a": [call(p=0, o=0)]}, extra=extra), 1.0, 1.0)
    assert s["vision_calls"] == 2 and s["cost_usd"] == 2.0 and s["per_tool"]["sam"]["calls"] == 1


def test_mock_runs_have_no_vision_calls_and_no_cost():
    mock = {"tool": "classifier", "label": "x", "confidence": 0.9, "cached": None}
    s = summarise(run_events({"a": [mock]}))
    assert s["vision_calls"] == 0 and s["cost_usd"] == 0 and s["cost_per_tile_usd"] is None


def test_calls_without_token_usage_are_flagged_not_silently_free():
    s = summarise(run_events({"a": [{**call(), "prompt_tokens": None, "output_tokens": None}]}))
    assert s["billed_without_usage"] == 1 and s["cost_usd"] == 0


def test_projection_scales_cost_linearly_and_time_by_workers():
    s = summarise(run_events({f"t{i}": [call(p=1000, o=100, lat=4.0)] for i in range(4)}), 1.0, 1.0)
    p = project(s, tiles=1000, workers=8)
    assert p["cost_usd"] == pytest.approx(round(s["cost_per_tile_usd"] * 1000, 2))
    assert p["minutes"] == pytest.approx(round(4.0 * 1000 / 8 / 60, 1))
    assert project(summarise(run_events({"a": []})), 10, 4) is None                 # nothing measured: no projection


def test_show_prints_a_warning_when_nothing_was_billed(capsys):
    from scripts.cost_report import show

    show(summarise(run_events({"a": [call(cached=True)]})), "r1", None)
    assert "everything was replayed from the cache" in capsys.readouterr().out


# --- prompt printer ----------------------------------------------------------------------------------


def test_every_vision_prompt_is_printed_verbatim():
    from tools.vision.prompts import ACCESS_PROMPT, DAMAGE_FRACTIONS_PROMPT, DAMAGE_PROMPT, SECOND_PASS_PROMPT

    out = render()
    for prompt in (DAMAGE_PROMPT, DAMAGE_FRACTIONS_PROMPT, SECOND_PASS_PROMPT, ACCESS_PROMPT):
        assert prompt in out
    assert "no-damage" in out and "BEFORE" in out and "Agent 2, planner" in out and "No LLM" in out


def test_selector_prompts_are_included_when_the_module_exists(monkeypatch):
    fake = types.ModuleType("agents.perception.prompts")
    fake.SYSTEM_PROMPT = "SELECTOR SYSTEM PROMPT TEXT"
    fake.build_user_prompt = lambda history, remaining: f"history={len(history)} remaining={[t.name for t in remaining]}"
    monkeypatch.setitem(sys.modules, "agents.perception.prompts", fake)
    out = render()
    assert "SELECTOR SYSTEM PROMPT TEXT" in out and "history=1 remaining=['detector', 'sam']" in out


def test_a_missing_selector_module_does_not_break_the_printer(monkeypatch):
    monkeypatch.setitem(sys.modules, "agents.perception.prompts", None)       # forces ImportError
    assert "not available here" in render()
