"""Accuracy evaluation against xBD ground truth, and vision mode end to end (fake model)."""
from __future__ import annotations

import json
import struct
import zlib

import pytest

from agents.orchestrator.evaluation import (
    evaluate,
    evaluate_events,
    predictions_from_events,
    spearman,
)
from agents.orchestrator.runner import RunOptions, execute_run
from shared.agent_factory import TOOLS_MODES, build_perception_agent, build_tools


# --- spearman ------------------------------------------------------------------------------


def test_spearman_known_values():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0
    assert spearman([1, 2, 3, 4, 5], [2, 1, 4, 3, 5]) == 0.8


def test_spearman_handles_ties_and_degenerate_input():
    assert spearman([1, 1, 2, 2], [1, 2, 3, 4]) == pytest.approx(0.894, abs=0.001)   # tied ranks averaged
    assert spearman([1, 1, 1, 1], [1, 2, 3, 4]) is None                              # constant side
    assert spearman([1, 2], [1, 2]) is None                                          # too few points
    assert spearman([1, 2, 3], [1, 2]) is None


# --- predictions from events -----------------------------------------------------------------


def tile_event(tile_id, status="assessed", calls=()):
    return {"type": "tile_finished", "tile": {
        "tile_id": tile_id, "status": status,
        "tool_calls": [{"tool": t, "label": l, "confidence": c} for t, l, c in calls]}}


def test_predictions_use_the_best_confidence_damage_reading_and_ignore_the_detector():
    events = [
        tile_event("a", calls=[("classifier", "minor-damage", 0.6), ("detector", "access-blocked", 0.99), ("sam", "destroyed", 0.9)]),
        tile_event("b", calls=[("classifier", "no-damage", 0.8)]),
        tile_event("c", status="escalated", calls=[("classifier", "major-damage", 0.4)]),
        tile_event("d", calls=[("detector", "access-clear", 0.9)]),         # no damage reading at all
    ]
    p = predictions_from_events(events)
    assert p["scores"] == {"a": 3.0, "b": 0.0}
    assert p["escalated"] == ["c", "d"]


def test_reinspection_replaces_the_tile_and_last_ranking_wins():
    events = [
        tile_event("a", calls=[("classifier", "no-damage", 0.9)]),
        {"type": "zones_ranked", "zones": [{"zone_id": "z1", "tile_ids": ["a"], "damage_score": 0}]},
        {"type": "tile_reinspected", "after": {"tile_id": "a", "status": "assessed",
            "tool_calls": [{"tool": "sam", "label": "destroyed", "confidence": 0.8}]}},
        {"type": "zones_ranked", "zones": [{"zone_id": "z1", "tile_ids": ["a"], "damage_score": 3}]},
    ]
    p = predictions_from_events(events)
    assert p["scores"] == {"a": 3.0} and p["zones"][0]["damage_score"] == 3


# --- metrics ---------------------------------------------------------------------------------


def truth(mean, buildings=10):
    return {"mean": mean, "buildings": buildings, "worst": 3}


def test_a_perfect_assessor_scores_perfectly():
    preds = {"scores": {"a": 0.0, "b": 1.0, "c": 2.0, "d": 3.0}, "escalated": [], "zones": []}
    t = {"a": truth(0.0), "b": truth(1.0), "c": truth(2.0), "d": truth(3.0)}
    m = evaluate(preds, t)["tiles"]
    assert (m["spearman"], m["mean_abs_error"], m["within_one_level"], m["tiles_compared"]) == (1.0, 0.0, 1.0, 4)


def test_a_reversed_assessor_scores_badly_and_baseline_is_reported():
    preds = {"scores": {"a": 3.0, "b": 2.0, "c": 1.0, "d": 0.0}, "escalated": ["x"], "zones": []}
    t = {"a": truth(0.0), "b": truth(1.0), "c": truth(2.0), "d": truth(3.0)}
    m = evaluate(preds, t)["tiles"]
    assert m["spearman"] == -1.0 and m["mean_abs_error"] == 2.0 and m["within_one_level"] == 0.5
    assert m["tiles_escalated"] == 1 and m["baseline_mean_abs_error"] == 1.0   # always guessing the median truth (2.0)


def test_tiles_without_labelled_buildings_are_skipped():
    preds = {"scores": {"a": 1.0, "b": 2.0}, "escalated": [], "zones": []}
    m = evaluate(preds, {"a": truth(1.0), "b": {"mean": None, "buildings": 0, "worst": None}})["tiles"]
    assert m["tiles_compared"] == 1


def test_zone_metric_compares_zone_damage_with_building_weighted_truth():
    zones = [{"tile_ids": ["a"], "damage_score": 0.5}, {"tile_ids": ["b", "c"], "damage_score": 2.0},
             {"tile_ids": ["d"], "damage_score": 3.0}]
    t = {"a": truth(0.2), "b": truth(1.0, 10), "c": truth(3.0, 30), "d": truth(2.9)}
    z = evaluate({"scores": {}, "escalated": [], "zones": zones}, t)["zones"]
    assert z["zones_compared"] == 3 and z["spearman"] == 1.0


# --- vision mode end to end, with a fake model --------------------------------------------------


def make_png() -> bytes:
    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\x00\x80\x80\x80")) + chunk(b"IEND", b""))


@pytest.fixture
def demo(tmp_path):
    root = tmp_path / "demo"
    d = root / "storm"
    for sub in ("images", "pre", "labels"):
        (d / sub).mkdir(parents=True)
    geo = {}
    for i in range(6):
        tile = f"storm_{i:08d}_post_disaster"
        (d / "images" / f"{tile}.png").write_bytes(make_png())
        (d / "pre" / f"storm_{i:08d}_pre_disaster.png").write_bytes(make_png())
        sub = "destroyed" if i >= 3 else "no-damage"                       # truth: tiles 3-5 are wrecked
        (d / "labels" / f"{tile}.json").write_text(json.dumps({"features": {"lng_lat": [
            {"properties": {"subtype": sub}} for _ in range(5)]}}))
        geo[tile] = {"latitude": 10 + (i // 3) * 0.004, "longitude": 20 + (i % 3) * 0.004}
    (d / "geo.json").write_text(json.dumps(geo))
    (d / "demo.json").write_text(json.dumps({"adjacency_radius_m": 700}))
    return root


def fake_vision(monkeypatch):
    """Gemini stand-in that 'sees' the truth: tiles 3-5 destroyed, 0-2 undamaged."""
    import tools.vision.client as vc

    def make(project=None, location=None, model=None):
        def generate(images, prompt, schema):
            if "ROAD ACCESS" in prompt:                       # answer the question that was asked
                return json.dumps({"label": "access-degraded", "confidence": 0.7, "evidence": "debris on road"})
            return json.dumps({"label": "destroyed", "confidence": 0.9, "evidence": "collapsed roofs"})
        return generate

    monkeypatch.setattr(vc, "make_vertex_generate_fn", make)


def test_tools_modes_and_factory():
    assert TOOLS_MODES == ("mock", "vision")
    assert [t.name for t in build_tools("mock")] == ["classifier", "detector", "sam"]
    v = build_tools("vision")
    assert [t.name for t in v] == ["classifier", "detector", "sam"]
    assert [t.provides_damage for t in v] == [True, False, True]
    assert type(v[0]).__name__ == "GeminiDamageTool" and type(v[1]).__name__ == "GeminiAccessTool"
    with pytest.raises(ValueError):
        build_tools("telepathy")
    with pytest.raises(ValueError):
        execute_run(RunOptions(disaster="storm", tools="telepathy"))


def test_vision_run_records_tools_evidence_and_reads_through_the_cache(demo, tmp_path, monkeypatch):
    fake_vision(monkeypatch)
    monkeypatch.setenv("TRIAGE_VISION_CACHE", str(tmp_path / "cache"))
    opts = RunOptions(disaster="storm", demo_root=demo, offline=True, tools="vision",
                      events_dir=tmp_path / "runs", ledger_path=tmp_path / "l.jsonl")
    result = execute_run(opts)

    events = [json.loads(l) for l in (tmp_path / "runs" / result.run_id / "events.jsonl").read_text().splitlines()]
    assert next(e for e in events if e["type"] == "run_started")["tools"] == "vision"
    finished = [e["tile"] for e in events if e["type"] == "tile_finished"]
    assert all(t["status"] == "assessed" and t["damage_severity"] == "destroyed" for t in finished)
    assert all(t["tool_errors"] == 0 for t in finished)                   # the access tool answered too
    assert all(any(c["evidence"] == "collapsed roofs" for c in t["tool_calls"]) for t in finished)
    assert list((tmp_path / "cache").glob("*.json"))                      # answers were cached


def test_a_second_vision_run_makes_no_model_calls(demo, tmp_path, monkeypatch):
    import tools.vision.client as vc

    calls = {"n": 0}

    def make(project=None, location=None, model=None):
        def generate(images, prompt, schema):
            calls["n"] += 1
            if "ROAD ACCESS" in prompt:
                return json.dumps({"label": "access-clear", "confidence": 0.8, "evidence": "x"})
            return json.dumps({"label": "minor-damage", "confidence": 0.8, "evidence": "x"})
        return generate

    monkeypatch.setattr(vc, "make_vertex_generate_fn", make)
    monkeypatch.setenv("TRIAGE_VISION_CACHE", str(tmp_path / "cache"))
    opts = lambda: RunOptions(disaster="storm", demo_root=demo, offline=True, tools="vision",
                              events_dir=tmp_path / "runs", ledger_path=tmp_path / "l.jsonl", rounds=0)
    execute_run(opts())
    first = calls["n"]
    assert first > 0
    execute_run(opts())
    assert calls["n"] == first                                           # fully served from cache, even offline


def test_evaluation_of_a_run_against_labels(demo, tmp_path, monkeypatch):
    fake_vision(monkeypatch)
    monkeypatch.setenv("TRIAGE_VISION_CACHE", str(tmp_path / "cache"))
    result = execute_run(RunOptions(disaster="storm", demo_root=demo, offline=True, tools="vision", rounds=0,
                                    events_dir=tmp_path / "runs", ledger_path=tmp_path / "l.jsonl"))
    events = [json.loads(l) for l in (tmp_path / "runs" / result.run_id / "events.jsonl").read_text().splitlines()]
    report = evaluate_events(events, demo / "storm" / "labels")

    assert report["tiles"]["tiles_compared"] == 6
    # the fake says "destroyed" everywhere: right for tiles 3-5 (error 0), wrong for 0-2 (error 3)
    assert report["tiles"]["mean_abs_error"] == 1.5 and report["tiles"]["within_one_level"] == 0.5
    assert report["tiles"]["spearman"] is None                           # constant predictions carry no rank information


def test_default_runs_are_mock_and_say_so(demo, tmp_path):
    result = execute_run(RunOptions(disaster="storm", demo_root=demo, offline=True, rounds=0,
                                    events_dir=tmp_path / "runs", ledger_path=tmp_path / "l.jsonl"))
    started = json.loads((tmp_path / "runs" / result.run_id / "events.jsonl").read_text().splitlines()[1])
    assert started["type"] == "run_started" and started["tools"] == "mock"
