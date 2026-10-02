"""execute_run: the one function behind both `scripts.run_e2e` and the dashboard's Run button."""
from __future__ import annotations

import json
import struct
import zlib

import pytest

from agents.orchestrator.events import read_events
from agents.orchestrator.runner import RunOptions, execute_run, load_specs


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
    (d / "images").mkdir(parents=True)
    geo = {}
    for i in range(8):
        tile = f"storm_{i:08d}_post_disaster"
        (d / "images" / f"{tile}.png").write_bytes(make_png())
        geo[tile] = {"latitude": 10 + (i // 4) * 0.004, "longitude": 20 + (i % 4) * 0.004}
    (d / "geo.json").write_text(json.dumps(geo))
    (d / "demo.json").write_text(json.dumps({"adjacency_radius_m": 700}))
    return root


def options(demo, tmp_path, **kw):
    return RunOptions(disaster="storm", demo_root=demo, offline=True, events_dir=tmp_path / "runs",
                      ledger_path=tmp_path / "ledger.jsonl", **kw)


def test_load_specs_reads_tiles_and_geo(demo):
    specs = load_specs(demo / "storm")
    assert len(specs) == 8 and all(s.latitude is not None for s in specs)


def test_run_writes_an_event_log_the_dashboard_can_use(demo, tmp_path):
    result = execute_run(options(demo, tmp_path))

    events = read_events(tmp_path / "runs" / result.run_id / "events.jsonl")
    started = next(e for e in events if e["type"] == "run_started")
    assert started["disaster"] == "storm"
    assert len(started["tiles"]) == 8 and started["tiles"][0]["latitude"] is not None
    assert started["adjacency_radius_m"] == 700          # taken from demo.json
    assert events[0]["type"] == "transport_selected"
    assert events[-1]["type"] == "run_finished"
    assert result.versions and result.batch_id.startswith("storm-")


def test_streaming_run_through_execute_run(demo, tmp_path):
    result = execute_run(options(demo, tmp_path, wave_size=3, wave_interval_s=0.0))

    events = read_events(tmp_path / "runs" / result.run_id / "events.jsonl")
    assert [e["wave"] for e in events if e["type"] == "wave_arrived"] == [1, 2, 3]
    assert len(result.versions) >= 3                      # a version per wave, plus re-inspection


def test_sweep_order_is_applied_when_streaming(demo, tmp_path):
    result = execute_run(options(demo, tmp_path, wave_size=4, wave_interval_s=0.0))
    events = read_events(tmp_path / "runs" / result.run_id / "events.jsonl")
    first_wave = next(e for e in events if e["type"] == "wave_arrived")["tile_ids"]
    started = next(e for e in events if e["type"] == "run_started")["tiles"]
    lat = {t["tile_id"]: t["latitude"] for t in started}
    assert {lat[t] for t in first_wave} == {10.004}       # northern row arrives first


def test_explicit_radius_and_batch_id_win(demo, tmp_path):
    result = execute_run(options(demo, tmp_path, radius_m=123.0, batch_id="my-batch"))
    started = next(e for e in read_events(tmp_path / "runs" / result.run_id / "events.jsonl")
                   if e["type"] == "run_started")
    assert result.batch_id == "my-batch" and started["adjacency_radius_m"] == 123.0


def test_missing_tiles_and_bad_transport_raise_value_error(demo, tmp_path):
    with pytest.raises(ValueError, match="No tiles at"):
        execute_run(RunOptions(disaster="nope", demo_root=demo, offline=True))
    with pytest.raises(ValueError, match="transport"):
        execute_run(options(demo, tmp_path, transport="carrier-pigeon"))
    with pytest.raises(ValueError, match="agent URLs"):
        execute_run(options(demo, tmp_path, transport="a2a"))


def test_a2a_run_through_execute_run(demo, tmp_path):
    pytest.importorskip("google.adk")
    result = execute_run(options(demo, tmp_path, transport="a2a", spawn_local=True))
    events = read_events(tmp_path / "runs" / result.run_id / "events.jsonl")
    assert next(e for e in events if e["type"] == "transport_selected")["transport"] == "a2a"
    assert "transport_fallback" not in [e["type"] for e in events]
    assert result.versions
