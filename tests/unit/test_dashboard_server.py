"""Dashboard backend: run listing, event feed, images, ground truth, and the Run button."""
from __future__ import annotations

import json
import struct
import threading
import zlib

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from agents.orchestrator.runner import RunOptions, execute_run  # noqa: E402
from dashboard.server import create_app, ground_truth, read_events_after  # noqa: E402


def make_png() -> bytes:
    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\x00\x80\x80\x80")) + chunk(b"IEND", b""))


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("world")
    demo, runs = root / "demo", root / "runs"
    d = demo / "storm"
    for sub in ("images", "pre", "labels"):
        (d / sub).mkdir(parents=True)
    geo = {}
    for i in range(6):
        tile = f"storm_{i:08d}_post_disaster"
        (d / "images" / f"{tile}.png").write_bytes(make_png())
        (d / "pre" / f"storm_{i:08d}_pre_disaster.png").write_bytes(make_png())
        subtypes = ["no-damage", "destroyed", "minor-damage", "un-classified"]
        (d / "labels" / f"{tile}.json").write_text(json.dumps({"features": {"lng_lat": [
            {"properties": {"feature_type": "building", "subtype": s}} for s in subtypes]}}))
        geo[tile] = {"latitude": 10 + (i // 3) * 0.004, "longitude": 20 + (i % 3) * 0.004}
    (d / "geo.json").write_text(json.dumps(geo))
    (d / "demo.json").write_text(json.dumps({"tile_count": 6, "adjacency_radius_m": 700}))
    result = execute_run(RunOptions(disaster="storm", demo_root=demo, offline=True, events_dir=runs,
                                    ledger_path=root / "ledger.jsonl", wave_size=3, wave_interval_s=0.0))
    return {"demo": demo, "runs": runs, "run_id": result.run_id, "root": root}


@pytest.fixture
def client(world):
    return TestClient(create_app(world["runs"], world["demo"], launcher=lambda req: "fake-run"))


def test_health_and_static_index(client):
    assert client.get("/api/health").json() == {"ok": True, "allow_run": True}
    page = client.get("/")
    assert page.status_code == 200 and "TRIAGE" in page.text
    assert client.get("/static/state.js").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def test_disasters_lists_demo_sets(client):
    assert client.get("/api/disasters").json() == [
        {"disaster": "storm", "tile_count": 6, "adjacency_radius_m": 700}]


def test_runs_listing_summarises_finished_run(client, world):
    runs = client.get("/api/runs").json()
    r = next(x for x in runs if x["run_id"] == world["run_id"])
    assert r["status"] == "finished" and r["disaster"] == "storm" and r["tile_count"] == 6
    assert r["streaming"] is True and r["transport"] == "inprocess" and r["versions"]


def test_events_endpoint_supports_incremental_polling(client, world):
    everything = client.get(f"/api/runs/{world['run_id']}/events").json()
    assert everything["finished"] is True
    seqs = [e["seq"] for e in everything["events"]]
    assert seqs == sorted(seqs) and seqs[0] == 1
    tail = client.get(f"/api/runs/{world['run_id']}/events?after={seqs[-3]}").json()
    assert [e["seq"] for e in tail["events"]] == seqs[-2:]
    assert client.get(f"/api/runs/{world['run_id']}/events?after={seqs[-1]}").json()["events"] == []


def test_unknown_run_is_404_and_bad_ids_are_rejected(client):
    assert client.get("/api/runs/nope/events").status_code == 404
    assert client.get("/api/runs/..%2f..%2fetc/events").status_code in (400, 404)
    assert client.get("/api/runs/bad id/events").status_code == 400


def test_half_written_last_line_is_ignored(tmp_path):
    p = tmp_path / "events.jsonl"
    p.write_text('{"seq": 1, "type": "a"}\n{"seq": 2, "type": "b"}\n{"seq": 3, "ty')
    assert [e["seq"] for e in read_events_after(p)] == [1, 2]


def test_images_serve_post_and_pre_and_reject_traversal(client):
    tile = "storm_00000001_post_disaster"
    assert client.get(f"/api/image/storm/post/{tile}").headers["content-type"] == "image/png"
    assert client.get(f"/api/image/storm/pre/{tile}").status_code == 200   # maps to the _pre_ file
    assert client.get(f"/api/image/storm/post/{tile}?w=64").status_code == 200
    assert client.get("/api/image/storm/post/missing").status_code == 404
    assert client.get("/api/image/storm/sideways/" + tile).status_code == 400
    assert client.get("/api/image/..%2fetc/post/x").status_code in (400, 404)
    assert client.get("/api/image/storm/post/..%2f..%2fsecret").status_code in (400, 404)


def test_ground_truth_counts_and_scores(client, tmp_path):
    t = client.get("/api/truth/storm/storm_00000001_post_disaster").json()
    assert t["counts"] == {"no-damage": 1, "minor-damage": 1, "major-damage": 0, "destroyed": 1}
    assert t["buildings"] == 3 and t["worst"] == 3 and t["mean"] == pytest.approx(1.33, abs=0.01)
    assert client.get("/api/truth/storm/ghost").status_code == 404
    empty = tmp_path / "e.json"
    empty.write_text(json.dumps({"features": {"lng_lat": []}}))
    assert ground_truth(empty)["worst"] is None


# --- the Run button --------------------------------------------------------------------


def test_run_endpoint_launches_and_validates(world):
    launched = []
    app = create_app(world["runs"], world["demo"], launcher=lambda req: launched.append(req) or "run-123")
    c = TestClient(app)
    ok = c.post("/api/runs", json={"disaster": "storm", "transport": "a2a", "wave_size": 3})
    assert ok.status_code == 200 and ok.json() == {"run_id": "run-123"}
    assert launched[0].transport == "a2a" and launched[0].wave_size == 3
    assert c.post("/api/runs", json={"disaster": "storm", "transport": "pigeon"}).status_code == 400
    assert c.post("/api/runs", json={"disaster": "../x"}).status_code == 400
    assert c.post("/api/runs", json={"disaster": "storm", "wave_size": 0}).status_code == 422


def test_run_endpoint_can_be_disabled_for_public_deployments(world):
    c = TestClient(create_app(world["runs"], world["demo"], launcher=lambda r: "x", allow_run=False))
    assert c.get("/api/health").json()["allow_run"] is False
    assert c.post("/api/runs", json={"disaster": "storm"}).status_code == 403


def test_unknown_disaster_gives_a_clear_400_with_the_real_launcher(world):
    c = TestClient(create_app(world["runs"], world["demo"]))
    r = c.post("/api/runs", json={"disaster": "atlantis"})
    assert r.status_code == 400 and "atlantis" in r.json()["detail"]


def test_real_launcher_runs_in_background_and_a_second_run_is_refused(world, tmp_path):
    runs = tmp_path / "runs"
    c = TestClient(create_app(runs, world["demo"]))
    first = c.post("/api/runs", json={"disaster": "storm", "wave_size": 2, "wave_interval_s": 0.4})
    assert first.status_code == 200
    run_id = first.json()["run_id"]
    assert c.post("/api/runs", json={"disaster": "storm"}).status_code == 409   # one at a time

    import time
    deadline = time.time() + 30
    while time.time() < deadline:
        if c.get(f"/api/runs/{run_id}/events").json()["finished"]:
            break
        time.sleep(0.2)
    events = c.get(f"/api/runs/{run_id}/events").json()["events"]
    assert events[-1]["type"] == "run_finished"
    assert c.get("/api/runs").json()[0]["status"] == "finished"
