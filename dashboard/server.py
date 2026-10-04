"""TRIAGE dashboard backend (FastAPI): serves the single-page UI and a small read API
over the run event logs, plus a Run button that launches the pipeline.

Everything the UI shows is derived from `data/runs/<run_id>/events.jsonl`, so the same
page serves live runs (polling) and finished runs (replay), and works with no Firestore.

Run it:   python -m dashboard.server            (http://localhost:8080)
"""
from __future__ import annotations

import json
import os
import re
import threading
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from shared.truth import DAMAGE_SCORES, ground_truth  # noqa: F401  (ground_truth re-exported for tests)

STATIC_DIR = Path(__file__).parent / "static"
SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


class RunRequest(BaseModel):
    disaster: str
    transport: str = "inprocess"
    wave_size: int | None = Field(default=3, ge=1, le=50)
    wave_interval_s: float = Field(default=2.0, ge=0.0, le=30.0)
    use_llm_selector: bool = False
    tools: str = "mock"
    vision_variant: str | None = None


def _safe(name: str) -> str:
    if not SAFE_NAME.match(name) or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="invalid name")
    return name


def read_events_after(path: Path, after: int = 0) -> list[dict]:
    if not path.exists():
        return []
    events = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:  # a half-written last line during a live run
                continue
            if event["seq"] > after:
                events.append(event)
    return events


def summarise_run(run_dir: Path) -> dict | None:
    events = read_events_after(run_dir / "events.jsonl")
    if not events:
        return None
    started = next((e for e in events if e["type"] == "run_started"), None)
    transport = next((e for e in events if e["type"] == "transport_selected"), None)
    finished = next((e for e in reversed(events) if e["type"] == "run_finished"), None)
    return {
        "run_id": run_dir.name,
        "batch_id": started["batch_id"] if started else None,
        "disaster": started.get("disaster") if started else None,
        "started_at": events[0]["ts"],
        "status": "finished" if finished else "running",
        "tile_count": started["tile_count"] if started else None,
        "versions": finished["versions"] if finished else None,
        "transport": transport["transport"] if transport else None,
        "streaming": bool(started and started.get("wave_size")),
        "tools": started.get("tools", "mock") if started else None,
    }


def create_app(
    runs_dir: str | Path = "data/runs",
    demo_dir: str | Path = "data/demo",
    launcher: Callable[[RunRequest], str] | None = None,
    allow_run: bool | None = None,
) -> FastAPI:
    runs_dir, demo_dir = Path(runs_dir), Path(demo_dir)
    if allow_run is None:
        allow_run = os.environ.get("TRIAGE_ALLOW_RUN", "1") == "1"
    app = FastAPI(title="TRIAGE dashboard")
    run_lock = threading.Lock()
    active = {"run_id": None, "thread": None}

    def default_launcher(req: RunRequest) -> str:
        from agents.orchestrator.pipeline import new_run_id
        from agents.orchestrator.runner import RunOptions, execute_run

        run_id = new_run_id()
        # When TRIAGE_AGENT1_URL / TRIAGE_AGENT2_URL are set (Cloud Run), A2A runs call those
        # separate services; otherwise the agents are started on localhost for the run.
        agent1, agent2 = os.environ.get("TRIAGE_AGENT1_URL"), os.environ.get("TRIAGE_AGENT2_URL")
        remote_agents = bool(agent1 and agent2)
        opts = RunOptions(
            disaster=req.disaster, demo_root=demo_dir, run_id=run_id, events_dir=runs_dir,
            transport=req.transport, spawn_local=req.transport == "a2a" and not remote_agents,
            agent1_url=agent1, agent2_url=agent2,
            a2a_auth=os.environ.get("TRIAGE_A2A_AUTH") == "id_token",
            vision_variant=req.vision_variant or os.environ.get("TRIAGE_VISION_VARIANT") or None,
            wave_size=req.wave_size, wave_interval_s=req.wave_interval_s,
            use_llm_selector=req.use_llm_selector, tools=req.tools,
            offline=os.environ.get("TRIAGE_STORE", "memory") != "firestore",
        )
        # validate synchronously so the UI gets a clear error instead of a silent failure
        if not (opts.demo_root / opts.disaster / "images").is_dir():
            raise ValueError(f"No demo tiles for '{req.disaster}'")

        def work() -> None:
            try:
                execute_run(opts)
            except Exception as exc:  # surface in the event log; never kill the server
                from agents.orchestrator.events import EventLog

                EventLog(run_id, runs_dir / run_id / "events.jsonl").emit(
                    "run_failed", error=f"{type(exc).__name__}: {str(exc)[:300]}")

        t = threading.Thread(target=work, daemon=True)
        active.update(run_id=run_id, thread=t)
        t.start()
        return run_id

    launch = launcher or default_launcher

    @app.get("/api/health")
    def health():
        remote = bool(os.environ.get("TRIAGE_AGENT1_URL") and os.environ.get("TRIAGE_AGENT2_URL"))
        return {"ok": True, "allow_run": allow_run, "agents": "remote" if remote else "local"}

    @app.get("/api/disasters")
    def disasters():
        out = []
        if demo_dir.is_dir():
            for d in sorted(p for p in demo_dir.iterdir() if (p / "images").is_dir()):
                meta = {}
                if (d / "demo.json").exists():
                    meta = json.loads((d / "demo.json").read_text())
                out.append({"disaster": d.name, "tile_count": meta.get("tile_count"),
                            "adjacency_radius_m": meta.get("adjacency_radius_m")})
        return out

    @app.get("/api/runs")
    def runs():
        summaries = []
        if runs_dir.is_dir():
            for d in runs_dir.iterdir():
                if d.is_dir():
                    s = summarise_run(d)
                    if s:
                        summaries.append(s)
        return sorted(summaries, key=lambda s: s["started_at"], reverse=True)

    @app.get("/api/runs/{run_id}/events")
    def run_events(run_id: str, after: int = Query(0, ge=0)):
        _safe(run_id)
        path = runs_dir / run_id / "events.jsonl"
        if not path.exists():
            raise HTTPException(status_code=404, detail="unknown run")
        events = read_events_after(path, after)
        done = any(e["type"] in ("run_finished", "run_failed") for e in events) or (
            after > 0 and summarise_run(runs_dir / run_id) is not None
            and summarise_run(runs_dir / run_id)["status"] == "finished")
        return {"events": events, "finished": done}

    @app.post("/api/runs")
    def start_run(req: RunRequest):
        if not allow_run:
            raise HTTPException(status_code=403, detail="starting runs is disabled on this server")
        _safe(req.disaster)
        if req.transport not in ("inprocess", "a2a"):
            raise HTTPException(status_code=400, detail="transport must be inprocess or a2a")
        if req.tools not in ("mock", "vision"):
            raise HTTPException(status_code=400, detail="tools must be mock or vision")
        if not run_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="a run is already starting")
        try:
            t = active["thread"]
            if t is not None and t.is_alive():
                raise HTTPException(status_code=409, detail="a run is already in progress")
            try:
                return {"run_id": launch(req)}
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
        finally:
            run_lock.release()

    @app.get("/api/image/{disaster}/{kind}/{tile_id}")
    def image(disaster: str, kind: str, tile_id: str, w: int | None = Query(None, ge=32, le=1024)):
        _safe(disaster), _safe(tile_id)
        if kind not in ("post", "pre"):
            raise HTTPException(status_code=400, detail="kind must be post or pre")
        base = demo_dir / disaster
        path = (base / "images" / f"{tile_id}.png") if kind == "post" else (
            base / "pre" / f"{tile_id.replace('_post_disaster', '_pre_disaster')}.png")
        if not path.is_file():
            raise HTTPException(status_code=404, detail="image not found")
        if w:
            try:
                from PIL import Image

                cache = base / ".thumbs" / f"{kind}_{tile_id}_{w}.jpg"
                if not cache.exists():
                    cache.parent.mkdir(exist_ok=True)
                    with Image.open(path) as im:
                        im.convert("RGB").resize((w, int(im.height * w / im.width))).save(cache, quality=82)
                return FileResponse(cache, media_type="image/jpeg")
            except ImportError:
                pass  # no Pillow: serve the full-size image
        return FileResponse(path, media_type="image/png")

    @app.get("/api/runs/{run_id}/evaluation")
    def run_evaluation(run_id: str):
        """Agreement of this run's readings with xBD ground truth (see agents.orchestrator.evaluation)."""
        _safe(run_id)
        events = read_events_after(runs_dir / run_id / "events.jsonl")
        started = next((e for e in events if e["type"] == "run_started"), None)
        if started is None or not started.get("disaster"):
            raise HTTPException(status_code=404, detail="unknown run")
        from agents.orchestrator.evaluation import evaluate_events

        result = evaluate_events(events, demo_dir / started["disaster"] / "labels")
        result["run"] = {"run_id": run_id, "disaster": started["disaster"], "tools": started.get("tools", "mock")}
        return result

    @app.get("/api/footprints/{disaster}")
    def footprints(disaster: str):
        """True ground footprint of each tile (see scripts.build_footprints); {} if not built."""
        _safe(disaster)
        path = demo_dir / disaster / "footprints.json"
        return JSONResponse(json.loads(path.read_text()) if path.exists() else {})

    @app.get("/api/truth/{disaster}/{tile_id}")
    def truth(disaster: str, tile_id: str):
        _safe(disaster), _safe(tile_id)
        label = demo_dir / disaster / "labels" / f"{tile_id}.json"
        if not label.is_file():
            raise HTTPException(status_code=404, detail="no label file")
        return JSONResponse(ground_truth(label))

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8080)))
    parser.add_argument("--runs-dir", default="data/runs")
    parser.add_argument("--demo-dir", default="data/demo")
    parser.add_argument("--no-run", action="store_true", help="Disable the Run button (view/replay only).")
    args = parser.parse_args()
    app = create_app(args.runs_dir, args.demo_dir, allow_run=False if args.no_run else None)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
