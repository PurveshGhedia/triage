"""Build and execute one orchestrated run from plain options.

The command-line script (scripts.run_e2e) and the dashboard's Run button both go
through `execute_run`, so a run behaves identically whichever way it was started.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from agents.orchestrator.events import EventLog
from agents.orchestrator.models import TileSpec, order_for_arrival
from agents.orchestrator.pipeline import RunConfig, RunResult, new_run_id, run_pipeline
from agents.orchestrator.ports import InProcessPerception, InProcessPlanner
from shared.agent_factory import TOOLS_MODES, build_perception_agent, build_second_opinion_agent
from shared.local_zone_store import LocalZoneRankingStore

TILE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
TRANSPORTS = ("inprocess", "a2a")


@dataclass
class RunOptions:
    disaster: str
    demo_root: Path = Path("data/demo")
    demo_dir: Path | None = None
    batch_id: str | None = None
    radius_m: float | None = None
    rounds: int = 2
    per_tile_cap: int = 2
    concurrency: int = 4
    wave_size: int | None = None
    wave_interval_s: float = 2.0
    arrival_order: str = "sweep"
    use_llm_selector: bool = False
    tools: str = "mock"
    transport: str = "inprocess"
    agent1_url: str | None = None
    agent2_url: str | None = None
    spawn_local: bool = False
    fallback: bool = True
    offline: bool = False
    events_dir: Path = Path("data/runs")
    ledger_path: Path | None = None
    run_id: str | None = None


def load_specs(demo_dir: Path) -> list[TileSpec]:
    images = sorted(p for p in (demo_dir / "images").iterdir() if p.suffix.lower() in TILE_EXTENSIONS)
    geo_path = demo_dir / "geo.json"
    geo = json.loads(geo_path.read_text()) if geo_path.exists() else {}
    return [
        TileSpec(
            tile_id=p.stem,
            tile_path=str(p),
            latitude=geo.get(p.stem, {}).get("latitude"),
            longitude=geo.get(p.stem, {}).get("longitude"),
        )
        for p in images
    ]


def new_batch_id(disaster: str) -> str:
    return f"{disaster}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"


def execute_run(
    opts: RunOptions,
    listener: Callable[[dict], None] | None = None,
    store=None,
) -> RunResult:
    """Run the whole pipeline. Raises ValueError for bad input (missing tiles, bad transport)."""
    if opts.transport not in TRANSPORTS:
        raise ValueError(f"transport must be one of {TRANSPORTS}")
    if opts.tools not in TOOLS_MODES:
        raise ValueError(f"tools must be one of {TOOLS_MODES}")
    demo_dir = opts.demo_dir or opts.demo_root / opts.disaster
    if not (demo_dir / "images").is_dir():
        raise ValueError(
            f"No tiles at {demo_dir}/images. Run: python -m scripts.select_demo_tiles "
            f"--disaster {opts.disaster}"
        )
    specs = load_specs(demo_dir)
    if not specs:
        raise ValueError(f"No tile images in {demo_dir}/images")

    meta_path = demo_dir / "demo.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    radius = opts.radius_m or meta.get("adjacency_radius_m") or 500.0

    run_id = opts.run_id or new_run_id()
    batch_id = opts.batch_id or new_batch_id(opts.disaster)
    config = RunConfig(
        batch_id=batch_id, run_id=run_id, adjacency_radius_m=radius, max_rounds=opts.rounds,
        per_tile_cap=opts.per_tile_cap, disaster=opts.disaster, tools=opts.tools,
        wave_size=opts.wave_size, wave_interval_s=opts.wave_interval_s,
    )
    if opts.wave_size:
        specs = order_for_arrival(specs, opts.arrival_order)

    if store is None:
        if opts.offline:
            from shared.memory_store import InMemoryTileStore

            store = InMemoryTileStore(batch_id=batch_id)
        else:
            from shared.firestore_client import TileAssessmentStore

            store = TileAssessmentStore(batch_id=batch_id)

    events = EventLog(run_id, opts.events_dir / run_id / "events.jsonl", listener=listener)
    in_process_perception = InProcessPerception(
        agent=build_perception_agent(opts.use_llm_selector, tools_mode=opts.tools),
        reinspection_agent=build_second_opinion_agent(opts.use_llm_selector, tools_mode=opts.tools),
        max_workers=opts.concurrency,
    )
    in_process_planner = InProcessPlanner()
    perception, planner = in_process_perception, in_process_planner
    services = None

    if opts.transport == "a2a":
        import warnings

        warnings.filterwarnings("ignore", message=r"\[EXPERIMENTAL\]")
        from agents.orchestrator.a2a_client import (
            A2APerception, A2APlanner, ResilientPerception, ResilientPlanner,
        )

        if opts.spawn_local:
            from agents.orchestrator.local_services import start_local_services

            services = start_local_services(in_process_perception, in_process_planner)
            agent1_url, agent2_url = services.perception_url, services.planner_url
        else:
            agent1_url, agent2_url = opts.agent1_url, opts.agent2_url
            if not (agent1_url and agent2_url):
                raise ValueError("transport 'a2a' needs spawn_local, or both agent URLs")

        def on_fallback(agent: str, exc: Exception) -> None:
            events.emit("transport_fallback", agent=agent,
                        error=f"{type(exc).__name__}: {str(exc)[:160]}")

        perception, planner = A2APerception(agent1_url), A2APlanner(agent2_url)
        if opts.fallback:
            perception = ResilientPerception(perception, in_process_perception, on_fallback)
            planner = ResilientPlanner(planner, in_process_planner, on_fallback)
        events.emit("transport_selected", transport="a2a", agent1_url=agent1_url, agent2_url=agent2_url)
    else:
        events.emit("transport_selected", transport="inprocess")

    ledger = LocalZoneRankingStore(opts.ledger_path) if opts.ledger_path else LocalZoneRankingStore()
    try:
        return run_pipeline(specs, config, perception, planner, store, ledger, events)
    finally:
        if services:
            services.stop()
