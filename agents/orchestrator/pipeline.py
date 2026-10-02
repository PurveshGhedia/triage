"""The orchestrator: Agent 1 over a batch -> Agent 2 -> targeted re-inspection
rounds -> re-plan, with every step written to the event log.

This answers "what triggers Agent 2?": the orchestrator does, once Agent 1 has
reported for every tile it submitted. Because it holds the manifest, it also
knows exactly which tiles never reported (SYS-04) and plans anyway.

Streaming mode (`wave_size`): tiles are fed to Agent 1 in waves, as if arriving from a
satellite pass, and Agent 2 re-plans after every wave. Each wave writes a new ledger
version, so the ledger records how the ranking evolved as data came in, and Agent 2 is
never made to wait for the whole batch. Re-inspection runs once, after the last wave.

The re-inspection loop is bounded three ways: `max_rounds` per run, `per_tile_cap`
attempts per tile, and it stops as soon as a round changes no tile's result (a
second opinion that only confirms the first adds no information, so re-planning
would just write an identical ledger version).
"""
from __future__ import annotations

import time
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from agents.orchestrator.events import EventLog
from agents.orchestrator.models import PlanOutcome, TileSpec
from agents.orchestrator.ports import PerceptionPort, PlannerPort
from agents.perception.schemas import TileAssessment, TileStatus
from agents.planner.schemas import ZoneRanking


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]


@dataclass
class RunConfig:
    batch_id: str
    run_id: str = field(default_factory=new_run_id)
    adjacency_radius_m: float = 500.0
    max_rounds: int = 2
    per_tile_cap: int = 2
    disaster: str | None = None       # label only; lets the dashboard find images
    wave_size: int | None = None      # None: all tiles at once (one wave)
    wave_interval_s: float = 0.0      # pause between waves (simulated arrival)


@dataclass
class RunResult:
    run_id: str
    batch_id: str
    versions: list[int]
    final_rankings: list[ZoneRanking]
    escalated_tile_ids: list[str]
    rounds_used: int
    reinspected_tile_ids: list[str]


def tile_summary(a: TileAssessment) -> dict:
    return {
        "tile_id": a.tile_id,
        "status": a.status.value,
        "damage_severity": a.damage_severity,
        "confidence": a.confidence,
        "latitude": a.latitude,
        "longitude": a.longitude,
        "failure_reason": a.failure_reason,
        "tool_errors": len(a.tool_errors),
        "tool_calls": [
            {
                "tool": r.tool_name,
                "label": r.label,
                "confidence": r.confidence,
                "reasoning": r.selection_reasoning,
            }
            for r in a.tool_call_history
        ],
    }


def _result_key(a: TileAssessment | None):
    """What a re-inspection can change about a tile, for the 'did anything change' test."""
    if a is None:
        return None
    return (
        a.status,
        a.damage_severity,
        a.confidence,
        tuple((r.tool_name, r.label, r.confidence) for r in a.tool_call_history),
    )


def zone_summary(r: ZoneRanking) -> dict:
    return {
        "zone_id": r.zone_id,
        "version": r.version,
        "tile_ids": r.tile_ids,
        "damage_score": r.damage_score,
        "accessibility_penalty": r.accessibility_penalty,
        "priority_score": r.priority_score,
        "status": r.status.value,
        "escalated_tile_ids": r.escalated_tile_ids,
        "ungeolocated_tile_ids": r.ungeolocated_tile_ids,
        "conflicting_tile_pairs": r.conflicting_tile_pairs,
    }


def run_pipeline(
    specs: list[TileSpec],
    config: RunConfig,
    perception: PerceptionPort,
    planner: PlannerPort,
    store,
    ledger,
    events: EventLog,
    sleep: Callable[[float], None] = time.sleep,
) -> RunResult:
    batch_id = config.batch_id
    specs_by_id = {s.tile_id: s for s in specs}
    handled: set[str] = set()

    events.emit(
        "run_started",
        batch_id=batch_id,
        tile_count=len(specs),
        adjacency_radius_m=config.adjacency_radius_m,
        max_rounds=config.max_rounds,
        tile_ids=[s.tile_id for s in specs],
        disaster=config.disaster,
        tiles=[{"tile_id": s.tile_id, "latitude": s.latitude, "longitude": s.longitude} for s in specs],
        wave_size=config.wave_size,
        wave_interval_s=config.wave_interval_s,
    )

    # ---- Agent 2 helper (used after every wave and after every re-inspection round) --
    versions: list[int] = []
    attempts: Counter[str] = Counter()
    reinspected: list[str] = []

    def plan_once(round_no: int, wave_no: int | None = None) -> PlanOutcome:
        tiles = store.list_by_batch(batch_id)  # batch-scoped: never another batch's tiles
        version = ledger.next_version(batch_id)
        events.emit("planner_started", batch_id=batch_id, version=version,
                    tile_count=len(tiles), round=round_no, wave=wave_no)
        outcome = planner.plan(batch_id, tiles, version, config.adjacency_radius_m)
        ledger.append_all(outcome.rankings)
        versions.append(version)
        events.emit(
            "zones_ranked",
            batch_id=batch_id,
            version=version,
            round=round_no,
            wave=wave_no,
            zones=[zone_summary(r) for r in outcome.rankings],
            reinspection_requests=[r.model_dump() for r in outcome.reinspection_requests],
        )
        return outcome

    # ---- Agent 1: tiles in waves (one wave unless streaming), concurrently within a wave
    def on_first_pass(a: TileAssessment) -> None:
        store.save(a)
        handled.add(a.tile_id)
        events.emit("tile_finished", batch_id=batch_id, tile=tile_summary(a))

    if config.wave_size and config.wave_size > 0:
        waves = [specs[i:i + config.wave_size] for i in range(0, len(specs), config.wave_size)]
    else:
        waves = [specs]
    waves = [w for w in waves if w] or [[]]  # never zero waves, even for an empty batch

    outcome: PlanOutcome | None = None
    for wave_no, wave_specs in enumerate(waves, start=1):
        events.emit("wave_arrived", batch_id=batch_id, wave=wave_no, waves=len(waves),
                    tile_ids=[s.tile_id for s in wave_specs])

        returned = perception.assess(batch_id, wave_specs, reinspection=False, on_result=on_first_pass)
        for a in returned:  # transports without per-tile callbacks report at the end
            if a.tile_id not in handled:
                on_first_pass(a)

        # SYS-04: a tile that never reported must not stall or silently vanish.
        for spec in wave_specs:
            if spec.tile_id not in handled:
                missing = TileAssessment(
                    tile_id=spec.tile_id,
                    batch_id=batch_id,
                    tile_path=spec.tile_path,
                    latitude=spec.latitude,
                    longitude=spec.longitude,
                    status=TileStatus.ESCALATED,
                    failure_reason="Agent 1 returned no result for this tile",
                )
                on_first_pass(missing)
                events.emit("tile_missing", batch_id=batch_id, tile_id=spec.tile_id)

        outcome = plan_once(0, wave_no)
        if wave_no < len(waves) and config.wave_interval_s > 0:
            sleep(config.wave_interval_s)

    rounds_used = 0
    no_change = False

    for round_no in range(1, config.max_rounds + 1):
        eligible = [r for r in outcome.reinspection_requests
                    if attempts[r.tile_id] < config.per_tile_cap and r.tile_id in specs_by_id]
        if not eligible:
            break
        rounds_used = round_no
        events.emit(
            "reinspection_requested",
            batch_id=batch_id,
            round=round_no,
            requests=[r.model_dump() for r in eligible],
        )
        before = {a.tile_id: a for a in store.list_by_batch(batch_id)}

        def on_reinspected(a: TileAssessment) -> None:
            store.save(a)
            old = before.get(a.tile_id)
            events.emit(
                "tile_reinspected",
                batch_id=batch_id,
                round=round_no,
                before=tile_summary(old) if old else None,
                after=tile_summary(a),
            )

        results = perception.assess(
            batch_id,
            [specs_by_id[r.tile_id] for r in eligible],
            reinspection=True,
            on_result=on_reinspected,
        )
        for a in results:
            attempts[a.tile_id] += 1
            reinspected.append(a.tile_id)

        changed = [a.tile_id for a in results if _result_key(before.get(a.tile_id)) != _result_key(a)]
        if not changed:
            events.emit("reinspection_no_change", batch_id=batch_id, round=round_no,
                        tile_ids=[a.tile_id for a in results])
            no_change = True
            break
        outcome = plan_once(round_no)

    if outcome.reinspection_requests and rounds_used >= config.max_rounds and not no_change:
        events.emit(
            "reinspection_exhausted",
            batch_id=batch_id,
            unresolved=[r.tile_id for r in outcome.reinspection_requests],
        )

    final_tiles = store.list_by_batch(batch_id)
    escalated = sorted(a.tile_id for a in final_tiles if a.status == TileStatus.ESCALATED)
    events.emit(
        "run_finished",
        batch_id=batch_id,
        versions=versions,
        zone_count=len(outcome.rankings),
        tile_count=len(final_tiles),
        escalated_tile_ids=escalated,
        reinspected_tile_ids=sorted(set(reinspected)),
        rounds_used=rounds_used,
    )
    return RunResult(
        run_id=config.run_id,
        batch_id=batch_id,
        versions=versions,
        final_rankings=outcome.rankings,
        escalated_tile_ids=escalated,
        rounds_used=rounds_used,
        reinspected_tile_ids=sorted(set(reinspected)),
    )
