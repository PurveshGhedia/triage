"""One command: Agent 1 over a demo batch -> Agent 2 -> re-inspection rounds ->
re-plan, with every step written to an event log the dashboard can replay.

Usage (run as a module from the repo root):
    python -m scripts.run_e2e --disaster hurricane-harvey
    python -m scripts.run_e2e --disaster hurricane-harvey --use-llm-selector
    python -m scripts.run_e2e --disaster hurricane-harvey --offline     # no Firestore

Reads data/demo/<disaster>/ (made by scripts.select_demo_tiles): the post-disaster
images, geo.json and demo.json (which holds the recommended adjacency radius).
Each run gets its own batch id (<disaster>-<timestamp>) unless you pass --batch-id,
so repeated demo runs never mix versions. Events go to data/runs/<run_id>/events.jsonl.

--tools vision swaps the mock tools for Gemini-vision tools that look at the real before/after
imagery (cached on disk, so a repeat run is instant and identical). Without it the tools are
random stand-ins that ignore the image: fine for testing the machinery, not for judging damage.

--transport a2a runs Agent 1 and Agent 2 as real Google ADK A2A services and talks to them
over HTTP (the orchestrator uses ADK's RemoteA2aAgent). With --spawn-local the two services
are started on localhost for you; otherwise pass --agent1-url / --agent2-url. If an A2A call
fails, the run falls back to the in-process agents (disable with --no-fallback) and says so.

Simulated live arrival: --wave-size 3 --wave-interval-s 2 feeds tiles in waves (default order:
a north-to-south "sweep", like a satellite pass) and re-plans after every wave, so zones form
and priorities shift as data comes in. Each wave is a new ledger version.

--offline keeps tile records in memory (no Firestore/network); the zone ledger and
event log are still written to disk, so the dashboard works the same way.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agents.orchestrator.models import ARRIVAL_ORDERS
from agents.orchestrator.runner import RunOptions, execute_run


def print_event(event: dict) -> None:
    t = event["type"]
    if t == "run_started":
        print(f"[run] tools: {event.get('tools', 'mock')}")
        print(f"[run] {event['tile_count']} tiles, batch {event['batch_id']}, "
              f"radius {event['adjacency_radius_m']:.0f} m")
    elif t == "tile_finished":
        tile = event["tile"]
        sev = tile["damage_severity"] or "-"
        extra = f"  ({tile['failure_reason']})" if tile["failure_reason"] else ""
        print(f"  [agent 1] {tile['tile_id']}: {tile['status']} {sev} "
              f"conf={tile['confidence']}{extra}")
    elif t == "wave_arrived":
        print(f"[feed] wave {event['wave']}/{event['waves']}: {len(event['tile_ids'])} tile(s) arrived")
    elif t == "zones_ranked":
        print(f"[agent 2] version {event['version']}: {len(event['zones'])} zone(s), "
              f"{len(event['reinspection_requests'])} re-inspection request(s)")
        for z in event["zones"][:5]:
            print(f"    {z['zone_id']}: priority {z['priority_score']} "
                  f"(damage {z['damage_score']} + access {z['accessibility_penalty']}) {z['status']}")
    elif t == "reinspection_requested":
        print(f"[loop] round {event['round']}: re-inspecting "
              f"{[r['tile_id'] for r in event['requests']]}")
    elif t == "tile_reinspected":
        b, a = event["before"], event["after"]
        print(f"  [agent 1] {a['tile_id']}: {b['damage_severity'] if b else '?'} -> "
              f"{a['damage_severity']}")
    elif t == "transport_selected":
        print(f"[transport] {event['transport']}"
              + (f"  agent1={event['agent1_url']}  agent2={event['agent2_url']}"
                 if event["transport"] == "a2a" else ""))
    elif t == "transport_fallback":
        print(f"[transport] !! A2A call to the {event['agent']} agent failed "
              f"({event['error']}); using the in-process agent instead")
    elif t == "reinspection_no_change":
        print(f"[loop] round {event['round']}: second opinion changed nothing; stopping")
    elif t == "reinspection_exhausted":
        print(f"[loop] cap reached; still unresolved: {event['unresolved']}")
    elif t == "run_finished":
        print(f"[run] done: versions {event['versions']}, {event['zone_count']} zone(s), "
              f"{len(event['escalated_tile_ids'])} escalated tile(s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disaster", required=True)
    parser.add_argument("--demo-dir", type=Path, default=None)
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--radius-m", type=float, default=None)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--per-tile-cap", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--wave-size", type=int, default=None,
                        help="Feed tiles in waves of this size (simulated live arrival).")
    parser.add_argument("--wave-interval-s", type=float, default=2.0)
    parser.add_argument("--arrival-order", choices=ARRIVAL_ORDERS, default="sweep")
    parser.add_argument("--use-llm-selector", action="store_true",
                        help="Let Gemini choose which tool to try next (default: fixed order).")
    parser.add_argument("--vision-variant", choices=["classes", "fractions"], default=None,
                        help="Vision prompt: 'classes' (one level per tile) or 'fractions' (share of "
                             "buildings per level -> continuous score). Default: classes.")
    parser.add_argument("--tools", choices=["mock", "vision"], default="mock",
                        help="mock: random stand-in tools. vision: Gemini looks at the real imagery.")
    parser.add_argument("--transport", choices=["inprocess", "a2a"], default="inprocess")
    parser.add_argument("--agent1-url", default=None)
    parser.add_argument("--agent2-url", default=None)
    parser.add_argument("--spawn-local", action="store_true",
                        help="With --transport a2a: start both agents on localhost.")
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--events-dir", type=Path, default=Path("data/runs"))
    args = parser.parse_args()

    if args.vision_variant:
        import os

        os.environ["TRIAGE_VISION_VARIANT"] = args.vision_variant
    opts = RunOptions(
        disaster=args.disaster, demo_dir=args.demo_dir, batch_id=args.batch_id,
        radius_m=args.radius_m, rounds=args.rounds, per_tile_cap=args.per_tile_cap,
        concurrency=args.concurrency, wave_size=args.wave_size,
        wave_interval_s=args.wave_interval_s, arrival_order=args.arrival_order,
        use_llm_selector=args.use_llm_selector, tools=args.tools, transport=args.transport,
        agent1_url=args.agent1_url, agent2_url=args.agent2_url,
        spawn_local=args.spawn_local, fallback=not args.no_fallback,
        offline=args.offline, events_dir=args.events_dir,
    )
    try:
        result = execute_run(opts, listener=print_event)
    except ValueError as exc:
        sys.exit(str(exc))
    print(f"\nEvent log: {opts.events_dir / result.run_id / 'events.jsonl'}")


if __name__ == "__main__":
    main()
