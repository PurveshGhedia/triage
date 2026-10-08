"""Cost and time of a run, from what the model calls actually reported (tokens and seconds).

    python -m scripts.cost_report                       # the newest run
    python -m scripts.cost_report --run-id 20261004-101500-ab12
    python -m scripts.cost_report --tiles 1000          # project the cost/time of a bigger area

Only vision calls are measured. A call served from the answer cache costs nothing NOW, so it is
counted separately (with what it originally cost). To measure real cost, run with an empty cache:
    TRIAGE_VISION_CACHE=$(mktemp -d) python -m scripts.run_e2e --disaster hurricane-harvey --tools vision --offline

Prices (USD per million tokens) default to the published Gemini 2.5 Flash rates reported by third-party
trackers in mid-2026 ($0.30 input, $2.50 output). They change: check Google's pricing page and override with
GEMINI_PRICE_IN_PER_M / GEMINI_PRICE_OUT_PER_M or --price-in / --price-out. Gemini tool-selection calls
(small text-only prompts) are not measured and are excluded.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from agents.orchestrator.events import read_events

DEFAULT_PRICE_IN = 0.30
DEFAULT_PRICE_OUT = 2.50


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo), 3)


def _calls(events: list[dict]) -> list[dict]:
    """Every tool call that produced a reading: first-pass tiles and second-opinion re-inspections."""
    calls = []
    for e in events:
        tile = e.get("tile") if e["type"] == "tile_finished" else e.get("after") if e["type"] == "tile_reinspected" else None
        for c in (tile or {}).get("tool_calls", []):
            calls.append({**c, "tile_id": tile["tile_id"]})
    return calls


def summarise(events: list[dict], price_in: float = DEFAULT_PRICE_IN, price_out: float = DEFAULT_PRICE_OUT) -> dict:
    started = next((e for e in events if e["type"] == "run_started"), None)
    finished = next((e for e in reversed(events) if e["type"] == "run_finished"), None)
    tiles = started["tile_count"] if started else 0
    wall = None
    if started and finished:
        wall = (datetime.fromisoformat(finished["ts"]) - datetime.fromisoformat(started["ts"])).total_seconds()

    vision = [c for c in _calls(events) if c.get("cached") is not None]
    billed = [c for c in vision if c["cached"] is False]
    replayed = [c for c in vision if c["cached"] is True]
    cost = lambda cs: sum((c.get("prompt_tokens") or 0) * price_in + (c.get("output_tokens") or 0) * price_out for c in cs) / 1e6
    latencies = [c["latency_s"] for c in billed if c.get("latency_s") is not None]

    per_tool: dict[str, dict] = {}
    for c in billed:
        t = per_tool.setdefault(c["tool"], {"calls": 0, "prompt_tokens": 0, "output_tokens": 0})
        t["calls"] += 1
        t["prompt_tokens"] += c.get("prompt_tokens") or 0
        t["output_tokens"] += c.get("output_tokens") or 0
    for name, t in per_tool.items():
        t["cost_usd"] = round(cost([c for c in billed if c["tool"] == name]), 5)

    seq = sum(latencies)
    return {
        "tiles": tiles,
        "wall_seconds": round(wall, 1) if wall is not None else None,
        "vision_calls": len(vision),
        "billed_calls": len(billed),
        "replayed_calls": len(replayed),
        "billed_without_usage": sum(1 for c in billed if c.get("prompt_tokens") is None),
        "prompt_tokens": sum(c.get("prompt_tokens") or 0 for c in billed),
        "output_tokens": sum(c.get("output_tokens") or 0 for c in billed),
        "cost_usd": round(cost(billed), 5),
        "replayed_originally_cost_usd": round(cost(replayed), 5),
        "latency_mean_s": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "latency_median_s": _percentile(latencies, 0.5),
        "latency_p95_s": _percentile(latencies, 0.95),
        "latency_max_s": round(max(latencies), 2) if latencies else None,
        "calls_per_tile": round(len(vision) / tiles, 2) if tiles else None,
        "cost_per_tile_usd": round(cost(billed) / tiles, 5) if tiles and billed else None,
        "wall_seconds_per_tile": round(wall / tiles, 2) if tiles and wall is not None else None,
        "sequential_seconds_per_tile": round(seq / tiles, 2) if tiles and latencies else None,
        "per_tool": per_tool,
        "prices_per_million": {"input": price_in, "output": price_out},
    }


def project(summary: dict, tiles: int, workers: int) -> dict | None:
    """Scale the measured per-tile figures to `tiles` tiles (a PROJECTION: it assumes the same per-tile
    behaviour and that time shrinks linearly with workers, up to the model's rate limits)."""
    if not summary["cost_per_tile_usd"] or not summary["sequential_seconds_per_tile"]:
        return None
    return {
        "tiles": tiles,
        "workers": workers,
        "cost_usd": round(summary["cost_per_tile_usd"] * tiles, 2),
        "minutes": round(summary["sequential_seconds_per_tile"] * tiles / workers / 60, 1),
    }


def show(s: dict, run_id: str, proj: dict | None) -> None:
    f = lambda v, unit="": "n/a" if v is None else f"{v}{unit}"
    print(f"Run {run_id}: {s['tiles']} tiles, wall-clock {f(s['wall_seconds'], ' s')}")
    print(f"  vision calls: {s['vision_calls']} ({s['calls_per_tile']} per tile): "
          f"{s['billed_calls']} billed now, {s['replayed_calls']} replayed from cache (free now)")
    if s["billed_calls"] == 0 and s["replayed_calls"]:
        print("  !! everything was replayed from the cache, so this run measured nothing. For real numbers use an empty cache:")
        print("     TRIAGE_VISION_CACHE=$(mktemp -d) python -m scripts.run_e2e --disaster <area> --tools vision --offline")
    if s["billed_without_usage"]:
        print(f"  note: {s['billed_without_usage']} billed call(s) reported no token usage, so cost is understated")
    print(f"  tokens billed: {s['prompt_tokens']:,} in, {s['output_tokens']:,} out")
    pr = s["prices_per_million"]
    print(f"  COST this run: ${s['cost_usd']}  (at ${pr['input']}/M in, ${pr['output']}/M out)  ->  {f(s['cost_per_tile_usd'], ' USD')} per tile")
    if s["replayed_originally_cost_usd"]:
        print(f"  (replayed answers originally cost ${s['replayed_originally_cost_usd']})")
    print(f"  per call: mean {f(s['latency_mean_s'], ' s')}, median {f(s['latency_median_s'], ' s')}, "
          f"p95 {f(s['latency_p95_s'], ' s')}, max {f(s['latency_max_s'], ' s')}")
    print(f"  TIME per tile: {f(s['wall_seconds_per_tile'], ' s')} wall-clock (parallel); "
          f"{f(s['sequential_seconds_per_tile'], ' s')} of model time if done one at a time")
    for name, t in s["per_tool"].items():
        print(f"    {name:<11}{t['calls']:>4} calls  {t['prompt_tokens']:>7,} in  {t['output_tokens']:>6,} out  ${t['cost_usd']}")
    if proj:
        print(f"  PROJECTION for {proj['tiles']:,} tiles with {proj['workers']} workers: about ${proj['cost_usd']} and "
              f"{proj['minutes']} minutes (assumes the same per-tile behaviour and linear speed-up)")


def newest_run(runs_dir: Path) -> str | None:
    found = [d for d in runs_dir.iterdir() if (d / "events.jsonl").exists()] if runs_dir.is_dir() else []
    return max(found, key=lambda d: d.stat().st_mtime).name if found else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--runs-dir", type=Path, default=Path("data/runs"))
    parser.add_argument("--price-in", type=float, default=float(os.environ.get("GEMINI_PRICE_IN_PER_M", DEFAULT_PRICE_IN)))
    parser.add_argument("--price-out", type=float, default=float(os.environ.get("GEMINI_PRICE_OUT_PER_M", DEFAULT_PRICE_OUT)))
    parser.add_argument("--tiles", type=int, default=None, help="Also project cost/time for this many tiles.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or newest_run(args.runs_dir)
    if not run_id:
        sys.exit("No runs found.")
    events = read_events(args.runs_dir / run_id / "events.jsonl")
    if not events:
        sys.exit(f"No events for run {run_id}.")
    s = summarise(events, args.price_in, args.price_out)
    proj = project(s, args.tiles, args.workers) if args.tiles else None
    if args.json:
        print(json.dumps({"run_id": run_id, **s, "projection": proj}, indent=1))
    else:
        show(s, run_id, proj)


if __name__ == "__main__":
    main()
