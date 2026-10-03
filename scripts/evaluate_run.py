"""Score a finished run against xBD ground truth.

Usage (run as a module from the repo root):
    python -m scripts.evaluate_run                        # the newest run
    python -m scripts.evaluate_run --run-id 20261003-101500-ab12
    python -m scripts.evaluate_run --compare RUN_A RUN_B  # e.g. a mock run vs a vision run

Truth for a tile = the mean xBD damage (0-3) over its labelled buildings. The output shows
rank correlation (does the run order tiles/zones by damage like the experts did?), average
error on the 0-3 scale against a "constant guess" baseline, and how many tiles are within one
damage level. Run it on a mock run and a vision run to see what real perception adds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agents.orchestrator.evaluation import evaluate_events
from agents.orchestrator.events import read_events


def newest_run(runs_dir: Path) -> str | None:
    candidates = [d for d in runs_dir.iterdir() if (d / "events.jsonl").exists()] if runs_dir.is_dir() else []
    return max(candidates, key=lambda d: d.stat().st_mtime).name if candidates else None


def evaluate_run(run_id: str, runs_dir: Path, demo_root: Path, details: bool = False) -> dict:
    events = read_events(runs_dir / run_id / "events.jsonl")
    started = next((e for e in events if e["type"] == "run_started"), None)
    if started is None or not started.get("disaster"):
        raise ValueError(f"run {run_id} has no disaster recorded")
    result = evaluate_events(events, demo_root / started["disaster"] / "labels", details=details)
    result["run"] = {"run_id": run_id, "disaster": started["disaster"], "tools": started.get("tools", "mock")}
    return result


def show(r: dict) -> None:
    t, z, run = r["tiles"], r["zones"], r["run"]
    fmt = lambda v: "n/a" if v is None else v
    print(f"{run['run_id']}  {run['disaster']}  tools={run['tools']}")
    print(f"  tiles compared: {t['tiles_compared']}  (escalated, not scored: {t['tiles_escalated']})")
    print(f"  tile rank correlation (Spearman): {fmt(t['spearman'])}   (1 = same order as the experts, 0 = unrelated)")
    print(f"  mean abs error: {fmt(t['mean_abs_error'])}  vs constant-guess baseline {fmt(t['baseline_mean_abs_error'])}   (0-3 scale; lower is better)")
    print(f"  tiles within one damage level: {fmt(t['within_one_level'])}")
    print(f"  zone rank correlation (Spearman, {z['zones_compared']} zones): {fmt(z['spearman'])}"
          + ("   (too few zones to read much into)" if z["zones_compared"] < 8 else ""))
    rows = r.get("details")
    if rows:
        print("  worst disagreements (predicted vs xBD mean damage, 0-3):")
        print(f"    {'tile':>6}  {'pred':>5}  {'truth':>5}  {'worst':>5}  {'bldgs':>5}  {'error':>6}")
        for row in rows[:12]:
            tid = row["tile_id"].split("_")[-3].lstrip("0") or "0"
            print(f"    {tid:>6}  {row['predicted']:>5}  {row['truth_mean']:>5}  {row['truth_worst']:>5}  {row['buildings']:>5}  {row['error']:>+6}")
        over = sum(1 for x in rows if x["error"] > 0.5)
        under = sum(1 for x in rows if x["error"] < -0.5)
        print(f"  over-calls (pred > truth by >0.5): {over}   under-calls: {under}   of {len(rows)} tiles")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--compare", nargs="+", default=None)
    parser.add_argument("--runs-dir", type=Path, default=Path("data/runs"))
    parser.add_argument("--demo-root", type=Path, default=Path("data/demo"))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--details", action="store_true", help="Per-tile table of the worst disagreements.")
    args = parser.parse_args()

    run_ids = args.compare or [args.run_id or newest_run(args.runs_dir)]
    if not run_ids or run_ids[0] is None:
        sys.exit("No runs found. Run: python -m scripts.run_e2e --disaster hurricane-harvey --offline")
    results = []
    for rid in run_ids:
        try:
            results.append(evaluate_run(rid, args.runs_dir, args.demo_root, details=args.details))
        except (ValueError, FileNotFoundError) as exc:
            sys.exit(str(exc))
    for r in results:
        print(json.dumps(r, indent=1)) if args.json else show(r)


if __name__ == "__main__":
    main()
