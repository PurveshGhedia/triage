"""Compare vision prompt variants on the SAME tiles, with real Gemini calls, in one command.

    python -m scripts.ab_vision --disasters hurricane-harvey santa-rosa-wildfire midwest-flooding
    python -m scripts.ab_vision --disasters hurricane-michael socal-fire --demo-root data/heldout --live

For each disaster and each variant it runs perception only (no live feed, no re-inspection, so the
numbers isolate what Gemini sees), scores the result against xBD ground truth, and prints a table
plus pooled numbers over the informative areas. It also reports wall-clock time and how many fresh
Gemini answers each run produced.

--live uses an empty answer cache, so EVERY call is a real, billed Gemini call (nothing replayed).
Without it, answers already cached from earlier real runs are reused (faster, same answers).

The variants are tried on identical tiles, so a difference is the prompt, not the data. To keep a
fair fight no run sees another's answers: each variant has its own cache entries by construction.
"""
from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path

from agents.orchestrator.evaluation import (
    evaluate,
    predictions_from_events,
    truth_for_tiles,
)
from agents.orchestrator.events import read_events
from agents.orchestrator.runner import RunOptions, execute_run

MIN_BUILDINGS = 10


def run_one(disaster: str, variant: str, demo_root: Path, live: bool, work: Path) -> dict:
    """One perception-only run; returns its metrics and the data needed for pooling."""
    # live: a brand-new empty folder for EVERY run, so nothing can ever be replayed from an earlier one
    cache = (Path(tempfile.mkdtemp(prefix=f"cache-{variant}-", dir=work)) if live
             else Path(os.environ.get("TRIAGE_VISION_CACHE", "data/cache/vision")))
    before = len(list(cache.glob("*.json"))) if cache.is_dir() else 0
    previous = os.environ.get("TRIAGE_VISION_CACHE")
    os.environ["TRIAGE_VISION_CACHE"] = str(cache)
    started = time.time()
    try:
        result = execute_run(RunOptions(
            disaster=disaster, demo_root=demo_root, offline=True, tools="vision", vision_variant=variant,
            rounds=0, events_dir=work / "runs", ledger_path=work / "ledger.jsonl"))
    finally:
        if previous is None:
            os.environ.pop("TRIAGE_VISION_CACHE", None)
        else:
            os.environ["TRIAGE_VISION_CACHE"] = previous
    seconds = time.time() - started
    events = read_events(work / "runs" / result.run_id / "events.jsonl")
    predictions = predictions_from_events(events)
    truth = truth_for_tiles(demo_root / disaster / "labels", list(predictions["scores"]) + predictions["escalated"])
    metrics = evaluate(predictions, truth, min_buildings=MIN_BUILDINGS)
    after = len(list(cache.glob("*.json"))) if cache.is_dir() else 0
    return {"disaster": disaster, "variant": variant, "metrics": metrics, "predictions": predictions,
            "truth": truth, "seconds": round(seconds, 1), "new_answers": after - before}


def pool(rows: list[dict]) -> dict:
    """Metrics over all tiles of the informative areas in `rows` (one variant)."""
    scores, truth = {}, {}
    for r in rows:
        if not r["metrics"]["tiles"]["informative"]:
            continue
        for t, v in r["predictions"]["scores"].items():
            scores[f"{r['disaster']}/{t}"] = v
        for t, v in r["truth"].items():
            truth[f"{r['disaster']}/{t}"] = v
    if not scores:
        return {}
    return evaluate({"scores": scores, "escalated": [], "zones": []}, truth, min_buildings=MIN_BUILDINGS)["tiles"]


def fmt(v) -> str:
    return "n/a" if v is None else f"{v:.2f}"


def table(results: list[dict], variants: list[str]) -> str:
    lines = [f"{'area':<22}{'variant':<11}{'tiles':>6}{'MAE':>7}{'guess':>7}{'in-1':>6}{'rho':>7}{'time':>7}{'new':>5}  note"]
    for r in results:
        t = r["metrics"]["tiles"]
        note = "" if t["informative"] else "NOT INFORMATIVE (control)"
        lines.append(f"{r['disaster']:<22}{r['variant']:<11}{t['tiles_compared']:>6}{fmt(t['mean_abs_error']):>7}"
                     f"{fmt(t['baseline_mean_abs_error']):>7}{fmt(t['within_one_level']):>6}{fmt(t['spearman']):>7}"
                     f"{r['seconds']:>6.0f}s{r['new_answers']:>5}  {note}")
    lines.append("")
    lines.append("Pooled over the informative areas (tiles with >= 10 labelled buildings):")
    for v in variants:
        p = pool([r for r in results if r["variant"] == v])
        if p:
            lines.append(f"  {v:<10} tiles {p['tiles_compared']:>3}   MAE {fmt(p['mean_abs_error'])} (guess {fmt(p['baseline_mean_abs_error'])})"
                         f"   within one level {fmt(p['within_one_level'])}   Spearman {fmt(p['spearman'])}")
    lines.append("")
    lines.append("MAE = mean absolute error on the 0-3 damage scale (lower is better); guess = always predicting the middle truth;")
    lines.append("in-1 = share of tiles within one damage level; rho = rank correlation; new = fresh Gemini answers this run.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disasters", nargs="+", required=True)
    parser.add_argument("--variants", nargs="+", default=["classes", "fractions"], choices=["classes", "fractions"])
    parser.add_argument("--demo-root", type=Path, default=Path("data/demo"))
    parser.add_argument("--live", action="store_true", help="Empty cache: every call is a real Gemini call.")
    args = parser.parse_args()

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for disaster in args.disasters:
            for variant in args.variants:
                print(f"running {disaster} / {variant} ...", flush=True)
                try:
                    results.append(run_one(disaster, variant, args.demo_root, args.live, work))
                except ValueError as exc:
                    print(f"  skipped: {exc}")
        print()
        print(table(results, args.variants))


if __name__ == "__main__":
    main()
