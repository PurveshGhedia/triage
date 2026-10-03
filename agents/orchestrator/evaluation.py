"""How well did a run's damage readings agree with xBD ground truth?

Truth for a tile is the mean damage (0-3) over its labelled buildings. The prediction is the
damage score Agent 2 would use: the highest-confidence reading from a damage tool (classifier /
SAM), on the same 0-3 scale. Pure functions, so they are easy to test and to run on any run.

Honest limits: a tile's mean is a summary of many buildings, so even a perfect tile-level
assessor will not match it exactly; compare runs (mock vs vision) rather than reading one
number in isolation. The 'baseline' shows what a constant guess would score.
"""
from __future__ import annotations

from pathlib import Path

from agents.planner.scoring import DAMAGE_LABEL_SCORES, DAMAGE_TOOL_NAMES
from shared.truth import ground_truth


def _ranks(values: list[float]) -> list[float]:
    """Average ranks (ties share the mean of their positions)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation; None if fewer than 3 points or either side is constant."""
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return None
    return round(cov / (vx * vy) ** 0.5, 3)


def predictions_from_events(events: list[dict]) -> dict:
    """Final per-tile state from a run's event log: scores for assessed tiles, plus the list
    of escalated tiles and the last published zone ranking."""
    tiles: dict[str, dict] = {}
    zones: list[dict] = []
    for e in events:
        if e["type"] == "tile_finished":
            tiles[e["tile"]["tile_id"]] = e["tile"]
        elif e["type"] == "tile_reinspected":
            tiles[e["after"]["tile_id"]] = e["after"]
        elif e["type"] == "zones_ranked":
            zones = e["zones"]
    scores: dict[str, float] = {}
    escalated: list[str] = []
    for tile_id, t in tiles.items():
        calls = [c for c in t.get("tool_calls", []) if c["tool"] in DAMAGE_TOOL_NAMES]
        if t["status"] == "escalated" or not calls:
            escalated.append(tile_id)
            continue
        best = max(calls, key=lambda c: c["confidence"])
        scores[tile_id] = best["score"] if best.get("score") is not None else DAMAGE_LABEL_SCORES.get(best["label"], 0.0)
    return {"scores": scores, "escalated": sorted(escalated), "zones": zones}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def evaluate(predictions: dict, truth: dict[str, dict]) -> dict:
    """Compare predictions with truth ({tile_id: ground_truth(...)}); see module docstring."""
    pairs = [(predictions["scores"][t], truth[t]["mean"]) for t in predictions["scores"]
             if t in truth and truth[t].get("mean") is not None]
    preds = [p for p, _ in pairs]
    means = [m for _, m in pairs]
    errors = [abs(p - m) for p, m in pairs]
    median_truth = sorted(means)[len(means) // 2] if means else 0.0

    tile_metrics = {
        "tiles_compared": len(pairs),
        "tiles_escalated": len(predictions["escalated"]),
        "spearman": spearman(preds, means),
        "mean_abs_error": round(_mean(errors), 2) if errors else None,
        "baseline_mean_abs_error": round(_mean([abs(median_truth - m) for m in means]), 2) if means else None,
        "within_one_level": round(sum(1 for x in errors if x <= 1.0) / len(errors), 2) if errors else None,
    }

    zone_pairs = []
    for z in predictions["zones"]:
        member = [(truth[t]["mean"], truth[t]["buildings"]) for t in z["tile_ids"]
                  if t in truth and truth[t].get("mean") is not None and truth[t]["buildings"]]
        weight = sum(b for _, b in member)
        if weight:
            zone_pairs.append((z["damage_score"], sum(m * b for m, b in member) / weight))
    zone_metrics = {
        "zones_compared": len(zone_pairs),
        "spearman": spearman([p for p, _ in zone_pairs], [t for _, t in zone_pairs]),
    }
    return {"tiles": tile_metrics, "zones": zone_metrics}


def tile_table(predictions: dict, truth: dict[str, dict]) -> list[dict]:
    """One row per scored tile, worst disagreements first: where is the run wrong, and how?"""
    rows = []
    for tile_id, pred in predictions["scores"].items():
        t = truth.get(tile_id)
        if not t or t.get("mean") is None:
            continue
        rows.append({"tile_id": tile_id, "predicted": round(pred, 2), "truth_mean": t["mean"],
                     "truth_worst": t["worst"], "buildings": t["buildings"],
                     "error": round(pred - t["mean"], 2)})
    return sorted(rows, key=lambda r: -abs(r["error"]))


def truth_for_tiles(labels_dir: str | Path, tile_ids: list[str]) -> dict[str, dict]:
    truth = {}
    for tile_id in tile_ids:
        label = Path(labels_dir) / f"{tile_id}.json"
        if label.is_file():
            truth[tile_id] = ground_truth(label)
    return truth


def evaluate_events(events: list[dict], labels_dir: str | Path, details: bool = False) -> dict:
    predictions = predictions_from_events(events)
    ids = list(predictions["scores"]) + predictions["escalated"]
    truth = truth_for_tiles(labels_dir, ids)
    result = evaluate(predictions, truth)
    if details:
        result["details"] = tile_table(predictions, truth)
    return result
