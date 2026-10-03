"""xBD ground truth for a tile, from its label file: per-building damage counts and a
0-3 summary. Used by the dashboard's ground-truth toggle and by the accuracy evaluation."""
from __future__ import annotations

import json
from pathlib import Path

DAMAGE_SCORES = {"no-damage": 0, "minor-damage": 1, "major-damage": 2, "destroyed": 3}


def ground_truth(label_path: str | Path) -> dict:
    """Per-building xBD damage counts for one tile, and a 0-3 tile summary.
    Buildings labelled 'un-classified' are ignored."""
    data = json.loads(Path(label_path).read_text())
    counts = {k: 0 for k in DAMAGE_SCORES}
    for feature in data.get("features", {}).get("lng_lat", []):
        subtype = feature.get("properties", {}).get("subtype")
        if subtype in counts:
            counts[subtype] += 1
    scores = [DAMAGE_SCORES[k] for k, n in counts.items() for _ in range(n)]
    return {
        "counts": counts,
        "buildings": len(scores),
        "worst": max(scores) if scores else None,
        "mean": round(sum(scores) / len(scores), 2) if scores else None,
    }
