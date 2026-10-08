"""Exploratory data analysis over a folder of xBD label JSONs.

Produces summary statistics and charts for a "Data preprocessing & EDA"
slide: damage-class distribution, buildings-per-tile distribution, disaster
coverage, and image metadata (sensor, GSD).

Usage:
    python -m scripts.eda_xbd --labels-dir ~/Downloads/test/labels --out-dir data/eda

Point --labels-dir at your FULL downloaded xBD labels folder, not just the
handful of sample tiles in the repo — a 5-tile sample isn't representative
enough to call "EDA." The repo's data/sample_labels/ still works if that's
genuinely all you have, but the charts will be thin.
"""
from __future__ import annotations
import matplotlib.pyplot as plt

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed, just save files


def load_labels(labels_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in labels_dir.glob("*.json")]


def summarize(labels: list[dict]) -> dict:
    pre_count = sum(
        1 for l in labels if "pre_disaster" in l.get("metadata", {}).get("img_name", "")
    )
    post_count = sum(
        1 for l in labels if "post_disaster" in l.get("metadata", {}).get("img_name", "")
    )

    disasters = Counter(l.get("metadata", {}).get(
        "disaster", "unknown") for l in labels)
    disaster_types = Counter(l.get("metadata", {}).get(
        "disaster_type", "unknown") for l in labels)
    sensors = Counter(l.get("metadata", {}).get(
        "sensor", "unknown") for l in labels)
    gsds = [l.get("metadata", {}).get("gsd")
            for l in labels if l.get("metadata", {}).get("gsd") is not None]

    buildings_per_tile = []
    damage_subtypes = Counter()
    for label in labels:
        features = label.get("features", {}).get("lng_lat", [])
        buildings_per_tile.append(len(features))
        for feature in features:
            subtype = feature.get("properties", {}).get("subtype")
            if subtype:
                damage_subtypes[subtype] += 1

    return {
        "total_labels": len(labels),
        "pre_disaster_count": pre_count,
        "post_disaster_count": post_count,
        "disasters": disasters,
        "disaster_types": disaster_types,
        "sensors": sensors,
        "gsd_min": min(gsds) if gsds else None,
        "gsd_max": max(gsds) if gsds else None,
        "gsd_avg": (sum(gsds) / len(gsds)) if gsds else None,
        "buildings_per_tile": buildings_per_tile,
        "damage_subtypes": damage_subtypes,
    }


def make_charts(summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if summary["damage_subtypes"]:
        labels, counts = zip(
            *sorted(summary["damage_subtypes"].items(), key=lambda kv: -kv[1]))
        plt.figure(figsize=(7, 4))
        plt.bar(labels, counts, color="#4C72B0")
        plt.title("Building damage subtype distribution")
        plt.ylabel("Building count")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(out_dir / "damage_subtype_distribution.png", dpi=150)
        plt.close()

    if summary["buildings_per_tile"]:
        plt.figure(figsize=(7, 4))
        plt.hist(summary["buildings_per_tile"], bins=20, color="#55A868")
        plt.title("Labeled buildings per tile")
        plt.xlabel("Buildings in tile")
        plt.ylabel("Number of tiles")
        plt.tight_layout()
        plt.savefig(out_dir / "buildings_per_tile_histogram.png", dpi=150)
        plt.close()

    if len(summary["disaster_types"]) > 1:
        labels, counts = zip(*summary["disaster_types"].items())
        plt.figure(figsize=(7, 4))
        plt.bar(labels, counts, color="#C44E52")
        plt.title("Tiles per disaster type")
        plt.ylabel("Tile count")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(out_dir / "disaster_type_distribution.png", dpi=150)
        plt.close()


def print_summary(summary: dict) -> None:
    print(f"Total label files: {summary['total_labels']}")
    print(f"  pre_disaster:  {summary['pre_disaster_count']}")
    print(f"  post_disaster: {summary['post_disaster_count']}")
    print()
    print("Disasters represented:")
    for name, count in summary["disasters"].most_common():
        print(f"  {name}: {count}")
    print()
    print("Disaster types:")
    for name, count in summary["disaster_types"].most_common():
        print(f"  {name}: {count}")
    print()
    print("Sensors:")
    for name, count in summary["sensors"].most_common():
        print(f"  {name}: {count}")
    print()
    if summary["gsd_avg"] is not None:
        print(
            f"GSD (ground sample distance, meters/pixel): "
            f"min={summary['gsd_min']:.3f}, max={summary['gsd_max']:.3f}, "
            f"avg={summary['gsd_avg']:.3f}"
        )
        print()
    if summary["buildings_per_tile"]:
        bpt = summary["buildings_per_tile"]
        print(
            f"Buildings per tile: min={min(bpt)}, max={max(bpt)}, "
            f"avg={sum(bpt) / len(bpt):.1f}, total={sum(bpt)}"
        )
        print()
    print("Damage subtype distribution (post-disaster labels only):")
    total_buildings = sum(summary["damage_subtypes"].values())
    for name, count in summary["damage_subtypes"].most_common():
        pct = 100 * count / total_buildings if total_buildings else 0
        print(f"  {name}: {count} ({pct:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/eda"))
    args = parser.parse_args()

    labels = load_labels(args.labels_dir)
    if not labels:
        raise SystemExit(f"No label JSON files found in {args.labels_dir}")

    summary = summarize(labels)
    print_summary(summary)
    make_charts(summary, args.out_dir)
    print(f"\nCharts saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
