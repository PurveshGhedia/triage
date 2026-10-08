"""Pick a geographically CLUSTERED set of tiles for one disaster, for the demo.

Random tiles from an xBD disaster are kilometres apart, so Agent 2's 500 m
adjacency graph leaves every tile in its own zone and the map shows isolated
dots. This picks a seed tile in the densest part of the disaster and its nearest
neighbours, copies the post- and pre-disaster images plus label files to an
output folder, and writes a matching geo file. It also prints how many zones the
selection forms at several adjacency radii, to help choose --adjacency-radius-m.

Usage (run as a module from the repo root):
    python -m scripts.select_demo_tiles --disaster hurricane-harvey --count 16
    python -m scripts.select_demo_tiles --disaster socal-fire --count 20 --seed-tile socal-fire_00000123_post_disaster

Defaults assume the symlink  data/xbd_test -> ~/Downloads/test  and the geo file
built by scripts.build_geo_file. Output goes to data/demo/<disaster>/ :
    images/ (post-disaster tiles to run Agent 1 on)   pre/   labels/   geo.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import networkx as nx

from agents.planner.adjacency import haversine_distance_m

POST_SUFFIX = "_post_disaster"
RADII_TO_REPORT = (300.0, 500.0, 800.0, 1200.0)
RADIUS_SEARCH = tuple(float(r) for r in range(300, 2601, 100))


def _dist(geo: dict, a: str, b: str) -> float:
    return haversine_distance_m(
        geo[a]["latitude"], geo[a]["longitude"], geo[b]["latitude"], geo[b]["longitude"]
    )


def post_tiles_for(geo: dict, disaster: str) -> list[str]:
    prefix = f"{disaster}_"
    return sorted(t for t in geo if t.startswith(prefix) and t.endswith(POST_SUFFIX))


def pick_seed(geo: dict, candidates: list[str], cluster_radius_m: float) -> str:
    """The tile with the most neighbours within `cluster_radius_m` (ties: alphabetical)."""
    def neighbours(t: str) -> int:
        return sum(1 for o in candidates if o != t and _dist(geo, t, o) <= cluster_radius_m)

    return max(candidates, key=lambda t: (neighbours(t), [-ord(c) for c in t]))


def select_clustered(
    geo: dict,
    disaster: str,
    count: int,
    cluster_radius_m: float = 1500.0,
    seed_tile: str | None = None,
) -> list[str]:
    """`count` post-disaster tiles of `disaster`: the seed plus its nearest neighbours."""
    candidates = post_tiles_for(geo, disaster)
    if not candidates:
        raise ValueError(f"No geolocated post-disaster tiles found for '{disaster}'")
    if seed_tile is not None and seed_tile not in candidates:
        raise ValueError(f"Seed tile '{seed_tile}' is not a geolocated tile of '{disaster}'")
    seed = seed_tile or pick_seed(geo, candidates, cluster_radius_m)
    ordered = sorted(candidates, key=lambda t: (_dist(geo, seed, t), t))
    return ordered[: min(count, len(ordered))]


def damage_mix(tiles: list[str], truth: dict[str, dict], min_buildings: int = 10) -> dict:
    """How many of `tiles` are truly damaged (xBD mean >= 0.5), undamaged, or too small to judge."""
    damaged = undamaged = small = 0
    for t in tiles:
        info = truth.get(t)
        if not info or info.get("mean") is None or info.get("buildings", 0) < min_buildings:
            small += 1
        elif info["mean"] >= 0.5:
            damaged += 1
        else:
            undamaged += 1
    return {"damaged": damaged, "undamaged": undamaged, "too_small": small}


def select_informative(
    geo: dict,
    disaster: str,
    count: int,
    truth: dict[str, dict],
    min_buildings: int = 10,
) -> list[str]:
    """The compact group of `count` tiles that contains the best MIX of damaged and undamaged tiles.

    Every post-disaster tile is tried as a seed; its `count` nearest neighbours are scored by how
    evenly they split between damaged and undamaged (0.5 = perfectly mixed), minus a small penalty
    for being spread out. xBD truth is used here only to choose WHICH area to show, never by the
    agents; it does mean the demo area is not a random sample, which should be said in the report.
    Falls back to the densest cluster if no area has both kinds of tile."""
    candidates = post_tiles_for(geo, disaster)
    if not candidates:
        raise ValueError(f"No geolocated post-disaster tiles found for '{disaster}'")
    best: tuple[float, str, list[str]] | None = None
    for seed in candidates:
        ordered = sorted(candidates, key=lambda t: (_dist(geo, seed, t), t))[: min(count, len(candidates))]
        mix = damage_mix(ordered, truth, min_buildings)
        judged = mix["damaged"] + mix["undamaged"]
        if judged < max(4, len(ordered) // 2) or mix["damaged"] < 2 or mix["undamaged"] < 2:
            continue
        balance = min(mix["damaged"], mix["undamaged"]) / judged
        spread_km = sum(_dist(geo, seed, t) for t in ordered) / len(ordered) / 1000
        score = balance - 0.1 * spread_km
        if best is None or (score, seed) > (best[0], best[1]):
            best = (score, seed, ordered)
    if best is None:
        return select_clustered(geo, disaster, count)
    return best[2]


def zones_at_radii(geo: dict, tiles: list[str], radii=RADII_TO_REPORT) -> dict[float, int]:
    """How many connected zones these tiles form at each adjacency radius."""
    result = {}
    for radius in radii:
        graph = nx.Graph()
        graph.add_nodes_from(tiles)
        for i, a in enumerate(tiles):
            for b in tiles[i + 1:]:
                if _dist(geo, a, b) <= radius:
                    graph.add_edge(a, b)
        result[radius] = nx.number_connected_components(graph)
    return result


def recommend_radius(
    geo: dict, tiles: list[str], target_zones: int, radii=RADIUS_SEARCH
) -> tuple[float, int, bool]:
    """Smallest adjacency radius that merges `tiles` into at most `target_zones`
    zones. Returns (radius_m, zones_at_that_radius, cap_hit). `cap_hit` means even
    the largest radius searched could not reach the target (the largest is returned).
    """
    counts = zones_at_radii(geo, tiles, radii)
    for radius in radii:
        if counts[radius] <= target_zones:
            return radius, counts[radius], False
    last = radii[-1]
    return last, counts[last], True


GENERATED = ("images", "pre", "labels", ".thumbs")
GENERATED_FILES = ("geo.json", "demo.json", "footprints.json")


def reset_output_dir(out_dir: Path) -> list[str]:
    """Remove what an earlier sampler run generated in `out_dir`, and nothing else.

    Without this, re-running with a different --count or seed leaves old tiles behind, and the demo
    quietly contains tiles that are no longer in geo.json. Returns the names it removed."""
    removed = []
    for name in GENERATED:
        path = out_dir / name
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(name)
    for name in GENERATED_FILES:
        path = out_dir / name
        if path.is_file():
            path.unlink()
            removed.append(name)
    return removed


def _copy_if_exists(src: Path, dst_dir: Path) -> bool:
    if not src.exists():
        return False
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / src.name)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disaster", required=True, help="e.g. hurricane-harvey")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--geo-file", type=Path, default=Path("data/xbd_test_geo.json"))
    parser.add_argument("--xbd-dir", type=Path, default=Path("data/xbd_test"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--cluster-radius-m", type=float, default=1500.0)
    parser.add_argument("--seed-tile", default=None)
    parser.add_argument("--informative", action="store_true",
                        help="Choose the area with the best mix of damaged and undamaged tiles (uses xBD "
                             "labels to PICK the area; the agents never see them).")
    parser.add_argument("--keep-existing", action="store_true",
                        help="Do not clear files from an earlier run in the output folder first.")
    parser.add_argument(
        "--target-zones", type=int, default=None,
        help="Aim for at most this many zones when recommending a radius "
             "(default: count // 4, at least 2).",
    )
    args = parser.parse_args()

    geo = json.loads(args.geo_file.read_text())
    truth: dict[str, dict] = {}
    try:
        if args.informative and not args.seed_tile:
            from shared.truth import ground_truth

            for tile in post_tiles_for(geo, args.disaster):
                label = args.xbd_dir / "labels" / f"{tile}.json"
                if label.is_file():
                    truth[tile] = ground_truth(label)
            tiles = select_informative(geo, args.disaster, args.count, truth)
        else:
            tiles = select_clustered(
                geo, args.disaster, args.count, args.cluster_radius_m, args.seed_tile
            )
    except ValueError as exc:
        sys.exit(str(exc))

    out_dir = args.out_dir or Path("data/demo") / args.disaster
    if not args.keep_existing and reset_output_dir(out_dir):
        print(f"  cleared an earlier selection in {out_dir}")
    copied = {"post": 0, "pre": 0, "labels": 0}
    for tile in tiles:
        pre = tile.replace(POST_SUFFIX, "_pre_disaster")
        copied["post"] += _copy_if_exists(args.xbd_dir / "images" / f"{tile}.png", out_dir / "images")
        copied["pre"] += _copy_if_exists(args.xbd_dir / "images" / f"{pre}.png", out_dir / "pre")
        copied["labels"] += _copy_if_exists(args.xbd_dir / "labels" / f"{tile}.json", out_dir / "labels")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "geo.json").write_text(json.dumps({t: geo[t] for t in tiles}, indent=2))

    print(f"{args.disaster}: selected {len(tiles)} tiles -> {out_dir}")
    if truth:
        mix = damage_mix(tiles, truth)
        print(f"  xBD truth in this area: {mix['damaged']} damaged, {mix['undamaged']} undamaged, "
              f"{mix['too_small']} too small to judge (chosen for variety, not random)")
    print(f"  copied: {copied['post']} post, {copied['pre']} pre, {copied['labels']} label files")
    print("  zones formed at each adjacency radius (fewer = better connected):")
    for radius, n in zones_at_radii(geo, tiles).items():
        print(f"    {radius:>6.0f} m -> {n} zone(s)")

    target = args.target_zones or max(2, len(tiles) // 4)
    radius, n_zones, cap_hit = recommend_radius(geo, tiles, target)
    seed = args.seed_tile or pick_seed(geo, post_tiles_for(geo, args.disaster), args.cluster_radius_m)
    (out_dir / "demo.json").write_text(json.dumps({
        "disaster": args.disaster,
        "seed_tile": seed,
        "tile_count": len(tiles),
        "target_zones": target,
        "adjacency_radius_m": radius,
        "zones_at_radius": n_zones,
        "radius_cap_hit": cap_hit,
    }, indent=2))
    note = "  (!! cap reached: target not met; try a larger --count)" if cap_hit else ""
    print(f"  recommended adjacency radius: {radius:.0f} m -> {n_zones} zone(s){note}")
    print(f"  wrote {out_dir}/demo.json")
    print("Run the full pipeline on it with:")
    print(f"  python -m scripts.run_e2e --disaster {args.disaster}")


if __name__ == "__main__":
    main()
