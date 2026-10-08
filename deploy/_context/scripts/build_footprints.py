"""Compute each demo tile's true ground footprint from its xBD label file and write
data/demo/<disaster>/footprints.json, which the dashboard uses to draw the real
post-disaster imagery on the map at the right place.

Usage (run as a module from the repo root):
    python -m scripts.build_footprints                      # every folder under data/demo
    python -m scripts.build_footprints --disaster hurricane-harvey

Tiles whose labels have too few buildings to fit are skipped (the dashboard falls back
to a square around the tile's centre). The summary shows each disaster's fit error, in
metres: it is the honest measure of how well the imagery lines up.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from shared.georef import footprint_from_file


def build_for(demo_dir: Path) -> tuple[dict, list[str]]:
    footprints: dict[str, dict] = {}
    skipped: list[str] = []
    for label in sorted((demo_dir / "labels").glob("*.json")):
        fp = footprint_from_file(label)
        if fp is None:
            skipped.append(label.stem)
        else:
            footprints[label.stem] = fp
    (demo_dir / "footprints.json").write_text(json.dumps(footprints, indent=1))
    return footprints, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-root", type=Path, default=Path("data/demo"))
    parser.add_argument("--disaster", default=None)
    args = parser.parse_args()

    dirs = ([args.demo_root / args.disaster] if args.disaster
            else sorted(p for p in args.demo_root.iterdir() if (p / "labels").is_dir()))
    for d in dirs:
        footprints, skipped = build_for(d)
        if footprints:
            rms = sorted(f["rms_m"] for f in footprints.values())
            size = sorted(f["width_m"] for f in footprints.values())
            print(f"{d.name}: {len(footprints)} footprints (skipped {len(skipped)}) · "
                  f"fit error median {rms[len(rms) // 2]} m, worst {rms[-1]} m · "
                  f"tile width ~{size[len(size) // 2]:.0f} m")
        else:
            print(f"{d.name}: no footprints could be computed ({len(skipped)} skipped)")


if __name__ == "__main__":
    main()
