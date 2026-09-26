"""Build a --geo-file JSON for scripts/run_local_batch.py from xBD post-disaster
label JSONs, by centroid-averaging each tile's labeled building footprints.

xBD label json structure (per the official dataset):
{
  "features": {
    "lng_lat": [
      {"properties": {"feature_type": "building", "uid": "...", "subtype": "..."},
       "wkt": "POLYGON ((<lng> <lat>, ...))"},
      ...
    ],
    "xy": [...]  # same buildings in pixel coordinates — not used here
  },
  "metadata": {"img_name": "...", "disaster": "...", ...}
}

Usage:
    python -m scripts.build_geo_file \
        --labels-dir data/xbd/hurricane-harvey/labels \
        --tiles-dir data/sample_tiles \
        --out data/sample_tiles_geo.json

Tiles are matched by filename stem (e.g. tile image
"hurricane-harvey_00000000_post_disaster.png" matches label
"hurricane-harvey_00000000_post_disaster.json" in --labels-dir). A tile with
no labeled buildings gets no entry in the output — run_local_batch.py (and
later Agent 2) will treat it as missing geolocation, same as any tile you
haven't geo-tagged at all.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import shapely.wkt

TILE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def extract_centroid(label_path: Path) -> tuple[float, float] | None:
    """Returns (latitude, longitude) as the mean centroid of all labeled
    building polygons in this tile's label file, or None if there are none.
    """
    data = json.loads(label_path.read_text())
    building_features = data.get("features", {}).get("lng_lat", [])
    if not building_features:
        return None

    lats: list[float] = []
    lngs: list[float] = []
    for feature in building_features:
        wkt = feature.get("wkt")
        if not wkt:
            continue
        centroid = shapely.wkt.loads(wkt).centroid
        lngs.append(centroid.x)  # WKT convention: x = longitude
        lats.append(centroid.y)  # y = latitude

    if not lats:
        return None
    return sum(lats) / len(lats), sum(lngs) / len(lngs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--tiles-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tile_ids = sorted(
        p.stem for p in args.tiles_dir.iterdir() if p.suffix.lower() in TILE_EXTENSIONS
    )

    geo: dict[str, dict[str, float]] = {}
    skipped: list[str] = []
    for tile_id in tile_ids:
        label_path = args.labels_dir / f"{tile_id}.json"
        if not label_path.exists():
            skipped.append(tile_id)
            continue
        centroid = extract_centroid(label_path)
        if centroid is None:
            skipped.append(tile_id)
            continue
        lat, lng = centroid
        geo[tile_id] = {"latitude": lat, "longitude": lng}

    args.out.write_text(json.dumps(geo, indent=2))
    print(f"Wrote {len(geo)} geolocations to {args.out}")
    if skipped:
        preview = skipped[:10]
        more = "..." if len(skipped) > 10 else ""
        print(
            f"Skipped {len(skipped)} tiles with no label or no buildings: {preview}{more}")


if __name__ == "__main__":
    main()
