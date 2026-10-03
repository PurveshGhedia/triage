"""Work out where an xBD tile sits on the ground.

Every xBD label file lists each building twice: once in image pixels (`xy`) and once
in longitude/latitude (`lng_lat`), with matching uids. Fitting an affine transform
(pixel -> lon/lat) through those matched points gives the tile's true corners, so its
image can be drawn on a map exactly where it was taken — instead of guessing a square
around the average building position.

The fit is least-squares and reports its own error (`rms_m`, metres). Imagery taken far
off-nadir is slanted, so the error is worth showing rather than assuming it is zero.
Pure Python (no numpy/shapely) so it is trivially testable.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

_NUMBER_PAIR = re.compile(r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
METRES_PER_DEG_LAT = 111_320.0
MIN_POINTS = 6


def wkt_points(wkt: str) -> list[tuple[float, float]]:
    """All (x, y) vertices in a WKT string such as 'POLYGON ((x y, x y, ...))'."""
    return [(float(a), float(b)) for a, b in _NUMBER_PAIR.findall(wkt or "")]


def _solve3(m: list[list[float]], v: list[float]) -> list[float] | None:
    """Solve a 3x3 linear system by Gaussian elimination; None if (near) singular."""
    a = [row[:] + [v[i]] for i, row in enumerate(m)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-9:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(3):
            if r != col:
                f = a[r][col] / a[col][col]
                a[r] = [a[r][k] - f * a[col][k] for k in range(4)]
    return [a[i][3] / a[i][i] for i in range(3)]


def fit_affine(
    pixels: list[tuple[float, float]], lonlats: list[tuple[float, float]]
) -> tuple[list[float], list[float]] | None:
    """Least-squares  lon = a*x + b*y + c  and  lat = d*x + e*y + f. None if degenerate."""
    if len(pixels) != len(lonlats) or len(pixels) < 3:
        return None
    sxx = sxy = sx = syy = sy = 0.0
    n = float(len(pixels))
    for x, y in pixels:
        sxx += x * x; sxy += x * y; sx += x; syy += y * y; sy += y
    normal = [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, n]]
    sol = []
    for axis in (0, 1):
        rhs = [0.0, 0.0, 0.0]
        for (x, y), ll in zip(pixels, lonlats):
            t = ll[axis]
            rhs[0] += x * t; rhs[1] += y * t; rhs[2] += t
        coeffs = _solve3(normal, rhs)
        if coeffs is None:
            return None
        sol.append(coeffs)
    return sol[0], sol[1]


def _apply(coeffs: tuple[list[float], list[float]], x: float, y: float) -> tuple[float, float]:
    (a, b, c), (d, e, f) = coeffs
    return a * x + b * y + c, d * x + e * y + f  # (lon, lat)


def _metres(lat: float, dlon: float, dlat: float) -> float:
    return math.hypot(dlon * METRES_PER_DEG_LAT * math.cos(math.radians(lat)), dlat * METRES_PER_DEG_LAT)


def tile_footprint(label: dict) -> dict | None:
    """Footprint of one tile from its parsed xBD label JSON, or None if it cannot be
    determined (too few buildings, or buildings all on one line).

    Returns {"bounds": [[south, west], [north, east]], "corners": [[lat, lon] x4 in the
    order NW, NE, SE, SW of the image], "center": [lat, lon], "width_m", "height_m",
    "rms_m", "points"}.
    """
    features = label.get("features", {})
    xy_feats, ll_feats = features.get("xy", []), features.get("lng_lat", [])
    by_uid = {f.get("properties", {}).get("uid"): f for f in ll_feats if f.get("properties", {}).get("uid")}

    pixels: list[tuple[float, float]] = []
    lonlats: list[tuple[float, float]] = []
    for i, xf in enumerate(xy_feats):
        uid = xf.get("properties", {}).get("uid")
        lf = by_uid.get(uid) or (ll_feats[i] if i < len(ll_feats) and not uid else None)
        if lf is None:
            continue
        px, ll = wkt_points(xf.get("wkt")), wkt_points(lf.get("wkt"))
        if len(px) == len(ll):
            pixels.extend(px)
            lonlats.extend(ll)
    if len(pixels) < MIN_POINTS:
        return None

    coeffs = fit_affine(pixels, lonlats)
    if coeffs is None:
        return None

    meta = label.get("metadata", {})
    width, height = float(meta.get("width") or 1024), float(meta.get("height") or 1024)
    corners_px = [(0, 0), (width, 0), (width, height), (0, height)]  # NW, NE, SE, SW
    corners = []
    for x, y in corners_px:
        lon, lat = _apply(coeffs, x, y)
        corners.append([lat, lon])

    lats = [c[0] for c in corners]
    lons = [c[1] for c in corners]
    centre_lat = sum(lats) / 4
    errors = []
    for (x, y), (lon, lat) in zip(pixels, lonlats):
        plon, plat = _apply(coeffs, x, y)
        errors.append(_metres(lat, plon - lon, plat - lat) ** 2)
    return {
        "bounds": [[min(lats), min(lons)], [max(lats), max(lons)]],
        "corners": corners,
        "center": [centre_lat, sum(lons) / 4],
        "width_m": round(_metres(centre_lat, corners[1][1] - corners[0][1], corners[1][0] - corners[0][0]), 1),
        "height_m": round(_metres(centre_lat, corners[3][1] - corners[0][1], corners[3][0] - corners[0][0]), 1),
        "rms_m": round(math.sqrt(sum(errors) / len(errors)), 2),
        "points": len(pixels),
    }


def footprint_from_file(path: str | Path) -> dict | None:
    return tile_footprint(json.loads(Path(path).read_text()))
