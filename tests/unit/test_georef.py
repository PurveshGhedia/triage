"""Tile footprints from xBD labels: pixel <-> lon/lat affine fit."""
from __future__ import annotations

import math

import pytest

from shared.georef import fit_affine, tile_footprint, wkt_points

M_PER_DEG = 111_320.0


def make_label(a, b, c, d, e, f, n=8, width=1024, height=1024, noise=0.0, uids=True):
    """A synthetic xBD label whose buildings follow a known pixel->lon/lat transform."""
    xy, ll = [], []
    for i in range(n):
        px = [(80 + i * 110, 90 + (i % 3) * 250), (110 + i * 110, 90 + (i % 3) * 250),
              (110 + i * 110, 130 + (i % 3) * 250), (80 + i * 110, 130 + (i % 3) * 250)]
        lonlat = [(a * x + b * y + c + noise * math.sin(x * 12.9898 + y * 78.233), d * x + e * y + f) for x, y in px]
        props = {"feature_type": "building", "subtype": "no-damage"}
        if uids:
            props["uid"] = f"u{i}"
        fmt = lambda pts: "POLYGON ((" + ", ".join(f"{p[0]!r} {p[1]!r}" for p in pts) + "))"
        xy.append({"properties": dict(props), "wkt": fmt(px)})
        ll.append({"properties": dict(props), "wkt": fmt(lonlat)})
    return {"features": {"xy": xy, "lng_lat": ll}, "metadata": {"width": width, "height": height}}


# a tile ~400 m across, north-up, near Houston
LAT0, LON0 = 29.80, -95.60
PX_DEG_LAT = -400 / 1024 / M_PER_DEG                      # y grows downward -> latitude falls
PX_DEG_LON = 400 / 1024 / (M_PER_DEG * math.cos(math.radians(LAT0)))


def test_wkt_points_parses_polygons_and_scientific_notation():
    assert wkt_points("POLYGON ((1.5 2.5, -3 4, 5e-1 6))") == [(1.5, 2.5), (-3.0, 4.0), (0.5, 6.0)]
    assert wkt_points("") == [] and wkt_points(None) == []


def test_fit_affine_recovers_a_known_transform_exactly():
    pix = [(0, 0), (10, 0), (0, 10), (10, 10), (5, 3)]
    ll = [(2 * x + 0.5 * y + 100, -3 * x + y - 7) for x, y in pix]
    (a, b, c), (d, e, f) = fit_affine(pix, ll)
    assert (round(a, 6), round(b, 6), round(c, 6)) == (2.0, 0.5, 100.0)
    assert (round(d, 6), round(e, 6), round(f, 6)) == (-3.0, 1.0, -7.0)


def test_fit_affine_rejects_degenerate_input():
    assert fit_affine([(1, 1), (2, 2), (3, 3)], [(1, 1), (2, 2), (3, 3)]) is None   # collinear
    assert fit_affine([(0, 0), (1, 1)], [(0, 0), (1, 1)]) is None                    # too few
    assert fit_affine([(0, 0), (1, 1), (2, 5)], [(0, 0)]) is None                    # length mismatch


def test_footprint_recovers_tile_corners_size_and_centre():
    label = make_label(PX_DEG_LON, 0, LON0, 0, PX_DEG_LAT, LAT0)
    fp = tile_footprint(label)
    assert fp["rms_m"] < 0.01
    assert fp["width_m"] == pytest.approx(400, abs=1) and fp["height_m"] == pytest.approx(400, abs=1)
    assert fp["bounds"][1][0] == pytest.approx(LAT0, abs=1e-6)                       # north edge = image top
    assert fp["center"][0] == pytest.approx(LAT0 - 200 / M_PER_DEG, abs=1e-5)
    assert fp["corners"][0][1] < fp["corners"][1][1]                                 # NW is west of NE
    assert fp["corners"][0][0] > fp["corners"][3][0]                                 # NW is north of SW


def test_footprint_reports_its_own_fit_error_when_data_is_noisy():
    clean = tile_footprint(make_label(PX_DEG_LON, 0, LON0, 0, PX_DEG_LAT, LAT0))
    noisy = tile_footprint(make_label(PX_DEG_LON, 0, LON0, 0, PX_DEG_LAT, LAT0, noise=2e-5))
    assert noisy["rms_m"] > clean["rms_m"] + 0.2
    assert noisy["width_m"] == pytest.approx(400, abs=25)                           # still close


def test_slightly_rotated_imagery_is_handled():
    theta = math.radians(4)
    a, b = PX_DEG_LON * math.cos(theta), PX_DEG_LON * math.sin(theta)
    d, e = -PX_DEG_LAT * math.sin(theta) * 1.0, PX_DEG_LAT * math.cos(theta)
    fp = tile_footprint(make_label(a, b, LON0, d, e, LAT0))
    assert fp["rms_m"] < 0.01 and fp["width_m"] == pytest.approx(400, abs=2)


def test_buildings_matched_by_uid_even_when_lists_are_in_different_order():
    label = make_label(PX_DEG_LON, 0, LON0, 0, PX_DEG_LAT, LAT0)
    label["features"]["lng_lat"].reverse()
    fp = tile_footprint(label)
    assert fp is not None and fp["rms_m"] < 0.01


def test_falls_back_to_list_order_when_there_are_no_uids():
    fp = tile_footprint(make_label(PX_DEG_LON, 0, LON0, 0, PX_DEG_LAT, LAT0, uids=False))
    assert fp is not None and fp["rms_m"] < 0.01


def test_too_few_buildings_or_no_buildings_gives_none():
    assert tile_footprint(make_label(PX_DEG_LON, 0, LON0, 0, PX_DEG_LAT, LAT0, n=1)) is None   # 4 points < 6
    assert tile_footprint({"features": {"xy": [], "lng_lat": []}, "metadata": {}}) is None
    assert tile_footprint({}) is None


def test_uses_metadata_size_for_non_1024_tiles():
    fp = tile_footprint(make_label(PX_DEG_LON, 0, LON0, 0, PX_DEG_LAT, LAT0, width=512, height=512))
    assert fp["width_m"] == pytest.approx(200, abs=1)


def test_fit_uses_every_building_not_just_the_first_few():
    """Noise of ~2e-5 deg (~1.9 m peak) per vertex: a fit through all buildings averages it
    out (<1.5 m rms); a fit through just three points would extrapolate it (~8 m)."""
    fp = tile_footprint(make_label(PX_DEG_LON, 0, LON0, 0, PX_DEG_LAT, LAT0, noise=2e-5))
    assert 0.3 < fp["rms_m"] < 1.5
    assert fp["points"] == 32  # all 8 buildings x 4 vertices contributed
