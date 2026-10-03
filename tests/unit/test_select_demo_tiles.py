"""scripts.select_demo_tiles: clustered tile selection for the demo."""
from __future__ import annotations

import pytest

from scripts.select_demo_tiles import pick_seed, post_tiles_for, select_clustered, zones_at_radii

# ~111 m per 0.001 deg of latitude
def geo_grid():
    geo = {}
    # a tight cluster of 6 tiles for "storm", ~100 m apart
    for i in range(6):
        geo[f"storm_{i:08d}_post_disaster"] = {"latitude": 10.0 + i * 0.001, "longitude": 20.0}
    # far-away outliers for the same disaster
    geo["storm_00000100_post_disaster"] = {"latitude": 30.0, "longitude": 40.0}
    geo["storm_00000101_post_disaster"] = {"latitude": -30.0, "longitude": -40.0}
    # pre-disaster tiles and another disaster must be ignored
    geo["storm_00000000_pre_disaster"] = {"latitude": 10.0, "longitude": 20.0}
    geo["quake_00000000_post_disaster"] = {"latitude": 10.0, "longitude": 20.0}
    return geo


def test_only_post_tiles_of_the_requested_disaster_are_candidates():
    tiles = post_tiles_for(geo_grid(), "storm")
    assert len(tiles) == 8
    assert all(t.startswith("storm_") and t.endswith("_post_disaster") for t in tiles)


def test_prefix_does_not_match_a_longer_disaster_name():
    geo = {"storm_1_post_disaster": {"latitude": 0, "longitude": 0},
           "storm-two_1_post_disaster": {"latitude": 0, "longitude": 0}}
    assert post_tiles_for(geo, "storm") == ["storm_1_post_disaster"]


def test_selection_is_the_dense_cluster_not_the_outliers():
    picked = select_clustered(geo_grid(), "storm", count=5)
    assert len(picked) == 5
    assert not {"storm_00000100_post_disaster", "storm_00000101_post_disaster"} & set(picked)


def test_explicit_seed_is_honoured_and_ordered_nearest_first():
    seed = "storm_00000100_post_disaster"
    picked = select_clustered(geo_grid(), "storm", count=3, seed_tile=seed)
    assert picked[0] == seed


def test_count_larger_than_available_returns_everything():
    assert len(select_clustered(geo_grid(), "storm", count=999)) == 8


def test_unknown_disaster_or_bad_seed_raises():
    with pytest.raises(ValueError):
        select_clustered(geo_grid(), "volcano", count=3)
    with pytest.raises(ValueError):
        select_clustered(geo_grid(), "storm", count=3, seed_tile="nope")


def test_selection_is_deterministic():
    assert select_clustered(geo_grid(), "storm", 5) == select_clustered(geo_grid(), "storm", 5)


def test_pick_seed_is_inside_the_cluster():
    geo = geo_grid()
    seed = pick_seed(geo, post_tiles_for(geo, "storm"), 1500.0)
    assert seed in {f"storm_{i:08d}_post_disaster" for i in range(6)}


def test_zone_counts_shrink_as_radius_grows():
    geo = geo_grid()
    tiles = select_clustered(geo, "storm", count=6)
    zones = zones_at_radii(geo, tiles, radii=(50.0, 150.0, 1000.0))
    assert zones[50.0] == 6      # nothing linked at 50 m
    assert zones[150.0] == 1     # chain linked at ~111 m spacing
    assert zones[1000.0] == 1


# --- radius recommendation ------------------------------------------------------

from scripts.select_demo_tiles import recommend_radius


def test_recommend_radius_returns_smallest_radius_meeting_the_target():
    geo = geo_grid()
    tiles = select_clustered(geo, "storm", count=6)  # chain ~111 m apart
    radius, zones, cap_hit = recommend_radius(geo, tiles, target_zones=1)
    assert (radius, zones, cap_hit) == (300.0, 1, False)  # 300 m is the smallest radius searched


def test_recommend_radius_flags_when_the_target_cannot_be_met():
    geo = {
        "far_1_post_disaster": {"latitude": 0.0, "longitude": 0.0},
        "far_2_post_disaster": {"latitude": 40.0, "longitude": 40.0},
    }
    radius, zones, cap_hit = recommend_radius(
        geo, ["far_1_post_disaster", "far_2_post_disaster"], target_zones=1, radii=(300.0, 600.0)
    )
    assert cap_hit is True and zones == 2 and radius == 600.0


# --- clearing stale output ------------------------------------------------------------------------

from scripts.select_demo_tiles import reset_output_dir


def test_reset_output_dir_clears_only_what_the_sampler_generates(tmp_path):
    for sub in ("images", "pre", "labels", ".thumbs"):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "stale.png").write_bytes(b"x")
    for name in ("geo.json", "demo.json", "footprints.json"):
        (tmp_path / name).write_text("{}")
    (tmp_path / "my_notes.txt").write_text("keep me")
    (tmp_path / "other_dir").mkdir()

    removed = reset_output_dir(tmp_path)

    assert set(removed) == {"images", "pre", "labels", ".thumbs", "geo.json", "demo.json", "footprints.json"}
    assert (tmp_path / "my_notes.txt").read_text() == "keep me" and (tmp_path / "other_dir").is_dir()
    assert not (tmp_path / "images").exists()


def test_reset_output_dir_on_an_empty_or_missing_folder_is_a_no_op(tmp_path):
    assert reset_output_dir(tmp_path) == []
    assert reset_output_dir(tmp_path / "does-not-exist") == []
