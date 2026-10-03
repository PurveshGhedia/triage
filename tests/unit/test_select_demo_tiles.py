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


# --- informative selection (pick an area with a mix of damage) -----------------------------------------

from scripts.select_demo_tiles import damage_mix, select_informative


def mixed_world():
    """Cluster A: 8 tiles, all undamaged, very dense. Cluster B: 8 tiles, half damaged, half not."""
    geo, truth = {}, {}
    for i in range(8):
        t = f"storm_{i:08d}_post_disaster"
        geo[t] = {"latitude": 10.0 + i * 0.0005, "longitude": 20.0}                 # ~55 m apart
        truth[t] = {"mean": 0.0, "buildings": 40, "worst": 0}
    for i in range(8):
        t = f"storm_{100 + i:08d}_post_disaster"
        geo[t] = {"latitude": 11.0 + i * 0.003, "longitude": 21.0}                  # ~330 m apart
        truth[t] = {"mean": 0.1 if i % 2 else 2.5, "buildings": 40, "worst": 3}
    return geo, truth


def test_damage_mix_counts_damaged_undamaged_and_too_small():
    truth = {"a": {"mean": 2.0, "buildings": 30}, "b": {"mean": 0.1, "buildings": 30},
             "c": {"mean": 3.0, "buildings": 2}, "d": {"mean": None, "buildings": 0}}
    assert damage_mix(["a", "b", "c", "d", "ghost"], truth) == {"damaged": 1, "undamaged": 1, "too_small": 3}


def test_informative_selection_prefers_the_mixed_area_over_the_denser_quiet_one():
    geo, truth = mixed_world()
    plain = select_clustered(geo, "storm", 8)
    picked = select_informative(geo, "storm", 8, truth)
    assert all(int(t.split("_")[1]) < 100 for t in plain)                           # plain picks the dense quiet cluster
    assert all(int(t.split("_")[1]) >= 100 for t in picked)                         # informative picks the mixed one
    mix = damage_mix(picked, truth)
    assert mix["damaged"] == 4 and mix["undamaged"] == 4


def test_informative_selection_is_deterministic_and_respects_the_count():
    geo, truth = mixed_world()
    assert select_informative(geo, "storm", 8, truth) == select_informative(geo, "storm", 8, truth)
    assert len(select_informative(geo, "storm", 6, truth)) == 6


def test_informative_selection_falls_back_to_the_densest_cluster_when_nothing_is_mixed():
    geo, _ = mixed_world()
    quiet = {t: {"mean": 0.0, "buildings": 40, "worst": 0} for t in geo}
    assert select_informative(geo, "storm", 8, quiet) == select_clustered(geo, "storm", 8)


def test_tiny_tiles_do_not_make_an_area_look_informative():
    geo, truth = mixed_world()
    for t in truth:
        if int(t.split("_")[1]) >= 100:
            truth[t]["buildings"] = 3                                              # every damaged tile has 3 buildings
    picked = select_informative(geo, "storm", 8, truth)
    assert picked == select_clustered(geo, "storm", 8)                              # nothing credible -> fallback


def test_informative_selection_unknown_disaster_raises():
    geo, truth = mixed_world()
    with pytest.raises(ValueError):
        select_informative(geo, "volcano", 8, truth)
