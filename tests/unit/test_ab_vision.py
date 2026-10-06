"""scripts.ab_vision: variants compared on identical tiles, with live vs cached answers made explicit."""
from __future__ import annotations

import json
import struct
import threading
import zlib
from pathlib import Path

import pytest

import tools.vision.client as vc
from scripts.ab_vision import pool, run_one, table


def make_png(shade: int = 128) -> bytes:
    """A 1x1 PNG whose PIXEL is `shade`. Distinct pixels matter: the vision cache is keyed on the decoded
    image, so identical-looking tiles would share cache entries and make call counts depend on thread timing."""
    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes([0, shade, shade, shade]))) + chunk(b"IEND", b""))


def build_area(root: Path, name: str, damaged: int, undamaged: int, base_shade: int = 10) -> None:
    d = root / name
    for sub in ("images", "pre", "labels"):
        (d / sub).mkdir(parents=True)
    geo = {}
    n = damaged + undamaged
    for i in range(n):
        tile = f"{name}_{i:08d}_post_disaster"
        (d / "images" / f"{tile}.png").write_bytes(make_png(base_shade + 9 * i))       # a different pixel per tile
        (d / "pre" / f"{name}_{i:08d}_pre_disaster.png").write_bytes(make_png(200))
        sub = "destroyed" if i < damaged else "no-damage"
        (d / "labels" / f"{tile}.json").write_text(json.dumps({"features": {"lng_lat": [{"properties": {"subtype": sub}}] * 12}}))
        geo[tile] = {"latitude": 10 + (i // 4) * 0.004, "longitude": 20 + (i % 4) * 0.004}
    (d / "geo.json").write_text(json.dumps(geo))
    (d / "demo.json").write_text(json.dumps({"adjacency_radius_m": 900, "tile_count": n}))


@pytest.fixture
def world(tmp_path, monkeypatch):
    root = tmp_path / "demo"
    build_area(root, "mixed", damaged=4, undamaged=4, base_shade=10)
    build_area(root, "quiet", damaged=0, undamaged=6, base_shade=120)
    calls = {"n": 0}
    lock = threading.Lock()          # Agent 1 calls the model from several worker threads at once

    def make(project=None, location=None, model=None):
        def generate(images, prompt, schema):
            with lock:               # `calls["n"] += 1` alone can lose updates under contention
                calls["n"] += 1
            if "ROAD ACCESS" in prompt:
                return json.dumps({"label": "access-clear", "confidence": 0.8, "evidence": "x"})
            # a 'smart' model: tiles with brighter pixels (the fake can't see them) -> use call parity trick instead
            fractions = "SHARE" in prompt.upper()
            return (json.dumps({"no_damage": 0.0, "minor_damage": 0.0, "major_damage": 0.0, "destroyed": 1.0,
                                "confidence": 0.9, "evidence": "all destroyed"}) if fractions else
                    json.dumps({"label": "no-damage", "confidence": 0.9, "evidence": "fine"}))
        return generate

    monkeypatch.setattr(vc, "make_vertex_generate_fn", make)
    monkeypatch.setenv("TRIAGE_VISION_CACHE", str(tmp_path / "shared-cache"))
    return root, calls, tmp_path


def test_variants_run_on_identical_tiles_and_differ_only_in_the_prompt(world):
    root, calls, tmp = world
    work = tmp / "work"; work.mkdir()
    classes = run_one("mixed", "classes", root, live=False, work=work)
    fractions = run_one("mixed", "fractions", root, live=False, work=work)

    assert set(classes["predictions"]["scores"]) == set(fractions["predictions"]["scores"])   # same 8 tiles
    assert classes["metrics"]["tiles"]["tiles_compared"] == fractions["metrics"]["tiles"]["tiles_compared"] == 8
    assert set(classes["predictions"]["scores"].values()) == {0.0}                            # 'classes' fake says fine everywhere
    assert set(fractions["predictions"]["scores"].values()) == {3.0}                          # 'fractions' fake says destroyed everywhere
    assert classes["metrics"]["tiles"]["mean_abs_error"] == 1.5 and fractions["metrics"]["tiles"]["mean_abs_error"] == 1.5


def test_live_mode_uses_an_empty_cache_so_every_call_is_real(world):
    root, calls, tmp = world
    work = tmp / "work"; work.mkdir()
    first = run_one("mixed", "classes", root, live=True, work=work)
    # 8 tiles x (one damage reading + one road-access check) = 16 real calls, every time
    assert calls["n"] == 16 and first["new_answers"] == 16
    again = run_one("mixed", "classes", root, live=True, work=work)
    assert calls["n"] == 32                                           # a fresh empty cache: nothing replayed
    assert again["new_answers"] == 16


def test_cached_mode_replays_earlier_answers_without_calling_the_model(world):
    root, calls, tmp = world
    work = tmp / "work"; work.mkdir()
    first = run_one("mixed", "classes", root, live=False, work=work)
    assert calls["n"] == 16 and first["new_answers"] == 16
    second = run_one("mixed", "classes", root, live=False, work=work)
    assert calls["n"] == 16 and second["new_answers"] == 0            # served entirely from cache


def test_the_environment_variable_is_restored_after_a_run(world):
    root, _, tmp = world
    work = tmp / "work"; work.mkdir()
    import os
    before = os.environ["TRIAGE_VISION_CACHE"]
    run_one("mixed", "classes", root, live=True, work=work)
    assert os.environ["TRIAGE_VISION_CACHE"] == before


def test_pooling_ignores_uninformative_areas(world):
    root, _, tmp = world
    work = tmp / "work"; work.mkdir()
    mixed = run_one("mixed", "classes", root, live=False, work=work)
    quiet = run_one("quiet", "classes", root, live=False, work=work)
    assert quiet["metrics"]["tiles"]["informative"] is False
    pooled = pool([mixed, quiet])
    assert pooled["tiles_compared"] == 8                              # only the mixed area contributes
    assert pool([quiet]) == {}                                        # nothing informative -> nothing to pool


def test_table_labels_controls_and_shows_pooled_numbers(world):
    root, _, tmp = world
    work = tmp / "work"; work.mkdir()
    results = [run_one("mixed", "classes", root, False, work), run_one("quiet", "classes", root, False, work)]
    out = table(results, ["classes"])
    assert "mixed" in out and "quiet" in out and "NOT INFORMATIVE (control)" in out
    assert "Pooled over the informative areas" in out and "classes" in out.split("Pooled")[1]
