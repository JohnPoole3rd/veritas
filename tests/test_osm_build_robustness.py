"""Offline robustness tests for the OSM-Sentinel 1M builder (no STAC, no network).

They pin down: multi-granule mosaic alignment and source tracking, chip -> source granule mapping,
scene ranking, and main()'s resume / per-region cap / retry-queue / per-region RNG behaviour, so the
multi-process refactor of main() and the per-chip optimisations can be checked against them.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

from experiments.osm_sentinel import build_dataset as bd

UTM = "EPSG:32617"
X0, Y0 = 300000.0, 4500000.0  # grid origin shared by every synthetic granule (10 m)


# ----------------------------------------------------------------------------------------------
# synthetic granules
# ----------------------------------------------------------------------------------------------
def _encode(gcol, grow):
    """Pixel value encodes its global grid position, so any off-by-one shows up as a value mismatch."""
    return np.stack([gcol % 250 + 1, grow % 250 + 1, np.full_like(gcol, 7)]).astype(np.uint8)


class Granules:
    """In-memory GeoTIFFs; keeps the MemoryFiles alive for the duration of a test."""

    def __init__(self):
        self.mems = []

    def make(self, item_id, col0, row0, width, height, *, valid_cols=None, crs=UTM, dtype="uint8",
             shift_m=0.0, value=None, mgrs=None):
        transform = from_origin(X0 + col0 * 10 + shift_m, Y0 - row0 * 10, 10, 10)
        gcol, grow = np.meshgrid(np.arange(col0, col0 + width), np.arange(row0, row0 + height))
        data = _encode(gcol, grow) if value is None else np.full((3, height, width), value)
        data = data.astype(dtype)
        if valid_cols is not None:  # zero (nodata) outside [lo, hi) local columns
            lo, hi = valid_cols
            data[:, :, :lo] = 0
            data[:, :, hi:] = 0
        mem = MemoryFile()
        with mem.open(driver="GTiff", width=width, height=height, count=3, dtype=dtype, crs=crs,
                      transform=transform, nodata=0) as dst:
            dst.write(data)
        self.mems.append(mem)
        props = {"eo:cloud_cover": 1.0}
        if mgrs:
            props.update({"mgrs:utm_zone": mgrs[:2], "mgrs:latitude_band": mgrs[2], "mgrs:grid_square": mgrs[3:]})
        return SimpleNamespace(id=item_id, properties=props, datetime=datetime(2024, 6, 1),
                               assets={"visual": SimpleNamespace(href=mem.name)}, geometry=None)

    def close(self):
        for mem in self.mems:
            mem.close()


@pytest.fixture
def granules():
    g = Granules()
    yield g
    g.close()


def _scene_bbox(col_lo, col_hi, row_lo, row_hi):
    """lon/lat bbox of a UTM pixel box (the builder reprojects it back, densified)."""
    return transform_bounds(UTM, "EPSG:4326", X0 + col_lo * 10, Y0 - row_hi * 10, X0 + col_hi * 10,
                            Y0 - row_lo * 10)


def _assert_aligned(rgb, transform, source_index):
    rows, cols = np.nonzero(source_index)
    gcol = np.round((transform.c - X0) / 10).astype(int) + cols
    grow = np.round((Y0 - transform.f) / 10).astype(int) + rows
    expected = _encode(gcol, grow).T  # (n, 3)
    np.testing.assert_array_equal(rgb[rows, cols], expected)
    assert transform.a == 10 and transform.e == -10
    assert (transform.c - X0) % 10 == 0 and (Y0 - transform.f) % 10 == 0  # on the 10 m grid


def test_mosaic_fills_nodata_from_second_granule_pixel_exact(granules):
    # A: global cols 0..199, data only in cols 0..119. B: global cols 80..279, all valid.
    a = granules.make("S2A_17TAA_20240601_0_L2A", 0, 0, 200, 200, valid_cols=(0, 120), mgrs="17TAA")
    b = granules.make("S2B_17TBB_20240601_0_L2A", 80, 0, 200, 200, mgrs="17TBB")
    bbox = _scene_bbox(30, 250, 30, 170)
    out = bd.read_region_mosaic([(a, 0.6), (b, 0.5)], bbox, log=lambda m: None)
    assert out is not None
    rgb, transform, crs, src_idx, used = out
    assert [it.id for it in used] == [a.id, b.id]
    assert str(crs) == UTM and src_idx.dtype == np.uint8
    _assert_aligned(rgb, transform, src_idx)
    gcol = np.round((transform.c - X0) / 10).astype(int) + np.arange(src_idx.shape[1])
    inside_rows = slice(5, src_idx.shape[0] - 5)  # away from the reprojected-bbox fringe
    # A keeps its own pixels (first granule wins), B fills only A's nodata, beyond both -> 0
    assert (src_idx[inside_rows][:, (gcol >= 30) & (gcol < 120)] == 1).all()
    assert (src_idx[inside_rows][:, (gcol >= 120) & (gcol < 250)] == 2).all()
    assert (src_idx[:, gcol >= 280] == 0).all()
    assert (rgb[src_idx == 0] == 0).all()


def test_mosaic_scales_non_uint8_fill_granule(granules):
    a = granules.make("S2A_17TAA_20240601_0_L2A", 0, 0, 200, 200, valid_cols=(0, 100), mgrs="17TAA")
    b = granules.make("S2B_17TBB_20240601_0_L2A", 0, 0, 200, 200, dtype="uint16", value=2000, mgrs="17TBB")
    rgb, _t, _c, src_idx, used = bd.read_region_mosaic([(a, 1), (b, 1)], _scene_bbox(20, 180, 20, 180),
                                                       log=lambda m: None)
    assert len(used) == 2 and rgb.dtype == np.uint8
    filled = src_idx == 2
    assert filled.any()
    assert (rgb[filled] == int(2000 / 4000 * 255)).all()


@pytest.mark.parametrize("kind", ["half_pixel_shift", "other_crs", "same_mgrs_tile"])
def test_mosaic_rejects_unalignable_or_duplicate_granules(granules, kind):
    a = granules.make("S2A_17TAA_20240601_0_L2A", 0, 0, 200, 200, valid_cols=(0, 100), mgrs="17TAA")
    if kind == "half_pixel_shift":
        b = granules.make("S2B_17TBB_20240601_0_L2A", 0, 0, 200, 200, shift_m=5.0, mgrs="17TBB")
    elif kind == "other_crs":
        b = granules.make("S2B_18TBB_20240601_0_L2A", 0, 0, 200, 200, crs="EPSG:32618", mgrs="18TBB")
    else:
        b = granules.make("S2B_17TAA_20240611_0_L2A", 0, 0, 200, 200, mgrs="17TAA")
    rgb, transform, _c, src_idx, used = bd.read_region_mosaic([(a, 1), (b, 1)], _scene_bbox(20, 180, 20, 180),
                                                              log=lambda m: None)
    assert [it.id for it in used] == [a.id]
    assert src_idx.max() == 1
    _assert_aligned(rgb, transform, src_idx)


def test_mosaic_skips_sliver_and_unreadable_granules(granules):
    # sliver covers 2 columns of a ~160-column window (<2 %): rejected as the grid-defining base
    sliver = granules.make("S2A_17TSL_20240601_0_L2A", 0, 0, 200, 200, valid_cols=(0, 2), mgrs="17TSL")
    broken = SimpleNamespace(id="S2A_17TXX_20240601_0_L2A", properties={}, datetime=datetime(2024, 6, 1),
                             assets={"visual": SimpleNamespace(href="/vsimem/does-not-exist.tif")})
    good = granules.make("S2B_17TGG_20240601_0_L2A", 0, 0, 200, 200, mgrs="17TGG")
    out = bd.read_region_mosaic([(sliver, 1), (broken, 1), (good, 1)], _scene_bbox(20, 180, 20, 180),
                                log=lambda m: None)
    assert [it.id for it in out[4]] == [good.id]


def test_mosaic_returns_none_when_nothing_usable(granules):
    empty = granules.make("S2A_17TAA_20240601_0_L2A", 0, 0, 50, 50, value=0, mgrs="17TAA")
    assert bd.read_region_mosaic([(empty, 1)], _scene_bbox(10, 40, 10, 40), log=lambda m: None) is None
    assert bd.read_region_mosaic([], _scene_bbox(10, 40, 10, 40), log=lambda m: None) is None


def test_mosaic_network_error_raises_network_down_after_retries(monkeypatch, granules):
    item = granules.make("S2A_17TAA_20240601_0_L2A", 0, 0, 50, 50, mgrs="17TAA")
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise RuntimeError("CURL error: Could not resolve host")

    monkeypatch.setattr(bd, "_mosaic_step", boom)
    monkeypatch.setattr(bd.time, "sleep", lambda s: None)
    with pytest.raises(bd.NetworkDown):
        bd.read_region_mosaic([(item, 1)], _scene_bbox(10, 40, 10, 40), log=lambda m: None)
    assert len(calls) == 3


def test_is_network_error():
    assert bd._is_network_error(RuntimeError("CURL error: Failed to connect to host"))
    assert not bd._is_network_error(RuntimeError("not a valid TIFF"))


# ----------------------------------------------------------------------------------------------
# chip_source
# ----------------------------------------------------------------------------------------------
def _latlon(x, y, crs=UTM):
    from rasterio.warp import transform as warp

    lon, lat = warp(crs, "EPSG:4326", [x], [y])
    return lat[0], lon[0]


def test_chip_source_row_col_order_and_crs_transform():
    idx = np.zeros((20, 30), dtype=np.uint8)
    idx[3, 25] = 2  # row 3, col 25: asymmetric so a row/col swap is caught
    transform = from_origin(X0, Y0, 10, 10)
    lat, lon = _latlon(X0 + 25 * 10 + 5, Y0 - 3 * 10 - 5)
    assert bd.chip_source(idx, transform, UTM, lat, lon) == 2
    lat, lon = _latlon(X0 + 3 * 10 + 5, Y0 - 25 * 10 - 5)
    assert bd.chip_source(idx, transform, UTM, lat, lon) == 0  # swapped -> out of range


def test_chip_source_edges():
    idx = np.ones((20, 30), dtype=np.uint8)
    transform = from_origin(X0, Y0, 10, 10)
    assert bd.chip_source(idx, transform, UTM, *_latlon(X0 + 1, Y0 - 1)) == 1  # first pixel
    assert bd.chip_source(idx, transform, UTM, *_latlon(X0 + 299, Y0 - 199)) == 1  # last pixel
    assert bd.chip_source(idx, transform, UTM, *_latlon(X0 + 301, Y0 - 100)) == 0  # right of raster
    assert bd.chip_source(idx, transform, UTM, *_latlon(X0 + 100, Y0 - 201)) == 0  # below raster
    # within one pixel above / left of the raster: int() truncation used to map these to pixel 0
    assert bd.chip_source(idx, transform, UTM, *_latlon(X0 - 4, Y0 - 100)) == 0
    assert bd.chip_source(idx, transform, UTM, *_latlon(X0 + 100, Y0 + 4)) == 0


# ----------------------------------------------------------------------------------------------
# find_scenes
# ----------------------------------------------------------------------------------------------
def _stac_item(item_id, bbox, month, cloud, href="https://example/x.tif", dt=True):
    w, s, e, n = bbox
    geom = {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}
    when = datetime(2024, month, 15)
    props = {"eo:cloud_cover": cloud, "datetime": when.isoformat() + "Z"}
    return SimpleNamespace(id=item_id, properties=props, datetime=when if dt else None, geometry=geom,
                           assets={"visual": SimpleNamespace(href=href)})


def _fake_stac(monkeypatch, items):
    seen = {}

    class Search:
        def items(self):
            return iter(items)

    class Client:
        def search(self, **kw):
            seen.update(kw)
            return Search()

    monkeypatch.setattr(bd, "_STAC_CLIENT", Client())
    return seen


def test_find_scenes_ranking_and_filters(monkeypatch):
    region = (-83.2, 40.0, -83.0, 40.2)
    full, half = region, (-83.2, 40.0, -83.1, 40.2)
    items = [
        _stac_item("s3_placeholder", full, 7, 0.0, href="s3://sentinel-s2-l2a/x.jp2"),
        _stac_item("half_summer_clear", half, 7, 0.0),
        _stac_item("full_winter_clear", full, 1, 0.5),
        _stac_item("full_summer_cloudy", full, 7, 15.0),
        _stac_item("full_summer_clear", full, 6, 2.0),
        _stac_item("nodatetime_full", full, 8, 9.0, dt=False),  # item.datetime None must not crash
    ]
    seen = _fake_stac(monkeypatch, items)
    ranked = bd.find_scenes(region, "2024-01-01", "2024-12-31", 20.0)
    ids = [it.id for it, _ in ranked]
    assert "s3_placeholder" not in ids
    assert ids == ["full_summer_clear", "nodatetime_full", "full_summer_cloudy", "full_winter_clear",
                   "half_summer_clear"]
    assert ranked[0][1] == pytest.approx(1.0) and ranked[-1][1] == pytest.approx(0.5)
    assert seen["query"]["s2:nodata_pixel_percentage"] == {"lt": 50.0}
    assert seen["query"]["eo:cloud_cover"] == {"lt": 20.0}


@pytest.mark.parametrize("lat,month,expected", [
    (45, 7, True), (45, 1, False), (-40, 1, True), (-40, 7, False), (10, 1, True),
])
def test_in_season(lat, month, expected):
    assert bd._in_season(SimpleNamespace(datetime=datetime(2024, month, 1), properties={}), lat) is expected


def test_in_season_without_any_datetime_does_not_raise():
    assert bd._in_season(SimpleNamespace(datetime=None, properties={}), 45) is True


def test_with_retry_retries_then_raises(monkeypatch):
    monkeypatch.setattr(bd.time, "sleep", lambda s: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("503")
        return "ok"

    assert bd._with_retry(flaky, "x") == "ok" and len(calls) == 3
    with pytest.raises(RuntimeError):
        bd._with_retry(lambda: (_ for _ in ()).throw(RuntimeError("down")), "x", tries=2)


# ----------------------------------------------------------------------------------------------
# drop_fit_regions (train.py and compare_eurosat.py must drop the same rows)
# ----------------------------------------------------------------------------------------------
def test_drop_fit_regions_only_drops_train_rows_of_excluded_regions():
    from experiments.osm_sentinel import train

    rows = [
        {"tile_id": "a", "split": "train", "region_id": "us_columbus"},
        {"tile_id": "b", "split": "test", "region_id": "us_columbus"},
        {"tile_id": "c", "split": "train", "region_id": "other"},
        {"tile_id": "d", "split": "val", "region_id": "us_columbus"},
    ]
    assert [r["tile_id"] for r in train.drop_fit_regions(rows, train.DEFAULT_EXCLUDE_FIT)] == ["b", "c", "d"]
    assert [r["tile_id"] for r in train.drop_fit_regions(rows, [])] == ["a", "b", "c", "d"]
    assert [r["tile_id"] for r in train.drop_fit_regions(rows, None)] == ["a", "b", "c", "d"]
    source = (Path(train.__file__).parent / "compare_eurosat.py").read_text()
    assert "osm_train.drop_fit_regions(osm_train.load_rows(" in source
    assert "osm_train.DEFAULT_EXCLUDE_FIT" in source


# ----------------------------------------------------------------------------------------------
# main(): resume, cap, retry queue, per-region RNG
# ----------------------------------------------------------------------------------------------
REGIONS = [
    ("r_a", "A", "temperate", "train", 40.0, -83.0),
    ("r_b", "B", "temperate", "val", 41.0, -84.0),
    ("r_c", "C", "temperate", "test", 42.0, -85.0),
]
RADIUS_KM = 1.5  # ~17 grid cells of 640 m
SCENE = SimpleNamespace(id="S2A_17TAA_20240601_0_L2A", datetime=datetime(2024, 6, 1),
                        properties={"eo:cloud_cover": 1.5})


class FakeWorld:
    """Monkeypatches every I/O dependency of main(); counts and optionally fails scene searches."""

    def __init__(self, monkeypatch, tmp_path):
        self.tmp = tmp_path
        self.out = tmp_path / "out"
        self.csv = tmp_path / "regions.csv"
        with self.csv.open("w", newline="") as h:
            w = csv.writer(h)
            w.writerow(["region_id", "country", "biome", "split", "lat", "lon", "radius_km", "notes"])
            for rid, country, biome, split, lat, lon in REGIONS:
                w.writerow([rid, country, biome, split, lat, lon, RADIUS_KM, ""])
        self.searches: list[float] = []
        self.fail: dict[float, int] = {}  # region lat -> number of failures left (-1 = always)
        self.caption_calls = 0
        self.crash_after_captions: int | None = None
        self.monkeypatch = monkeypatch
        monkeypatch.setattr(bd, "find_scenes", self.find_scenes)
        monkeypatch.setattr(bd, "load_region_features", lambda rid, bbox, cache: SimpleNamespace(
            for_bbox=lambda b: ([], [])))
        monkeypatch.setattr(bd, "read_region_mosaic", self.mosaic)
        monkeypatch.setattr(bd, "build_caption_from_features", self.caption)

    def find_scenes(self, bbox, *a, **k):
        lat = round((bbox[1] + bbox[3]) / 2)
        self.searches.append(lat)
        left = self.fail.get(lat, 0)
        if left:
            self.fail[lat] = left - 1 if left > 0 else -1
            raise RuntimeError("stac 502")
        return [(SCENE, 1.0)]

    def mosaic(self, ranked, scene_bbox, max_scenes=4, target_valid=0.97, log=print):
        west, south, east, north = scene_bbox
        res = 0.0005
        w, h = math.ceil((east - west) / res), math.ceil((north - south) / res)
        rgb = np.full((h, w, 3), 100, dtype=np.uint8)
        return rgb, from_origin(west, north, res, res), "EPSG:4326", np.ones((h, w), np.uint8), [SCENE]

    def caption(self, features, buildings, chip_bbox, worldcover=None):
        self.caption_calls += 1
        if self.crash_after_captions is not None and self.caption_calls > self.crash_after_captions:
            raise KeyboardInterrupt  # simulates the process being killed mid-region
        return "Buildings.", "built", {"buildings": 1}, 1

    def run(self, *extra, cap=10, max_chips=0):
        argv = ["build_dataset.py", "--regions-csv", str(self.csv), "--out-dir", str(self.out),
                "--chip-px", "8", "--chips-per-region", str(cap), "--no-worldcover", "--log-every", "100000",
                "--max-chips", str(max_chips), *extra]
        self.monkeypatch.setattr(sys, "argv", argv)
        bd.main()

    def rows(self):
        with (self.out / "manifest.csv").open(newline="") as h:
            return list(csv.DictReader(h))

    def stats(self):
        return json.loads((self.out / "dataset_stats.json").read_text())

    def status(self):
        p = self.out / "region_status.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def check_consistent(self, cap):
        rows = self.rows()
        ids = [r["tile_id"] for r in rows]
        assert len(ids) == len(set(ids)), "duplicate tile_ids in manifest"
        per = {}
        for r in rows:
            per[r["region_id"]] = per.get(r["region_id"], 0) + 1
            assert r["split"] == {x[0]: x[3] for x in REGIONS}[r["region_id"]]
            assert r["scene_id"] == SCENE.id and r["datetime"] == "2024-06-01"
            assert (self.out / r["image"]).exists()
        assert all(n <= cap for n in per.values()), per
        assert self.stats()["n"] == len(rows)
        captions = (self.out / "captions.jsonl").read_text().splitlines()
        assert [json.loads(c)["tile_id"] for c in captions] == ids
        return per


@pytest.fixture
def world(monkeypatch, tmp_path):
    return FakeWorld(monkeypatch, tmp_path)


def test_main_full_run_then_resume_does_no_io(world):
    world.run()
    assert world.check_consistent(10) == {"r_a": 10, "r_b": 10, "r_c": 10}
    assert set(world.status()) == {"r_a", "r_b", "r_c"}
    assert all(s["attempted"] and s["policy"] == bd.SCENE_POLICY for s in world.status().values())
    world.searches.clear()
    world.run()
    assert world.searches == []  # complete regions: no STAC / OSM / scene I/O
    assert len(world.rows()) == 30


@pytest.mark.parametrize("prefetch", [True, False])
def test_main_max_chips_stop_mid_region_resumes_without_duplicates(world, prefetch):
    extra = () if prefetch else ("--no-prefetch",)
    world.run(*extra, max_chips=15)
    per = world.check_consistent(10)
    assert sum(per.values()) == 15 and per == {"r_a": 10, "r_b": 5}
    # r_b was interrupted by --max-chips: it must NOT be marked attempted
    assert set(world.status()) == {"r_a"}
    world.searches.clear()
    world.run(*extra)
    assert world.check_consistent(10) == {"r_a": 10, "r_b": 10, "r_c": 10}
    assert sorted(world.searches) == [41, 42]  # r_a skipped, r_b finished, r_c done


def test_main_killed_mid_region_resumes_to_exact_cap(world):
    world.crash_after_captions = 13
    with pytest.raises(KeyboardInterrupt):
        world.run("--no-prefetch")
    # dataset_stats.json is only rewritten at region boundaries, so after a kill it lags the manifest
    # (10 vs 13); the resume below rebuilds n from the manifest.
    assert world.stats()["n"] == 10
    rows = world.rows()
    assert len(rows) == 13 and sum(r["region_id"] == "r_b" for r in rows) == 3
    assert set(world.status()) == {"r_a"}  # status is only written after a region finishes
    world.crash_after_captions = None
    world.run()
    assert world.check_consistent(10) == {"r_a": 10, "r_b": 10, "r_c": 10}


def test_main_torn_manifest_row_is_dropped_on_resume(world):
    world.run(max_chips=12)
    manifest = world.out / "manifest.csv"
    manifest.write_bytes(manifest.read_bytes() + b"r_b_g+0001_+0000,images/x.jpg,r_b,B")  # torn append
    world.run()
    assert world.check_consistent(10) == {"r_a": 10, "r_b": 10, "r_c": 10}


@pytest.mark.parametrize("prefetch", [True, False])
def test_main_retry_queue_recovers_transient_failure(world, prefetch):
    world.fail[40] = 1  # r_a's first scene search fails
    world.run(*(() if prefetch else ("--no-prefetch",)))
    assert world.check_consistent(10) == {"r_a": 10, "r_b": 10, "r_c": 10}
    assert world.searches == [40, 41, 42, 40]  # retried once, at the end of the run
    assert world.stats()["failed_regions"] == []


@pytest.mark.parametrize("prefetch", [True, False])
def test_main_retry_queue_terminates_on_permanent_failure_and_recovers_next_run(world, prefetch):
    extra = () if prefetch else ("--no-prefetch",)
    world.fail[41] = -1  # r_b always fails
    world.run(*extra)
    assert world.searches.count(41) == 2  # exactly one retry, no infinite loop
    assert world.check_consistent(10) == {"r_a": 10, "r_c": 10}
    assert world.stats()["failed_regions"] == ["r_b"]
    assert "r_b" not in world.status()  # not marked attempted, so the next run retries it
    world.fail.clear()
    world.searches.clear()
    world.run(*extra)
    assert world.searches == [41]
    assert world.check_consistent(10) == {"r_a": 10, "r_b": 10, "r_c": 10}
    assert world.stats()["failed_regions"] == []


def test_main_retry_queue_last_region_failure_with_prefetch(world):
    world.fail[42] = 1  # the last region fails: its retry is appended after the prefetch slot passed
    world.run()
    assert world.searches == [40, 41, 42, 42]
    assert world.check_consistent(10) == {"r_a": 10, "r_b": 10, "r_c": 10}


def test_main_per_region_rng_is_independent_of_other_regions(world, tmp_path):
    world.run("--region", "r_b")
    alone = sorted(r["tile_id"] for r in world.rows())
    world.out = tmp_path / "out2"
    world.fail[40] = -1  # r_a failing must not change r_b's sample either
    world.run()
    together = sorted(r["tile_id"] for r in world.rows() if r["region_id"] == "r_b")
    assert alone == together and len(alone) == 10


@pytest.mark.xfail(strict=True, reason="BUG: region_status 'attempted' ignores a raised --chips-per-region; "
                                        "see the proposed patch in the agent-5 review")
def test_main_raising_chips_per_region_resumes_attempted_regions(world):
    world.run(cap=5)
    assert world.check_consistent(5) == {"r_a": 5, "r_b": 5, "r_c": 5}
    world.run(cap=10)
    assert world.check_consistent(10) == {"r_a": 10, "r_b": 10, "r_c": 10}
