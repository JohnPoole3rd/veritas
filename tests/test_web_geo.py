"""Offline tests for web/geo.py (owner D): grid maths, scene ranking, rendering and the disk cache.

No test here touches the network; anything that would is replaced by a function that fails the test.
"""

import io
import math
from datetime import datetime, timezone

import numpy as np
import pytest

pytest.importorskip("pyproj")
pytest.importorskip("rasterio")
pytest.importorskip("shapely")

from PIL import Image  # noqa: E402

from web import geo  # noqa: E402

UA_BBOX = (-83.110, 39.985, -83.037, 40.070)  # src/local_collect.py BBOX, as (W, S, E, N)


def _no_network(*args, **kwargs):
    raise AssertionError("this test must not touch the network")


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(geo, "CACHE_DIR", tmp_path / "web_cache")
    monkeypatch.setattr(geo, "_stac_search", _no_network)
    monkeypatch.setattr(geo, "_get_item", _no_network)
    monkeypatch.setattr(geo, "_read_window", _no_network)
    monkeypatch.setattr(geo, "_pc_sign", _no_network)
    with geo._LOCK:
        geo._MEM.clear()
        geo._MEM_EXTRA.clear()
    yield tmp_path / "web_cache"
    with geo._LOCK:
        geo._MEM.clear()
        geo._MEM_EXTRA.clear()


def _synthetic_region(rows=2, cols=3, bbox=UA_BBOX, seed=0, region_id="0123456789ab"):
    rng = np.random.default_rng(seed)
    x0, y1 = 318_000.0, 4_436_000.0
    corners, tiles = geo._tile_table(32617, x0, y1, rows, cols)
    rgb = rng.integers(1, 255, (rows * geo.TILE_PX, cols * geo.TILE_PX, 3), dtype=np.uint8)
    scene = {"id": "S2X_TEST", "datetime": "2025-08-01T16:32:38.097000Z", "cloud_cover": 0.24,
             "platform": "sentinel-2c", "epsg": 32617, "collection": geo.S2_COLLECTION, "source": geo.S2_SOURCE}
    return geo.Region(id=region_id, bbox=tuple(bbox), scene=scene, epsg=32617,
                      transform=(10.0, 0.0, x0, 0.0, -10.0, y1), rows=rows, cols=cols, rgb=rgb,
                      corners=corners, tiles=tiles, assets={"dn_offset": 0},
                      scl=np.full((rows * 32, cols * 32), 4, np.uint8))


# --------------------------------------------------------------------------- grid maths
def test_grid_estimate_upper_arlington():
    g = geo.grid_estimate(UA_BBOX)
    assert g["epsg"] == 32617
    assert (g["rows"], g["cols"]) == (15, 11)  # 9.4 km x 6.2 km box, grown to whole 640 m tiles
    assert g["n_tiles"] == g["rows"] * g["cols"] == 165
    assert g["width_km"] == pytest.approx(g["cols"] * 0.64)
    assert g["height_km"] == pytest.approx(g["rows"] * 0.64)
    # the snapped window is never smaller than the box, and never more than one tile bigger
    assert 0 <= g["width_km"] - g["bbox_width_km"] < 0.64 + 0.35  # + grid-convergence skew of the box
    assert 0 <= g["height_km"] - g["bbox_height_km"] < 0.64 + 0.35


def test_grid_estimate_grows_and_has_a_floor():
    w, s, e, n = UA_BBOX
    prev = 0
    for k in (0.0, 0.01, 0.03, 0.08):
        g = geo.grid_estimate((w - k, s - k, e + k, n + k))
        assert g["n_tiles"] >= prev
        prev = g["n_tiles"]
    tiny = geo.grid_estimate((-83.0601, 40.0200, -83.0600, 40.0201))
    assert (tiny["rows"], tiny["cols"], tiny["n_tiles"]) == (1, 1, 1)
    assert geo.grid_estimate((-47.95, -15.82, -47.91, -15.78))["epsg"] == 32723  # Brasilia: southern zone 23


@pytest.mark.parametrize("bad", [(1, 2, 3), (10, 5, 9, 6), (10, 5, 11, 4), ("a", 0, 1, 1), (0, 85, 1, 86),
                                 (179, 0, 181, 1), (float("nan"), 0, 1, 1)])
def test_bad_boxes_are_plain_errors(bad):
    with pytest.raises(geo.GeoError):
        geo.grid_estimate(bad)


def test_snap_window_contains_box_and_stays_on_the_pixel_grid():
    from pyproj import Transformer

    for origin in [(0.0, 0.0), (300_000.0, 4_500_000.0), (399_960.0, 4_500_000.0)]:
        x0, y1, rows, cols = geo._snap(UA_BBOX, 32617, origin)
        x1, y0 = x0 + cols * geo.TILE_M, y1 - rows * geo.TILE_M
        w, s, e, n = UA_BBOX
        xs, ys = Transformer.from_crs(4326, 32617, always_xy=True).transform([w, e, e, w], [n, n, s, s])
        assert x0 <= min(xs) and max(xs) <= x1 and y0 <= min(ys) and max(ys) <= y1
        # on the 20 m grid relative to the scene origin (so 10 m bands and the 20 m SCL band both align)
        assert ((x0 - origin[0]) / 20).is_integer() and ((y1 - origin[1]) / 20).is_integer()
        # padding is split evenly: neither side gets more than half a tile plus one 20 m step
        assert min(xs) - x0 <= geo.TILE_M / 2 + 20 and x1 - max(xs) <= geo.TILE_M / 2 + 20


def test_tile_table_corner_order_and_row_major():
    rows, cols = 3, 4
    corners, tiles = geo._tile_table(32617, 318_000.0, 4_436_000.0, rows, cols)
    tl, tr, br, bl = corners
    assert tl[0] < tr[0] and bl[0] < br[0]  # west < east
    assert tl[1] > bl[1] and tr[1] > br[1]  # north > south
    assert len(tiles) == rows * cols and [t["i"] for t in tiles] == list(range(rows * cols))
    first, last = tiles[0], tiles[-1]
    assert (first["row"], first["col"]) == (0, 0) and (last["row"], last["col"]) == (rows - 1, cols - 1)
    assert first["corners"][0] == tl and last["corners"][2] == br  # tile 0 is the north-west one
    assert tiles[1]["center"][0] > first["center"][0]  # i + 1 moves east
    assert tiles[cols]["center"][1] < first["center"][1]  # i + cols moves south
    for t in tiles:
        (a, b, c, d), (cx, cy) = t["corners"], t["center"]
        assert a[0] < cx < b[0] and d[0] < cx < c[0] and d[1] < cy < a[1] and c[1] < cy < b[1]
    # neighbours share an edge exactly, and one tile is 640 m wide
    assert tiles[0]["corners"][1] == tiles[1]["corners"][0]
    width_m = (tiles[0]["corners"][1][0] - tiles[0]["corners"][0][0]) * 111_320 * math.cos(math.radians(40.05))
    assert width_m == pytest.approx(640, rel=0.01)


def test_region_tile_image_is_the_right_chip():
    r = _synthetic_region(rows=2, cols=3)
    assert r.n_tiles == 6
    img = r.tile_image(4)  # row 1, col 1
    assert img.size == (64, 64) and img.mode == "RGB"
    assert np.array_equal(np.asarray(img), r.rgb[64:128, 64:128])
    x0, y0, x1, y1 = r.bounds
    assert (x1 - x0, y1 - y0) == (3 * 640, 2 * 640)
    with pytest.raises(IndexError):
        r.tile_image(6)


def test_block_reductions_ignore_nan():
    a = np.arange(16, dtype=np.float32).reshape(4, 4)
    a[0, 0] = np.nan
    a[2:, 2:] = np.nan
    m = geo._block_mean(a, 2, 2)
    assert m[0, 0] == pytest.approx((1 + 4 + 5) / 3) and m[0, 1] == pytest.approx((2 + 3 + 6 + 7) / 4)
    assert np.isnan(m[1, 1])
    f = geo._block_frac(np.isfinite(a), 2, 2)
    assert f.tolist() == [[0.75, 1.0], [1.0, 0.0]]


# --------------------------------------------------------------------------- scene choice
def _item(scene_id, date, cloud, epsg=32617, square=(-84.0, 39.0, -82.0, 41.0)):
    w, s, e, n = square
    return {"id": scene_id, "geometry": {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]},
            "properties": {"datetime": f"{date}T16:00:00Z", "eo:cloud_cover": cloud, "proj:epsg": epsg, "platform": "sentinel-2b"}}


def test_scene_ranking_order():
    items = [
        _item("cloudy_summer_recent", "2026-07-20", 12.0),
        _item("clear_winter", "2026-01-10", 0.2),
        _item("clear_summer_old", "2024-08-02", 0.6),
        _item("clear_summer_new", "2025-08-01", 0.3),
        _item("clear_summer_other_zone", "2026-08-30", 0.4, epsg=32616),
        _item("clearest_but_half_covered", "2026-08-15", 0.0, square=(-84.0, 40.02, -82.0, 41.0)),
        _item("elsewhere", "2026-08-16", 0.0, square=(10.0, 10.0, 11.0, 11.0)),
        _item("lightly_cloudy_summer", "2026-08-20", 3.0),
    ]
    order = [f["id"] for f in geo._rank_s2(items, UA_BBOX)]
    assert "elsewhere" not in order
    # full cover first; then cloud bucket; then leaf-on season; then the box's own UTM zone; then recency
    assert order == ["clear_summer_new", "clear_summer_old", "clear_summer_other_zone", "clear_winter",
                     "lightly_cloudy_summer", "cloudy_summer_recent", "clearest_but_half_covered"]
    # a fixed month switches the season rule off: recency decides inside the <1 % bucket
    assert geo._rank_s2(items[1:4], UA_BBOX, month_fixed=True)[0]["id"] == "clear_winter"
    # southern hemisphere: December is leaf-on
    south = (-58.50, -34.65, -58.45, -34.60)
    sq = (-59.0, -35.0, -58.0, -34.0)
    pair = [_item("july", "2026-07-01", 0.1, 32721, sq), _item("december", "2025-12-20", 0.5, 32721, sq)]
    assert geo._rank_s2(pair, south)[0]["id"] == "december"
    # near the equator there is no season rule: the newer pass wins
    eq = (36.80, -1.31, 36.84, -1.27)
    sq = (36.0, -2.0, 37.5, -0.5)
    pair = [_item("july", "2026-07-01", 0.1, 32737, sq), _item("december", "2025-12-20", 0.5, 32737, sq)]
    assert geo._rank_s2(pair, eq)[0]["id"] == "july"


def test_when_range_and_offsets():
    now = datetime(2026, 11, 15, tzinfo=timezone.utc)
    start, end, fixed = geo._when_range(None, now)
    assert end == now and (now - start).days == 365 * geo.SEARCH_YEARS and fixed is False
    start, end, fixed = geo._when_range("2024", now)
    assert (start.year, end.year, fixed) == (2024, 2025, False)
    start, end, fixed = geo._when_range("2024-12", now)
    assert (start.month, end.year, end.month, fixed) == (12, 2025, 1, True)
    for bad in ("24", "2024-13", "2014", "2031", "July"):
        with pytest.raises(geo.GeoError):
            geo._when_range(bad, now)
    # Earth Search already removed the +1000 DN offset when it says so; otherwise baseline >= 04.00 needs it
    assert geo._dn_offset({"earthsearch:boa_offset_applied": True, "s2:processing_baseline": "05.11"}) == 0
    assert geo._dn_offset({"earthsearch:boa_offset_applied": False, "s2:processing_baseline": "05.11"}) == 1000
    assert geo._dn_offset({"s2:processing_baseline": "04.00"}) == 1000
    assert geo._dn_offset({"s2:processing_baseline": "03.01"}) == 0
    assert geo._item_epsg({"proj:code": "EPSG:32617"}) == 32617 and geo._item_epsg({"proj:epsg": 32733}) == 32733
    assert geo._parse_dt("2025-08-01T16:32:38.097000Z").tzinfo is not None
    assert geo._pinned_scene(UA_BBOX) == "S2C_17TLE_20250801_0_L2A" and geo._pinned_scene((2.2, 48.8, 2.4, 48.9)) is None


def test_indices_maths():
    rows, cols = 1, 2
    shape = (64, 128)
    red = np.full(shape, 500, np.uint16)
    green = np.full(shape, 600, np.uint16)
    nir = np.full(shape, 4500, np.uint16)  # vegetation: NDVI = 0.8
    red[:, 64:], green[:, 64:], nir[:, 64:] = 300, 500, 100  # right tile is water: NDWI = 0.667
    nir[0, 0] = red[0, 0] = green[0, 0] = 0  # one no-data pixel
    scl = np.full((32, 64), 4, np.uint8)
    scl[:8, :32] = 9  # a cloud over a quarter of the left tile
    out = geo._indices_from_bands(red, green, nir, scl, rows, cols)
    assert out["ndvi"].shape == shape and out["ndvi"].dtype == np.float32
    assert np.isnan(out["ndvi"][0, 0]) and np.isnan(out["ndvi"][10, 10])
    assert out["tile_ndvi"][0] == pytest.approx(0.8, abs=1e-4)
    assert out["tile_ndvi"][1] == pytest.approx((100 - 300) / 400, abs=1e-4)
    assert out["valid_frac"].tolist() == pytest.approx([0.75, 1.0])
    assert out["tile_water_frac"].tolist() == pytest.approx([0.0, 1.0])
    # with the +1000 offset still in the numbers, the same reflectances need it taken off first
    shifted = geo._indices_from_bands(red + 1000, green + 1000, nir + 1000, None, rows, cols, offset=1000)
    assert shifted["tile_ndvi"][0] == pytest.approx(0.8, abs=1e-4)


# --------------------------------------------------------------------------- rendering
def test_colorize_png_with_transparent_nan():
    a = np.array([[0.0, 0.5, 1.0], [np.nan, -3.0, 9.0]], dtype=np.float32)
    for palette in ("ndvi", "heat"):
        img = Image.open(io.BytesIO(geo.colorize(a, 0.0, 1.0, palette)))
        assert img.format == "PNG" and img.mode == "RGBA" and img.size == (3, 2)
        px = np.asarray(img)
        assert px[1, 0].tolist() == [0, 0, 0, 0]  # NaN is fully transparent
        assert (px[..., 3][np.isfinite(a)] == 255).all()
        assert px[1, 1].tolist() == px[0, 0].tolist() and px[1, 2].tolist() == px[0, 2].tolist()  # clipped
        assert len({tuple(p) for p in px[0]}) == 3
    ndvi = np.asarray(Image.open(io.BytesIO(geo.colorize(a, 0.0, 1.0, "ndvi"))))
    assert ndvi[0, 2, 1] > ndvi[0, 2, 0] and ndvi[0, 0, 0] > ndvi[0, 0, 1]  # high = green, low = brown
    heat = np.asarray(Image.open(io.BytesIO(geo.colorize(a, 0.0, 1.0, "heat"))))
    assert heat[0, 0, 2] > heat[0, 0, 0] and heat[0, 2, 0] > heat[0, 2, 2]  # cold = blue, hot = red
    match = np.asarray(Image.open(io.BytesIO(geo.colorize(a, 0.0, 1.0, "match"))))
    assert match[0, 0, 3] == 0 and match[0, 0, 3] < match[0, 1, 3] < match[0, 2, 3]  # alpha grows with the match
    big = Image.open(io.BytesIO(geo.colorize(a, 0.0, 1.0, "heat", scale=4)))
    assert big.size == (12, 8) and np.asarray(big)[7, 0, 3] == 0
    flat = Image.open(io.BytesIO(geo.colorize(np.ones((2, 2)), 1.0, 1.0, "heat")))  # vmin == vmax must not crash
    assert flat.size == (2, 2)
    with pytest.raises(ValueError):
        geo.colorize(a, 0, 1, "rainbow")
    with pytest.raises(ValueError):
        geo.colorize(np.zeros(3), 0, 1, "heat")
    stops = geo.palette_stops("heat", 5)
    assert [s[0] for s in stops] == [0.0, 0.25, 0.5, 0.75, 1.0] and all(s[1].startswith("#") for s in stops)


def test_rgb_png_round_trip():
    r = _synthetic_region()
    img = Image.open(io.BytesIO(geo.rgb_png(r)))
    assert img.mode == "RGB" and img.size == (r.cols * 64, r.rows * 64)
    assert np.array_equal(np.asarray(img), r.rgb)


# --------------------------------------------------------------------------- cache
def test_region_disk_cache_round_trip(cache):
    r = _synthetic_region()
    assert geo.load_cached(r.id) is None and geo.cached_region_for_bbox(UA_BBOX) is None
    geo._save_region(r, geo._request_key(r.bbox, None))
    assert (cache / r.id / "region.npz").exists() and (cache / "index.json").exists()
    with geo._LOCK:
        geo._MEM.clear()  # force the disk path
    back = geo.load_cached(r.id)
    assert back is not None and back is not r
    assert np.array_equal(back.rgb, r.rgb) and np.array_equal(back.scl, r.scl)
    for name in ("id", "bbox", "scene", "epsg", "transform", "rows", "cols", "corners", "tiles", "assets"):
        assert getattr(back, name) == getattr(r, name), name
    assert geo.load_cached(r.id) is back  # second hit comes from memory
    assert geo.cached_region_for_bbox(UA_BBOX).id == r.id
    assert geo.cached_region_for_bbox(tuple(v + 1e-6 for v in UA_BBOX)).id == r.id  # same box to ~10 m
    assert geo.cached_region_for_bbox(UA_BBOX, when="2024") is None  # a different date is a different request
    assert geo.cached_region_for_bbox((-83.2, 39.985, -83.037, 40.070)) is None
    listed = geo.list_cached()
    assert [d["id"] for d in listed] == [r.id] and listed[0]["layers"] == [] and listed[0]["when"] is None
    # ids come from URLs: anything that isn't a 12-hex id is refused, never used as a path
    for evil in ("../index", "0123456789aZ", "", None, "0123456789ab/..", "a" * 13):
        assert geo.load_cached(evil) is None


def test_fetch_region_uses_cache_and_checks_size_first(cache, monkeypatch):
    with pytest.raises(geo.GeoError, match="limit here is 100"):  # 165 tiles > 100, before any network call
        geo.fetch_region(UA_BBOX, max_tiles=100)
    r = _synthetic_region(rows=15, cols=11)
    geo._save_region(r, geo._request_key(UA_BBOX, None))
    said = []
    got = geo.fetch_region(UA_BBOX, progress=lambda stage, detail: said.append((stage, detail)))
    assert got.id == r.id and said and said[0][0] == "cache" and "1 Aug 2025" in said[0][1]

    def offline(*a, **k):
        raise geo.GeoError("Couldn't reach the Sentinel-2 archive. Check the internet connection.", network=True)

    monkeypatch.setattr(geo, "_fetch_region_online", offline)
    assert geo.fetch_region(UA_BBOX, refresh=True).id == r.id  # network down: the saved copy still answers
    with pytest.raises(geo.GeoError, match="internet"):
        geo.fetch_region((2.30, 48.85, 2.34, 48.87))  # never fetched and no network: a plain error


def test_layers_come_from_cache_without_network(cache):
    r = _synthetic_region()
    geo._save_region(r)
    n, H, W = r.n_tiles, r.rows * 64, r.cols * 64
    rng = np.random.default_rng(1)
    ndvi = rng.uniform(-0.2, 0.9, (H, W)).astype(np.float32)
    ndvi[0, :5] = np.nan
    indices = {"ndvi": ndvi, "ndwi": -ndvi, "tile_ndvi": geo._block_mean(ndvi, r.rows, r.cols).ravel(),
               "tile_water_frac": np.zeros(n, np.float32), "valid_frac": np.ones(n, np.float32)}
    lst_c = rng.uniform(25, 45, (r.rows * 20, r.cols * 20)).astype(np.float32)
    lst = {"lst_c": lst_c, "tile_lst_c": geo._block_mean(lst_c, r.rows, r.cols).ravel(),
           "scene": {"id": "LC09_TEST", "datetime": "2025-07-28T16:10:00Z", "cloud_cover": 1.2,
                     "platform": "landsat-9", "source": geo.LST_SOURCE}, "days_from_s2": -4}
    geo._save_extra(r.id, "indices", indices)
    geo._save_extra(r.id, "lst", lst)
    with geo._LOCK:
        geo._MEM.clear()
        geo._MEM_EXTRA.clear()  # force the disk path; the fixture makes any network call fail the test
    back = geo.load_cached(r.id)
    got = geo.fetch_indices(back)
    assert set(got) == {"ndvi", "ndwi", "tile_ndvi", "tile_water_frac", "valid_frac"}
    assert np.allclose(got["ndvi"], ndvi, equal_nan=True) and got["ndvi"].dtype == np.float32
    assert np.allclose(got["tile_ndvi"], indices["tile_ndvi"])
    got = geo.fetch_lst(back)
    assert np.allclose(got["lst_c"], lst_c) and got["tile_lst_c"].shape == (n,)
    assert got["scene"] == lst["scene"] and got["days_from_s2"] == -4
    assert geo.fetch_lst(back) is got  # then from memory
    assert sorted(geo.list_cached(), key=lambda d: d["id"]) == []  # saved without a request key: not listed


def test_memory_cache_is_bounded(cache):
    for k in range(geo.MEM_REGIONS + 3):
        geo._save_region(_synthetic_region(rows=1, cols=1, seed=k, region_id=f"{k:012x}"))
    with geo._LOCK:
        assert len(geo._MEM) == geo.MEM_REGIONS and f"{0:012x}" not in geo._MEM
    assert geo.load_cached(f"{0:012x}") is not None  # evicted from memory, still on disk
