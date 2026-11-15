"""Offline tests for web/server.py (owner D): the HTTP API in web/API.md, with MockBackend and a fake
satellite region. No network, no model downloads."""

import csv
import io
import json
import time

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pyproj")
pytest.importorskip("rasterio")
pytest.importorskip("shapely")

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from app import MockBackend  # noqa: E402
from src.local_tiles import EUROSAT_CLASSES  # noqa: E402
from web import analysis, geo, server  # noqa: E402

UA_BBOX = [-83.110, 39.985, -83.037, 40.070]
RID = "0123456789ab"


def _no_network(*args, **kwargs):
    raise AssertionError("this test must not touch the network")


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(geo, "CACHE_DIR", tmp_path / "web_cache")
    for name in ("_stac_search", "_get_item", "_read_window", "_pc_sign"):
        monkeypatch.setattr(geo, name, _no_network)
    with geo._LOCK:
        geo._MEM.clear()
        geo._MEM_EXTRA.clear()
    yield tmp_path / "web_cache"
    with geo._LOCK:
        geo._MEM.clear()
        geo._MEM_EXTRA.clear()


@pytest.fixture
def client(cache):
    app = server.create_app(MockBackend(), mount_classic=False)
    with TestClient(app) as c:
        yield c


def _fake_region(rows=3, cols=4, region_id=RID):
    rng = np.random.default_rng(0)
    x0, y1 = 318_000.0, 4_436_000.0
    corners, tiles = geo._tile_table(32617, x0, y1, rows, cols)
    rgb = rng.integers(1, 255, (rows * geo.TILE_PX, cols * geo.TILE_PX, 3), dtype=np.uint8)
    rgb[:geo.TILE_PX, :geo.TILE_PX] = 0  # tile 0 has no pixels
    for t in tiles:
        t["nodata_frac"], t["cloud_frac"] = (1.0 if t["i"] == 0 else 0.0), 0.0
    scene = {"id": "S2X_TEST", "datetime": "2025-08-01T16:32:38Z", "cloud_cover": 0.24, "platform": "sentinel-2c",
             "epsg": 32617, "collection": geo.S2_COLLECTION, "source": geo.S2_SOURCE}
    return geo.Region(id=region_id, bbox=tuple(UA_BBOX), scene=scene, epsg=32617,
                      transform=(10.0, 0.0, x0, 0.0, -10.0, y1), rows=rows, cols=cols, rgb=rgb,
                      corners=corners, tiles=tiles, assets={"dn_offset": 0}, scl=None)


@pytest.fixture
def fake_geo(cache, monkeypatch):
    region = _fake_region()
    n = region.n_tiles
    rng = np.random.default_rng(1)
    calls = {"region": 0, "indices": 0, "lst": 0}

    def fetch_region(bbox, max_tiles=400, when=None, progress=None, refresh=False):
        calls["region"] += 1
        progress("search", "fake search")
        geo._save_region(region, geo._request_key(geo._check_bbox(bbox), when))
        return region

    def fetch_indices(reg, progress=None):
        calls["indices"] += 1
        ndvi = rng.uniform(0.1, 0.8, (reg.rows * 64, reg.cols * 64)).astype(np.float32)
        tile_ndvi = np.linspace(0.1, 0.8, n).astype(np.float32)
        water = np.zeros(n, np.float32)
        water[-1] = 0.9
        return {"ndvi": ndvi, "ndwi": -ndvi, "tile_ndvi": tile_ndvi, "tile_water_frac": water,
                "valid_frac": np.ones(n, np.float32)}

    def fetch_lst(reg, progress=None):
        calls["lst"] += 1
        tile = (45 - 15 * np.linspace(0.1, 0.8, n)).astype(np.float32)  # greener = cooler
        lst = np.repeat(np.repeat(tile.reshape(reg.rows, reg.cols), 20, 0), 20, 1)
        return {"lst_c": lst, "tile_lst_c": tile, "days_from_s2": -9,
                "scene": {"id": "LC08_TEST", "datetime": "2025-07-23T16:00:00Z", "cloud_cover": 0.03,
                          "platform": "landsat-8", "source": geo.LST_SOURCE}}

    monkeypatch.setattr(geo, "fetch_region", fetch_region)
    monkeypatch.setattr(geo, "fetch_indices", fetch_indices)
    monkeypatch.setattr(geo, "fetch_lst", fetch_lst)
    return region, calls


def _wait(client, job_id, timeout=30.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] != "running":
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


# --------------------------------------------------------------------------- meta
def test_class_meta_order_matches_eurosat_classes():
    assert analysis.CLASS_NAMES == list(EUROSAT_CLASSES)


def test_presets_fit_the_tile_limit():
    keys = [p["key"] for p in server.PRESETS]
    assert len(keys) == len(set(keys)) and "upper-arlington" in keys
    for p in server.PRESETS:
        assert geo.grid_estimate(p["bbox"])["n_tiles"] <= server.MAX_TILES, p["key"]


def test_meta_shape(client):
    r = client.get("/api/meta")
    assert r.status_code == 200
    m = r.json()
    for key in ("mode", "disclaimer", "device", "classes", "galleries", "examples", "metrics", "local_tiles",
                "local_examples", "probe", "limits", "presets", "geo_available"):
        assert key in m, key
    assert m["mode"] == "MOCK" and m["device"] is None
    assert [c["name"] for c in m["classes"]] == list(EUROSAT_CLASSES)
    assert m["limits"] == {"max_tiles": 400, "tile_m": 640, "tile_px": 64}
    assert all(set(p) >= {"key", "name", "note", "bbox", "cached"} for p in m["presets"])
    assert not any(p["cached"] for p in m["presets"])  # empty tmp cache
    met = m["metrics"]
    for key in ("majority", "resnet18", "caption_classification", "retrieval", "local", "local_by_category",
                "local_label_counts", "per_class", "confusion", "top_confusions", "failure_taxonomy"):
        assert key in met, key
    if met["caption_classification"]:
        assert set(met["caption_classification"]["procrustes"]) == {"accuracy", "top5", "macro_f1", "n_images"}
    if met["confusion"]:
        c = met["confusion"]
        assert len(c["matrix"]) == len(c["classes"]) and all(len(r) == len(c["classes"]) for r in c["matrix"])
    if met["local"] and "oracle_majority_local" in met["local"]:
        assert met["local"]["oracle_majority_local"]["label"]
    if m["local_tiles"]:
        t = m["local_tiles"][0]
        assert t["image"] == f"/api/tiles/local/{t['tile_id']}.png"
        assert isinstance(t["lat"], float)
    json.dumps(m)


def test_read_metrics_missing_files_are_null(tmp_path):
    met = server.read_metrics(tmp_path)
    assert met["majority"] is None and met["confusion"] is None and met["local"] is None


# --------------------------------------------------------------------------- search, tiles, classify
def test_search_validation(client):
    assert client.get("/api/search", params={"q": ""}).status_code == 400
    assert client.get("/api/search", params={"q": "   "}).json()["error"]
    assert client.get("/api/search", params={"q": "forest", "k": 0}).status_code == 400
    assert client.get("/api/search", params={"q": "forest", "k": 25}).status_code == 400
    assert client.get("/api/search", params={"q": "forest", "k": "x"}).status_code == 400
    assert client.get("/api/search", params={"q": "forest", "gallery": "mars"}).status_code == 404


def test_search_local(client):
    srv = client.app.state.server
    if "local" not in srv.galleries:
        pytest.skip("no local manifest")
    r = client.get("/api/search", params={"q": "river corridor", "gallery": "local", "k": 5})
    assert r.status_code == 200
    d = r.json()
    assert d["k"] == 5 and len(d["hits"]) == 5 and "MOCK" in d["note"]
    scores = [h["score"] for h in d["hits"]]
    assert scores == sorted(scores, reverse=True)
    h = d["hits"][0]
    assert set(h) >= {"tile_id", "label", "score", "image", "lat", "lon"}
    png = client.get(h["image"])
    assert png.status_code == 200 and png.headers["content-type"] == "image/png"
    assert "max-age" in png.headers["cache-control"]
    assert Image.open(io.BytesIO(png.content)).size == (64, 64)


def test_tile_ids_are_validated(client):
    for path in ("/api/tiles/local/..%2F..%2Fapp.png", "/api/tiles/local/nope.png", "/api/tiles/local/..png",
                 "/api/tiles/../app.py", "/api/tiles/etc/passwd.png", "/api/classify/local/..%2Fx"):
        r = client.get(path)
        assert r.status_code in (400, 404), path
        assert "error" in r.json()


def test_classify_upload_and_local(client):
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (20, 90, 30)).save(buf, format="PNG")
    r = client.post("/api/classify", files={"file": ("t.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200
    d = r.json()
    assert len(d["top"]) == 3 and d["disclaimer"]
    assert all(t["label"] in EUROSAT_CLASSES for t in d["top"])
    bad = client.post("/api/classify", files={"file": ("x.png", b"not an image", "image/png")})
    assert bad.status_code == 400 and "error" in bad.json()
    srv = client.app.state.server
    if "local" in srv.galleries:
        tid = srv.galleries["local"]["items"][0]["tile_id"]
        assert len(client.get(f"/api/classify/local/{tid}").json()["top"]) == 3


def test_index_and_static(client):
    r = client.get("/")
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        assert r.headers["cache-control"] == "no-cache"
    else:
        assert "error" in r.json()
    assert "error" in client.get("/api/nothing-here").json()


# --------------------------------------------------------------------------- region explorer
def test_region_estimate(client):
    r = client.post("/api/region/estimate", json={"bbox": UA_BBOX})
    assert r.status_code == 200
    d = r.json()
    assert (d["rows"], d["cols"], d["n_tiles"], d["ok"], d["max_tiles"]) == (15, 11, 165, True, 400)
    big = client.post("/api/region/estimate", json={"bbox": [-84, 39, -82, 41]}).json()
    assert big["ok"] is False and "limit" in big["message"]
    assert client.post("/api/region/estimate", json={"bbox": [1, 2, 3]}).status_code == 400
    assert client.post("/api/region/estimate", json={"bbox": [3, 2, 1, 4]}).status_code == 422
    assert client.post("/api/region/estimate", content=b"nope").status_code == 400


def test_region_rejects_bad_requests(client):
    assert client.post("/api/region", json={"bbox": [-84, 39, -82, 41]}).status_code == 422
    assert client.post("/api/region", json={"bbox": UA_BBOX, "when": "1999"}).status_code == 422
    assert client.get("/api/region/zzz").status_code == 404
    assert client.get("/api/region/ffffffffffff").status_code == 404
    assert client.get("/api/jobs/nope").status_code == 404


def test_region_job_flow(client, fake_geo, cache):
    region, calls = fake_geo
    r = client.post("/api/region", json={"bbox": UA_BBOX, "when": None})
    assert r.status_code == 200
    job = _wait(client, r.json()["job"])
    assert job["status"] == "done", job["error"]
    assert job["kind"] == "region" and job["stages"][0]["stage"] == "search"
    assert all(set(s) == {"stage", "detail", "t"} for s in job["stages"])
    res = job["result"]
    assert res["id"] == RID and res["mode"] == "MOCK" and res["n_tiles"] == region.n_tiles == len(res["tiles"])
    assert res["image"] == f"/api/region/{RID}/image.png" and len(res["corners"]) == 4
    t = res["tiles"][1]
    assert set(t) >= {"i", "row", "col", "corners", "center", "nodata_frac", "cloud_frac", "cls", "score", "top3"}
    assert sum(c["n"] for c in res["composition"]) == region.n_tiles - 1  # tile 0 has no pixels
    assert sum(c["n"] for c in res["ipcc"]) == region.n_tiles - 1
    assert (cache / RID / "summary.mock.json").exists() and (cache / RID / "aligned.mock.npy").exists()
    assert client.get(f"/api/region/{RID}").json()["id"] == RID

    img = client.get(f"/api/region/{RID}/image.png")
    assert Image.open(io.BytesIO(img.content)).size == (4 * 64, 3 * 64)
    assert Image.open(io.BytesIO(client.get(f"/api/region/{RID}/tile/5.png").content)).size == (64, 64)
    assert client.get(f"/api/region/{RID}/tile/99.png").status_code == 404

    s = client.get(f"/api/region/{RID}/search", params={"q": "parking lots", "k": 3}).json()
    assert len(s["scores"]) == region.n_tiles and len(s["top"]) == 3
    assert all(h["i"] != 0 for h in s["top"])  # the empty tile never ranks
    assert s["min"] <= s["max"]
    assert client.get(f"/api/region/{RID}/search", params={"q": ""}).status_code == 400
    sim = client.get(f"/api/region/{RID}/similar", params={"i": 5, "k": 4}).json()
    assert sim["query"] == 5 and len(sim["top"]) == 4 and all(h["i"] != 5 for h in sim["top"])
    assert client.get(f"/api/region/{RID}/similar", params={"i": 99}).status_code == 400

    # Climate
    assert client.get(f"/api/region/{RID}/layer/ndvi.png").status_code == 404
    cj = _wait(client, client.post(f"/api/region/{RID}/climate").json()["job"])
    assert cj["status"] == "done", cj["error"]
    c = cj["result"]
    assert len(c["ndvi"]["tile"]) == region.n_tiles and c["ndvi"]["vmin"] == 0.0 and c["ndvi"]["vmax"] == 0.9
    assert c["ndvi"]["palette"] and c["lst"]["palette"] and c["lst_error"] is None
    assert c["lst"]["vmin"] < c["lst"]["vmax"] and c["lst"]["days_from_s2"] == -9
    st = c["stats"]
    assert st["n_water"] == 1 and st["n_land"] == region.n_tiles - 2
    assert st["regression"]["slope_per_0p1_ndvi"] < 0 and st["tercile"]["hot_minus_green_c"] > 0
    for name in ("ndvi", "lst"):
        layer = client.get(f"/api/region/{RID}/layer/{name}.png")
        assert layer.status_code == 200
        assert Image.open(io.BytesIO(layer.content)).mode == "RGBA"
    assert client.get(f"/api/region/{RID}/layer/foo.png").status_code == 404

    # A second climate POST completes immediately from the cache.
    again = client.get(f"/api/jobs/{client.post(f'/api/region/{RID}/climate').json()['job']}").json()
    assert again["status"] == "done" and calls["indices"] == 1

    # Exports
    ex = client.get(f"/api/region/{RID}/export.csv")
    assert "attachment" in ex.headers["content-disposition"] and ex.headers["content-disposition"].endswith('.csv"')
    rows = list(csv.DictReader(io.StringIO(ex.text)))
    assert len(rows) == region.n_tiles and rows[0]["mode"] == "MOCK" and rows[1]["lst_c"] != ""
    gj = client.get(f"/api/region/{RID}/export.geojson")
    assert "attachment" in gj.headers["content-disposition"]
    fc = json.loads(gj.text)
    assert fc["type"] == "FeatureCollection" and len(fc["features"]) == region.n_tiles
    assert fc["features"][0]["geometry"]["type"] == "Polygon"


def test_region_survives_restart(client, fake_geo, cache):
    job = _wait(client, client.post("/api/region", json={"bbox": UA_BBOX}).json()["job"])
    assert job["status"] == "done"
    _wait(client, client.post(f"/api/region/{RID}/climate").json()["job"])
    # A fresh server (same disk cache) answers GETs without recomputing.
    with TestClient(server.create_app(MockBackend(), mount_classic=False)) as c2:
        d = c2.get(f"/api/region/{RID}")
        assert d.status_code == 200 and d.json()["n_tiles"] == 12
        cj = c2.get(f"/api/jobs/{c2.post(f'/api/region/{RID}/climate').json()['job']}").json()
        assert cj["status"] == "done"
        assert len(c2.get(f"/api/region/{RID}/search", params={"q": "trees"}).json()["scores"]) == 12
        # POSTing the same box again reuses the saved labels.
        again = _wait(c2, c2.post("/api/region", json={"bbox": UA_BBOX}).json()["job"])
        assert again["result"]["from_cache"] is True
    # Pixels saved but never labelled (only region.npz): 404 that tells the client to POST again.
    geo._save_region(_fake_region(region_id="aaaaaaaaaaaa"))
    r = client.get("/api/region/aaaaaaaaaaaa")
    assert r.status_code == 404 and "POST" in r.json()["error"]


def test_climate_without_lst(client, fake_geo, monkeypatch):
    def no_lst(region, progress=None):
        raise geo.GeoError("No clear Landsat pass with surface temperature covers this area near that date.")

    monkeypatch.setattr(geo, "fetch_lst", no_lst)
    _wait(client, client.post("/api/region", json={"bbox": UA_BBOX}).json()["job"])
    c = _wait(client, client.post(f"/api/region/{RID}/climate").json()["job"])
    assert c["status"] == "done"
    assert c["result"]["lst"] is None and "Landsat" in c["result"]["lst_error"]
    assert c["result"]["stats"]["regression"] is None and c["result"]["stats"]["ndvi_mean"] is not None
    assert client.get(f"/api/region/{RID}/layer/lst.png").status_code == 404


def test_region_job_errors_are_plain(client, cache, monkeypatch):
    def offline(*a, **k):
        raise geo.GeoError("Couldn't reach the Sentinel-2 archive. Check the internet connection.", network=True)

    monkeypatch.setattr(geo, "fetch_region", offline)
    job = _wait(client, client.post("/api/region", json={"bbox": UA_BBOX}).json()["job"])
    assert job["status"] == "error" and "internet" in job["error"]

    def crash(*a, **k):
        raise ZeroDivisionError("secret internals")

    monkeypatch.setattr(geo, "fetch_region", crash)
    job = _wait(client, client.post("/api/region", json={"bbox": UA_BBOX}).json()["job"])
    assert job["status"] == "error" and "secret" not in job["error"] and "Traceback" not in job["error"]


def test_geocode_offline_and_cached(client, monkeypatch):
    import requests

    def boom(*a, **k):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(requests, "get", boom)
    assert client.get("/api/geocode", params={"q": "Phoenix"}).json() == []
    assert client.get("/api/geocode", params={"q": ""}).json() == []

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return [{"display_name": "Phoenix, Arizona", "lat": "33.45", "lon": "-112.07",
                     "boundingbox": ["33.29", "33.92", "-112.32", "-111.92"], "addresstype": "city"}]

    seen = []

    def fake_get(url, params=None, headers=None, timeout=None):
        seen.append((url, params, headers, timeout))
        return Resp()

    monkeypatch.setattr(requests, "get", fake_get)
    out = client.get("/api/geocode", params={"q": "Tempe"}).json()
    assert out == [{"name": "Phoenix, Arizona", "lat": 33.45, "lon": -112.07,
                    "bbox": [-112.32, 33.29, -111.92, 33.92], "type": "city"}]
    client.get("/api/geocode", params={"q": "tempe"})
    assert len(seen) == 1  # second call came from the cache
    assert seen[0][2]["User-Agent"] == "platonic-earth-hackathon/0.1" and seen[0][3] == 6


def test_regions_cached_lists_saved_requests(client, fake_geo):
    _wait(client, client.post("/api/region", json={"bbox": UA_BBOX}).json()["job"])
    listed = client.get("/api/regions/cached").json()
    assert [d["id"] for d in listed] == [RID]
    presets = {p["key"]: p for p in client.get("/api/meta").json()["presets"]}
    assert presets["upper-arlington"]["cached"] is True
