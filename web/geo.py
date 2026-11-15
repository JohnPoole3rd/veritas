"""Satellite I/O for the map "region explorer" (owner: D, subagent D1).

A user drags a lon/lat box on the map. This module
  1. picks a clear Sentinel-2 L2A pass for it (Element84 Earth Search STAC, public COGs on AWS),
  2. reads the true-colour pixels and cuts 64 x 64 px tiles (640 m at 10 m per pixel, the same
     footprint as EuroSAT and A's Upper Arlington tiles) on the scene's native UTM pixel grid,
  3. reads measured climate layers for the same footprint: NDVI / NDWI from Sentinel-2 bands and
     land-surface temperature from Landsat 8/9 Collection 2 Level-2 (Microsoft Planetary Computer),
  4. renders arrays as PNG overlays.

No model code lives here: embedding, rotation and text search are in web/server.py.

Everything fetched is cached under data/raw/web_cache/ (gitignored), so an area that was fetched
once keeps working with no internet. Heavy geo packages (rasterio, pyproj, shapely) are imported
lazily, so importing this module never fails on a machine that only runs the core demo.

`progress(stage, detail)` callbacks get short sentences for the UI. Stages: search, scene, pixels,
tiles, cache (region); indices (NDVI/NDWI); lst-search, lst-scene, lst-pixels (surface temperature).
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "raw" / "web_cache"  # tests monkeypatch this

TILE_PX = 64
GSD_M = 10.0
TILE_M = TILE_PX * GSD_M  # 640 m
LST_PX_PER_TILE = 20  # surface temperature is resampled to 32 m so one tile is exactly 20 x 20 pixels
LST_GSD_M = TILE_M / LST_PX_PER_TILE

S2_STAC = "https://earth-search.aws.element84.com/v1"
S2_COLLECTION = "sentinel-2-l2a"
S2_SOURCE = "Copernicus Sentinel-2 L2A via Element84 Earth Search"
PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
PC_SAS = "https://planetarycomputer.microsoft.com/api/sas/v1"
LANDSAT_COLLECTION = "landsat-c2-l2"
LST_SOURCE = "USGS Landsat Collection 2 Level-2 ST via Microsoft Planetary Computer"
LST_SCALE, LST_OFFSET = 0.00341802, 149.0  # Kelvin = DN * scale + offset (USGS C2 L2 ST_B10)

HEADERS = {"User-Agent": "platonic-earth-hackathon/0.1"}
HTTP_TIMEOUT = (8, 30)  # connect, read seconds: nothing here may hang
GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIRANGE="YES",
    GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES", GDAL_HTTP_TIMEOUT="30", GDAL_HTTP_CONNECTTIMEOUT="8",
    GDAL_HTTP_MAX_RETRY="2", GDAL_HTTP_RETRY_DELAY="1", VSI_CACHE="TRUE",
)
S2_GDAL_ENV = dict(GDAL_ENV, CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF")

# Sentinel-2 scene classification (SCL) codes.
SCL_INVALID = (0, 1, 3, 8, 9, 10, 11)  # no data, saturated, cloud shadow, cloud x2, cirrus, snow
SCL_CLOUD = (3, 8, 9, 10)
# Landsat QA_PIXEL bits to drop: dilated cloud, cirrus, cloud, cloud shadow, snow.
LANDSAT_BAD_BITS = (1, 2, 3, 4, 5)

SEARCH_YEARS = 3
MEM_REGIONS = 8
RETRY_CLOUD, RETRY_NODATA = 0.20, 0.02  # try the next-best pass if this much of the box is cloud / missing

# Where the project already has labelled tiles, use the pass those tiles came from, so explorer chips
# are pixel-identical to A's (src/local_collect.py SCENE_ID). Only when no date is asked for.
PINNED_SCENES = [((-83.25, 39.85, -82.85, 40.20), "S2C_17TLE_20250801_0_L2A")]

_LOCK = threading.RLock()
_MEM: "OrderedDict[str, Region]" = OrderedDict()
_MEM_EXTRA: dict[str, dict] = {}
_PC_TOKEN: dict = {}


class GeoError(RuntimeError):
    """A problem worth showing to the user. str(e) is plain language."""

    def __init__(self, message: str, network: bool = False):
        super().__init__(message)
        self.network = network  # True: the internet/archive failed (a saved copy may still work)


@dataclass
class Region:
    id: str            # stable short hash of (scene id, snapped pixel window)
    bbox: tuple        # requested (west, south, east, north), lon/lat
    scene: dict        # id, datetime, cloud_cover, platform, epsg, collection, source
    epsg: int
    transform: tuple   # affine (a, b, c, d, e, f) of the snapped window, rasterio order
    rows: int
    cols: int
    rgb: np.ndarray    # uint8 [rows*64, cols*64, 3]
    corners: list      # [[lon, lat] x 4]: TL, TR, BR, BL of the whole window
    tiles: list        # row-major dicts: i, row, col, corners, center, nodata_frac, cloud_frac
    assets: dict = field(default_factory=dict, repr=False)  # band hrefs + DN offset (for fetch_indices)
    scl: np.ndarray | None = field(default=None, repr=False)  # 20 m scene classification, uint8

    @property
    def n_tiles(self) -> int:
        return self.rows * self.cols

    @property
    def bounds(self) -> tuple:
        """(x0, y0, x1, y1) in the region's UTM CRS, metres."""
        a, _, c, _, e, f = self.transform
        return (c, f + e * self.rows * TILE_PX, c + a * self.cols * TILE_PX, f)

    def tile_array(self, i: int) -> np.ndarray:
        if not 0 <= i < self.n_tiles:
            raise IndexError(f"tile {i} is outside 0..{self.n_tiles - 1}")
        r, c = divmod(int(i), self.cols)
        return self.rgb[r * TILE_PX:(r + 1) * TILE_PX, c * TILE_PX:(c + 1) * TILE_PX]

    def tile_image(self, i: int):
        """The 64 x 64 RGB chip for tile i (what the image encoder sees)."""
        from PIL import Image

        return Image.fromarray(np.ascontiguousarray(self.tile_array(i)))


# --------------------------------------------------------------------------- small helpers
def available() -> str | None:
    """None if the explorer can run here, else a plain message saying what is missing."""
    try:
        import pyproj  # noqa: F401
        import rasterio  # noqa: F401
        import requests  # noqa: F401
        import shapely  # noqa: F401
    except ImportError as e:
        return f"The map explorer needs the optional packages in requirements.txt (missing: {e.name})."
    return None


def _need_deps() -> None:
    msg = available()
    if msg:
        raise GeoError(msg)


def _check_bbox(bbox) -> tuple:
    try:
        w, s, e, n = (float(v) for v in bbox)
    except (TypeError, ValueError):
        raise GeoError("That box isn't valid. Drag a new one on the map.") from None
    if not all(math.isfinite(v) for v in (w, s, e, n)) or e <= w or n <= s:
        raise GeoError("That box isn't valid. Drag a new one on the map.")
    if w < -180 or e > 180:
        raise GeoError("Boxes across the date line aren't supported. Draw it on one side.")
    if s < -80 or n > 84:
        raise GeoError("Sentinel-2 tiles are only mapped between 80°S and 84°N.")
    return (w, s, e, n)


def _utm_epsg(lon: float, lat: float) -> int:
    zone = min(max(int((lon + 180) // 6) + 1, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


def _parse_dt(text: str) -> datetime:
    t = text.strip().replace("Z", "+00:00")
    m = re.match(r"^(.*?T\d\d:\d\d:\d\d)(\.\d+)?(.*)$", t)
    if m:  # normalise fractional seconds to 6 digits (older Pythons are strict)
        frac = (m.group(2) or ".0")[:7].ljust(7, "0")
        t = m.group(1) + frac + (m.group(3) or "+00:00")
    dt = datetime.fromisoformat(t)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _nice_date(dt: datetime) -> str:
    return f"{dt.day} {dt:%b %Y}"


def _nice_platform(p: str) -> str:
    p = (p or "").lower()
    if p.startswith("sentinel-2"):
        return "Sentinel-2" + p[len("sentinel-2"):].upper()
    if p.startswith("landsat-"):
        return "Landsat " + p.split("-", 1)[1]
    return p or "satellite"


def _item_epsg(props: dict) -> int:
    v = props.get("proj:epsg")
    if v:
        return int(v)
    m = re.search(r"(\d{4,6})$", str(props.get("proj:code", "")))
    if m:
        return int(m.group(1))
    raise GeoError("That Sentinel-2 scene has no map projection listed.")


def _in_season(month: int, lat: float) -> bool:
    """Leaf-on months for the hemisphere; no season rule within 15 degrees of the equator."""
    if abs(lat) < 15:
        return True
    return month in (6, 7, 8, 9) if lat > 0 else month in (12, 1, 2, 3)


def _dn_offset(props: dict) -> int:
    """Sentinel-2 processing baseline >= 04.00 adds 1000 to every DN. Earth Search removes it again in
    its COGs and says so with earthsearch:boa_offset_applied; only subtract when that has not happened."""
    if props.get("earthsearch:boa_offset_applied") is True:
        return 0
    try:
        baseline = float(str(props.get("s2:processing_baseline", "0")))
    except ValueError:
        baseline = 0.0
    return 1000 if baseline >= 4.0 else 0


def _when_range(when, now: datetime | None = None) -> tuple:
    """(start, end, month_fixed). when: None | 'YYYY' | 'YYYY-MM'."""
    now = now or datetime.now(timezone.utc)
    if when in (None, "", "latest"):
        return now - timedelta(days=365 * SEARCH_YEARS), now, False
    m = re.fullmatch(r"(\d{4})(?:-(\d{2}))?", str(when).strip())
    if not m or (m.group(2) and not 1 <= int(m.group(2)) <= 12) or not 2015 <= int(m.group(1)) <= now.year:
        raise GeoError("Pick a year like 2024 or a month like 2024-07 (Sentinel-2 starts in 2015).")
    y = int(m.group(1))
    if m.group(2):
        mo = int(m.group(2))
        end = datetime(y + (mo == 12), mo % 12 + 1, 1, tzinfo=timezone.utc)
        return datetime(y, mo, 1, tzinfo=timezone.utc), end, True
    return datetime(y, 1, 1, tzinfo=timezone.utc), datetime(y + 1, 1, 1, tzinfo=timezone.utc), False


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- grid maths
def _snap(bbox: tuple, epsg: int, origin: tuple = (0.0, 0.0)) -> tuple:
    """Snapped pixel window for a lon/lat box in a UTM CRS.

    Envelope of the box in the CRS -> snapped outwards to the 20 m grid (still on the native 10 m
    grid, and it keeps the 20 m SCL band aligned) -> grown symmetrically to whole 64 px tiles.
    Returns (x0, y1, rows, cols): top-left corner in metres and the tile counts.
    """
    from pyproj import Transformer

    w, s, e, n = bbox
    k = 9
    lons = np.concatenate([np.linspace(w, e, k), np.full(k, e), np.linspace(e, w, k), np.full(k, w)])
    lats = np.concatenate([np.full(k, n), np.linspace(n, s, k), np.full(k, s), np.linspace(s, n, k)])
    xs, ys = Transformer.from_crs(4326, epsg, always_xy=True).transform(lons, lats)
    ox, oy = origin
    step = 2 * GSD_M
    px0 = 2 * math.floor((float(np.min(xs)) - ox) / step + 1e-9)
    px1 = 2 * math.ceil((float(np.max(xs)) - ox) / step - 1e-9)
    py0 = 2 * math.floor((float(np.min(ys)) - oy) / step + 1e-9)
    py1 = 2 * math.ceil((float(np.max(ys)) - oy) / step - 1e-9)
    cols = max(1, math.ceil((px1 - px0) / TILE_PX))
    rows = max(1, math.ceil((py1 - py0) / TILE_PX))
    px0 -= ((cols * TILE_PX - (px1 - px0)) // 4) * 2  # half the padding, kept even
    py1 += ((rows * TILE_PX - (py1 - py0)) // 4) * 2
    return (ox + px0 * GSD_M, oy + py1 * GSD_M, rows, cols)


def grid_estimate(bbox) -> dict:
    """Tile grid a box would get, with no network: the same snapping as fetch_region, in the UTM zone
    of the box centre (a scene delivered in a neighbouring zone can differ by one row or column)."""
    _need_deps()
    w, s, e, n = bbox = _check_bbox(bbox)
    epsg = _utm_epsg((w + e) / 2, (s + n) / 2)
    _, _, rows, cols = _snap(bbox, epsg)
    lat = math.radians((s + n) / 2)
    return {
        "rows": rows, "cols": cols, "n_tiles": rows * cols,
        "width_km": round(cols * TILE_M / 1000, 2), "height_km": round(rows * TILE_M / 1000, 2),
        "epsg": epsg,
        "bbox_width_km": round((e - w) * 111.32 * math.cos(lat), 2), "bbox_height_km": round((n - s) * 110.57, 2),
    }


def _tile_table(epsg: int, x0: float, y1: float, rows: int, cols: int) -> tuple:
    """(window corners, per-tile dicts) in lon/lat. Corner order is TL, TR, BR, BL; tiles are row-major
    from the north-west."""
    from pyproj import Transformer

    tf = Transformer.from_crs(epsg, 4326, always_xy=True)
    gx = x0 + np.arange(cols + 1) * TILE_M
    gy = y1 - np.arange(rows + 1) * TILE_M
    X, Y = np.meshgrid(gx, gy)
    lon, lat = tf.transform(X, Y)
    cx, cy = np.meshgrid(gx[:-1] + TILE_M / 2, gy[:-1] - TILE_M / 2)
    clon, clat = tf.transform(cx, cy)

    def pt(r, c):
        return [round(float(lon[r, c]), 6), round(float(lat[r, c]), 6)]

    tiles = []
    for r in range(rows):
        for c in range(cols):
            tiles.append({
                "i": r * cols + c, "row": r, "col": c,
                "corners": [pt(r, c), pt(r, c + 1), pt(r + 1, c + 1), pt(r + 1, c)],
                "center": [round(float(clon[r, c]), 6), round(float(clat[r, c]), 6)],
                "nodata_frac": 0.0, "cloud_frac": None,
            })
    return [pt(0, 0), pt(0, cols), pt(rows, cols), pt(rows, 0)], tiles


def _block_mean(a: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Mean of each tile block, ignoring NaN. [rows*k, cols*k] -> [rows, cols]; all-NaN blocks -> NaN."""
    k = a.shape[0] // rows
    blocks = a[:rows * k, :cols * k].reshape(rows, k, cols, k).astype(np.float64)
    ok = np.isfinite(blocks)
    cnt = ok.sum(axis=(1, 3))
    tot = np.where(ok, blocks, 0.0).sum(axis=(1, 3))
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan).astype(np.float32)


def _block_frac(mask: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Fraction of True pixels in each tile block."""
    k = mask.shape[0] // rows
    return mask[:rows * k, :cols * k].reshape(rows, k, cols, k).mean(axis=(1, 3)).astype(np.float32)


# --------------------------------------------------------------------------- network: catalogues
def _stac_search(base: str, body: dict, what: str, max_items: int = 1500) -> list:
    import requests

    feats, url, payload = [], f"{base}/search", dict(body)
    limit = int(body.get("limit", 100))
    for _ in range(8):
        try:
            r = requests.post(url, json=payload, timeout=HTTP_TIMEOUT, headers=HEADERS)
            r.raise_for_status()
            data = r.json()
        except (requests.RequestException, ValueError) as e:
            raise GeoError(f"Couldn't reach the {what}. Check the internet connection.", network=True) from e
        page = data.get("features") or []
        feats += page
        nxt = next((l for l in data.get("links", []) if l.get("rel") == "next"), None)
        if not nxt or len(page) < limit or len(feats) >= max_items or str(nxt.get("method", "GET")).upper() != "POST":
            break
        payload = {**payload, **nxt.get("body", {})} if nxt.get("merge") else nxt.get("body", payload)
        url = nxt.get("href", url)
    return feats


def _get_item(scene_id: str) -> dict:
    import requests

    try:
        r = requests.get(f"{S2_STAC}/collections/{S2_COLLECTION}/items/{scene_id}", timeout=HTTP_TIMEOUT, headers=HEADERS)
        if r.status_code == 404:
            raise GeoError("That Sentinel-2 pass is no longer in the archive.")
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError) as e:
        raise GeoError("Couldn't reach the Sentinel-2 archive. Check the internet connection.", network=True) from e


def _cover_area(bbox: tuple, margin_m: float = TILE_M + 60):
    """The box plus the most the tile snapping can add, as a lon/lat polygon."""
    from shapely.geometry import box

    w, s, e, n = bbox
    dlat = margin_m / 110_570
    dlon = margin_m / (111_320 * max(math.cos(math.radians((s + n) / 2)), 0.1))
    return box(w - dlon, s - dlat, e + dlon, n + dlat)


def _cover_frac(geometry: dict, area) -> float:
    from shapely.geometry import shape

    try:
        g = shape(geometry)
        if not g.is_valid:
            g = g.buffer(0)
        return float(g.intersection(area).area / area.area)
    except Exception:
        return 0.0


def _rank_s2(features: list, bbox: tuple, month_fixed: bool = False) -> list:
    """Order candidate passes: covers the box > least cloud (<1, <5, <10, <20 %) > leaf-on season >
    same UTM zone as the box centre (keeps grid_estimate exact) > most recent."""
    w, s, e, n = bbox
    area, lat = _cover_area(bbox), (s + n) / 2
    zone = _utm_epsg((w + e) / 2, lat)
    scored = []
    for f in features:
        p = f.get("properties", {})
        try:
            dt = _parse_dt(p["datetime"])
        except (KeyError, ValueError):
            continue
        cover = _cover_frac(f.get("geometry"), area)
        if cover <= 0:
            continue
        cloud = float(p.get("eo:cloud_cover", 100) or 0)
        bucket = 0 if cloud < 1 else 1 if cloud < 5 else 2 if cloud < 10 else 3 if cloud < 20 else 4
        full = cover >= 0.999
        try:
            same_zone = _item_epsg(p) == zone
        except GeoError:
            same_zone = False
        scored.append(((0 if full else 1, 0.0 if full else -round(cover, 2), bucket,
                        0 if (month_fixed or _in_season(dt.month, lat)) else 1,
                        0 if same_zone else 1, -dt.timestamp()), f))
    scored.sort(key=lambda t: t[0])
    return [f for _, f in scored]


def _pinned_scene(bbox: tuple) -> str | None:
    w, s, e, n = bbox
    cx, cy = (w + e) / 2, (s + n) / 2
    for (pw, ps, pe, pn), scene_id in PINNED_SCENES:
        if pw <= cx <= pe and ps <= cy <= pn:
            return scene_id
    return None


def _s2_candidates(bbox: tuple, when=None):
    """Full STAC items, best first."""
    seen = set()
    pin = _pinned_scene(bbox) if when in (None, "", "latest") else None
    if pin:
        try:
            item = _get_item(pin)
            if _cover_frac(item.get("geometry"), _cover_area(bbox)) >= 0.999:
                seen.add(pin)
                yield item
        except GeoError as e:
            if e.network:
                raise
    start, end, month_fixed = _when_range(when)
    feats = _stac_search(S2_STAC, {
        "collections": [S2_COLLECTION], "bbox": list(bbox), "datetime": f"{_iso(start)}/{_iso(end)}",
        "query": {"eo:cloud_cover": {"lt": 20}}, "limit": 500,
        "fields": {"include": ["id", "geometry", "properties.datetime", "properties.eo:cloud_cover",
                               "properties.proj:epsg", "properties.proj:code", "properties.platform"],
                   "exclude": ["assets", "links", "stac_extensions", "bbox"]},
    }, "Sentinel-2 archive")
    ranked = _rank_s2(feats, bbox, month_fixed)
    if not ranked and not seen:
        asked = f" in {when}" if when not in (None, "", "latest") else " in the last three years"
        raise GeoError(f"No clear Sentinel-2 pass covers that spot{asked}. Open ocean and the poles aren't imaged; "
                       "try land, or another date.")
    for f in ranked:
        if f["id"] not in seen:
            seen.add(f["id"])
            yield _get_item(f["id"])


def find_s2_scene(bbox, when=None) -> dict:
    """The best Sentinel-2 L2A pass for a lon/lat box, as a STAC item. when: None | 'YYYY' | 'YYYY-MM'."""
    _need_deps()
    bbox = _check_bbox(bbox)
    for item in _s2_candidates(bbox, when):
        return item
    raise GeoError("No clear Sentinel-2 pass covers that spot. Try land, or another date.")


# --------------------------------------------------------------------------- network: pixels
def _read_window(href: str, x0: float, y1: float, width: int, height: int, indexes, env: dict = S2_GDAL_ENV) -> np.ndarray:
    """[bands, height, width] pixels of a remote COG whose top-left corner is (x0, y1) in the file's CRS,
    at the file's own pixel size. Parts outside the file are 0 (no data)."""
    import rasterio
    from rasterio.windows import Window

    def read_part(win):  # own handle per thread: one HTTP connection per internal COG block
        with rasterio.Env(**env), rasterio.open(href) as part:
            return part.read(indexes, window=win)

    try:
        with rasterio.Env(**env), rasterio.open(href) as src:
            t = src.transform
            col, row = int(round((x0 - t.c) / t.a)), int(round((y1 - t.f) / t.e))
            c0, r0 = max(col, 0), max(row, 0)
            c1, r1 = min(col + width, src.width), min(row + height, src.height)
            out = np.zeros((len(indexes), height, width), dtype=src.dtypes[0])
            if c1 <= c0 or r1 <= r0:
                return out
            # Split along the file's internal blocks (1024 px for these COGs) and download them in
            # parallel; on a slow link this roughly halves the wait.
            bh, bw = src.block_shapes[0]
            xs, ys = [c0, c1], [r0, r1]
            if bh >= 256 and bw >= 256:
                xs = [c0, *range((c0 // bw + 1) * bw, c1, bw), c1]
                ys = [r0, *range((r0 // bh + 1) * bh, r1, bh), r1]
            parts = [Window(xa, ya, xb - xa, yb - ya) for ya, yb in zip(ys, ys[1:]) for xa, xb in zip(xs, xs[1:])]
            if len(parts) == 1 or len(parts) > 16:
                out[:, r0 - row:r1 - row, c0 - col:c1 - col] = src.read(indexes, window=Window(c0, r0, c1 - c0, r1 - r0))
                return out
        with ThreadPoolExecutor(max_workers=min(len(parts), 8)) as pool:
            for win, data in zip(parts, pool.map(read_part, parts)):
                rr, cc = int(win.row_off) - row, int(win.col_off) - col
                out[:, rr:rr + int(win.height), cc:cc + int(win.width)] = data
        return out
    except GeoError:
        raise
    except Exception as e:  # rasterio / GDAL / curl
        raise GeoError("Couldn't download the satellite pixels. Check the internet connection and try again.",
                       network=True) from e


def _too_big(n: int, limit: int) -> GeoError:
    return GeoError(f"That area is {n:,} tiles; the limit here is {limit:,}. Draw a smaller box.")


def _build_region(bbox: tuple, item: dict, max_tiles: int, say) -> Region:
    p, assets = item["properties"], item["assets"]
    epsg = _item_epsg(p)
    vis = assets.get("visual")
    if not vis:
        raise GeoError("That Sentinel-2 pass has no true-colour image.")
    tr = vis.get("proj:transform") or [GSD_M, 0, 0.0, 0, -GSD_M, 0.0]
    x0, y1, rows, cols = _snap(bbox, epsg, (float(tr[2]), float(tr[5])))
    if rows * cols > max_tiles:
        raise _too_big(rows * cols, max_tiles)
    dt = _parse_dt(p["datetime"])
    cloud = float(p.get("eo:cloud_cover", 0) or 0)
    say("scene", f"Found a {_nice_platform(p.get('platform'))} pass from {_nice_date(dt)}, {cloud:.1f}% cloud")
    W, H = cols * TILE_PX, rows * TILE_PX
    say("pixels", f"Reading {W:,} x {H:,} pixels")

    def read_scl():
        try:
            return _read_window(assets["scl"]["href"], x0, y1, W // 2, H // 2, [1])[0]
        except Exception:
            return None  # cloud_frac stays None

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_scl = pool.submit(read_scl)
        rgb = np.moveaxis(_read_window(vis["href"], x0, y1, W, H, [1, 2, 3]), 0, -1)
        scl = f_scl.result()
    rgb = np.ascontiguousarray(rgb.astype(np.uint8))

    corners, tiles = _tile_table(epsg, x0, y1, rows, cols)
    nodata = _block_frac(rgb.sum(axis=-1) == 0, rows, cols).ravel()
    cloudy = _block_frac(np.isin(scl, SCL_CLOUD), rows, cols).ravel() if scl is not None else None
    for t in tiles:
        t["nodata_frac"] = round(float(nodata[t["i"]]), 4)
        t["cloud_frac"] = None if cloudy is None else round(float(cloudy[t["i"]]), 4)
    say("tiles", f"Cut {rows * cols:,} tiles of 640 m")

    key = f"{item['id']}|{epsg}|{x0:.1f}|{y1:.1f}|{rows}|{cols}"
    return Region(
        id=hashlib.sha1(key.encode()).hexdigest()[:12], bbox=tuple(bbox),
        scene={"id": item["id"], "datetime": p["datetime"], "cloud_cover": round(cloud, 3),
               "platform": p.get("platform", ""), "epsg": epsg, "collection": S2_COLLECTION, "source": S2_SOURCE},
        epsg=epsg, transform=(GSD_M, 0.0, x0, 0.0, -GSD_M, y1), rows=rows, cols=cols, rgb=rgb,
        corners=corners, tiles=tiles,
        assets={"dn_offset": _dn_offset(p),
                **{k: assets[k]["href"] for k in ("visual", "red", "green", "nir", "scl") if k in assets}},
        scl=scl,
    )


def _region_quality(region: Region) -> tuple:
    """(missing fraction, cloud fraction) over the whole window."""
    nodata = float(np.mean([t["nodata_frac"] for t in region.tiles]))
    clouds = [t["cloud_frac"] for t in region.tiles if t["cloud_frac"] is not None]
    return nodata, (float(np.mean(clouds)) if clouds else 0.0)


def _fetch_region_online(bbox: tuple, max_tiles: int, when, say) -> Region:
    say("search", "Looking for the clearest Sentinel-2 pass")
    best, best_score = None, None
    for attempt, item in enumerate(_s2_candidates(bbox, when)):
        region = _build_region(bbox, item, max_tiles, say)
        nodata, cloud = _region_quality(region)
        score = 2 * nodata + cloud
        if best is None or score < best_score:
            best, best_score = region, score
        if (nodata <= RETRY_NODATA and cloud <= RETRY_CLOUD) or attempt >= 2:
            break
        say("search", "That pass is cloudy or incomplete over this box; trying the next clearest")
    if best is None:
        raise GeoError("No clear Sentinel-2 pass covers that spot. Try land, or another date.")
    return best


def fetch_region(bbox, max_tiles: int = 400, when=None, progress=None, refresh: bool = False) -> Region:
    """Sentinel-2 true-colour pixels for a lon/lat box, cut into 640 m tiles.

    A box that was fetched before comes straight from the disk cache (instant, works offline); pass
    refresh=True to look for a newer pass. If the network fails, the saved copy is used when there is one.
    """
    _need_deps()
    bbox = _check_bbox(bbox)
    say = progress or (lambda stage, detail: None)
    est = grid_estimate(bbox)
    if est["n_tiles"] > max_tiles:
        raise _too_big(est["n_tiles"], max_tiles)  # before any network call
    cached = cached_region_for_bbox(bbox, when)
    if cached is not None and cached.n_tiles > max_tiles:
        cached = None
    if cached is not None and not refresh:
        say("cache", f"Using the saved Sentinel-2 pass from {_nice_date(_parse_dt(cached.scene['datetime']))}")
        return cached
    try:
        region = _fetch_region_online(bbox, max_tiles, when, say)
    except GeoError as e:
        if e.network and cached is not None:
            say("cache", "Offline: using the saved copy of this area")
            return cached
        raise
    _save_region(region, _request_key(bbox, when))
    return region


# --------------------------------------------------------------------------- NDVI / NDWI
def _reflectance(dn: np.ndarray, offset: int) -> np.ndarray:
    r = (dn.astype(np.float32) - offset) / 10000.0
    r = np.clip(r, 0.0, None)
    r[dn == 0] = np.nan
    return r


def _ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b), NaN where undefined."""
    den = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 1e-6, (a - b) / den, np.nan)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _indices_from_bands(red, green, nir, scl, rows: int, cols: int, offset: int = 0) -> dict:
    """NDVI = (nir - red) / (nir + red); NDWI (McFeeters 1996) = (green - nir) / (green + nir); water is
    NDWI > 0. Pixels the scene classification marks as missing, saturated, shadow, cloud, cirrus or snow
    are NaN. Per-tile values use the valid pixels only."""
    red, green, nir = (_reflectance(b, offset) for b in (red, green, nir))
    ndvi, ndwi = _ratio(nir, red), _ratio(green, nir)
    if scl is not None:
        bad = np.isin(scl, SCL_INVALID).repeat(2, axis=0).repeat(2, axis=1)[:ndvi.shape[0], :ndvi.shape[1]]
        ndvi[bad] = np.nan
        ndwi[bad] = np.nan
    valid = np.isfinite(ndvi)
    valid_frac = _block_frac(valid, rows, cols).ravel()
    water = _block_frac(valid & (ndwi > 0), rows, cols).ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        water_frac = np.where(valid_frac > 0, water / np.maximum(valid_frac, 1e-9), np.nan).astype(np.float32)
    return {"ndvi": ndvi, "ndwi": ndwi, "tile_ndvi": _block_mean(ndvi, rows, cols).ravel(),
            "tile_water_frac": water_frac, "valid_frac": valid_frac}


def fetch_indices(region: Region, progress=None) -> dict:
    """Measured greenness and water for a region, from the same Sentinel-2 pass as its picture.

    Returns ndvi, ndwi (float32 [H, W], NaN = invalid), tile_ndvi, tile_water_frac, valid_frac (float32 [N]).
    """
    hit = _extra_get(region.id, "indices") or _load_extra(region.id, "indices")
    if hit is not None:
        return hit
    _need_deps()
    say = progress or (lambda stage, detail: None)
    a = dict(region.assets or {})
    if not all(k in a for k in ("red", "green", "nir")):
        item = _get_item(region.scene["id"])
        a = {"dn_offset": _dn_offset(item["properties"]), **{k: v["href"] for k, v in item["assets"].items() if "href" in v}}
    x0, _, _, y1 = region.bounds
    W, H = region.cols * TILE_PX, region.rows * TILE_PX
    say("indices", "Reading the red, green and near-infrared bands")

    def band(name):
        return _read_window(a[name], x0, y1, W, H, [1])[0]

    def scl_band():
        if region.scl is not None:
            return region.scl
        try:
            return _read_window(a["scl"], x0, y1, W // 2, H // 2, [1])[0]
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(band, n) for n in ("red", "green", "nir")]
        f_scl = pool.submit(scl_band)
        red, green, nir = (f.result() for f in futs)
        scl = f_scl.result()
    out = _indices_from_bands(red, green, nir, scl, region.rows, region.cols, int(a.get("dn_offset", 0)))
    say("indices", f"Measured greenness on {float(out['valid_frac'].mean()):.0%} of the pixels")
    _save_extra(region.id, "indices", out)
    return out


# --------------------------------------------------------------------------- land-surface temperature
def _pc_sign(href: str) -> str:
    """Planetary Computer blobs need a short-lived SAS token; one collection token is reused until it expires."""
    import requests

    now = datetime.now(timezone.utc)
    with _LOCK:
        tok, exp = _PC_TOKEN.get("token"), _PC_TOKEN.get("expiry")
    if tok and exp and exp - now > timedelta(minutes=5):
        return f"{href}?{tok}"
    try:
        r = requests.get(f"{PC_SAS}/token/{LANDSAT_COLLECTION}", timeout=HTTP_TIMEOUT, headers=HEADERS)
        r.raise_for_status()
        d = r.json()
        with _LOCK:
            _PC_TOKEN.update(token=d["token"], expiry=_parse_dt(d["msft:expiry"]))
        return f"{href}?{d['token']}"
    except Exception:
        pass
    try:
        r = requests.get(f"{PC_SAS}/sign", params={"href": href}, timeout=HTTP_TIMEOUT, headers=HEADERS)
        r.raise_for_status()
        return r.json()["href"]
    except (requests.RequestException, ValueError, KeyError) as e:
        raise GeoError("Couldn't get access to the Landsat archive. Check the internet connection.", network=True) from e


def _landsat_search(poly, start: datetime, end: datetime) -> list:
    w, s, e, n = poly.bounds
    return _stac_search(PC_STAC, {
        "collections": [LANDSAT_COLLECTION], "bbox": [w, s, e, n], "datetime": f"{_iso(start)}/{_iso(end)}",
        "query": {"eo:cloud_cover": {"lt": 15}, "platform": {"in": ["landsat-8", "landsat-9"]}}, "limit": 100,
    }, "Landsat archive", max_items=300)


def _usable_landsat(features: list, poly) -> list:
    """Daytime passes with a surface-temperature band that cover the whole region."""
    out = []
    for f in features:
        a, p = f.get("assets", {}), f.get("properties", {})
        if "lwir11" not in a or "qa_pixel" not in a:
            continue
        sun = p.get("view:sun_elevation")
        if sun is not None and float(sun) <= 0:
            continue
        if _cover_frac(f.get("geometry"), poly) >= 0.999:
            out.append(f)
    return out


def _rank_landsat(region: Region, say) -> list:
    """Closest pass in time to the Sentinel-2 picture: within 60 days first, then the same time of year
    in the two years either side."""
    from shapely.geometry import Polygon

    poly = Polygon(region.corners)
    s2 = _parse_dt(region.scene["datetime"])
    say("lst-search", "Looking for a clear Landsat pass with surface temperature")
    near = _usable_landsat(_landsat_search(poly, s2 - timedelta(days=60), s2 + timedelta(days=60)), poly)
    if near:
        return sorted(near, key=lambda f: (abs((_parse_dt(f["properties"]["datetime"]) - s2).total_seconds()),
                                           f["properties"].get("eo:cloud_cover", 0)))
    now = datetime.now(timezone.utc)
    spans = []
    for k in (-1, 1, -2, 2):
        mid = s2 + timedelta(days=365.25 * k)
        if mid - timedelta(days=45) < now:
            spans.append((mid - timedelta(days=45), min(mid + timedelta(days=45), now)))
    with ThreadPoolExecutor(max_workers=4) as pool:
        pages = list(pool.map(lambda sp: _landsat_search(poly, *sp), spans))
    wide = _usable_landsat([f for page in pages for f in page], poly)

    def key(f):
        dt = _parse_dt(f["properties"]["datetime"])
        doy = abs(dt.timetuple().tm_yday - s2.timetuple().tm_yday)
        return (min(doy, 365 - doy), abs(dt.year - s2.year))

    return sorted(wide, key=key)


def _read_lst(region: Region, item: dict) -> tuple:
    """(lst_c [rows*20, cols*20] in deg C with NaN, fraction of valid pixels) for one Landsat pass."""
    import rasterio
    from rasterio.transform import Affine
    from rasterio.warp import Resampling, reproject, transform_bounds
    from rasterio.windows import Window

    hrefs = {k: _pc_sign(item["assets"][k]["href"]) for k in ("lwir11", "qa_pixel")}
    x0, y0, x1, y1 = region.bounds
    dst_crs = f"EPSG:{region.epsg}"

    def read(key):
        with rasterio.Env(**GDAL_ENV), rasterio.open(hrefs[key]) as src:
            b = transform_bounds(dst_crs, src.crs, x0, y0, x1, y1, densify_pts=21)
            t = src.transform
            c0 = max(int(math.floor((b[0] - t.c) / t.a)) - 3, 0)
            c1 = min(int(math.ceil((b[2] - t.c) / t.a)) + 3, src.width)
            r0 = max(int(math.floor((b[3] - t.f) / t.e)) - 3, 0)
            r1 = min(int(math.ceil((b[1] - t.f) / t.e)) + 3, src.height)
            if c1 <= c0 or r1 <= r0:
                raise GeoError("That Landsat pass doesn't cover this area.")
            win = Window(c0, r0, c1 - c0, r1 - r0)
            return src.read(1, window=win), src.window_transform(win), src.crs

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_qa = pool.submit(read, "qa_pixel")
            dn, src_transform, src_crs = read("lwir11")
            qa, _, _ = f_qa.result()
    except GeoError:
        raise
    except Exception as e:
        raise GeoError("Couldn't download the Landsat surface temperature. Check the internet connection and try again.",
                       network=True) from e

    celsius = dn.astype(np.float32) * LST_SCALE + LST_OFFSET - 273.15
    bad = dn == 0
    for bit in LANDSAT_BAD_BITS:
        bad |= (qa >> bit) & 1 == 1
    celsius[bad] = np.nan

    h, w = region.rows * LST_PX_PER_TILE, region.cols * LST_PX_PER_TILE
    dst = np.full((h, w), np.nan, dtype=np.float32)
    with rasterio.Env():
        reproject(celsius, dst, src_transform=src_transform, src_crs=src_crs, src_nodata=np.nan,
                  dst_transform=Affine(LST_GSD_M, 0, x0, 0, -LST_GSD_M, y1), dst_crs=dst_crs, dst_nodata=np.nan,
                  resampling=Resampling.bilinear)
    return dst, float(np.isfinite(dst).mean())


def fetch_lst(region: Region, progress=None) -> dict:
    """Measured land-surface temperature for a region from the Landsat 8/9 pass closest in time to its
    Sentinel-2 picture.

    Returns lst_c (float32 deg C on a 32 m grid over the same footprint, [rows*20, cols*20], NaN = cloud or
    missing), tile_lst_c (float32 [N], NaN where under half the tile is valid), scene, days_from_s2.
    """
    hit = _extra_get(region.id, "lst") or _load_extra(region.id, "lst")
    if hit is not None:
        return hit
    _need_deps()
    say = progress or (lambda stage, detail: None)
    s2 = _parse_dt(region.scene["datetime"])
    best = None
    for attempt, item in enumerate(_rank_landsat(region, say)[:3]):
        p = item["properties"]
        dt = _parse_dt(p["datetime"])
        days = (dt.date() - s2.date()).days
        apart = "the same day as" if days == 0 else f"{abs(days)} day{'s' if abs(days) != 1 else ''} {'after' if days > 0 else 'before'}"
        say("lst-scene", f"Found a {_nice_platform(p.get('platform'))} pass from {_nice_date(dt)}, {apart} the Sentinel-2 picture")
        say("lst-pixels", "Reading surface temperature")
        lst, valid = _read_lst(region, item)
        if best is None or valid > best[2]:
            best = (item, lst, valid, days)
        if valid >= 0.8:
            break
    if best is None or best[2] <= 0:
        raise GeoError("No clear Landsat pass with surface temperature covers this area near that date.")
    item, lst, valid, days = best
    p = item["properties"]
    tile = _block_mean(lst, region.rows, region.cols)
    tile[_block_frac(np.isfinite(lst), region.rows, region.cols) < 0.5] = np.nan
    out = {
        "lst_c": lst, "tile_lst_c": tile.ravel(),
        "scene": {"id": item["id"], "datetime": p["datetime"], "cloud_cover": round(float(p.get("eo:cloud_cover", 0) or 0), 3),
                  "platform": p.get("platform", ""), "source": LST_SOURCE},
        "days_from_s2": int(days),
    }
    _save_extra(region.id, "lst", out)
    return out


# --------------------------------------------------------------------------- rendering
# (position, R, G, B, A) stops, linearly interpolated.
PALETTES = {
    "ndvi": [(0.00, 140, 90, 43, 255), (0.22, 201, 166, 107, 255), (0.42, 241, 236, 208, 255),
             (0.60, 169, 207, 122, 255), (0.80, 78, 154, 63, 255), (1.00, 15, 77, 36, 255)],
    "heat": [(0.00, 29, 59, 115, 255), (0.25, 75, 143, 179, 255), (0.50, 247, 239, 193, 255),
             (0.70, 242, 153, 74, 255), (0.86, 209, 73, 46, 255), (1.00, 122, 20, 32, 255)],
    "match": [(0.00, 11, 61, 30, 0), (0.35, 60, 140, 50, 70), (0.70, 150, 215, 60, 170), (1.00, 200, 255, 61, 235)],
}


def _lut(palette: str) -> np.ndarray:
    if palette not in PALETTES:
        raise ValueError(f"unknown palette {palette!r}; choose from {sorted(PALETTES)}")
    stops = np.array(PALETTES[palette], dtype=np.float64)
    x = np.linspace(0, 1, 256)
    return np.stack([np.interp(x, stops[:, 0], stops[:, k]) for k in range(1, 5)], axis=1).round().astype(np.uint8)


def palette_stops(palette: str, n: int = 9) -> list:
    """[(position 0..1, '#rrggbb', alpha 0..1)] for drawing a matching legend."""
    lut = _lut(palette)
    out = []
    for i in range(n):
        r, g, b, a = lut[round(i * 255 / (n - 1))]
        out.append((round(i / (n - 1), 3), f"#{r:02x}{g:02x}{b:02x}", round(a / 255, 3)))
    return out


def colorize(arr, vmin: float, vmax: float, palette: str, scale: int = 1) -> bytes:
    """A 2-D array as an RGBA PNG. NaN is transparent. scale > 1 repeats pixels (crisp upscaling)."""
    from PIL import Image

    a = np.asarray(arr, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError("colorize needs a 2-D array")
    lut = _lut(palette)
    span = float(vmax) - float(vmin)
    with np.errstate(invalid="ignore"):
        t = np.clip((a - float(vmin)) / (span if span else 1.0), 0.0, 1.0)
    nan = ~np.isfinite(t)
    rgba = lut[np.where(nan, 0, np.round(t * 255)).astype(np.uint8)]
    rgba[nan] = 0
    if scale > 1:
        rgba = rgba.repeat(int(scale), axis=0).repeat(int(scale), axis=1)
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgba)).save(buf, format="PNG", compress_level=4)
    return buf.getvalue()


def rgb_png(region: Region) -> bytes:
    """True-colour PNG of the whole region (north-west pixel first; place it with region.corners)."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(region.rgb)).save(buf, format="PNG", compress_level=4)
    return buf.getvalue()


# --------------------------------------------------------------------------- cache
def _request_key(bbox: tuple, when=None) -> str:
    return ",".join(f"{float(v):.4f}" for v in bbox) + "|" + ("" if when in (None, "", "latest") else str(when))


def _valid_id(region_id) -> bool:
    return isinstance(region_id, str) and re.fullmatch(r"[0-9a-f]{12}", region_id) is not None


def _atomic_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(tmp, "wb") as f:
            writer(f)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _remember(region: Region) -> None:
    with _LOCK:
        _MEM[region.id] = region
        _MEM.move_to_end(region.id)
        while len(_MEM) > MEM_REGIONS:
            old, _ = _MEM.popitem(last=False)
            _MEM_EXTRA.pop(old, None)


def _read_index() -> dict:
    try:
        return json.loads((CACHE_DIR / "index.json").read_text())
    except (OSError, ValueError):
        return {}


def _save_region(region: Region, request_key: str | None = None) -> None:
    """Keep a region in memory and on disk; optionally remember which request produced it."""
    _remember(region)
    meta = {k: getattr(region, k) for k in ("id", "scene", "epsg", "rows", "cols", "corners", "tiles", "assets")}
    meta.update(bbox=list(region.bbox), transform=list(region.transform))
    arrays = {"rgb": region.rgb, "meta": np.array(json.dumps(meta))}
    if region.scl is not None:
        arrays["scl"] = region.scl
    try:
        _atomic_write(CACHE_DIR / region.id / "region.npz", lambda f: np.savez_compressed(f, **arrays))
        if request_key:
            with _LOCK:
                index = _read_index()
                index[request_key] = region.id
                _atomic_write(CACHE_DIR / "index.json", lambda f: f.write(json.dumps(index, indent=1).encode()))
    except OSError:
        pass  # a read-only disk must not break the live path


def load_cached(region_id: str) -> Region | None:
    """A previously fetched region by id (memory first, then disk), or None."""
    if not _valid_id(region_id):
        return None
    with _LOCK:
        if region_id in _MEM:
            _MEM.move_to_end(region_id)
            return _MEM[region_id]
    path = CACHE_DIR / region_id / "region.npz"
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            region = Region(
                id=meta["id"], bbox=tuple(meta["bbox"]), scene=meta["scene"], epsg=int(meta["epsg"]),
                transform=tuple(meta["transform"]), rows=int(meta["rows"]), cols=int(meta["cols"]),
                rgb=z["rgb"], corners=meta["corners"], tiles=meta["tiles"], assets=meta.get("assets", {}),
                scl=z["scl"] if "scl" in z.files else None,
            )
    except (OSError, ValueError, KeyError):
        return None
    _remember(region)
    return region


def cached_region_for_bbox(bbox, when=None) -> Region | None:
    """Offline path: the region a previous identical request (same box, same date) produced, or None."""
    try:
        key = _request_key(_check_bbox(bbox), when)
    except GeoError:
        return None
    with _LOCK:
        region_id = _read_index().get(key)
    return load_cached(region_id) if region_id else None


def list_cached() -> list:
    """Saved requests, newest file first: [{'id', 'bbox', 'when', 'scene', 'rows', 'cols', 'layers'}]."""
    with _LOCK:
        index = _read_index()
    out = []
    for key, region_id in index.items():
        path = CACHE_DIR / str(region_id) / "region.npz"
        if not _valid_id(region_id) or not path.exists():
            continue
        try:
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"]))
        except (OSError, ValueError, KeyError):
            continue
        box, _, when = key.partition("|")
        out.append({"id": region_id, "bbox": [float(v) for v in box.split(",")], "when": when or None,
                    "scene": meta["scene"], "rows": meta["rows"], "cols": meta["cols"],
                    "layers": [n for n in ("indices", "lst") if (path.parent / f"{n}.npz").exists()],
                    "_mtime": path.stat().st_mtime})
    out.sort(key=lambda d: -d.pop("_mtime"))
    return out


_EXTRA_ARRAYS = {"indices": ("ndvi", "ndwi", "tile_ndvi", "tile_water_frac", "valid_frac"), "lst": ("lst_c", "tile_lst_c")}


def _extra_get(region_id: str, name: str):
    with _LOCK:
        return _MEM_EXTRA.get(region_id, {}).get(name)


def _save_extra(region_id: str, name: str, data: dict) -> None:
    with _LOCK:
        _MEM_EXTRA.setdefault(region_id, {})[name] = data
    arrays = {k: np.asarray(data[k], dtype=np.float32) for k in _EXTRA_ARRAYS[name]}
    arrays["meta"] = np.array(json.dumps({k: v for k, v in data.items() if k not in arrays}))
    if not _valid_id(region_id):
        return
    try:
        _atomic_write(CACHE_DIR / region_id / f"{name}.npz", lambda f: np.savez_compressed(f, **arrays))
    except OSError:
        pass


def _load_extra(region_id: str, name: str):
    if not _valid_id(region_id):
        return None
    path = CACHE_DIR / region_id / f"{name}.npz"
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            data = {k: z[k].astype(np.float32) for k in _EXTRA_ARRAYS[name]}
            data.update(json.loads(str(z["meta"])))
    except (OSError, ValueError, KeyError):
        return None
    with _LOCK:
        _MEM_EXTRA.setdefault(region_id, {})[name] = data
    return data
