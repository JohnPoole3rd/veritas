"""T10 — Collect 50 Upper Arlington Sentinel-2 tiles (owner: A).

Reproducible, scripted version of the Copernicus-Browser workflow in
plan/14-upper-arlington-data-collection.md:

1. City boundary: OpenStreetMap relation 181876 (Upper Arlington, Ohio).
2. Imagery: one Sentinel-2 L2A scene (true-colour TCI, native 10 m), read from
   the public Copernicus Sentinel-2 COG mirror on AWS (Element84 Earth Search).
3. Candidate 64x64-pixel windows (640 m x 640 m) on the native S2 pixel grid
   are scored with objective land-cover evidence:
   - ESA WorldCover 2021 v200 (10 m) — tree / grass / built / water fractions.
   - OpenStreetMap — landuse, parks, golf courses, water bodies, major roads.
4. Tiles are picked greedily per category, never overlapping each other, and
   (except for water, which sits on the western city boundary) fully inside
   the city boundary.

Outputs (paths relative to repo root):
    data/raw/local_tiles/ua_XXX.png             64x64 RGB model input
    data/raw/local_tiles/context/ua_XXX.png     320x320 context crop, tile box drawn
    data/local_tiles/collection_log.csv         per-tile evidence used for selection
    data/local_tiles/manifest_draft.csv         draft manifest (labels finalised in T11)

Run:  python -m src.local_collect
These tiles are an OUT-OF-DISTRIBUTION test set. Never fit or tune on them.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]

SCENE_ID = "S2C_17TLE_20250801_0_L2A"  # 2025-08-01, 0.24 % scene cloud, leaf-on summer
STAC_URL = "https://earth-search.aws.element84.com/v1"
WORLDCOVER_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_N39W084_Map.tif"
)
OSM_RELATION = 181876
OVERPASS_URLS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]
USER_AGENT = "platonic-earth-hackathon/0.1"
BBOX = (39.985, -83.110, 40.070, -83.037)  # S, W, N, E — boundary bbox + margin

TILE = 64  # pixels; 10 m/pixel -> 640 m, same as EuroSAT
# Upper Arlington (~25 km^2) holds only ~61 non-overlapping 640 m tiles in total, so 50
# category-pure tiles cannot all be disjoint. Candidates use a fine stride and each new tile may
# overlap previously chosen tiles by at most MAX_OVERLAP of its area (plan/14 §3: "avoid
# overlapping tiles where possible"). Overlap is recorded per tile in collection_log.csv.
STEP = 8
MAX_OVERLAP = 0.25
CONTEXT = 320  # context crop size in pixels
# The OSM city boundary follows the EAST bank of the Scioto River / Griggs Reservoir, so the
# water itself is outside Upper Arlington. Water tiles therefore straddle the western boundary
# ("immediately adjacent", as allowed by plan/14 §4.5); all other tiles are fully inside.
WATER_MIN_INSIDE = 0.25

BOUNDARY_PATH = ROOT / "data/local_tiles/ua_boundary.geojson"
CACHE = ROOT / "data/raw/local_tiles/_cache"
OUT_TILES = ROOT / "data/raw/local_tiles"
OUT_META = ROOT / "data/local_tiles"

# (local_category, sub-type, default EuroSAT label, count, id range start)
PLAN = [
    ("residential", "residential", "Residential", 10, 1),
    ("commercial_industrial", "commercial", "Industrial", 10, 11),
    ("road_highway", "road", "Highway", 10, 21),
    ("park_forest_herbaceous", "forest", "Forest", 5, 31),
    ("park_forest_herbaceous", "herbaceous", "HerbaceousVegetation", 5, 36),
    # Water: one pick of the 10 most water-rich tiles; River vs SeaLake is decided visually in
    # T11 (the Griggs Reservoir impoundment is channel-shaped at 640 m).
    ("water_river", "water", "River", 10, 41),
]
# Order in which categories claim space: scarcest first, abundant residential last.
PICK_ORDER = ["water", "commercial", "forest", "herbaceous", "road", "residential"]

ROAD_WIDTH_M = {  # approximate paved corridor width
    "motorway": 30, "trunk": 26, "primary": 20, "secondary": 16, "tertiary": 12,
    "motorway_link": 10, "trunk_link": 10, "primary_link": 10, "secondary_link": 9,
}

OVERPASS_QUERY = """
[out:json][timeout:120];
(
  way["landuse"]({b}); relation["landuse"]({b});
  way["leisure"~"park|golf_course|pitch|nature_reserve|recreation_ground"]({b});
  relation["leisure"~"park|golf_course|nature_reserve"]({b});
  way["natural"~"water|wood|scrub|grassland"]({b}); relation["natural"~"water|wood"]({b});
  way["waterway"~"river|riverbank|dam"]({b});
  way["highway"~"motorway|trunk|primary|secondary|tertiary|motorway_link|trunk_link|primary_link|secondary_link"]({b});
  way["amenity"="parking"]({b});
  way["building"]["building"!~"house|detached|residential|garage|shed|semidetached_house|apartments"]({b});
);
out geom tags;
"""


# --------------------------------------------------------------------------- inputs
def _get(url: str, **kw) -> requests.Response:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=180, **kw)
    r.raise_for_status()
    return r


def load_boundary():
    from shapely.geometry import shape

    if not BOUNDARY_PATH.exists():
        BOUNDARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        url = (
            "https://nominatim.openstreetmap.org/lookup?"
            f"osm_ids=R{OSM_RELATION}&format=json&polygon_geojson=1"
        )
        geom = _get(url).json()[0]["geojson"]
        fc = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {
                    "name": "Upper Arlington, Ohio",
                    "source": f"OpenStreetMap relation {OSM_RELATION} via Nominatim",
                    "license": "ODbL (c) OpenStreetMap contributors",
                },
                "geometry": geom,
            }],
        }
        BOUNDARY_PATH.write_text(json.dumps(fc))
    fc = json.loads(BOUNDARY_PATH.read_text())
    return shape(fc["features"][0]["geometry"])


def load_osm() -> list[dict]:
    path = CACHE / "osm.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        b = ",".join(str(x) for x in BBOX)
        q = OVERPASS_QUERY.format(b=b)
        last = None
        for url in OVERPASS_URLS:
            try:
                r = requests.post(url, data={"data": q}, headers={"User-Agent": USER_AGENT}, timeout=240)
                r.raise_for_status()
                path.write_bytes(r.content)
                break
            except Exception as e:  # try next mirror
                last = e
        else:
            raise RuntimeError(f"Overpass failed on all mirrors: {last}")
    return json.loads(path.read_text())["elements"]


def scene_asset_href(scene_id: str) -> tuple[str, str]:
    from pystac_client import Client

    item = next(Client.open(STAC_URL).search(ids=[scene_id], collections=["sentinel-2-l2a"]).items())
    return item.assets["visual"].href, item.properties["datetime"][:10]


# --------------------------------------------------------------------------- rasters
def read_scene(href: str, boundary_utm, crs, margin_m=1000):
    """Read the TCI window around the city, snapped to the native 10 m grid."""
    import rasterio
    from rasterio.windows import from_bounds

    with rasterio.open(href) as src:
        assert src.crs == crs, (src.crs, crs)
        minx, miny, maxx, maxy = boundary_utm.bounds
        win = from_bounds(minx - margin_m, miny - margin_m, maxx + margin_m, maxy + margin_m, src.transform)
        win = win.round_offsets().round_lengths()
        rgb = src.read(window=win)
        transform = src.window_transform(win)
    return np.moveaxis(rgb, 0, -1), transform


def worldcover_on_grid(shape_hw, transform, crs) -> np.ndarray:
    import rasterio
    from rasterio.warp import Resampling, reproject

    out = np.zeros(shape_hw, dtype=np.uint8)
    with rasterio.open(WORLDCOVER_URL) as src:
        from rasterio.warp import transform_bounds
        from rasterio.windows import from_bounds

        h, w = shape_hw
        left, top = transform * (0, 0)
        right, bottom = transform * (w, h)
        b = transform_bounds(crs, src.crs, left, bottom, right, top, densify_pts=21)
        win = from_bounds(b[0] - 0.01, b[1] - 0.01, b[2] + 0.01, b[3] + 0.01, src.transform)
        win = win.round_offsets().round_lengths()
        data = src.read(1, window=win)
        reproject(
            data, out, src_transform=src.window_transform(win), src_crs=src.crs,
            dst_transform=transform, dst_crs=crs, resampling=Resampling.nearest,
        )
    return out


def osm_layers(elements, shape_hw, transform, to_utm) -> dict[str, np.ndarray]:
    """Rasterise OSM evidence onto the scene grid. Returns boolean masks."""
    from rasterio.features import rasterize
    from shapely.geometry import LineString, Polygon
    from shapely.ops import transform as stransform, unary_union

    groups: dict[str, list] = {k: [] for k in
                               ["residential", "commercial", "park", "golf", "wood", "river", "lake", "road"]}

    def ring(geom):
        return [(p["lon"], p["lat"]) for p in geom]

    def polys(e):
        if e["type"] == "way" and "geometry" in e:
            pts = ring(e["geometry"])
            if len(pts) >= 4 and pts[0] == pts[-1]:
                yield Polygon(pts)
        elif e["type"] == "relation":
            outers = [ring(m["geometry"]) for m in e.get("members", [])
                      if m.get("role") == "outer" and "geometry" in m]
            # stitch outer ways into rings
            from shapely.ops import linemerge, polygonize
            lines = [LineString(o) for o in outers if len(o) >= 2]
            if lines:
                for p in polygonize(linemerge(lines) if len(lines) > 1 else lines):
                    yield p

    for e in elements:
        t = e.get("tags", {})
        lu, le, na, hw = t.get("landuse"), t.get("leisure"), t.get("natural"), t.get("highway")
        if hw in ROAD_WIDTH_M and e["type"] == "way" and "geometry" in e:
            line = stransform(to_utm, LineString(ring(e["geometry"])))
            groups["road"].append(line.buffer(ROAD_WIDTH_M[hw] / 2, cap_style=2))
            continue
        key = None
        if lu == "residential":
            key = "residential"
        elif lu in ("commercial", "retail", "industrial") or t.get("amenity") == "parking" or (
            t.get("building") and t["building"] not in ("yes", "roof", "carport", "school", "church")
        ):
            key = "commercial"
        elif le == "golf_course":
            key = "golf"
        elif le in ("park", "pitch", "recreation_ground", "nature_reserve") or lu in (
            "grass", "recreation_ground", "meadow", "cemetery", "village_green"
        ):
            key = "park"
        elif lu == "forest" or na in ("wood", "scrub"):
            key = "wood"
        elif na == "water" or t.get("waterway") == "riverbank":
            key = "river" if (t.get("water") == "river" or t.get("waterway") == "riverbank") else "lake"
        if key:
            for p in polys(e):
                if p.is_valid or p.buffer(0).is_valid:
                    groups[key].append(stransform(to_utm, p.buffer(0) if not p.is_valid else p))

    # Large non-house buildings tagged building=yes also signal commercial footprint.
    for e in elements:
        t = e.get("tags", {})
        if t.get("building") == "yes" and e["type"] == "way" and "geometry" in e:
            pts = ring(e["geometry"])
            if len(pts) >= 4:
                p = stransform(to_utm, Polygon(pts))
                if p.is_valid and p.area > 1500:  # > 1500 m^2 is not a house
                    groups["commercial"].append(p)

    out = {}
    for k, geoms in groups.items():
        if not geoms:
            out[k] = np.zeros(shape_hw, bool)
            continue
        out[k] = rasterize([(unary_union(geoms), 1)], out_shape=shape_hw, transform=transform,
                           fill=0, dtype="uint8").astype(bool)
    return out


def window_sums(mask: np.ndarray) -> np.ndarray:
    """Fraction of each TILExTILE window (top-left anchored) that is True."""
    ii = np.pad(mask.astype(np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    s = ii[TILE:, TILE:] - ii[:-TILE, TILE:] - ii[TILE:, :-TILE] + ii[:-TILE, :-TILE]
    return s / (TILE * TILE)


# --------------------------------------------------------------------------- selection
def build_features(rgb, wc, osm, inside):
    f = {
        "inside": window_sums(inside),
        "wc_tree": window_sums(wc == 10),
        "wc_grass": window_sums((wc == 30) | (wc == 40)),
        "wc_built": window_sums(wc == 50),
        "wc_water": window_sums(wc == 80),
        "osm_residential": window_sums(osm["residential"]),
        "osm_commercial": window_sums(osm["commercial"] & ~osm["road"]),
        "osm_road": window_sums(osm["road"]),
        "osm_park": window_sums(osm["park"] | osm["golf"]),
        "osm_golf": window_sums(osm["golf"]),
        "osm_wood": window_sums(osm["wood"]),
        "osm_river": window_sums(((wc == 80) | osm["river"]) & ~osm["lake"]),
        "osm_lake": window_sums(osm["lake"]),
        "water_any": window_sums((wc == 80) | osm["river"] | osm["lake"]),
        "tree_not_resid": window_sums((wc == 10) & ~osm["residential"]),
        "grass_open": window_sums(((wc == 30) | (wc == 40)) | ((osm["park"] | osm["golf"]) & (wc != 10) & (wc != 50))),
        "nodata": window_sums(rgb.sum(-1) == 0),
    }
    return f


def score(kind: str, f: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (score, eligible) arrays over candidate windows."""
    full = f["inside"] >= 0.999
    clean = f["nodata"] == 0
    if kind == "water":
        return f["water_any"], clean & (f["inside"] >= WATER_MIN_INSIDE)
    if kind == "commercial":
        s = f["osm_commercial"] + f["wc_built"] - f["water_any"] - 0.5 * f["osm_park"]
        return s, clean & full
    if kind == "road":
        s = f["osm_road"] - 0.2 * f["osm_commercial"] - f["water_any"] - 0.2 * f["osm_park"]
        return s, clean & full
    if kind == "forest":
        s = f["wc_tree"] + f["osm_wood"] - 1.5 * f["wc_built"] - f["water_any"] - f["osm_commercial"]
        return s, clean & full
    if kind == "herbaceous":
        s = f["grass_open"] - f["water_any"] - 0.5 * f["osm_commercial"] - 0.3 * f["wc_tree"]
        return s, clean & full
    if kind == "residential":
        other = f["osm_commercial"] + f["osm_park"] + f["water_any"] + f["osm_wood"] + 0.7 * f["osm_road"]
        s = 1.0 - other + 0.2 * f["osm_residential"]
        return s, clean & full
    raise ValueError(kind)


def pick(kind, n, f, taken: np.ndarray, rng) -> list[tuple[int, int, float]]:
    """Greedy pick of the n best-scoring windows. taken counts how many chosen tiles cover each
    pixel; a candidate is rejected if more than MAX_OVERLAP of it is already covered, or if it
    overlaps another tile of the same sub-type at all. Returns (row, col, overlap_frac)."""
    s, ok = score(kind, f)
    H, W = s.shape
    ys, xs = np.mgrid[0:H:STEP, 0:W:STEP]
    ys, xs = ys.ravel(), xs.ravel()
    cand = [(s[y, x] + rng.uniform(0, 1e-6), y, x) for y, x in zip(ys, xs) if ok[y, x]]
    cand.sort(reverse=True)
    chosen: list[tuple[int, int, float]] = []
    for _, y, x in cand:
        ov = float((taken[y:y + TILE, x:x + TILE] > 0).mean())
        if ov > MAX_OVERLAP:
            continue
        if any(abs(y - cy) < TILE and abs(x - cx) < TILE for cy, cx, _ in chosen):
            continue
        chosen.append((y, x, ov))
        taken[y:y + TILE, x:x + TILE] += 1
        if len(chosen) == n:
            break
    return chosen


# --------------------------------------------------------------------------- main
def main(argv=None):
    import rasterio.features
    from PIL import Image, ImageDraw
    from pyproj import Transformer
    from shapely.ops import transform as stransform

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scene", default=SCENE_ID)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)

    crs = "EPSG:32617"  # UTM 17N, native CRS of MGRS tile 17TLE
    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform

    boundary = load_boundary()
    boundary_utm = stransform(to_utm, boundary)
    href, date = scene_asset_href(args.scene)
    print(f"scene {args.scene} date {date}\n  {href}")
    from rasterio.crs import CRS
    rgb, transform = read_scene(href, boundary_utm, CRS.from_string(crs))
    H, W = rgb.shape[:2]
    print(f"scene window {W}x{H} px")

    wc = worldcover_on_grid((H, W), transform, crs)
    osm = osm_layers(load_osm(), (H, W), transform, to_utm)
    inside = rasterio.features.rasterize([(boundary_utm, 1)], out_shape=(H, W), transform=transform,
                                         fill=0, dtype="uint8").astype(bool)
    f = build_features(rgb, wc, osm, inside)

    plan = {p[1]: p for p in PLAN}
    taken = np.zeros((H, W), np.int16)
    picks = {}
    for kind in PICK_ORDER:
        picks[kind] = pick(kind, plan[kind][3], f, taken, rng)
        print(f"{kind:12s} picked {len(picks[kind])}/{plan[kind][3]}")

    OUT_TILES.mkdir(parents=True, exist_ok=True)
    (OUT_TILES / "context").mkdir(exist_ok=True)
    OUT_META.mkdir(parents=True, exist_ok=True)
    rows, manifest = [], []
    for cat, kind, label, n, start in PLAN:
        for i, (y, x, ov) in enumerate(picks[kind]):
            tid = f"ua_{start + i:03d}"
            tile = rgb[y:y + TILE, x:x + TILE]
            Image.fromarray(tile).save(OUT_TILES / f"{tid}.png")
            # context crop with the tile outlined
            c0y, c0x = max(0, y - (CONTEXT - TILE) // 2), max(0, x - (CONTEXT - TILE) // 2)
            ctx = Image.fromarray(rgb[c0y:c0y + CONTEXT, c0x:c0x + CONTEXT].copy())
            ImageDraw.Draw(ctx).rectangle([x - c0x, y - c0y, x - c0x + TILE - 1, y - c0y + TILE - 1],
                                          outline=(255, 0, 0), width=1)
            ctx.save(OUT_TILES / "context" / f"{tid}.png")
            cx, cy = transform * (x + TILE / 2, y + TILE / 2)
            lon, lat = to_ll(cx, cy)
            stats = {k: round(float(v[y, x]), 3) for k, v in f.items()}
            rows.append({"tile_id": tid, "local_category": cat, "subtype": kind, "lat": round(lat, 5),
                         "lon": round(lon, 5), "utm_x": round(cx, 1), "utm_y": round(cy, 1),
                         "scene_id": args.scene, "date": date, "overlap": round(ov, 3), "score": round(float(score(kind, f)[0][y, x]), 3),
                         **stats})
            manifest.append({"tile_id": tid, "path": f"data/raw/local_tiles/{tid}.png", "label": label,
                             "local_category": cat, "lat": round(lat, 5), "lon": round(lon, 5), "date": date,
                             "source": "Copernicus", "notes": ""})
    pd.DataFrame(rows).to_csv(OUT_META / "collection_log.csv", index=False)
    pd.DataFrame(manifest).to_csv(OUT_META / "manifest_draft.csv", index=False)
    print(f"wrote {len(manifest)} tiles")


if __name__ == "__main__":
    main()
