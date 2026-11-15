"""Model inference over map tiles, land-cover roll-ups and the climate statistics shown on the
website (owner: D, subagent D2). Pure functions: no web framework, no network.

Two kinds of numbers come out of here, and the site keeps them apart:

- **Model estimates.** A tile's land-cover class is the caption class with the highest cosine
  similarity after the one Procrustes rotation, exactly the rule `RealBackend.classify_image` and
  A's T12 evaluation use (`src.retrieval.class_scores`, agg="max"). Nothing is trained or tuned here.
- **Measurements.** NDVI, NDWI water fraction and land-surface temperature arrive as arrays from
  `web/geo.py`; `climate_stats` only summarises them (ordinary least squares, group means, ranks).

Everything returned for the API passes through `_clean`, so NaN never reaches JSON.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
from typing import Any, Callable, Sequence

import numpy as np

TILE_KM2 = 0.4096  # one 64 x 64 px tile at 10 m per pixel = 640 m x 640 m

# Order must match src.local_tiles.EUROSAT_CLASSES (checked in tests/test_web_server.py).
# `ipcc` is the IPCC 2006 Guidelines (Vol. 4, Ch. 3) land-use category the class rolls up to;
# `group` is the coarse surface type used for the temperature comparison.
CLASS_META: list[dict[str, str]] = [
    {"name": "AnnualCrop", "label": "Annual crop", "color": "#E3B23C", "ipcc": "Cropland", "group": "Cropland"},
    {"name": "Forest", "label": "Forest", "color": "#1E6B45", "ipcc": "Forest land", "group": "Vegetated"},
    {"name": "HerbaceousVegetation", "label": "Herbaceous vegetation", "color": "#6DB36A", "ipcc": "Grassland",
     "group": "Vegetated"},
    {"name": "Highway", "label": "Highway / road", "color": "#7B7F86", "ipcc": "Settlements", "group": "Built"},
    {"name": "Industrial", "label": "Industrial / commercial", "color": "#B4483A", "ipcc": "Settlements",
     "group": "Built"},
    {"name": "Pasture", "label": "Pasture", "color": "#B7D36B", "ipcc": "Grassland", "group": "Vegetated"},
    {"name": "PermanentCrop", "label": "Permanent crop", "color": "#B97A2B", "ipcc": "Cropland", "group": "Cropland"},
    {"name": "Residential", "label": "Residential", "color": "#E8957A", "ipcc": "Settlements", "group": "Built"},
    {"name": "River", "label": "River", "color": "#3E9BC7", "ipcc": "Wetlands (open water)", "group": "Water"},
    {"name": "SeaLake", "label": "Sea / lake", "color": "#1D5E8C", "ipcc": "Wetlands (open water)", "group": "Water"},
]
CLASS_NAMES: list[str] = [c["name"] for c in CLASS_META]
_META = {c["name"]: c for c in CLASS_META}
_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
GROUP_ORDER = ["Built", "Vegetated", "Cropland", "Water"]
IPCC_ORDER = ["Forest land", "Cropland", "Grassland", "Wetlands (open water)", "Settlements"]

WATER_TILE_FRAC = 0.5      # a tile is "water" when NDWI > 0 on more than half of it
MIN_FIT_TILES = 8          # fewer land tiles than this: no regression, no ranking
INVALID_NODATA_FRAC = 0.5  # tiles with more missing pixels than this are not counted anywhere

Progress = Callable[[str, str], None]


# --------------------------------------------------------------------------- JSON safety
def _clean(obj: Any) -> Any:
    """Plain JSON types only: numpy scalars/arrays become Python numbers/lists, NaN and inf become None."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return [_clean(v) for v in obj.tolist()]
    if isinstance(obj, (list, tuple, set)):
        return [_clean(v) for v in obj]
    return str(obj)


def _r(value, digits: int):
    """Rounded float, or None when the value is missing or not finite."""
    if value is None:
        return None
    v = float(value)
    return round(v, digits) if math.isfinite(v) else None


def _vec(values, n: int) -> np.ndarray:
    """A length-n float64 vector with NaN for missing entries (None, NaN, or no array at all)."""
    if values is None:
        return np.full(n, np.nan)
    a = np.array([np.nan if v is None else v for v in np.asarray(values, dtype=object).ravel()], dtype=np.float64)
    if a.size != n:
        raise ValueError(f"expected {n} per-tile values, got {a.size}")
    return a


def _seed(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16)


# --------------------------------------------------------------------------- model inference
class _TileDataset:
    """Map-style dataset: PIL tiles -> normalised tensors, built batch by batch (not all in memory)."""

    def __init__(self, images, transform):
        self.images, self.transform = images, transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        return self.transform(self.images[i])


def _mock_embed_and_score(backend, images) -> tuple[np.ndarray, np.ndarray]:
    """Rehearsal values for MockBackend: its own colour heuristic for the classes, and deterministic
    pseudo-random unit vectors as 'embeddings'. Nothing here is a model output."""
    n, c = len(images), len(CLASS_NAMES)
    emb = np.zeros((n, 384), np.float32)
    scores = np.zeros((n, c), np.float32)
    for i, im in enumerate(images):
        arr = np.asarray(im.resize((16, 16)), dtype=np.uint8)
        rng = np.random.default_rng(_seed("tile", hashlib.sha256(arr.tobytes()).hexdigest()))
        v = rng.normal(size=384)
        emb[i] = v / max(float(np.linalg.norm(v)), 1e-12)
        ranked = list(backend.classify_image(im)) if hasattr(backend, "classify_image") else []
        floor = min((float(s) for _, s in ranked), default=0.5)
        row = np.minimum(rng.uniform(0.05, 0.30, c), floor - 0.02)
        for label, s in ranked:
            if label in _INDEX:
                row[_INDEX[label]] = float(s)
        scores[i] = row
    return emb, scores


def embed_and_score(backend, images: Sequence, progress: Progress | None = None,
                    batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """(rotated image embeddings float32 [N, 384], class scores float32 [N, 10]) for PIL tiles.

    REAL backend: frozen DINOv2 through `src.data.get_transforms` + `src.encoders.extract_image_embeddings`
    in batches, then `src.alignment.transform_images` and `src.retrieval.class_scores` (agg="max"):
    the same code path as `RealBackend.classify_image`, so one tile gets the same answer either way.
    MOCK backend: fake values (see `_mock_embed_and_score`).

    The caller holds the model lock; this function does no locking of its own.
    """
    images = [im if im.mode == "RGB" else im.convert("RGB") for im in images]
    n = len(images)
    if n == 0:
        return np.zeros((0, 384), np.float32), np.zeros((0, len(CLASS_NAMES)), np.float32)
    if getattr(backend, "name", "MOCK") != "REAL":
        if progress:
            progress("embed", f"Mock mode: placeholder vectors for {n} tiles (no model loaded)")
            progress("rotate", "Mock mode: placeholder scores")
        return _mock_embed_and_score(backend, images)

    from src.alignment import transform_images
    from src.retrieval import class_scores

    if getattr(backend, "_image_encoder", None) is not None and getattr(backend, "_dinov2", None) is None:
        # An injected encoder (tests): there is no DINOv2 to batch through.
        if progress:
            progress("embed", f"Embedding {n} tiles")
        X = np.concatenate([backend._encode_image(im) for im in images])
    else:
        from src.data import get_transforms
        from src.encoders import extract_image_embeddings

        model, device = backend._dinov2_model()
        tf = get_transforms(int(backend.config["image"]["input_size"]), train=False)
        chunks, step = [], batch_size * 2
        for start in range(0, n, step):
            part = images[start:start + step]
            chunks.append(extract_image_embeddings(model, _TileDataset(part, tf), batch_size=batch_size, device=device))
            if progress:
                progress("embed", f"Embedding {n:,} tiles with frozen DINOv2 on {device} · {min(start + step, n):,} of {n:,}")
        X = np.concatenate(chunks)
    if progress:
        progress("rotate", f"Applying the one rotation and scoring {len(backend.caption_labels)} captions")
    aligned = transform_images(X, backend.params)
    return aligned.astype(np.float32), score_classes(backend, aligned)


def score_classes(backend, aligned: np.ndarray) -> np.ndarray:
    """Class scores [N, 10] (columns in CLASS_NAMES order) for already-rotated image embeddings."""
    from src.retrieval import class_scores

    names = list(backend.class_names)
    scores = np.asarray(class_scores(aligned, backend.captions_aligned, backend.caption_labels, names), np.float32)
    if names != CLASS_NAMES:  # keep our column order even if the backend's ever changes
        scores = scores[:, [names.index(c) for c in CLASS_NAMES]]
    return scores


def text_scores(backend, query: str, aligned: np.ndarray) -> np.ndarray:
    """Cosine similarity of every tile to a free-text query (float32 [N]). The caller holds the model lock."""
    n = len(aligned)
    if getattr(backend, "name", "MOCK") != "REAL":
        return np.random.default_rng(_seed("text", query.strip().lower())).uniform(0.05, 0.6, n).astype(np.float32)
    from src.alignment import l2_normalize, transform_texts

    q = transform_texts(backend._encode_text(query), backend.params)[0]
    return (l2_normalize(aligned) @ l2_normalize(q.reshape(1, -1))[0]).astype(np.float32)


def image_scores(aligned: np.ndarray, i: int) -> np.ndarray:
    """Cosine similarity of every tile to tile i, in the rotated space (float32 [N]; tile i itself is 1)."""
    a = np.asarray(aligned, np.float32)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    return (a @ a[int(i)]).astype(np.float32)


def top_k(scores, k: int, valid=None, exclude: int | None = None) -> list[dict]:
    """The k best-scoring tiles as [{"i", "score"}], skipping invalid tiles and (optionally) one index."""
    s = np.asarray(scores, np.float64)
    ok = np.isfinite(s)
    if valid is not None:
        ok &= np.asarray(valid, bool)
    if exclude is not None and 0 <= exclude < len(s):
        ok[exclude] = False
    order = [int(i) for i in np.argsort(-np.where(ok, s, -np.inf), kind="stable") if ok[i]]
    return [{"i": i, "score": round(float(s[i]), 3)} for i in order[:max(int(k), 0)]]


def tile_labels(scores: np.ndarray) -> list[dict]:
    """Per tile: {"cls", "score", "top3": [{"label", "score"} x 3]} from a class-score matrix."""
    out = []
    for row in np.asarray(scores, np.float64):
        order = np.argsort(-row, kind="stable")[:3]
        top3 = [{"label": CLASS_NAMES[j], "score": round(float(row[j]), 3)} for j in order]
        out.append({"cls": top3[0]["label"], "score": top3[0]["score"], "top3": top3})
    return out


def valid_tiles(tiles: Sequence[dict]) -> np.ndarray:
    """False for tiles that are mostly missing pixels (outside the satellite swath)."""
    return np.array([float(t.get("nodata_frac") or 0.0) <= INVALID_NODATA_FRAC for t in tiles], dtype=bool)


# --------------------------------------------------------------------------- land-cover roll-ups
def _rollup(keys: Sequence[str], order: Sequence[str], field: str) -> list[dict]:
    counts: dict[str, int] = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    total = sum(counts.values())
    rank = {name: i for i, name in enumerate(order)}
    rows = sorted(counts.items(), key=lambda kv: (-kv[1], rank.get(kv[0], len(rank))))
    return [{field: k, "n": n, "share": round(n / total, 4), "km2": round(n * TILE_KM2, 2)} for k, n in rows]


def composition(classes: Sequence[str], valid=None) -> list[dict]:
    """[{"cls", "n", "share", "km2"}], largest first, over the tiles that have pixels. Model estimates."""
    keep = np.ones(len(classes), bool) if valid is None else np.asarray(valid, bool)
    return _rollup([c for c, ok in zip(classes, keep) if ok], CLASS_NAMES, "cls")


def ipcc_rollup(classes: Sequence[str], valid=None) -> list[dict]:
    """The same tiles grouped into IPCC land-use categories: [{"category", "n", "share", "km2"}]."""
    keep = np.ones(len(classes), bool) if valid is None else np.asarray(valid, bool)
    return _rollup([_META[c]["ipcc"] for c, ok in zip(classes, keep) if ok and c in _META], IPCC_ORDER, "category")


# --------------------------------------------------------------------------- climate statistics
def climate_stats(classes: Sequence[str], ndvi, water_frac, lst, centers: Sequence | None = None,
                  valid=None) -> dict:
    """Every field of `stats` in web/API.md, from per-tile vectors (None/NaN = missing).

    Definitions (all descriptive; nothing is fitted to labels):
    - water tile: NDWI > 0 on more than half the tile. Everything else with pixels is a land tile.
    - regression, tercile contrast and the two shortlists use land tiles that have both NDVI and
      surface temperature. Water is left out because it is cool for a different reason than shade.
    - regression: ordinary least squares of surface temperature on NDVI; the slope is reported per
      +0.1 NDVI. None with fewer than 8 tiles or no variation.
    - tercile: mean temperature of the least-green third minus the greenest third.
    - by_class / by_group: mean temperature and NDVI of the tiles the model put in each class
      (classes are model estimates; the temperatures are measurements).
    - priority: land tiles in the hottest quarter, ranked by z(temperature) - z(NDVI): hot and bare.
      refuges: the lowest of the same score: cool and green.
    """
    n = len(classes)
    ndvi, water, lst = _vec(ndvi, n), _vec(water_frac, n), _vec(lst, n)
    ok = np.ones(n, bool) if valid is None else np.asarray(valid, bool)
    is_water = ok & np.isfinite(water) & (water > WATER_TILE_FRAC)
    is_land = ok & ~is_water
    has_lst, has_ndvi = ok & np.isfinite(lst), ok & np.isfinite(ndvi)
    fit = is_land & has_lst & has_ndvi

    def mean(a, mask):
        return float(a[mask].mean()) if mask.any() else None

    out: dict[str, Any] = {
        "n_land": int(is_land.sum()), "n_water": int(is_water.sum()),
        "lst_mean": _r(mean(lst, has_lst), 1),
        "lst_min": _r(lst[has_lst].min(), 1) if has_lst.any() else None,
        "lst_max": _r(lst[has_lst].max(), 1) if has_lst.any() else None,
        "ndvi_mean": _r(mean(ndvi, has_ndvi), 2),
        "regression": None, "tercile": None, "by_class": [], "by_group": [],
        "built_minus_vegetated_c": None, "priority": [], "refuges": [],
    }

    # Means by estimated land cover. With a temperature layer: tiles that have a temperature.
    # Without one (no usable Landsat pass): tiles that have NDVI, and lst_mean stays None.
    base = has_lst if has_lst.any() else has_ndvi

    def group_row(key: str, name: str, mask: np.ndarray) -> dict | None:
        m = base & mask
        if not m.any():
            return None
        return {key: name, "n": int(m.sum()), "lst_mean": _r(mean(lst, m & has_lst), 1),
                "ndvi_mean": _r(mean(ndvi, m & has_ndvi), 2)}

    cls = np.asarray(list(classes), dtype=object)
    rows = [group_row("cls", c, cls == c) for c in CLASS_NAMES]
    out["by_class"] = sorted([r for r in rows if r],
                             key=lambda r: (-(r["lst_mean"] if r["lst_mean"] is not None else -1e9), -r["n"]))
    grp = np.array([_META[c]["group"] if c in _META else "" for c in classes], dtype=object)
    groups = {g: group_row("group", g, grp == g) for g in GROUP_ORDER}
    out["by_group"] = [groups[g] for g in GROUP_ORDER if groups[g]]
    b, v = groups.get("Built"), groups.get("Vegetated")
    if b and v and b["n"] >= 3 and v["n"] >= 3 and b["lst_mean"] is not None and v["lst_mean"] is not None:
        out["built_minus_vegetated_c"] = round(b["lst_mean"] - v["lst_mean"], 1)

    idx = np.flatnonzero(fit)
    x, y = ndvi[idx], lst[idx]
    if len(idx) >= 6:
        order = np.argsort(x, kind="stable")
        k = len(idx) // 3
        least, most = _r(y[order[:k]].mean(), 1), _r(y[order[-k:]].mean(), 1)
        out["tercile"] = {"hot_minus_green_c": round(least - most, 1), "least_green_mean": least,
                          "greenest_mean": most, "n_each": int(k)}
    if len(idx) < MIN_FIT_TILES:
        return _clean(out)

    sx, sy = float(x.std()), float(y.std())
    if sx > 1e-9 and sy > 1e-9:
        cov = float(((x - x.mean()) * (y - y.mean())).mean())
        slope = cov / (sx * sx)
        out["regression"] = {
            "slope_per_0p1_ndvi": round(slope * 0.1, 2), "intercept": round(float(y.mean() - slope * x.mean()), 1),
            "r": round(max(-1.0, min(1.0, cov / (sx * sy))), 2), "n": int(len(idx)),
        }

    z_t = (y - y.mean()) / sy if sy > 1e-9 else np.zeros(len(idx))
    z_g = (x - x.mean()) / sx if sx > 1e-9 else np.zeros(len(idx))
    exposure = z_t - z_g  # high = hot and bare, low = cool and green

    def entry(j: int) -> dict:
        i = int(idx[j])
        center = list(centers[i]) if centers is not None and centers[i] is not None else None
        return {"i": i, "lst": _r(y[j], 1), "ndvi": _r(x[j], 2), "center": center, "cls": classes[i]}

    hot = y >= np.percentile(y, 75)
    first = [int(j) for j in np.argsort(-exposure, kind="stable") if hot[j]][:8]
    out["priority"] = [entry(j) for j in first]
    taken = set(first)
    out["refuges"] = [entry(int(j)) for j in np.argsort(exposure, kind="stable") if int(j) not in taken][:5]
    return _clean(out)


# --------------------------------------------------------------------------- exports
EXPORT_COLUMNS = [
    "region_id", "tile", "row", "col", "lon", "lat", "class", "class_label", "score", "second", "third",
    "ipcc_category", "surface_group", "ndvi", "water_frac", "lst_c", "nodata_frac", "cloud_frac",
    "s2_scene", "s2_datetime", "landsat_scene", "landsat_datetime", "mode",
]


def _export_rows(summary: dict, climate: dict | None) -> list[dict]:
    """One flat record per tile. Class columns are model estimates; ndvi, water_frac and lst_c are measured."""
    climate = climate or {}
    n = len(summary["tiles"])
    ndvi = ((climate.get("ndvi") or {}).get("tile")) or [None] * n
    water = ((climate.get("water") or {}).get("tile")) or [None] * n
    lst_layer = climate.get("lst") or {}
    lst = lst_layer.get("tile") or [None] * n
    ls = lst_layer.get("scene") or {}
    s2 = summary.get("scene") or {}
    rows = []
    for t in summary["tiles"]:
        i = t["i"]
        top = [e["label"] for e in t.get("top3", [])] + [None, None, None]
        meta = _META.get(t.get("cls"), {})
        lon, lat = (t.get("center") or [None, None])[:2]
        rows.append({
            "region_id": summary.get("id"), "tile": i, "row": t.get("row"), "col": t.get("col"),
            "lon": lon, "lat": lat, "class": t.get("cls"), "class_label": meta.get("label"),
            "score": t.get("score"), "second": top[1], "third": top[2],
            "ipcc_category": meta.get("ipcc"), "surface_group": meta.get("group"),
            "ndvi": ndvi[i], "water_frac": water[i], "lst_c": lst[i],
            "nodata_frac": t.get("nodata_frac"), "cloud_frac": t.get("cloud_frac"),
            "s2_scene": s2.get("id"), "s2_datetime": s2.get("datetime"),
            "landsat_scene": ls.get("id"), "landsat_datetime": ls.get("datetime"),
            "mode": summary.get("mode"),
        })
    return rows


def export_csv(summary: dict, climate: dict | None = None) -> str:
    """Per-tile table as CSV text (one header row, no comment lines, empty cell = no value)."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=EXPORT_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in _export_rows(_clean(summary), _clean(climate)):
        writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
    return buf.getvalue()


def export_geojson(summary: dict, climate: dict | None = None, disclaimer: str = "") -> dict:
    """Per-tile polygons (WGS84, counter-clockwise rings) with the same columns as the CSV."""
    summary, climate = _clean(summary), _clean(climate)
    features = []
    for t, row in zip(summary["tiles"], _export_rows(summary, climate)):
        tl, tr, br, bl = t["corners"]
        features.append({
            "type": "Feature", "id": t["i"],
            "geometry": {"type": "Polygon", "coordinates": [[tl, bl, br, tr, tl]]},
            "properties": {k: v for k, v in row.items() if k not in ("region_id", "mode")},
        })
    lst_scene = ((climate or {}).get("lst") or {}).get("scene")
    return {
        "type": "FeatureCollection",
        "name": f"platonic-earth-{summary.get('id')}",
        "region_id": summary.get("id"), "mode": summary.get("mode"), "bbox_requested": summary.get("bbox"),
        "tile_m": summary.get("tile_m"), "sentinel2_scene": summary.get("scene"), "landsat_scene": lst_scene,
        "notes": [
            "class, score, second, third, ipcc_category and surface_group are model estimates "
            "(frozen DINOv2 + frozen MiniLM joined by one Procrustes rotation fit on EuroSAT).",
            "ndvi and water_frac are measured from Sentinel-2 L2A; lst_c is Landsat 8/9 Collection 2 Level-2 "
            "surface temperature in degrees Celsius (skin temperature at the overpass, not air temperature).",
            disclaimer,
        ],
        "features": features,
    }
