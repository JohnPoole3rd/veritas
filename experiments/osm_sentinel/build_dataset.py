#!/usr/bin/env python
"""Build an OSM-captioned Sentinel-2 dataset.

For each region in `regions.csv`:
  1. find a low-cloud Sentinel-2 L2A scene with STAC (Element84 Earth Search);
  2. load a cropped local OSM PBF with pyrosm (one download per region);
  3. sample chips, read them from the scene's COG, and caption each chip from OSM.

The output is intentionally plain and crash-safe:
  images/<tile_id>.jpg
  manifest.csv       (one row appended per completed chip)
  captions.jsonl     (one caption appended per completed chip)
  dataset_stats.json

Rerunning without `--overwrite` resumes from `manifest.csv`: completed tile IDs
are skipped and the caption JSONL is repaired from the manifest. This matters at
100k-1M-chip scale, where losing the manifest on an interruption is expensive.

Example:
    python experiments/osm_sentinel/build_dataset.py \
      --region us_columbus --chips-per-region 1 --max-chips 1

Scale up with more regions, `--chips-per-region`, or a longer region list.
PBF/GeoParquet caches live under `<out-dir>/_cache/`.
"""

from __future__ import annotations

import argparse
import atexit
import csv
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import io
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
THIS = Path(__file__).resolve().parent
for path in (ROOT, THIS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from osm_captions import AREA_SUBJECTS, build_caption_from_features  # noqa: E402
from osm_source import load_region_features  # noqa: E402
from worldcover import RegionWorldCover  # noqa: E402

STAC_URL = "https://earth-search.aws.element84.com/v1"
FIELDS = [
    "tile_id", "image", "region_id", "country", "biome", "split", "lat", "lon",
    "scene_id", "datetime", "cloud_cover", "label", "caption", "n_features",
    "building_count", "feature_counts", "worldcover_class", "worldcover_fraction",
]


def bbox_from_center(lat: float, lon: float, radius_km: float) -> tuple[float, float, float, float]:
    """Return (west, south, east, north)."""
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * max(math.cos(math.radians(lat)), 1e-6))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def region_geometry(region: dict[str, Any], args) -> tuple[float, float, float, tuple[float, float, float, float]]:
    """Return lat, lon, radius, bbox for a region and the current overrides."""
    lat, lon = float(region["lat"]), float(region["lon"])
    radius_km = args.radius_km if args.radius_km is not None else float(region["radius_km"])
    return lat, lon, radius_km, bbox_from_center(lat, lon, radius_km)


# --- A (a/osm-1m-fixes): scene selection ------------------------------------------------
# One shared STAC client, retries with backoff, and a ranked candidate list instead of the single
# lowest-cloud item. The old rule picked s3:// placeholder items (100 % nodata, cloud = 0) and winter
# scenes; it also ignored how much of the region a granule covers.
_STAC_CLIENT = None
SCENE_POLICY = 2  # bump when scene selection changes; recorded in region_status.json


def _stac_client():
    global _STAC_CLIENT
    if _STAC_CLIENT is None:
        from pystac_client import Client

        _STAC_CLIENT = Client.open(STAC_URL)
    return _STAC_CLIENT


def _with_retry(fn, what: str, tries: int = 4, base_s: float = 5.0):
    """Call fn(); on failure wait 5, 15, 45 s and retry (polite to the STAC API)."""
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:  # network / 403 / 5xx
            if attempt == tries - 1:
                raise
            wait = base_s * (3 ** attempt)
            print(f"[stac] {what} failed ({type(exc).__name__}: {exc}); retry in {wait:.0f}s", flush=True)
            time.sleep(wait)


def _mgrs_tile(item) -> str:
    p = item.properties
    tile = f"{p.get('mgrs:utm_zone', '')}{p.get('mgrs:latitude_band', '')}{p.get('mgrs:grid_square', '')}"
    return tile or item.id.split("_")[1]


def _in_season(item, lat: float) -> bool:
    """Leaf-on months: May–Sep north of 30°N, Nov–Mar south of 30°S, any month in between."""
    # item.datetime is None for range-only STAC items; fall back to the properties, and treat an
    # unparseable date as in season instead of crashing the whole scene search.
    stamp = item.datetime or item.properties.get("datetime") or item.properties.get("start_datetime")
    try:
        month = int(str(stamp)[5:7])
    except ValueError:
        return True
    if lat > 30:
        return 5 <= month <= 9
    if lat < -30:
        return month >= 11 or month <= 3
    return True


def find_scenes(bbox: tuple[float, float, float, float], start: str, end: str, max_cloud: float,
                max_items: int = 80, max_nodata: float = 50.0) -> list[tuple[Any, float]]:
    """Readable Sentinel-2 L2A items for a region, best first, as (item, region_coverage).

    Only https COGs (the s3:// sentinel-s2-l2a JP2 items need credentials and are empty
    placeholders) with nodata below `max_nodata` %. Ranked by region coverage (0.1 buckets),
    then growing season for the region's latitude, then cloud cover.
    """
    from shapely.geometry import box, shape

    region = box(*bbox)
    lat = (bbox[1] + bbox[3]) / 2

    def search():
        return list(_stac_client().search(
            collections=["sentinel-2-l2a"],
            bbox=list(bbox),
            datetime=f"{start}/{end}",
            query={"eo:cloud_cover": {"lt": max_cloud}, "s2:nodata_pixel_percentage": {"lt": max_nodata}},
            sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
            max_items=max_items,
        ).items())

    ranked = []
    for item in _with_retry(search, "scene search"):
        asset = item.assets.get("visual")
        if asset is None or not str(asset.href).startswith("http"):
            continue
        try:
            coverage = shape(item.geometry).intersection(region).area / region.area
        except Exception:
            coverage = 0.0
        ranked.append((item, coverage))
    ranked.sort(key=lambda t: (-round(t[1], 1), not _in_season(t[0], lat),
                               float(t[0].properties.get("eo:cloud_cover", 100))))
    return ranked


def find_scene(bbox: tuple[float, float, float, float], start: str, end: str, max_cloud: float):
    """Best single readable item (kept for compatibility; see find_scenes)."""
    ranked = find_scenes(bbox, start, end, max_cloud)
    return ranked[0][0] if ranked else None


def read_region(src, bbox: tuple[float, float, float, float]):
    """Read the region's RGB window once from the Sentinel COG.

    Reading a 1.28 km or 640 m window directly from S3 for every chip costs
    ~0.2 s/chip. One region-sized read followed by local crops removes that
    latency.
    """
    import numpy as np
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    west, south, east, north = bbox
    bounds = transform_bounds("EPSG:4326", src.crs, west, south, east, north, densify_pts=21)
    window = from_bounds(*bounds, src.transform).round_offsets().round_lengths()
    if window.width < 8 or window.height < 8:
        return None
    # boundless=True keeps the requested georeference when a 120 km region
    # extends past one Sentinel granule; outside pixels are zero and chips that
    # fall entirely outside are rejected later.
    data = src.read(window=window, boundless=True, fill_value=0)
    if data.size == 0:
        return None
    if data.shape[0] >= 3:
        rgb = np.transpose(data[:3], (1, 2, 0))
    else:
        rgb = np.repeat(np.transpose(data[:1], (1, 2, 0)), 3, axis=2)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb.astype("float32") / 4000.0 * 255.0, 0, 255).astype("uint8")
    if rgb.max() < 5:
        return None
    return np.ascontiguousarray(rgb), src.window_transform(window), src.crs


_NETWORK_MARKERS = ("CURL error", "Could not resolve", "Failed to connect", "timed out", "Connection reset",
                    "Temporary failure", "Network is unreachable")


class NetworkDown(RuntimeError):
    """The image host is unreachable; abort the region (it is retried later) instead of burning candidates."""


def _is_network_error(exc: Exception) -> bool:
    return any(marker in str(exc) for marker in _NETWORK_MARKERS)


def read_region_mosaic(ranked, scene_bbox, max_scenes: int = 4, target_valid: float = 0.97, log=print):
    """Region raster mosaicked from up to `max_scenes` granules (A, a/osm-1m-fixes).

    One MGRS granule is ~110 km, so a single scene covered a median ~4 % of a 60 km region. The first
    usable candidate defines the grid; further candidates on *other* MGRS tiles in the same CRS (same
    10 m UTM grid, so windows align exactly) fill its nodata pixels. Unreadable or empty granules
    (e.g. s3:// placeholders) are skipped instead of failing the region.

    Returns (rgb, transform, crs, source_index, used_items) or None; source_index[r, c] = k means
    the pixel came from used_items[k-1] (0 = nodata).
    """
    import numpy as np
    import rasterio
    from rasterio.transform import array_bounds
    from rasterio.windows import from_bounds

    base = None
    used: list = []
    tiles: set[str] = set()
    for item, _coverage in ranked:
        if len(used) >= max_scenes:
            break
        tile = _mgrs_tile(item)
        if tile in tiles:
            continue
        last_exc = None
        for attempt in range(3):  # A: retry transient network errors (10 s, 30 s) on the same granule
            try:
                outcome = _mosaic_step(item, base, used, scene_bbox, log, rasterio, np, array_bounds, from_bounds)
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                if not _is_network_error(exc) or attempt == 2:
                    break
                wait = 10 * (3 ** attempt)
                log(f"[scene] {item.id} network error ({type(exc).__name__}); retry in {wait}s")
                time.sleep(wait)
        if last_exc is not None:
            if _is_network_error(last_exc):
                raise NetworkDown(f"image host unreachable while reading {item.id}: {last_exc}")
            log(f"[scene] {item.id} unreadable, trying next: {type(last_exc).__name__}: {last_exc}")
            continue
        if outcome is None:
            continue
        base = outcome
        used.append(item)
        tiles.add(tile)
        if (base[3] > 0).mean() >= target_valid:
            break
    if base is None:
        return None
    return base[0], base[1], base[2], base[3], used


def _mosaic_step(item, base, used, scene_bbox, log, rasterio, np, array_bounds, from_bounds):
    """Read one granule into the mosaic. Returns the updated base, or None if it adds nothing."""
    with rasterio.open(item.assets["visual"].href) as src:
        if base is None:
            res = read_region(src, scene_bbox)
            if res is None:
                return None
            rgb, transform, crs = res
            valid = rgb.max(axis=2) > 0
            if valid.mean() < 0.02:
                log(f"[scene] {item.id} covers <2 % of the region window; trying the next candidate")
                return None
            return [np.array(rgb), transform, crs, valid.astype(np.uint8)]
        rgb, transform, crs, src_idx = base
        if src.crs != crs:
            return None  # different UTM zone: grids do not align
        h, w = src_idx.shape
        window = from_bounds(*array_bounds(h, w, transform), src.transform)
        window = window.round_offsets().round_lengths()
        if (int(window.height), int(window.width)) != (h, w) or not src.window_transform(
            window
        ).almost_equals(transform, precision=1e-3):
            return None
        data = src.read(window=window, boundless=True, fill_value=0)
        add = np.transpose(data[:3], (1, 2, 0))
        if add.dtype != np.uint8:
            add = np.clip(add.astype("float32") / 4000.0 * 255.0, 0, 255).astype("uint8")
        gap = (src_idx == 0) & (add.max(axis=2) > 0)
        if not gap.any():
            return None
        rgb[gap] = add[gap]
        src_idx[gap] = len(used) + 1
        return base


def chip_source(source_index, transform, crs, lat: float, lon: float) -> int:
    """Index (1-based) of the mosaic scene at the chip centre; 0 if outside/nodata."""
    from rasterio.warp import transform as warp_transform

    xs, ys = warp_transform("EPSG:4326", crs, [lon], [lat])
    col, row = ~transform * (xs[0], ys[0])
    # floor, not int(): int() truncates toward zero, so a centre up to one pixel above/left of the
    # raster (row or col in (-1, 0)) would wrongly map to pixel 0 instead of "outside".
    r, c = math.floor(row), math.floor(col)
    if 0 <= r < source_index.shape[0] and 0 <= c < source_index.shape[1]:
        return int(source_index[r, c])
    return 0


def read_chip(region_rgb, region_transform, region_crs, lat: float, lon: float, chip_m: float, chip_px: int):
    """Crop a chip from an in-memory region raster."""
    from PIL import Image
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    half_lat = chip_m / 2 / 111_320
    half_lon = chip_m / 2 / (111_320 * max(math.cos(math.radians(lat)), 1e-6))
    west, south, east, north = lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat
    bounds = transform_bounds("EPSG:4326", region_crs, west, south, east, north, densify_pts=21)
    window = from_bounds(*bounds, region_transform).round_offsets().round_lengths()
    row0 = int(round(window.row_off))
    col0 = int(round(window.col_off))
    row1 = row0 + max(1, int(round(window.height)))
    col1 = col0 + max(1, int(round(window.width)))
    if row0 < 0 or col0 < 0 or row1 > region_rgb.shape[0] or col1 > region_rgb.shape[1]:
        return None
    if row1 - row0 < 8 or col1 - col0 < 8:
        return None
    crop = region_rgb[row0:row1, col0:col1]
    # Boundless scene reads leave zero fill outside the granule. Skip chips
    # that are mostly nodata (e.g. the edge of a 60 km region).
    if (crop.max(axis=2) == 0).mean() > 0.5:
        return None
    image = Image.fromarray(crop).resize((chip_px, chip_px), Image.BILINEAR)
    return image


def read_regions(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def tile_id(region_id: str, lat: float, lon: float) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{region_id}_{abs(lat):.4f}{ns}_{abs(lon):.4f}{ew}"


def grid_candidates(
    region_id: str,
    lat: float,
    lon: float,
    radius_km: float,
    chip_m: float,
):
    """Yield non-overlapping chip centers on a local lat/lon grid."""
    dlat = chip_m / 111_320
    dlon = chip_m / (111_320 * max(math.cos(math.radians(lat)), 1e-6))
    chip_km = chip_m / 1000.0
    ny = int(radius_km / chip_km) + 2
    nx = int(radius_km / chip_km) + 2
    for iy in range(-ny, ny + 1):
        for ix in range(-nx, nx + 1):
            chip_lat = lat + iy * dlat
            chip_lon = lon + ix * dlon
            dy = (chip_lat - lat) * 111.32
            dx = (chip_lon - lon) * 111.32 * max(math.cos(math.radians(lat)), 1e-6)
            if dx * dx + dy * dy > radius_km * radius_km:
                continue
            yield f"{region_id}_g{ix:+05d}_{iy:+05d}", chip_lat, chip_lon


def random_candidates(
    region_id: str,
    lat: float,
    lon: float,
    radius_km: float,
    chips_per_region: int,
    rng: random.Random,
):
    """Yield random chip centers; resume still uses the stable coordinate ID."""
    for _ in range(chips_per_region * 8):
        angle = rng.uniform(0, 2 * math.pi)
        distance = radius_km * math.sqrt(rng.random())
        chip_lat = lat + (distance / 111.32) * math.cos(angle)
        chip_lon = lon + (distance / (111.32 * math.cos(math.radians(lat)))) * math.sin(angle)
        yield tile_id(region_id, chip_lat, chip_lon), chip_lat, chip_lon


def new_stats() -> dict[str, Any]:
    return {
        "n": 0,
        "by_split": Counter(),
        "by_country": Counter(),
        "by_label": Counter(),
        "regions": set(),
        "failed_regions": set(),
        "skipped_chips": 0,
        "worldcover_missing": 0,
    }


def update_stats(stats: dict[str, Any], row: dict[str, Any]) -> None:
    stats["n"] += 1
    stats["by_split"][row["split"]] += 1
    stats["by_country"][row["country"]] += 1
    stats["by_label"][row["label"]] += 1
    stats["regions"].add(row["region_id"])


def stats_payload(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "n": stats["n"],
        "by_split": dict(stats["by_split"]),
        "by_country": dict(stats["by_country"]),
        "by_label": dict(stats["by_label"]),
        "regions": sorted(stats["regions"]),
        "failed_regions": sorted(stats["failed_regions"]),
        "skipped_chips": stats["skipped_chips"],
        "worldcover_missing": stats["worldcover_missing"],
    }


def write_stats(path: Path, stats: dict[str, Any]) -> None:
    # main() writes stats at every region end and at exit: flush buffered manifest/caption rows
    # first, so dataset_stats.json never counts chips that are not on disk yet.
    flush_appends()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(stats_payload(stats), indent=2) + "\n")
    tmp.replace(path)


def build_meta(args) -> dict[str, Any]:
    source = Path(args.regions_csv)
    return {
        "version": 1,
        "chip_m": float(args.chip_m),
        "chip_px": int(args.chip_px),
        "sample_mode": args.sample_mode,
        "radius_km_override": args.radius_km,
        "regions_sha256": hashlib.sha256(source.read_bytes()).hexdigest()[:16],
    }


def check_build_meta(out_dir: Path, args, overwrite: bool) -> None:
    """Refuse to append a different chip/sampling config into an existing build."""
    meta_path = out_dir / "build_meta.json"
    manifest_path = out_dir / "manifest.csv"
    expected = build_meta(args)
    if overwrite:
        for path in (meta_path, manifest_path, out_dir / "captions.jsonl"):
            if path.exists():
                path.unlink()
    elif meta_path.exists():
        existing = json.loads(meta_path.read_text())
        if existing != expected:
            raise SystemExit(
                "Existing build metadata does not match this command:\n"
                f"  existing: {existing}\n  requested: {expected}\n"
                "Use a new --out-dir or --overwrite."
            )
    elif manifest_path.exists():
        raise SystemExit(
            f"{manifest_path} exists but {meta_path} is missing (legacy build). "
            "Use a new --out-dir or --overwrite so chip size/sampling cannot be mixed."
        )
    tmp = meta_path.with_name(meta_path.name + ".tmp")
    tmp.write_text(json.dumps(expected, indent=2) + "\n")
    tmp.replace(meta_path)


def caption_record(row: dict[str, Any]) -> dict[str, Any]:
    try:
        counts = json.loads(row.get("feature_counts") or "{}")
    except json.JSONDecodeError:
        counts = {}
    return {
        "tile_id": row["tile_id"],
        "image": row.get("image", ""),
        "caption": row["caption"],
        "label": row["label"],
        "split": row["split"],
        "region_id": row["region_id"],
        "country": row.get("country", ""),
        "biome": row.get("biome", ""),
        "feature_counts": counts,
        "worldcover_class": row.get("worldcover_class", ""),
        "worldcover_fraction": row.get("worldcover_fraction", ""),
    }


def rewrite_captions(manifest_path: Path, captions_path: Path) -> None:
    """Repair captions from the manifest after an interrupted run."""
    tmp = captions_path.with_name(captions_path.name + ".tmp")
    with manifest_path.open(newline="") as source, tmp.open("w") as out:
        for row in csv.DictReader(source):
            if any(not row.get(key) for key in ("tile_id", "caption", "label", "split", "region_id")):
                continue
            out.write(json.dumps(caption_record(row), sort_keys=True) + "\n")
    tmp.replace(captions_path)


def load_existing(manifest_path: Path, captions_path: Path) -> tuple[set[str], dict[str, Any]]:
    flush_appends()
    stats = new_stats()
    done: set[str] = set()
    if manifest_path.exists():
        raw = manifest_path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            # A crash can leave a torn final CSV row. Drop it before parsing.
            cut = raw.rfind(b"\n")
            manifest_path.write_bytes(raw[: cut + 1] if cut >= 0 else b"")
            print(f"[resume] dropped a torn final manifest row from {manifest_path}", flush=True)
        with manifest_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                required = ("tile_id", "split", "country", "label", "region_id")
                if any(not row.get(key) for key in required):
                    print(f"[resume] skipped malformed manifest row: {row}", flush=True)
                    continue
                done.add(row["tile_id"])
                update_stats(stats, row)
        rewrite_captions(manifest_path, captions_path)
    elif captions_path.exists():
        captions_path.write_text("")
    return done, stats


# Rows are buffered in memory and written in batches: opening/flushing both files for every chip
# cost two open/close pairs per chip, and each blocking syscall hands the GIL to the region-prefetch
# thread (up to a 5 ms switch interval to get it back). Buffered rows are written every
# APPEND_FLUSH_ROWS chips, at every write_stats() (region end and run end), on load_existing(), and
# at interpreter exit. A hard kill loses at most APPEND_FLUSH_ROWS rows; their images are rewritten
# identically on resume, and load_existing() already repairs a torn final row / captions file.
APPEND_FLUSH_ROWS = 500
_PENDING: dict[tuple[Path, Path], tuple[list[str], list[str]]] = {}


def _flush_pending(key: tuple[Path, Path]) -> None:
    manifest_rows, caption_lines = _PENDING.pop(key)
    if not manifest_rows:
        return
    manifest_path, captions_path = key
    new_file = not manifest_path.exists() or manifest_path.stat().st_size == 0
    with manifest_path.open("a", newline="") as handle:
        if new_file:
            csv.DictWriter(handle, fieldnames=FIELDS).writeheader()
        handle.write("".join(manifest_rows))
    with captions_path.open("a") as handle:
        handle.write("".join(caption_lines))


def flush_appends() -> None:
    """Write every buffered manifest/caption row to disk (safe to call any time)."""
    for key in list(_PENDING):
        _flush_pending(key)


atexit.register(flush_appends)


def append_chip(
    manifest_path: Path,
    captions_path: Path,
    row: dict[str, Any],
) -> None:
    """Queue a completed chip for both outputs; see APPEND_FLUSH_ROWS for when it reaches disk."""
    key = (Path(manifest_path), Path(captions_path))
    manifest_rows, caption_lines = _PENDING.setdefault(key, ([], []))
    line = io.StringIO(newline="")
    csv.DictWriter(line, fieldnames=FIELDS).writerow(row)
    manifest_rows.append(line.getvalue())
    caption_lines.append(json.dumps(caption_record(row), sort_keys=True) + "\n")
    if len(manifest_rows) >= APPEND_FLUSH_ROWS:
        _flush_pending(key)


# --- region-level parallelism (a/speed-agent1) ------------------------------------------------
# A region is the unit of work: prepare (STAC search + OSM load + mosaic read) and the chip loop
# (crop, caption, JPEG write) run in a worker process. The main process stays the single writer of
# manifest.csv / captions.jsonl / dataset_stats.json / region_status.json. With --workers 1 the
# same functions run in-process with the old one-region prefetch thread and per-chip appends.
STAC_MIN_GAP_S = 1.0  # minimum gap between STAC searches across all worker processes


@contextmanager
def _file_lock(path: Path):
    """Exclusive cross-process lock (flock) on `path`; yields the open handle."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield handle
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _serialized_find_scenes(lock_path: Path, bbox, start: str, end: str, max_cloud: float):
    """find_scenes under a cross-process lock with >= STAC_MIN_GAP_S between searches (the
    Earth Search API returns 403 when hammered). Retries/backoff inside find_scenes also hold
    the lock, so a throttled worker slows every worker down, which is what we want."""
    with _file_lock(lock_path) as handle:
        handle.seek(0)
        try:
            last = float(handle.read().strip() or 0.0)
        except ValueError:
            last = 0.0
        wait = last + STAC_MIN_GAP_S - time.time()
        if wait > 0:
            time.sleep(wait)
        try:
            return find_scenes(bbox, start, end, max_cloud)
        finally:
            handle.seek(0)
            handle.truncate()
            handle.write(f"{time.time():.3f}\n")
            handle.flush()


def _load_features_locked(region_id: str, bbox, cache_dir: Path, require_cache: bool = False):
    """load_region_features, but a missing PBF is downloaded under a cross-process lock so two
    workers never fetch (or write) the same Geofabrik extract at once. Parsing stays parallel."""
    from osm_source import cache_key, ensure_region_pbf

    key = cache_key(region_id, bbox)
    parquet = cache_dir / "osm" / f"{key}.geoparquet"
    cropped = cache_dir / "pbf" / f"{key}.osm.pbf"
    if not parquet.exists():
        if require_cache and not cropped.exists():
            # ensure_region_pbf validates the provider source cache and fails clearly
            # without starting a network download.
            ensure_region_pbf(key, bbox, cache_dir, require_cache=True)
        elif not cropped.exists():
            with _file_lock(cache_dir / ".osm_download.lock"):
                ensure_region_pbf(key, bbox, cache_dir)
    return load_region_features(region_id, bbox, cache_dir, require_cache=require_cache)


def prepare_region(region: dict[str, Any], args, cache_dir: Path) -> dict[str, Any]:
    """Scene search + OSM load + mosaic read for one region."""
    rid = region["region_id"]
    lat, lon, radius_km, bbox = region_geometry(region, args)
    scene_bbox = bbox_from_center(lat, lon, radius_km + args.chip_m / 1000.0)
    t: Counter[str] = Counter()
    t0 = time.perf_counter()
    ranked = _serialized_find_scenes(cache_dir / ".stac.lock", bbox, args.date_start, args.date_end,
                                     args.max_cloud)
    t["scene_search"] += time.perf_counter() - t0
    if not ranked:
        return {"ranked": [], "timings": t}
    t0 = time.perf_counter()
    features = _load_features_locked(
        rid, bbox, cache_dir, require_cache=args.require_osm_cache
    )
    t["osm_load"] += time.perf_counter() - t0
    t0 = time.perf_counter()
    mosaic = read_region_mosaic(ranked, scene_bbox, log=lambda m: print(f"[{rid}] {m}", flush=True))
    t["scene_read"] += time.perf_counter() - t0
    return {"ranked": ranked, "features": features, "mosaic": mosaic, "timings": t}


def chip_region(region: dict[str, Any], args, prepared: dict[str, Any], out_dir: Path, done: set[str],
                existing: int, cap: int, budget: int | None, emit) -> dict[str, Any]:
    """Run the chip loop for one prepared region.

    `done` holds already-written tile IDs, `existing` the region's chips already in the manifest,
    `budget` the chips still allowed by --max-chips (None = no limit). Each finished row goes to
    `emit(row)` (append to the manifest in-process, or collect it in a worker). Images are written
    here. Returns counters and timings; `prepared["timings"]` is included.
    """
    region_id = region["region_id"]
    lat, lon, radius_km, bbox = region_geometry(region, args)
    timings: Counter[str] = Counter(prepared["timings"])
    skipped = 0
    worldcover_missing = 0
    region_features = prepared["features"]
    region_rgb, region_transform, region_crs, source_index, used_items = prepared["mosaic"]
    valid_pct = 100.0 * float((source_index > 0).mean())
    print(f"[{region_id}] scenes {[it.id for it in used_items]} valid={valid_pct:.0f}%", flush=True)
    # WorldCover is a fallback: load it lazily the first time a chip has no OSM land-cover polygon.
    region_wc = None
    region_wc_attempted = bool(args.no_worldcover)

    added = 0
    budget_hit = False
    chip_t0 = time.perf_counter()
    if args.sample_mode == "grid":
        candidates = list(grid_candidates(region_id, lat, lon, radius_km, args.chip_m))
        # per-region RNG: a region's order no longer depends on which earlier regions ran or failed
        random.Random(f"{args.seed}:{region_id}").shuffle(candidates)
    else:
        candidates = random_candidates(region_id, lat, lon, radius_km, args.chips_per_region,
                                       random.Random(f"{args.seed}:{region_id}"))
    for tid, chip_lat, chip_lon in candidates:
        if budget is not None and added >= budget:
            budget_hit = True
            break
        if existing + added >= cap:
            break
        if tid in done:
            continue
        try:
            t0 = time.perf_counter()
            image = read_chip(
                region_rgb, region_transform, region_crs,
                chip_lat, chip_lon, args.chip_m, args.chip_px,
            )
            timings["raster_read"] += time.perf_counter() - t0
        except Exception as exc:  # transient COG/network error: skip this chip, keep the run alive
            print(f"[{region_id}] {tid} read failed, skipped: {type(exc).__name__}: {exc}", flush=True)
            skipped += 1
            continue
        if image is None:
            skipped += 1
            continue
        half_lat = args.chip_m / 2 / 111_320
        half_lon = args.chip_m / 2 / (111_320 * max(math.cos(math.radians(chip_lat)), 1e-6))
        chip_bbox = (chip_lon - half_lon, chip_lat - half_lat, chip_lon + half_lon, chip_lat + half_lat)
        try:
            t0 = time.perf_counter()
            features, buildings = region_features.for_bbox(chip_bbox)
            caption, label, counts, building_count = build_caption_from_features(
                features, buildings, chip_bbox
            )
            timings["caption"] += time.perf_counter() - t0
        except Exception as exc:  # malformed OSM data for this chip: skip it, keep the run alive
            print(f"[{region_id}] {tid} caption failed, skipped: {type(exc).__name__}: {exc}", flush=True)
            skipped += 1
            continue
        k = chip_source(source_index, region_transform, region_crs, chip_lat, chip_lon)
        chip_item = used_items[k - 1] if k > 0 else used_items[0]
        wc = None
        needs_worldcover = not any(p in AREA_SUBJECTS for p in counts)
        if needs_worldcover and not region_wc_attempted:
            region_wc_attempted = True
            if not args.no_worldcover:
                try:
                    t0 = time.perf_counter()
                    region_wc = RegionWorldCover(bbox)
                    timings["worldcover_load"] += time.perf_counter() - t0
                    print(f"[{region_id}] WorldCover region loaded lazily", flush=True)
                except Exception as exc:
                    worldcover_missing += 1
                    print(
                        f"[{region_id}] WorldCover unavailable, continuing without it: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
        if needs_worldcover and region_wc is not None:
            try:
                wc = region_wc.for_bbox(chip_bbox)
            except Exception:
                wc = None
            if wc is not None:
                try:
                    caption, label, counts, building_count = build_caption_from_features(
                        features, buildings, chip_bbox, worldcover=wc
                    )
                except Exception:
                    wc = None
        if caption is None or len(counts) < 1:
            skipped += 1
            continue
        image_rel = f"images/{tid}.jpg"
        try:
            t0 = time.perf_counter()
            image.save(out_dir / image_rel, quality=92)
            timings["image_save"] += time.perf_counter() - t0
        except Exception as exc:
            print(f"[{region_id}] {tid} image save failed, skipped: {type(exc).__name__}: {exc}", flush=True)
            skipped += 1
            continue
        wc_class = "" if wc is None else str(wc[2])
        wc_fraction = "" if wc is None else f"{wc[1]:.4f}"
        row = {
            "tile_id": tid,
            "image": image_rel,
            "region_id": region_id,
            "country": region["country"],
            "biome": region["biome"],
            "split": region["split"],
            "lat": f"{chip_lat:.6f}",
            "lon": f"{chip_lon:.6f}",
            "scene_id": chip_item.id,
            "datetime": str(chip_item.datetime)[:10],
            "cloud_cover": chip_item.properties.get("eo:cloud_cover", ""),
            "label": label,
            "caption": caption,
            "n_features": len(counts),
            "building_count": building_count,
            "feature_counts": json.dumps(counts, sort_keys=True),
            "worldcover_class": wc_class,
            "worldcover_fraction": wc_fraction,
        }
        t0 = time.perf_counter()
        emit(row)
        timings["append"] += time.perf_counter() - t0
        added += 1
        if args.log_every <= 1 or added % args.log_every == 0:
            print(f"[{region_id}] {tid} -> {caption}", flush=True)
    if budget is not None and added >= budget:
        budget_hit = True
    return {
        "added": added,
        "skipped_chips": skipped,
        "worldcover_missing": worldcover_missing,
        "timings": timings,
        "chips_time": time.perf_counter() - chip_t0,
        "valid_pct": valid_pct,
        "scenes": [it.id for it in used_items],
        "budget_hit": budget_hit,
    }


def run_region_worker(region: dict[str, Any], args, out_dir: str, done: set[str], existing: int, cap: int,
                      budget: int | None) -> dict[str, Any]:
    """Worker-process entry: prepare + chip one region; rows are returned to the single writer."""
    out = Path(out_dir)
    region_id = region["region_id"]
    start = time.perf_counter()
    try:
        prepared = prepare_region(region, args, out / "_cache")
    except Exception as exc:  # STAC/Geofabrik/pyrosm/scene failure: retried once at the end of the run
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    if not prepared["ranked"]:
        return {"status": "no_scene", "timings": prepared["timings"]}
    if prepared["mosaic"] is None:
        return {"status": "no_mosaic", "n_ranked": len(prepared["ranked"]), "timings": prepared["timings"]}
    rows: list[dict[str, Any]] = []
    result = chip_region(region, args, prepared, out, done, existing, cap, budget, rows.append)
    del prepared
    result.update(status="ok", rows=rows, wall_time=time.perf_counter() - start)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions-csv", default=str(THIS / "regions.csv"))
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "raw" / "osm_sentinel"))
    parser.add_argument("--region", action="append", help="Only this region_id (repeatable)")
    parser.add_argument("--chips-per-region", type=int, default=8)
    parser.add_argument(
        "--sample-mode",
        choices=["grid", "random"],
        default="grid",
        help="grid = non-overlapping local grid (default); random = old random-point sampler",
    )
    parser.add_argument("--no-worldcover", action="store_true", help="Disable ESA WorldCover fallback")
    parser.add_argument("--max-chips", type=int, default=0, help="Stop after N total chips (0 = no limit)")
    parser.add_argument("--radius-km", type=float, default=None, help="Override region radius")
    parser.add_argument("--chip-m", type=float, default=640.0, help="Chip side length in metres")
    parser.add_argument("--chip-px", type=int, default=64)
    parser.add_argument("--date-start", default="2024-01-01")
    parser.add_argument("--date-end", default="2025-12-31")
    parser.add_argument("--max-cloud", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=1000, help="Print every N successful chips")
    parser.add_argument(
        "--no-prefetch",
        action="store_true",
        help="Disable background OSM PBF/GeoParquet prefetch for the next region (--workers 1 only)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Regions processed concurrently in worker processes (~1.5 GB RSS each); 1 = in-process",
    )
    parser.add_argument(
        "--require-osm-cache",
        action="store_true",
        help="Fail if a source PBF is not already in the prefetched provider cache",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    image_dir = out_dir / "images"
    cache_dir = out_dir / "_cache"
    image_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.csv"
    captions_path = out_dir / "captions.jsonl"
    check_build_meta(out_dir, args, args.overwrite)
    if args.overwrite:
        for path in (manifest_path, captions_path):
            if path.exists():
                path.unlink()
        stats = new_stats()
        done: set[str] = set()
    else:
        done, stats = load_existing(manifest_path, captions_path)
    stats_path = out_dir / "dataset_stats.json"
    if stats_path.exists() and not args.overwrite:
        try:
            previous = json.loads(stats_path.read_text())
        except (OSError, json.JSONDecodeError):
            previous = {}
        stats["failed_regions"].update(previous.get("failed_regions", []))
        stats["skipped_chips"] = previous.get("skipped_chips", 0)
        stats["worldcover_missing"] = previous.get("worldcover_missing", 0)
    write_stats(stats_path, stats)

    if args.max_chips and stats["n"] >= args.max_chips:
        print(f"already have {stats['n']} chips; nothing to do")
        return

    regions = read_regions(Path(args.regions_csv))
    if args.region:
        wanted = set(args.region)
        regions = [row for row in regions if row["region_id"] in wanted]
    if not regions:
        raise SystemExit("No regions selected")

    # --- A (a/osm-1m-fixes): resume skip, per-region RNG/cap, full-setup prefetch -----------------
    status_path = out_dir / "region_status.json"
    try:
        region_status: dict[str, Any] = json.loads(status_path.read_text()) if status_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        region_status = {}
    per_region: Counter[str] = Counter()  # chips already in the manifest, per region (cap + resume skip)
    done_by_region: dict[str, set[str]] = {}  # tile IDs per region, shipped to that region's worker
    if manifest_path.exists():
        with manifest_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("region_id"):
                    per_region[row["region_id"]] += 1
                    if row.get("tile_id"):
                        done_by_region.setdefault(row["region_id"], set()).add(row["tile_id"])

    def region_cap(region: dict[str, Any]) -> tuple[int, int]:
        lat, lon, radius_km, _ = region_geometry(region, args)
        if args.sample_mode != "grid":
            return args.chips_per_region, args.chips_per_region
        n_cells = sum(1 for _ in grid_candidates(region["region_id"], lat, lon, radius_km, args.chip_m))
        return min(args.chips_per_region, n_cells), n_cells

    todo = []
    for region in regions:
        rid = region["region_id"]
        cap, _ = region_cap(region)
        st = region_status.get(rid, {})
        if per_region[rid] >= cap:
            continue  # quota already in the manifest: no STAC/OSM/scene I/O on resume
        if st.get("policy") == SCENE_POLICY and st.get("attempted"):
            continue  # already tried every cell with the current scene policy
        todo.append(region)
    print(f"[resume] {len(regions) - len(todo)} regions complete or already attempted; {len(todo)} to process",
          flush=True)

    def save_region_status() -> None:
        tmp = status_path.with_name(status_path.name + ".tmp")
        tmp.write_text(json.dumps(region_status, indent=1, sort_keys=True) + "\n")
        tmp.replace(status_path)

    def remaining_budget() -> int | None:
        return max(args.max_chips - stats["n"], 0) if args.max_chips else None

    def max_reached() -> bool:
        return bool(args.max_chips and stats["n"] >= args.max_chips)

    def emit_row(row: dict[str, Any]) -> None:
        append_chip(manifest_path, captions_path, row)
        update_stats(stats, row)
        done.add(row["tile_id"])

    all_timings: Counter[str] = Counter()
    retried: set[str] = set()

    def handle_failure(region: dict[str, Any], status: str, detail: str, queue: list) -> None:
        """Failed-region bookkeeping shared by the sequential and parallel paths."""
        region_id = region["region_id"]
        if status == "failed":
            print(f"[{region_id}] REGION FAILED, skipped: {detail}", flush=True)
            stats["failed_regions"].add(region_id)
            if region_id not in retried:  # A: one automatic retry at the end of this run
                retried.add(region_id)
                queue.append(region)
                print(f"[{region_id}] queued for one retry at the end of this run", flush=True)
        elif status == "no_scene":
            print(f"[{region_id}] no readable scene below {args.max_cloud}% cloud; skipped", flush=True)
            stats["failed_regions"].add(region_id)
        else:
            print(f"[{region_id}] REGION FAILED, skipped: no usable scene among {detail} candidates", flush=True)
            stats["failed_regions"].add(region_id)

    def finish_region(region: dict[str, Any], result: dict[str, Any], wall_elapsed: float, cap: int,
                      n_cells: int) -> bool:
        """Book a finished region (rows already emitted). Returns True when --max-chips is reached."""
        region_id = region["region_id"]
        added = result["added"]
        timings = result["timings"]
        stats["skipped_chips"] += result["skipped_chips"]
        stats["worldcover_missing"] += result["worldcover_missing"]
        stats["failed_regions"].discard(region_id)
        elapsed = result["chips_time"]
        rate = added / elapsed if elapsed > 0 else 0.0
        wall_rate = added / wall_elapsed if wall_elapsed > 0 else 0.0
        all_timings.update(timings)
        per_region[region_id] += added
        print(
            f"[{region_id}] timing: {added} chips, chip={rate:.1f}/s, wall={wall_rate:.1f}/s, "
            f"chips_time={elapsed:.1f}s, setup={wall_elapsed - elapsed:.1f}s, "
            + ", ".join(f"{key}={value:.2f}s" for key, value in sorted(timings.items())),
            flush=True,
        )
        write_stats(stats_path, stats)
        if max_reached() or result.get("budget_hit"):
            return True
        region_status[region_id] = {
            "policy": SCENE_POLICY, "attempted": True, "chips": per_region[region_id], "cap": cap,
            "cells": n_cells, "valid_pct": round(result["valid_pct"], 1), "scenes": result["scenes"],
        }
        save_region_status()
        return False

    run_t0 = time.perf_counter()
    n_start = stats["n"]
    if args.workers <= 1:
        _run_sequential(args, todo, cache_dir, out_dir, done, per_region, region_cap, remaining_budget,
                        emit_row, handle_failure, finish_region)
    else:
        _run_parallel(args, todo, out_dir, done_by_region, per_region, region_cap, remaining_budget,
                      max_reached, emit_row, handle_failure, finish_region, stats, stats_path)

    write_stats(stats_path, stats)
    run_s = time.perf_counter() - run_t0
    made = stats["n"] - n_start
    print(f"run: {made} chips in {run_s:.1f}s ({made / run_s if run_s > 0 else 0:.1f} chips/s, "
          f"workers={args.workers})", flush=True)
    print("timings: " + json.dumps({key: round(value, 2) for key, value in sorted(all_timings.items())}), flush=True)
    print(json.dumps(stats_payload(stats), indent=2))


def _run_sequential(args, todo, cache_dir, out_dir, done, per_region, region_cap, remaining_budget,
                    emit_row, handle_failure, finish_region) -> None:
    """--workers 1: the original loop (one prefetch thread, per-chip manifest appends)."""
    executor: ThreadPoolExecutor | None = None
    if not args.no_prefetch and len(todo) > 1:
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="region-prefetch")
    next_future: Future | None = (executor.submit(prepare_region, todo[0], args, cache_dir)
                                  if executor and todo else None)
    index = 0
    while index < len(todo):  # todo can grow (one retry per failed region)
        region = todo[index]
        region_id = region["region_id"]
        cap, n_cells = region_cap(region)
        existing = per_region[region_id]
        region_start = time.perf_counter()
        print(f"[{region_id}] preparing (have {existing}/{cap} chips, {n_cells} cells)", flush=True)
        try:
            prepared = next_future.result() if next_future is not None else prepare_region(region, args, cache_dir)
            error = ""
        except Exception as exc:
            prepared, error = None, f"{type(exc).__name__}: {exc}"
        # start the next region's setup now so it overlaps this region's chip loop
        next_future = (executor.submit(prepare_region, todo[index + 1], args, cache_dir)
                       if executor is not None and index + 1 < len(todo) else None)
        index += 1
        if prepared is None:
            handle_failure(region, "failed", error, todo)
            if next_future is None and executor is not None and index < len(todo):
                next_future = executor.submit(prepare_region, todo[index], args, cache_dir)
            continue
        if not prepared["ranked"]:
            handle_failure(region, "no_scene", "", todo)
            continue
        if prepared["mosaic"] is None:
            handle_failure(region, "no_mosaic", str(len(prepared["ranked"])), todo)
            continue
        result = chip_region(region, args, prepared, out_dir, done, existing, cap, remaining_budget(), emit_row)
        del prepared
        if finish_region(region, result, time.perf_counter() - region_start, cap, n_cells):
            break
    if executor is not None:
        executor.shutdown(wait=False, cancel_futures=True)


def _run_parallel(args, todo, out_dir, done_by_region, per_region, region_cap, remaining_budget, max_reached,
                  emit_row, handle_failure, finish_region, stats, stats_path) -> None:
    """--workers N: N regions in flight in worker processes; this process appends their rows."""
    import multiprocessing as mp
    import signal
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    queue = list(todo)
    pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                               max_tasks_per_child=8)

    def kill_workers() -> None:
        for proc in list((getattr(pool, "_processes", None) or {}).values()):
            try:
                proc.terminate()
            except Exception:
                pass

    def on_term(signum, frame):  # pkill of the main process must not leave orphan workers behind
        kill_workers()
        raise SystemExit(128 + signum)

    old_term = signal.signal(signal.SIGTERM, on_term)
    running: dict[Future, tuple[dict[str, Any], int, int, float]] = {}
    stop = False

    def refill(reserved: int) -> None:
        """Keep N regions in flight. `reserved` = rows received but not yet appended (--max-chips)."""
        while queue and len(running) < args.workers and not max_reached():
            budget = remaining_budget()
            if budget is not None:
                budget -= reserved
                if budget <= 0:
                    return
            region = queue.pop(0)
            region_id = region["region_id"]
            cap, n_cells = region_cap(region)
            existing = per_region[region_id]
            print(f"[{region_id}] submitted (have {existing}/{cap} chips, {n_cells} cells)", flush=True)
            fut = pool.submit(run_region_worker, region, args, str(out_dir),
                              done_by_region.get(region_id, set()), existing, cap, budget)
            running[fut] = (region, cap, n_cells, time.perf_counter())

    try:
        refill(0)
        while running and not stop:
            finished, _ = wait(list(running), return_when=FIRST_COMPLETED)
            ok = []
            for fut in finished:
                region, cap, n_cells, started = running.pop(fut)
                try:
                    result = fut.result()
                except Exception as exc:
                    if type(exc).__name__ == "BrokenProcessPool":
                        raise  # a worker died (OOM kill?): stop; a rerun resumes from the manifest
                    # unexpected chip-loop crash in a worker: treat it as a failed region
                    result = {"status": "failed", "error": f"worker crashed: {type(exc).__name__}: {exc}"}
                if result["status"] == "ok":
                    ok.append((region, cap, n_cells, started, result))
                else:
                    handle_failure(region, result["status"],
                                   result.get("error", str(result.get("n_ranked", ""))), queue)
                    write_stats(stats_path, stats)
            # resubmit before the (slower) manifest appends so no worker idles meanwhile
            refill(sum(len(item[4]["rows"]) for item in ok))
            for region, cap, n_cells, started, result in ok:
                kept = 0
                for row in result.pop("rows"):
                    if max_reached():
                        result["budget_hit"] = True
                        break
                    emit_row(row)
                    kept += 1
                result["added"] = kept
                if finish_region(region, result, time.perf_counter() - started, cap, n_cells):
                    stop = True
                    break
            if not stop:
                refill(0)
    finally:
        signal.signal(signal.SIGTERM, old_term)
        if stop or running:
            kill_workers()
        pool.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    main()
