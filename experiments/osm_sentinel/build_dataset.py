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
import csv
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
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


def find_scene(bbox: tuple[float, float, float, float], start: str, end: str, max_cloud: float):
    from pystac_client import Client

    catalog = Client.open(STAC_URL)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=list(bbox),
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
        limit=50,
    )
    items = list(search.items())
    if not items:
        return None
    return min(items, key=lambda item: float(item.properties.get("eo:cloud_cover", 100)))


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


def append_chip(
    manifest_path: Path,
    captions_path: Path,
    row: dict[str, Any],
) -> None:
    """Append a completed chip to both outputs and flush immediately."""
    new_file = not manifest_path.exists() or manifest_path.stat().st_size == 0
    with manifest_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
    with captions_path.open("a") as handle:
        handle.write(json.dumps(caption_record(row), sort_keys=True) + "\n")
        handle.flush()


def main() -> None:
    import rasterio

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
        help="Disable background OSM PBF/GeoParquet prefetch for the next region",
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

    rng = random.Random(args.seed)
    all_timings: Counter[str] = Counter()
    executor: ThreadPoolExecutor | None = None
    prefetched: dict[str, Future] = {}
    if not args.no_prefetch and len(regions) > 1:
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="osm-prefetch")

    def prefetch_next(index: int) -> None:
        """Download/parse the next region's OSM while this region crops chips."""
        if executor is None or index >= len(regions):
            return
        nxt = regions[index]
        nxt_id = nxt["region_id"]
        if nxt_id in prefetched:
            return
        _, _, _, nxt_bbox = region_geometry(nxt, args)
        prefetched[nxt_id] = executor.submit(load_region_features, nxt_id, nxt_bbox, cache_dir)

    for index, region in enumerate(regions):
        region_id = region["region_id"]
        future = prefetched.pop(region_id, None)
        lat, lon, radius_km, bbox = region_geometry(region, args)
        scene_bbox = bbox_from_center(lat, lon, radius_km + args.chip_m / 1000.0)
        timings: Counter[str] = Counter()
        region_start = time.perf_counter()
        print(f"[{region_id}] searching Sentinel-2 scene", flush=True)
        try:
            t0 = time.perf_counter()
            item = find_scene(bbox, args.date_start, args.date_end, args.max_cloud)
            timings["scene_search"] += time.perf_counter() - t0
            if item is None:
                print(f"[{region_id}] no scene below {args.max_cloud}% cloud; skipped", flush=True)
                stats["failed_regions"].add(region_id)
                continue
            print(f"[{region_id}] scene {item.id} cloud={item.properties.get('eo:cloud_cover')}", flush=True)
            asset = item.assets.get("visual")
            if asset is None:
                raise RuntimeError(f"Scene {item.id} has no visual asset")
            t0 = time.perf_counter()
            if future is not None:
                try:
                    region_features = future.result()
                except Exception as exc:
                    print(
                        f"[{region_id}] OSM prefetch failed ({type(exc).__name__}: {exc}); retrying synchronously",
                        flush=True,
                    )
                    region_features = load_region_features(region_id, bbox, cache_dir)
            else:
                region_features = load_region_features(region_id, bbox, cache_dir)
            timings["osm_load"] += time.perf_counter() - t0
        except Exception as exc:  # STAC/Geofabrik/pyrosm failure: skip the region (rerun to retry it)
            print(f"[{region_id}] REGION FAILED, skipped: {type(exc).__name__}: {exc}", flush=True)
            stats["failed_regions"].add(region_id)
            continue
        stats["failed_regions"].discard(region_id)
        print(f"[{region_id}] pyrosm region loaded", flush=True)
        prefetch_next(index + 1)
        # WorldCover is a fallback: load it lazily the first time a chip has
        # no OSM land-cover polygon. Natural/farm regions then pay zero
        # WorldCover I/O, while sparse urban chips still get a fallback.
        region_wc = None
        region_wc_attempted = bool(args.no_worldcover)

        try:
            t0 = time.perf_counter()
            scene = rasterio.open(asset.href)
            timings["scene_open"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            region_rgb, region_transform, region_crs = read_region(scene, scene_bbox)
            timings["scene_read"] += time.perf_counter() - t0
            scene.close()
            if region_rgb is None:
                raise RuntimeError("empty Sentinel region window")
        except Exception as exc:
            print(f"[{region_id}] scene read failed, skipped: {type(exc).__name__}: {exc}", flush=True)
            stats["failed_regions"].add(region_id)
            continue

        added = 0
        chip_t0 = time.perf_counter()
        if args.sample_mode == "grid":
            candidates = list(grid_candidates(region_id, lat, lon, radius_km, args.chip_m))
            rng.shuffle(candidates)
            candidates = candidates[: args.chips_per_region]
        else:
            candidates = random_candidates(region_id, lat, lon, radius_km, args.chips_per_region, rng)
        for tid, chip_lat, chip_lon in candidates:
            if args.max_chips and stats["n"] >= args.max_chips:
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
                stats["skipped_chips"] += 1
                continue
            if image is None:
                stats["skipped_chips"] += 1
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
                stats["skipped_chips"] += 1
                continue
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
                        stats["worldcover_missing"] += 1
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
                stats["skipped_chips"] += 1
                continue
            image_rel = f"images/{tid}.jpg"
            try:
                t0 = time.perf_counter()
                image.save(out_dir / image_rel, quality=92)
                timings["image_save"] += time.perf_counter() - t0
            except Exception as exc:
                print(f"[{region_id}] {tid} image save failed, skipped: {type(exc).__name__}: {exc}", flush=True)
                stats["skipped_chips"] += 1
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
                "scene_id": item.id,
                "datetime": str(item.datetime)[:10],
                "cloud_cover": item.properties.get("eo:cloud_cover", ""),
                "label": label,
                "caption": caption,
                "n_features": len(counts),
                "building_count": building_count,
                "feature_counts": json.dumps(counts, sort_keys=True),
                "worldcover_class": wc_class,
                "worldcover_fraction": wc_fraction,
            }
            t0 = time.perf_counter()
            append_chip(manifest_path, captions_path, row)
            timings["append"] += time.perf_counter() - t0
            update_stats(stats, row)
            done.add(tid)
            added += 1
            if args.log_every <= 1 or added % args.log_every == 0:
                print(f"[{region_id}] {tid} -> {caption}", flush=True)
        elapsed = time.perf_counter() - chip_t0
        wall_elapsed = time.perf_counter() - region_start
        rate = added / elapsed if elapsed > 0 else 0.0
        wall_rate = added / wall_elapsed if wall_elapsed > 0 else 0.0
        all_timings.update(timings)
        print(
            f"[{region_id}] timing: {added} chips, chip={rate:.1f}/s, wall={wall_rate:.1f}/s, "
            f"chips_time={elapsed:.1f}s, setup={wall_elapsed - elapsed:.1f}s, "
            + ", ".join(f"{key}={value:.2f}s" for key, value in sorted(timings.items())),
            flush=True,
        )
        write_stats(stats_path, stats)
        if args.max_chips and stats["n"] >= args.max_chips:
            break

    if executor is not None:
        executor.shutdown(wait=False, cancel_futures=True)
    write_stats(stats_path, stats)
    print("timings: " + json.dumps({key: round(value, 2) for key, value in sorted(all_timings.items())}), flush=True)
    print(json.dumps(stats_payload(stats), indent=2))


if __name__ == "__main__":
    main()
