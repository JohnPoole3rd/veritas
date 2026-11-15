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
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
THIS = Path(__file__).resolve().parent
for path in (ROOT, THIS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from osm_captions import build_caption_from_features  # noqa: E402
from osm_source import load_region_features  # noqa: E402

STAC_URL = "https://earth-search.aws.element84.com/v1"
FIELDS = [
    "tile_id", "image", "region_id", "country", "biome", "split", "lat", "lon",
    "scene_id", "datetime", "cloud_cover", "label", "caption", "n_features",
    "building_count", "feature_counts",
]


def bbox_from_center(lat: float, lon: float, radius_km: float) -> tuple[float, float, float, float]:
    """Return (west, south, east, north)."""
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * max(math.cos(math.radians(lat)), 1e-6))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


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


def read_chip(item, lat: float, lon: float, chip_m: float, chip_px: int):
    """Read a true-colour COG window and resize it to `chip_px`."""
    import numpy as np
    import rasterio
    from PIL import Image
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    asset = item.assets.get("visual")
    if asset is None:
        raise RuntimeError(f"Scene {item.id} has no visual asset")
    half_lat = chip_m / 2 / 111_320
    half_lon = chip_m / 2 / (111_320 * max(math.cos(math.radians(lat)), 1e-6))
    west, south, east, north = lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat

    with rasterio.open(asset.href) as src:
        bounds = transform_bounds("EPSG:4326", src.crs, west, south, east, north, densify_pts=21)
        window = from_bounds(*bounds, src.transform).round_offsets().round_lengths()
        if window.width < 8 or window.height < 8:
            return None
        data = src.read(window=window)
    if data.size == 0:
        return None
    if data.shape[0] >= 3:
        rgb = np.transpose(data[:3], (1, 2, 0))
    else:
        rgb = np.repeat(np.transpose(data[:1], (1, 2, 0)), 3, axis=2)
    if rgb.dtype != np.uint8:
        # TCI is uint8; this fallback handles raw reflectance assets.
        rgb = np.clip(rgb.astype("float32") / 4000.0 * 255.0, 0, 255).astype("uint8")
    image = Image.fromarray(rgb).resize((chip_px, chip_px), Image.BILINEAR)
    if np.asarray(image).max() < 5:
        return None
    return image


def read_regions(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def tile_id(region_id: str, lat: float, lon: float) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{region_id}_{abs(lat):.4f}{ns}_{abs(lon):.4f}{ew}"


def new_stats() -> dict[str, Any]:
    return {
        "n": 0,
        "by_split": Counter(),
        "by_country": Counter(),
        "by_label": Counter(),
        "regions": set(),
        "failed_regions": set(),
        "skipped_chips": 0,
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
    }


def write_stats(path: Path, stats: dict[str, Any]) -> None:
    path.write_text(json.dumps(stats_payload(stats), indent=2) + "\n")


def caption_record(row: dict[str, Any]) -> dict[str, Any]:
    try:
        counts = json.loads(row.get("feature_counts") or "{}")
    except json.JSONDecodeError:
        counts = {}
    return {
        "tile_id": row["tile_id"],
        "image": row["image"],
        "caption": row["caption"],
        "label": row["label"],
        "split": row["split"],
        "region_id": row["region_id"],
        "country": row["country"],
        "biome": row["biome"],
        "feature_counts": counts,
    }


def rewrite_captions(manifest_path: Path, captions_path: Path) -> None:
    """Repair captions from the manifest after an interrupted run."""
    tmp = captions_path.with_name(captions_path.name + ".tmp")
    with manifest_path.open(newline="") as source, tmp.open("w") as out:
        for row in csv.DictReader(source):
            out.write(json.dumps(caption_record(row), sort_keys=True) + "\n")
    tmp.replace(captions_path)


def load_existing(manifest_path: Path, captions_path: Path) -> tuple[set[str], dict[str, Any]]:
    stats = new_stats()
    done: set[str] = set()
    if manifest_path.exists():
        with manifest_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions-csv", default=str(THIS / "regions.csv"))
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "raw" / "osm_sentinel"))
    parser.add_argument("--region", action="append", help="Only this region_id (repeatable)")
    parser.add_argument("--chips-per-region", type=int, default=8)
    parser.add_argument("--max-chips", type=int, default=0, help="Stop after N total chips (0 = no limit)")
    parser.add_argument("--radius-km", type=float, default=None, help="Override region radius")
    parser.add_argument("--chip-m", type=float, default=1280.0, help="Chip side length in metres")
    parser.add_argument("--chip-px", type=int, default=128)
    parser.add_argument("--date-start", default="2024-01-01")
    parser.add_argument("--date-end", default="2025-12-31")
    parser.add_argument("--max-cloud", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    image_dir = out_dir / "images"
    cache_dir = out_dir / "_cache"
    image_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.csv"
    captions_path = out_dir / "captions.jsonl"
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
        previous = json.loads(stats_path.read_text())
        stats["failed_regions"].update(previous.get("failed_regions", []))
        stats["skipped_chips"] = previous.get("skipped_chips", 0)
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
    for region in regions:
        region_id = region["region_id"]
        lat, lon = float(region["lat"]), float(region["lon"])
        radius_km = args.radius_km if args.radius_km is not None else float(region["radius_km"])
        bbox = bbox_from_center(lat, lon, radius_km)
        print(f"[{region_id}] searching Sentinel-2 scene", flush=True)
        try:
            item = find_scene(bbox, args.date_start, args.date_end, args.max_cloud)
            if item is None:
                print(f"[{region_id}] no scene below {args.max_cloud}% cloud; skipped", flush=True)
                continue
            print(f"[{region_id}] scene {item.id} cloud={item.properties.get('eo:cloud_cover')}", flush=True)
            region_features = load_region_features(region_id, bbox, cache_dir)
        except Exception as exc:  # STAC/Geofabrik/pyrosm failure: skip the region (rerun to retry it)
            print(f"[{region_id}] REGION FAILED, skipped: {type(exc).__name__}: {exc}", flush=True)
            stats["failed_regions"].add(region_id)
            continue
        stats["failed_regions"].discard(region_id)
        print(f"[{region_id}] pyrosm region loaded", flush=True)

        added = 0
        attempts = 0
        while added < args.chips_per_region and attempts < args.chips_per_region * 8:
            attempts += 1
            if args.max_chips and stats["n"] >= args.max_chips:
                break
            angle = rng.uniform(0, 2 * math.pi)
            distance = radius_km * math.sqrt(rng.random())
            chip_lat = lat + (distance / 111.32) * math.cos(angle)
            chip_lon = lon + (distance / (111.32 * math.cos(math.radians(lat)))) * math.sin(angle)
            tid = tile_id(region_id, chip_lat, chip_lon)
            if tid in done:
                continue
            try:
                image = read_chip(item, chip_lat, chip_lon, args.chip_m, args.chip_px)
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
                features, buildings = region_features.for_bbox(chip_bbox)
                caption, label, counts, building_count = build_caption_from_features(
                    features, buildings, chip_bbox
                )
            except Exception as exc:  # malformed OSM data for this chip: skip it, keep the run alive
                print(f"[{region_id}] {tid} caption failed, skipped: {type(exc).__name__}: {exc}", flush=True)
                stats["skipped_chips"] += 1
                continue
            if caption is None or len(counts) < 1:
                stats["skipped_chips"] += 1
                continue
            image_rel = f"images/{tid}.jpg"
            image.save(out_dir / image_rel, quality=92)
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
            }
            append_chip(manifest_path, captions_path, row)
            update_stats(stats, row)
            done.add(tid)
            added += 1
            print(f"[{region_id}] {tid} -> {caption}", flush=True)
        write_stats(stats_path, stats)
        if args.max_chips and stats["n"] >= args.max_chips:
            break

    write_stats(stats_path, stats)
    print(json.dumps(stats_payload(stats), indent=2))


if __name__ == "__main__":
    main()
