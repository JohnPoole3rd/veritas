#!/usr/bin/env python
"""Build an OSM-captioned Sentinel-2 dataset.

For each region in `regions.csv`:
  1. find a low-cloud Sentinel-2 L2A scene with STAC (Element84 Earth Search);
  2. load a cropped local OSM PBF with pyrosm (one download per region);
  3. sample chips, read them from the scene's COG, and caption each chip from OSM.

The output is intentionally plain:
  images/<tile_id>.jpg
  manifest.csv
  captions.jsonl
  dataset_stats.json

Example smoke run:
    python experiments/osm_sentinel/build_dataset.py \
      --region us_columbus --chips-per-region 1 --max-chips 1

Scale up with more regions, `--chips-per-region`, or a longer region list.
Network data is cached under `<out-dir>/_cache/`; reruns skip existing chips.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions-csv", default=str(THIS / "regions.csv"))
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "raw" / "osm_sentinel"))
    parser.add_argument("--region", action="append", help="Only this region_id (repeatable)")
    parser.add_argument("--chips-per-region", type=int, default=8)
    parser.add_argument("--max-chips", type=int, default=0, help="Stop after N new chips (0 = no limit)")
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
    rows: list[dict[str, Any]] = []
    if manifest_path.exists() and not args.overwrite:
        with manifest_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    done = {row["tile_id"] for row in rows}
    captions: list[dict[str, Any]] = []
    if captions_path.exists() and not args.overwrite:
        captions = [json.loads(line) for line in captions_path.read_text().splitlines() if line.strip()]

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
        item = find_scene(bbox, args.date_start, args.date_end, args.max_cloud)
        if item is None:
            print(f"[{region_id}] no scene below {args.max_cloud}% cloud; skipped", flush=True)
            continue
        print(f"[{region_id}] scene {item.id} cloud={item.properties.get('eo:cloud_cover')}", flush=True)
        region_features = load_region_features(region_id, bbox, cache_dir)
        print(f"[{region_id}] pyrosm region loaded", flush=True)

        added = 0
        attempts = 0
        while added < args.chips_per_region and attempts < args.chips_per_region * 8:
            attempts += 1
            angle = rng.uniform(0, 2 * math.pi)
            distance = radius_km * math.sqrt(rng.random())
            chip_lat = lat + (distance / 111.32) * math.cos(angle)
            chip_lon = lon + (distance / (111.32 * math.cos(math.radians(lat)))) * math.sin(angle)
            tid = tile_id(region_id, chip_lat, chip_lon)
            if tid in done:
                continue
            image = read_chip(item, chip_lat, chip_lon, args.chip_m, args.chip_px)
            if image is None:
                continue
            half_lat = args.chip_m / 2 / 111_320
            half_lon = args.chip_m / 2 / (111_320 * max(math.cos(math.radians(chip_lat)), 1e-6))
            chip_bbox = (chip_lon - half_lon, chip_lat - half_lat, chip_lon + half_lon, chip_lat + half_lat)
            features, buildings = region_features.for_bbox(chip_bbox)
            caption, label, counts, building_count = build_caption_from_features(
                features, buildings, chip_bbox
            )
            if caption is None or len(counts) < 1:
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
            rows.append(row)
            captions.append({
                "tile_id": tid,
                "image": image_rel,
                "caption": caption,
                "label": label,
                "split": region["split"],
                "region_id": region_id,
                "country": region["country"],
                "biome": region["biome"],
                "feature_counts": counts,
            })
            done.add(tid)
            added += 1
            print(f"[{region_id}] {tid} -> {caption}", flush=True)
            if args.max_chips and len(rows) >= args.max_chips:
                break
        if args.max_chips and len(rows) >= args.max_chips:
            break

    fields = [
        "tile_id", "image", "region_id", "country", "biome", "split", "lat", "lon",
        "scene_id", "datetime", "cloud_cover", "label", "caption", "n_features",
        "building_count", "feature_counts",
    ]
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with captions_path.open("w") as handle:
        for caption in captions:
            handle.write(json.dumps(caption, sort_keys=True) + "\n")

    stats = {
        "n": len(rows),
        "by_split": dict(Counter(row["split"] for row in rows)),
        "by_country": dict(Counter(row["country"] for row in rows)),
        "by_label": dict(Counter(row["label"] for row in rows)),
        "regions": sorted({row["region_id"] for row in rows}),
    }
    (out_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
