#!/usr/bin/env python
"""Re-caption an existing OSM-Sentinel build with the current captioner.

This is a migration helper for builds made before the line-weight fix. It reads
the cached per-region OSM GeoParquet and each chip's lat/lon from `manifest.csv`,
then writes a parallel data directory. Images are symlinked, not downloaded
again.

    python experiments/osm_sentinel/recaption.py \
        --src data/raw/osm_sentinel --dst data/raw/osm_sentinel_recaption

    python experiments/osm_sentinel/train.py --data data/raw/osm_sentinel_recaption ...
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
THIS = Path(__file__).resolve().parent
for path in (ROOT, THIS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_dataset import FIELDS, caption_record  # noqa: E402
from osm_captions import build_caption_from_features  # noqa: E402
from osm_source import RegionFeatures  # noqa: E402
from worldcover import CLASSES  # noqa: E402

METRES_PER_DEG = 111_320.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(ROOT / "data/raw/osm_sentinel"))
    parser.add_argument("--dst", default=str(ROOT / "data/raw/osm_sentinel_recaption"))
    parser.add_argument("--chip-m", type=float, default=1280.0)
    args = parser.parse_args()

    import geopandas as gpd

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    if not (dst / "images").exists():
        (dst / "images").symlink_to((src / "images").resolve())

    rows = list(csv.DictReader((src / "manifest.csv").open(newline="")))
    by_region: dict[str, list[dict]] = {}
    for row in rows:
        by_region.setdefault(row["region_id"], []).append(row)

    out_rows: list[dict] = []
    label_changes: Counter[str] = Counter()
    dropped = 0
    for region_id, region_rows in by_region.items():
        parquet = src / "_cache" / "osm" / f"{region_id}.geoparquet"
        if not parquet.exists():
            raise SystemExit(f"Missing cached OSM features: {parquet}")
        features = RegionFeatures(gpd.read_parquet(parquet))
        for row in region_rows:
            lat, lon = float(row["lat"]), float(row["lon"])
            half_lat = args.chip_m / 2 / METRES_PER_DEG
            half_lon = half_lat / max(math.cos(math.radians(lat)), 1e-6)
            bbox = (lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat)
            feats, buildings = features.for_bbox(bbox)
            wc = None
            try:
                code = int(row.get("worldcover_class") or 0)
                if code in CLASSES:
                    phrase, label = CLASSES[code]
                    wc = (phrase, float(row.get("worldcover_fraction") or 0), code, label)
            except (TypeError, ValueError):
                wc = None
            caption, label, counts, building_count = build_caption_from_features(
                feats, buildings, bbox, worldcover=wc
            )
            if caption is None:
                dropped += 1
                continue
            if label != row["label"]:
                label_changes[f"{row['label']}->{label}"] += 1
            out_rows.append({
                **row,
                "caption": caption,
                "label": label,
                "n_features": len(counts),
                "building_count": building_count,
                "feature_counts": json.dumps(counts, sort_keys=True),
            })
        print(f"[{region_id}] {len(region_rows)} chips re-captioned", flush=True)

    with (dst / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)
    with (dst / "captions.jsonl").open("w") as handle:
        for row in out_rows:
            handle.write(json.dumps(caption_record(row), sort_keys=True) + "\n")

    summary = {
        "n": len(out_rows),
        "dropped": dropped,
        "labels_before": dict(Counter(row["label"] for row in rows)),
        "labels_after": dict(Counter(row["label"] for row in out_rows)),
        "label_changes": dict(label_changes.most_common()),
    }
    (dst / "recaption_stats.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
