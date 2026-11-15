#!/usr/bin/env python
"""Re-caption an existing OSM-Sentinel build with area-comparable feature weights (A, for J/D review).

Bug in osm_captions._weight (as built): polygons are weighted by AREA and lines by LENGTH, both in
degrees. A river line crossing a 1.28 km chip has "weight" ~1e-2 while a polygon covering the whole
chip has ~1.3e-4, so any mapped river/stream line outranks every land-cover polygon and becomes the
caption subject and label (e.g. every sampled Amazon chip was captioned "River.").

Fix used here: a line's weight is its length times a nominal on-the-ground width, so lines and
polygons are both areas (a river line across a chip ≈ 3 % of the chip, not 80x the chip). Everything
else (phrases, thresholds, ordering) is J's unchanged caption logic.

Images are not re-downloaded: this reads the cached per-region OSM GeoParquet and each chip's
lat/lon from manifest.csv, and writes a parallel data dir whose images/ is a symlink to the original.

    python experiments/osm_sentinel/recaption.py \
        --src data/raw/osm_sentinel --dst data/raw/osm_sentinel_areafix
    python experiments/osm_sentinel/train.py --data data/raw/osm_sentinel_areafix ...
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
THIS = Path(__file__).resolve().parent
for path in (ROOT, THIS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import osm_captions  # noqa: E402
from build_dataset import FIELDS, caption_record  # noqa: E402

METRES_PER_DEG = 111_320.0
# Nominal widths for linear features (metres). Anything else linear uses DEFAULT_LINE_WIDTH_M.
LINE_WIDTH_M = {"river": 30.0, "canal": 15.0, "stream": 4.0, "drainage channel": 3.0,
                "major road": 15.0, "railway": 8.0, "residential street": 8.0}
DEFAULT_LINE_WIDTH_M = 5.0


def _area_weight_factory():
    """Return a _weight(geom, chip) that yields comparable areas (deg²) for polygons and lines."""
    current_phrase = {"value": None}

    def weight(geom, chip) -> float:
        try:
            inter = geom.intersection(chip)
        except Exception:
            try:
                from shapely import make_valid

                inter = make_valid(geom).intersection(chip)
            except Exception:
                return 0.0
        if inter.is_empty:
            return 0.0
        if inter.geom_type in ("Polygon", "MultiPolygon"):
            return float(inter.area)
        if inter.geom_type in ("LineString", "MultiLineString"):
            width = LINE_WIDTH_M.get(current_phrase["value"], DEFAULT_LINE_WIDTH_M) / METRES_PER_DEG
            return float(inter.length) * width
        if inter.geom_type == "GeometryCollection":
            return float(sum(weight(g, chip) for g in inter.geoms))
        return 0.0

    return weight, current_phrase


def patched_build_caption(features, buildings, chip_bbox):
    """J's build_caption_from_features with the phrase-aware area weight."""
    from shapely.geometry import box

    weight, current = _area_weight_factory()
    chip = box(*chip_bbox)
    counts: Counter = Counter()
    weights: Counter = Counter()
    for tags, geom in features:
        phrases = osm_captions.visible_phrases(tags)
        if not phrases or geom is None or geom.is_empty:
            continue
        for phrase in phrases:  # only change vs J: the weight is an area for lines too
            current["value"] = phrase
            w = weight(geom, chip)
            if w <= 0:
                continue
            counts[phrase] += 1
            weights[phrase] += w
    building_count = sum(1 for g in buildings if g is not None and not g.is_empty and chip.contains(g.centroid))
    return osm_captions._finish_caption(counts, weights, building_count, chip.area)


def main() -> None:
    import math

    from osm_source import load_region_features

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data/raw/osm_sentinel"))
    ap.add_argument("--dst", default=str(ROOT / "data/raw/osm_sentinel_areafix"))
    ap.add_argument("--chip-m", type=float, default=1280.0)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    if not (dst / "images").exists():
        (dst / "images").symlink_to((src / "images").resolve())
    rows = list(csv.DictReader((src / "manifest.csv").open(newline="")))
    by_region: dict[str, list[dict]] = {}
    for r in rows:
        by_region.setdefault(r["region_id"], []).append(r)

    out_rows, changed, dropped = [], Counter(), 0
    for region_id, rr in by_region.items():
        rf = load_region_features(region_id, None, src / "_cache")  # cached GeoParquet; bbox unused
        for r in rr:
            lat, lon = float(r["lat"]), float(r["lon"])
            h = args.chip_m / 2 / METRES_PER_DEG
            hl = h / max(math.cos(math.radians(lat)), 1e-6)
            bbox = (lon - hl, lat - h, lon + hl, lat + h)
            feats, blds = rf.for_bbox(bbox)
            caption, label, counts, nb = patched_build_caption(feats, blds, bbox)
            if caption is None:
                dropped += 1
                continue
            if label != r["label"]:
                changed[(r["label"], label)] += 1
            out_rows.append({**r, "caption": caption, "label": label, "n_features": len(counts),
                             "building_count": nb, "feature_counts": json.dumps(counts, sort_keys=True)})
        print(f"[{region_id}] {len(rr)} chips re-captioned", flush=True)

    with (dst / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out_rows)
    with (dst / "captions.jsonl").open("w") as f:
        for r in out_rows:
            f.write(json.dumps(caption_record(r), sort_keys=True) + "\n")
    summary = {
        "n": len(out_rows), "dropped": dropped,
        "labels_before": dict(Counter(r["label"] for r in rows)),
        "labels_after": dict(Counter(r["label"] for r in out_rows)),
        "label_changes": {f"{a}->{b}": n for (a, b), n in changed.most_common()},
    }
    (dst / "recaption_stats.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
