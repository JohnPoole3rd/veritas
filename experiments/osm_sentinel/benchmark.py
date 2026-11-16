#!/usr/bin/env python
"""Shared benchmark harness for the OSM-Sentinel experiment.

The benchmark deliberately keeps the evaluation data fixed while allowing the
rotation (the only fitted component) to change.

Suites:
  B1  EuroSAT test (10 classes, cached embeddings)
  B2  Upper Arlington 50 / unambiguous 37
  B3  fixed, balanced sample of the final build's test regions
  B4  harmonized six-class view of B1/B2/B3

This module is intentionally standalone: `sample` creates the fixed B3 file and
`run` evaluates a snapshot/tag on B1/B2/B3/B4 and writes one JSON result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
THIS = Path(__file__).resolve().parent
for path in (ROOT, THIS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from osm_captions import LABEL_PROMPTS  # noqa: E402

B4_CLASSES = ("forest", "cropland", "grass/pasture", "built", "water", "other")
B4_PROMPTS = {
    "forest": "forest",
    "cropland": "cropland or farmland",
    "grass/pasture": "grassland or pasture",
    "built": "built-up area",
    "water": "water",
    "other": "other land cover",
}

B4_FROM_OSM = {
    "forest": "forest",
    "farmland": "cropland",
    "green": "grass/pasture",
    "water": "water",
    "built": "built",
    "commercial": "built",
    "industrial": "built",
    "residential": "built",
    "airport": "built",
    "transport": "other",
    "barren": "other",
    "coast": "other",
    "ice": "other",
    "other": "other",
}
B4_FROM_EUROSAT = {
    "AnnualCrop": "cropland",
    "PermanentCrop": "cropland",
    "Pasture": "grass/pasture",
    "HerbaceousVegetation": "grass/pasture",
    "Forest": "forest",
    "Residential": "built",
    "Industrial": "built",
    "Commercial": "built",
    "Highway": "other",
    "River": "water",
    "SeaLake": "water",
    "Other": "other",
}

COUNTRY_TO_CONTINENT = {
    # Africa
    "Algeria": "Africa", "Angola": "Africa", "Benin": "Africa",
    "Botswana": "Africa", "Burkina Faso": "Africa", "Burundi": "Africa",
    "Cameroon": "Africa", "Chad": "Africa", "Congo (Brazzaville)": "Africa",
    "Congo (Kinshasa)": "Africa", "DR Congo": "Africa", "Egypt": "Africa",
    "Eritrea": "Africa", "Ethiopia": "Africa", "Ghana": "Africa",
    "Guinea": "Africa", "Ivory Coast": "Africa", "Kenya": "Africa",
    "Lesotho": "Africa", "Libya": "Africa", "Madagascar": "Africa",
    "Malawi": "Africa", "Mali": "Africa", "Mauritania": "Africa",
    "Morocco": "Africa", "Mozambique": "Africa", "Namibia": "Africa",
    "Nigeria": "Africa", "Rwanda": "Africa", "Senegal": "Africa",
    "Sierra Leone": "Africa", "South Africa": "Africa", "Sudan": "Africa",
    "Tanzania": "Africa", "United Republic of Tanzania": "Africa",
    "Tunisia": "Africa", "Uganda": "Africa", "Zambia": "Africa",
    "Zimbabwe": "Africa",
    # Asia
    "Afghanistan": "Asia", "Bahrain": "Asia", "Bangladesh": "Asia",
    "Cambodia": "Asia", "China": "Asia", "India": "Asia",
    "Indonesia": "Asia", "Iran": "Asia", "Iraq": "Asia",
    "Israel": "Asia", "Japan": "Asia", "Jordan": "Asia",
    "Kazakhstan": "Asia", "Korea, North": "Asia", "Laos": "Asia",
    "Malaysia": "Asia", "Mongolia": "Asia", "Myanmar": "Asia",
    "Nepal": "Asia", "Oman": "Asia", "Pakistan": "Asia",
    "Philippines": "Asia", "Saudi Arabia": "Asia", "Singapore": "Asia",
    "South Korea": "Asia", "Sri Lanka": "Asia", "Taiwan": "Asia",
    "Thailand": "Asia", "Turkey": "Asia", "United Arab Emirates": "Asia",
    "Uzbekistan": "Asia", "Vietnam": "Asia", "Yemen": "Asia",
    # Europe
    "Austria": "Europe", "Belarus": "Europe", "Bulgaria": "Europe",
    "Croatia": "Europe", "Czechia": "Europe", "Denmark": "Europe",
    "Finland": "Europe", "France": "Europe", "French Republic": "Europe",
    "Germany": "Europe", "Greece": "Europe", "Hungary": "Europe",
    "Iceland": "Europe", "Ireland": "Europe", "Italy": "Europe",
    "Latvia": "Europe", "Lithuania": "Europe", "Malta": "Europe",
    "Moldova": "Europe", "Netherlands": "Europe", "Norway": "Europe",
    "Poland": "Europe", "Portugal": "Europe", "Romania": "Europe",
    "Russia": "Europe", "Serbia": "Europe", "Spain": "Europe",
    "Sweden": "Europe", "Switzerland": "Europe", "Ukraine": "Europe",
    "United Kingdom": "Europe",
    # North America
    "Canada": "North America", "Costa Rica": "North America",
    "Cuba": "North America", "Dominican Republic": "North America",
    "El Salvador": "North America", "Guatemala": "North America",
    "Honduras": "North America", "Mexico": "North America",
    "Panama": "North America", "United States": "North America",
    # South America
    "Argentina": "South America", "Bolivia": "South America",
    "Brazil": "South America", "Chile": "South America",
    "Colombia": "South America", "Ecuador": "South America",
    "Paraguay": "South America", "Peru": "South America",
    "Uruguay": "South America",
    # Oceania
    "Australia": "Oceania", "New Zealand": "Oceania",
    "Papua New Guinea": "Oceania",
}


def b4_label(label: str, source: str) -> str:
    mapping = B4_FROM_OSM if source == "osm" else B4_FROM_EUROSAT
    if label not in mapping:
        raise KeyError(f"unknown {source} label for B4 mapping: {label!r}")
    return mapping[label]


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _region_rng(seed: int, region_id: str) -> random.Random:
    return random.Random(f"{seed}:{region_id}")


def sample_b3(
    data_dir: str | Path,
    out_csv: str | Path,
    *,
    seed: int = 20261115,
    cell_cap: int = 200,
    region_cap: int = 300,
    source: str = "osm",
) -> dict[str, Any]:
    """Create one deterministic, balanced sample of the test split.

    Rows are grouped by (B4 label, continent), then regions are visited
    round-robin. The output includes scene-sharing information so a
    scene-disjoint variant can be filtered without resampling.
    """
    data_dir = Path(data_dir)
    rows = [r for r in _read_manifest(data_dir / "manifest.csv") if r.get("split") == "test"]
    if not rows:
        raise SystemExit("no test rows in manifest")
    train_scenes = {r.get("scene_id") for r in _read_manifest(data_dir / "manifest.csv")
                    if r.get("split") == "train" and r.get("scene_id")}
    groups: dict[tuple[str, str], dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        label = b4_label(row["label"], source)
        continent = COUNTRY_TO_CONTINENT.get(row.get("country", ""))
        if continent is None:
            raise SystemExit(f"unknown country for continent mapping: {row.get('country')!r}")
        groups[(label, continent)][row["region_id"]].append(row)

    selected: list[dict[str, str]] = []
    per_region: Counter[str] = Counter()
    for key in sorted(groups):
        region_rows = groups[key]
        regions = sorted(region_rows)
        chosen = 0
        # Round-robin one region at a time until the cell cap or all rows are
        # exhausted. Per-region order is deterministic.
        while chosen < cell_cap:
            progressed = False
            for region_id in regions:
                if chosen >= cell_cap:
                    break
                if per_region[region_id] >= region_cap:
                    continue
                ordered = sorted(region_rows[region_id], key=lambda r: r["tile_id"])
                rng = _region_rng(seed, region_id)
                rng.shuffle(ordered)
                for row in ordered:
                    if row["tile_id"] in {x["tile_id"] for x in selected}:
                        continue
                    selected.append(row)
                    per_region[region_id] += 1
                    chosen += 1
                    progressed = True
                    break
            if not progressed:
                break

    # Coverage guarantee: any represented test region gets at least two rows.
    for region_id in sorted({r["region_id"] for r in rows}):
        have = sum(1 for r in selected if r["region_id"] == region_id)
        if have >= 2:
            continue
        pool = [r for r in rows if r["region_id"] == region_id]
        pool.sort(key=lambda r: r["tile_id"])
        for row in pool:
            if row["tile_id"] in {x["tile_id"] for x in selected}:
                continue
            selected.append(row)
            per_region[region_id] += 1
            if per_region[region_id] >= 2:
                break

    selected.sort(key=lambda r: r["tile_id"])
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["tile_id", "image", "region_id", "country", "label", "b4_label",
              "continent", "scene_id", "scene_shared_with_train"]
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            writer.writerow({
                **{k: row.get(k, "") for k in fields},
                "b4_label": b4_label(row["label"], source),
                "continent": COUNTRY_TO_CONTINENT[row["country"]],
                "scene_shared_with_train": str(row.get("scene_id") in train_scenes).lower(),
            })
    digest = hashlib.sha256("\n".join(sorted(r["tile_id"] for r in selected)).encode()).hexdigest()
    (out_csv.with_suffix(out_csv.suffix + ".sha256")).write_text(digest + "\n")
    mapping = out_csv.parent / "b4_mapping.json"
    mapping.write_text(json.dumps({
        "classes": list(B4_CLASSES),
        "prompts": B4_PROMPTS,
        "osm": B4_FROM_OSM,
        "eurosat": B4_FROM_EUROSAT,
        "country_to_continent": COUNTRY_TO_CONTINENT,
    }, indent=2, sort_keys=True) + "\n")
    return {"n": len(selected), "sha256": digest, "rows": selected}


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(len(classes))),
                                   average="macro", zero_division=0)),
        "n": int(len(y_true)),
    }


def _cluster_bootstrap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: Iterable[str],
    classes: list[str],
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, list[float]]:
    groups = np.asarray(list(groups))
    unique = np.unique(groups)
    if len(unique) < 2:
        return {}
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_boot):
        picked = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in picked])
        if not len(idx):
            continue
        m = _metrics(y_true[idx], y_pred[idx], classes)
        for key in ("accuracy", "balanced_accuracy", "macro_f1"):
            values[key].append(m[key])
    return {key: [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
            for key, vals in values.items()}


def _wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return [max(0.0, centre - half), min(1.0, centre + half)]


def _eval_fit(
    fit: dict[str, Any],
    images: np.ndarray,
    labels: list[str],
    text: np.ndarray,
    caption_labels: list[str],
    classes: list[str],
    *,
    groups: Iterable[str] | None = None,
) -> dict[str, Any]:
    from src.alignment import transform_images, transform_texts
    from src.retrieval import class_scores

    y = np.array([classes.index(label) for label in labels])
    scores = class_scores(
        transform_images(images, fit),
        transform_texts(text, fit),
        caption_labels,
        classes,
    )
    pred = scores.argmax(axis=1)
    result: dict[str, Any] = _metrics(y, pred, classes)
    result["correct"] = int((pred == y).sum())
    result["per_class"] = {
        cls: {"support": int((y == i).sum()), "accuracy": float((pred[y == i] == i).mean()) if (y == i).any() else None}
        for i, cls in enumerate(classes)
    }
    if groups is not None:
        result["ci95_region_clustered"] = _cluster_bootstrap(y, pred, groups, classes)
    return result


def _load_eurosat_eval(config) -> dict[str, Any] | None:
    """Cached main-pipeline EuroSAT test embeddings, if present."""
    from src.config import resolve_path

    emb = resolve_path("artifacts/embeddings", config)
    split_path = resolve_path(config["data"]["splits_dir"], config) / "test.json"
    paths = {
        "images": emb / "dinov2_test.npy",
        "captions": emb / "minilm_captions.npy",
        "index": emb / "minilm_caption_index.json",
        "split": split_path,
    }
    if not all(path.exists() for path in paths.values()):
        return None
    split = json.loads(split_path.read_text())
    index = json.loads(paths["index"].read_text())
    return {
        "images": np.load(paths["images"]),
        "text": np.load(paths["captions"]),
        "caption_labels": [index[str(i)]["label"] for i in range(len(index))],
        "labels": [item["label"] for item in split["items"]],
        "classes": list(split["class_names"]),
        "n": len(split["items"]),
    }


def _load_ua_eval(config) -> dict[str, Any] | None:
    """Cached Upper Arlington embeddings and labels, if present."""
    from src.config import resolve_path
    from src.local_tiles import load_local_manifest

    emb = resolve_path("artifacts/embeddings", config)
    local = emb / "dinov2_local.npy"
    manifest_path = Path(config["data"]["local_manifest"])
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    if not (local.exists() and manifest_path.exists()):
        return None
    manifest = load_local_manifest(manifest_path, check_files=False)
    qa_path = ROOT / "data/local_tiles/labels_qa.csv"
    clear = np.ones(len(manifest), dtype=bool)
    if qa_path.exists():
        import pandas as pd

        qa = pd.read_csv(qa_path).set_index("tile_id")["ambiguous"].astype(str)
        ambiguous = manifest.tile_id.map(qa).str.lower().eq("true").to_numpy()
        clear = ~ambiguous
    return {
        "images": np.load(local),
        "labels": manifest.label.tolist(),
        "clear": clear,
        "tile_ids": manifest.tile_id.tolist(),
        "n": len(manifest),
    }


def _run_cached_suites(result: dict[str, Any], config, fit) -> None:
    """Evaluate EuroSAT/UA from cached main-pipeline embeddings when available."""
    from src.encoders import encode_texts, get_device, load_minilm

    device = get_device(None)
    text_model = load_minilm(config["text"]["model_name"], device=device)
    b4_text = encode_texts(text_model, [B4_PROMPTS[c] for c in B4_CLASSES])
    eu = _load_eurosat_eval(config)
    ua = _load_ua_eval(config)
    if eu:
        result["b1"] = {
            "ten_class": _eval_fit(fit, eu["images"], eu["labels"], eu["text"],
                                   eu["caption_labels"], eu["classes"]),
            "b4": _eval_fit(fit, eu["images"],
                            [b4_label(x, "eurosat") for x in eu["labels"]],
                            b4_text, list(B4_CLASSES), list(B4_CLASSES)),
        }
    if ua and eu:
        # Reuse the EuroSAT caption index as the UA label prompts, matching T12.
        def ua_eval(mask=None) -> dict[str, Any]:
            images = ua["images"] if mask is None else ua["images"][mask]
            labels = ua["labels"] if mask is None else [x for x, keep in zip(ua["labels"], mask) if keep]
            ten = _eval_fit(fit, images, labels, eu["text"], eu["caption_labels"], eu["classes"])
            ten["ci95_wilson"] = _wilson(ten["correct"], ten["n"])
            b4 = _eval_fit(fit, images, [b4_label(x, "eurosat") for x in labels],
                           b4_text, list(B4_CLASSES), list(B4_CLASSES))
            return {"ten_class": ten, "b4": b4}

        result["b2"] = ua_eval()
        clear = ua.get("clear")
        if clear is not None and bool(np.any(clear)):
            result["b2_unambiguous"] = ua_eval(clear)


def run_benchmark(
    data_dir: str | Path,
    b3_csv: str | Path,
    *,
    tag: str,
    device: str | None = None,
    dtype: str = "fp16",
    batch_size: int = 256,
    num_workers: int = 8,
    force: bool = False,
    out: str | Path | None = None,
    rotation_out: str | Path | None = None,
) -> dict[str, Any]:
    import train
    from src.alignment import fit_procrustes, save_procrustes, transform_images, transform_texts
    from src.config import load_config
    from src.encoders import device_summary, encode_texts, get_device, load_dinov2, load_minilm
    from src.retrieval import class_scores

    data_dir = Path(data_dir)
    config = load_config(str(ROOT / "configs" / "default.yaml"))
    device_resolved = get_device(device)
    args = SimpleNamespace(
        tag=tag,
        input_size=int(config["image"]["input_size"]),
        force=force,
        batch_size=batch_size,
        num_workers=num_workers,
        device_resolved=device_resolved,
        dtype=dtype,
        prefetch_factor=2,
    )
    rows = train.load_rows(data_dir)
    excluded = list(getattr(train, "DEFAULT_EXCLUDE_FIT", ()))
    rows = train.drop_fit_regions(rows, excluded)
    # Protocol: all non-test rows are fit data. The builder may label some rows
    # "val"; for this benchmark they are ordinary train data, not a tuning split.
    train_rows = [r for r in rows if r["split"] != "test"]
    if not train_rows:
        raise SystemExit("no non-test fit rows")
    b3_rows_by_id = {r["tile_id"]: r for r in train.load_rows(data_dir)}
    b3_meta = _read_manifest(Path(b3_csv))
    b3_rows = [b3_rows_by_id[r["tile_id"]] for r in b3_meta if r["tile_id"] in b3_rows_by_id]
    if not b3_rows:
        raise SystemExit("no B3 rows found in data dir")
    b3_meta_by_id = {r["tile_id"]: r for r in b3_meta}
    scene_disjoint = [
        row for row in b3_rows
        if b3_meta_by_id[row["tile_id"]].get("scene_shared_with_train", "").lower() != "true"
    ]

    model = load_dinov2(config["image"]["model_name"], device=device_resolved)
    text_model = load_minilm(config["text"]["model_name"], device=device_resolved)
    print(f"[bench] embedding train={len(train_rows)} b3={len(b3_rows)} with {device_summary(device_resolved)}", flush=True)
    x_train = train.embed(model, train_rows, "dinov2-small", f"bench_{tag}", args)
    x_b3 = train.embed(model, b3_rows, "dinov2-small", f"bench_{tag}_b3", args)
    text_unique, text_index = train.encode_captions(
        text_model, [r["caption"] for r in train_rows], config["text"]["model_name"], args
    )
    y_train = text_unique[text_index]
    fit = fit_procrustes(x_train, y_train, center=True)
    output = Path(out) if out else THIS / f"benchmark_{tag}.json"
    rotation_path = Path(rotation_out) if rotation_out else output.with_suffix(".npz")
    b3_sha = Path(str(b3_csv) + ".sha256")
    fit.update(
        tag=tag,
        data=str(data_dir),
        fit_split="all non-test rows (train + builder val)",
        n_fit=len(train_rows),
        n_b3=len(b3_rows),
        b3_sha256=b3_sha.read_text().strip() if b3_sha.exists() else None,
        model_name=config["image"]["model_name"],
        text_model=config["text"]["model_name"],
        input_size=args.input_size,
        device=device_summary(device_resolved),
        dtype=dtype,
    )
    save_procrustes(rotation_path, fit)
    rotation_sha = hashlib.sha256(rotation_path.read_bytes()).hexdigest()

    b3_labels = [r["label"] for r in b3_rows]
    b3_b4 = [b4_label(label, "osm") for label in b3_labels]
    b3_regions = [r["region_id"] for r in b3_rows]
    osm_classes = sorted(set(b3_labels))
    b4_classes = list(B4_CLASSES)
    osm_text = encode_texts(text_model, [f"{LABEL_PROMPTS.get(c, c)[:1].upper()}{LABEL_PROMPTS.get(c, c)[1:]}." for c in osm_classes])
    b4_text = encode_texts(text_model, [B4_PROMPTS[c] for c in b4_classes])
    result: dict[str, Any] = {
        "tag": tag,
        "data": str(data_dir),
        "device": device_summary(device_resolved),
        "dtype": dtype,
        "n_fit": len(train_rows),
        "fit_split": "all non-test rows (train + builder val)",
        "n_b3": len(b3_rows),
        "b3_rows_in_file": len(b3_meta),
        "b3_coverage": round(len(b3_rows) / max(len(b3_meta), 1), 4),
        "excluded_from_fit": excluded,
        "b3_sha256": fit.get("b3_sha256"),
        "rotation_path": str(rotation_path),
        "rotation_sha256": rotation_sha,
        "b3": {
            "osm": _eval_fit(fit, x_b3, b3_labels, osm_text, osm_classes, osm_classes, groups=b3_regions),
            "b4": _eval_fit(fit, x_b3, b3_b4, b4_text, b4_classes, b4_classes, groups=b3_regions),
        },
    }
    _run_cached_suites(result, config, fit)

    if scene_disjoint:
        labels = [r["label"] for r in scene_disjoint]
        b4_labels = [b4_label(label, "osm") for label in labels]
        regions = [r["region_id"] for r in scene_disjoint]
        keep = [i for i, row in enumerate(b3_rows) if row in scene_disjoint]
        x_scene = x_b3[keep]
        result["b3_scene_disjoint"] = {
            "n": len(scene_disjoint),
            "osm": _eval_fit(fit, x_scene, labels, osm_text, osm_classes, osm_classes, groups=regions),
            "b4": _eval_fit(fit, x_scene, b4_labels, b4_text, b4_classes, b4_classes, groups=regions),
        }

    # Controls on B3. Identity and random rotations are in the same 384-d space.
    dim = x_b3.shape[1]
    identity = {"W": np.eye(dim, dtype=np.float32), "mu_x": np.zeros(dim, np.float32),
                "mu_y": np.zeros(dim, np.float32)}
    rng = np.random.default_rng(int(config.get("seed", 42)))
    q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    random_fit = {"W": q.astype(np.float32), "mu_x": np.zeros(dim, np.float32),
                  "mu_y": np.zeros(dim, np.float32)}
    shuffled = fit_procrustes(x_train, y_train, center=True, shuffle_y=True, seed=int(config.get("seed", 42)))
    controls = {}
    for name, cfit in (("identity", identity), ("random_rotation", random_fit),
                       ("label_shuffled", shuffled)):
        controls[name] = {
            "osm": _eval_fit(cfit, x_b3, b3_labels, osm_text, osm_classes, osm_classes, groups=b3_regions),
            "b4": _eval_fit(cfit, x_b3, b3_b4, b4_text, b4_classes, b4_classes, groups=b3_regions),
        }
    result["controls"] = controls

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[bench] wrote {output}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="fixed benchmark sampler/evaluator")
    sub = parser.add_subparsers(dest="command", required=True)

    sample = sub.add_parser("sample", help="create the fixed B3 test CSV")
    sample.add_argument("--data", required=True)
    sample.add_argument("--out", required=True)
    sample.add_argument("--seed", type=int, default=20261115)
    sample.add_argument("--cell-cap", type=int, default=200)
    sample.add_argument("--region-cap", type=int, default=300)

    run = sub.add_parser("run", help="evaluate a snapshot on B3/B4")
    run.add_argument("--data", required=True)
    run.add_argument("--b3", required=True)
    run.add_argument("--tag", required=True)
    run.add_argument("--device", default=None)
    run.add_argument("--dtype", default="fp16")
    run.add_argument("--batch-size", type=int, default=256)
    run.add_argument("--num-workers", type=int, default=8)
    run.add_argument("--force", action="store_true")
    run.add_argument("--out", default=None)
    run.add_argument("--rotation-out", default=None,
                     help="Path for the portable .npz rotation (default: alongside --out)")

    args = parser.parse_args()
    if args.command == "sample":
        result = sample_b3(args.data, args.out, seed=args.seed,
                           cell_cap=args.cell_cap, region_cap=args.region_cap)
        print(json.dumps({"n": result["n"], "sha256": result["sha256"]}, indent=2))
    else:
        run_benchmark(args.data, args.b3, tag=args.tag, device=args.device, dtype=args.dtype,
                      batch_size=args.batch_size, num_workers=args.num_workers,
                      force=args.force, out=args.out, rotation_out=args.rotation_out)


if __name__ == "__main__":
    main()
