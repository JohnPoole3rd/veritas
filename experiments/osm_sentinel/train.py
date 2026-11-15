#!/usr/bin/env python
"""Fit an OSM-captioned Sentinel-2 rotation with DINOv3 (or DINOv2).

The method is the same closed-form Procrustes map used by the main project:
frozen image encoder + frozen MiniLM + one orthogonal rotation fitted on
labelled image-caption pairs. This experiment only changes the *training data*:
real OSM-grounded captions from many countries instead of EuroSAT templates.

Run:
    export DINOV3_REPO=/path/to/dinov3
    export DINOV3_WEIGHTS=/path/to/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
    python experiments/osm_sentinel/train.py --backbone dinov3-small

No DINOv3 access? Use `--backbone dinov2-small` for a CPU/MPS smoke run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
THIS = Path(__file__).resolve().parent
for path in (ROOT, THIS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from backbones import HF_MODELS, load_backbone  # noqa: E402
from osm_captions import LABEL_PROMPTS  # noqa: E402

CACHE = ROOT / "artifacts" / "embeddings" / "osm_sentinel"
ALIGN = ROOT / "artifacts" / "alignment"


class PILList:
    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from PIL import Image

        return self.transform(Image.open(self.paths[index]).convert("RGB"))


def load_rows(data_dir: Path) -> list[dict]:
    manifest = data_dir / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing {manifest}; run build_dataset.py first")
    captions = {}
    caption_path = data_dir / "captions.jsonl"
    if caption_path.exists():
        for line in caption_path.read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                captions[item["tile_id"]] = item
    rows = []
    for row in csv.DictReader(manifest.open(newline="")):
        item = captions.get(row["tile_id"], {})
        row = dict(row)
        row["path"] = data_dir / row["image"]
        row["caption"] = item.get("caption") or row["caption"]
        row["label"] = item.get("label") or row["label"]
        rows.append(row)
    return rows


def _cache_ok(path: Path, meta_path: Path, rows: list[dict], backbone: str) -> bool:
    if not (path.exists() and meta_path.exists()):
        return False
    meta = json.loads(meta_path.read_text())
    ids = [row["tile_id"] for row in rows]
    return meta.get("backbone") == backbone and meta.get("tile_ids") == ids


def embed(model, rows: list[dict], backbone: str, split: str, args) -> np.ndarray:
    from src.data import get_transforms
    from src.encoders import extract_image_embeddings

    path = CACHE / f"{backbone}_{split}.npy"
    meta_path = CACHE / f"{backbone}_{split}.json"
    if not args.force and _cache_ok(path, meta_path, rows, backbone):
        print(f"cache {path.relative_to(ROOT)} ({len(rows)} rows)", flush=True)
        return np.load(path)
    transform = get_transforms(224, train=False)
    dataset = PILList([row["path"] for row in rows], transform)
    print(f"embedding {split}: {len(rows)} images", flush=True)
    embeddings = extract_image_embeddings(
        model,
        dataset,
        batch_size=args.batch_size,
        device=args.device_resolved,
        num_workers=args.num_workers,
        dtype=args.dtype,
    )
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)
    meta_path.write_text(json.dumps({
        "backbone": backbone,
        "split": split,
        "tile_ids": [row["tile_id"] for row in rows],
        "n": len(rows),
        "image_hashes": [
            hashlib.sha256(str(row["path"]).encode()).hexdigest()[:12] for row in rows
        ],
    }, indent=2) + "\n")
    return embeddings


def class_scores(images, text, caption_labels: list[str], classes: list[str]):
    sims = images @ text.T
    index = np.array([classes.index(label) for label in caption_labels])
    scores = np.full((len(images), len(classes)), -np.inf, dtype=np.float32)
    for column, class_index in enumerate(index):
        scores[:, class_index] = np.maximum(scores[:, class_index], sims[:, column])
    return scores


def cls_metrics(scores, labels: list[str], classes: list[str]) -> dict:
    from sklearn.metrics import f1_score

    y = np.array([classes.index(label) for label in labels])
    pred = scores.argmax(1)
    return {
        "accuracy": round(float((pred == y).mean()), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
        "n": int(len(y)),
    }


def pair_recall(images, text, ks=(1, 10)) -> dict:
    sims = text @ images.T
    rank = (sims > sims[np.arange(len(sims)), np.arange(len(sims))][:, None]).sum(1)
    return {f"recall@{k}": round(float((rank < k).mean()), 4) for k in ks} | {"n": int(len(images))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "raw" / "osm_sentinel"))
    parser.add_argument("--backbone", default="dinov3-small", choices=list(HF_MODELS))
    parser.add_argument("--device", default=None, help="Override autodetection (cuda/mps/xpu/cpu)")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16", "auto"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Recompute cached embeddings")
    parser.add_argument("--eval-eurosat", action="store_true", help="Also score EuroSAT test (slower)")
    args = parser.parse_args()

    from src.alignment import fit_procrustes, orthogonality_error, save_procrustes, transform_images, transform_texts
    from src.config import load_config
    from src.encoders import device_summary, encode_texts, get_device, load_minilm

    config = load_config(str(ROOT / "configs" / "default.yaml"))
    args.device_resolved = get_device(args.device)
    print(f"backbone={args.backbone} device={device_summary(args.device_resolved)} dtype={args.dtype}", flush=True)
    model, model_label = load_backbone(args.backbone, args.device_resolved)

    rows = load_rows(Path(args.data))
    by_split = {split: [row for row in rows if row["split"] == split] for split in ("train", "val", "test")}
    for split, items in by_split.items():
        if not items:
            raise SystemExit(f"Split {split!r} is empty; check regions.csv and dataset_stats.json")
        print(f"{split}: {len(items)} chips, {len({row['country'] for row in items})} countries", flush=True)

    X = {split: embed(model, items, args.backbone, split, args) for split, items in by_split.items()}
    text_model = load_minilm(config["text"]["model_name"], device=args.device_resolved)
    Y = {split: encode_texts(text_model, [row["caption"] for row in items]) for split, items in by_split.items()}

    params = fit_procrustes(X["train"], Y["train"], center=True)
    random_params = fit_procrustes(X["train"], Y["train"], center=True, shuffle_y=True, seed=int(config["seed"]))
    params.update(backbone=args.backbone, model_name=model_label, data=str(Path(args.data)), train_n=len(by_split["train"]))
    ALIGN.mkdir(parents=True, exist_ok=True)
    save_procrustes(ALIGN / f"osm_sentinel_{args.backbone}.npz", params)

    labels = sorted({row["label"] for row in rows})
    prompt_text = [f"A 10 m Sentinel-2 satellite image of {LABEL_PROMPTS.get(label, label)}." for label in labels]
    Y_classes = encode_texts(text_model, prompt_text)

    results = {
        "backbone": args.backbone,
        "model_name": model_label,
        "device": device_summary(args.device_resolved),
        "dtype": args.dtype,
        "n_train": len(by_split["train"]),
        "n_val": len(by_split["val"]),
        "n_test": len(by_split["test"]),
        "countries": sorted({row["country"] for row in rows}),
        "orthogonality_error": round(orthogonality_error(params), 6),
        "classes": labels,
        "fits": {},
    }

    for name, fit in (("osm_fit", params), ("random_rotation", random_params)):
        fit_result = {}
        for split in ("val", "test"):
            items = by_split[split]
            images = transform_images(X[split], fit)
            text = transform_texts(Y[split], fit)
            fit_result[split] = {
                "caption_to_image": pair_recall(images, text),
                "image_to_label": cls_metrics(
                    class_scores(images, transform_texts(Y_classes, fit), labels, labels),
                    [row["label"] for row in items],
                    labels,
                ),
            }
            # One canonical caption per class, searched against the whole split.
            fit_result[split]["label_search"] = {
                "top1": round(float((
                    np.array([row["label"] for row in items])[
                        (transform_texts(Y_classes, fit) @ images.T).argmax(1)
                    ] == np.array(labels)
                ).mean()), 4),
                "queries": len(labels),
            }
        results["fits"][name] = fit_result

    local_manifest = ROOT / "data" / "local_tiles" / "manifest.csv"
    euro_captions = ROOT / "data" / "captions.jsonl"
    if local_manifest.exists() and euro_captions.exists():
        local_rows = list(csv.DictReader(local_manifest.open(newline="")))
        local_paths = [ROOT / row["path"] for row in local_rows]
        from src.data import get_transforms
        from src.encoders import extract_image_embeddings

        local_cache = CACHE / f"{args.backbone}_upper_arlington.npy"
        if not args.force and local_cache.exists():
            local_emb = np.load(local_cache)
        else:
            local_emb = extract_image_embeddings(
                model,
                PILList(local_paths, get_transforms(224, train=False)),
                batch_size=args.batch_size,
                device=args.device_resolved,
                num_workers=args.num_workers,
                dtype=args.dtype,
            )
            CACHE.mkdir(parents=True, exist_ok=True)
            np.save(local_cache, local_emb)
        euro_rows = [json.loads(line) for line in euro_captions.read_text().splitlines() if line.strip()]
        euro_classes = sorted({row["label"] for row in euro_rows})
        euro_text = encode_texts(text_model, [row["text"] for row in euro_rows])
        local_labels = [row["label"] for row in local_rows]
        for name, fit in (("osm_fit", params), ("random_rotation", random_params)):
            scores = class_scores(
                transform_images(local_emb, fit),
                transform_texts(euro_text, fit),
                [row["label"] for row in euro_rows],
                euro_classes,
            )
            results["fits"][name]["upper_arlington"] = cls_metrics(scores, local_labels, euro_classes)

    if args.eval_eurosat:
        split_path = ROOT / "data" / "splits" / "test.json"
        euro_root = ROOT / config["data"]["eurosat_root"]
        if split_path.exists():
            items = json.loads(split_path.read_text())["items"]
            from src.data import get_transforms
            from src.encoders import extract_image_embeddings

            cache_path = CACHE / f"{args.backbone}_eurosat_test.npy"
            if not args.force and cache_path.exists():
                euro_emb = np.load(cache_path)
            else:
                euro_emb = extract_image_embeddings(
                    model,
                    PILList([euro_root / item["path"] for item in items], get_transforms(224, train=False)),
                    batch_size=args.batch_size,
                    device=args.device_resolved,
                    num_workers=args.num_workers,
                    dtype=args.dtype,
                )
                CACHE.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, euro_emb)
            euro_rows = [json.loads(line) for line in euro_captions.read_text().splitlines() if line.strip()]
            euro_classes = sorted({row["label"] for row in euro_rows})
            euro_text = encode_texts(text_model, [row["text"] for row in euro_rows])
            for name, fit in (("osm_fit", params), ("random_rotation", random_params)):
                scores = class_scores(
                    transform_images(euro_emb, fit),
                    transform_texts(euro_text, fit),
                    [row["label"] for row in euro_rows],
                    euro_classes,
                )
                results["fits"][name]["eurosat_test"] = cls_metrics(
                    scores, [item["label"] for item in items], euro_classes
                )

    output = THIS / f"results_{args.backbone}.json"
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
