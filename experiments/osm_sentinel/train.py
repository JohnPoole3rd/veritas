#!/usr/bin/env python
"""Fit an OSM-captioned Sentinel-2 rotation with frozen DINOv2.

The method is the same closed-form Procrustes map used by the main project:
frozen image encoder + frozen MiniLM + one orthogonal rotation fitted on
labelled image-caption pairs. This experiment only changes the *training data*:
real OSM-grounded captions from many countries instead of EuroSAT templates.

DINOv2 is public and downloads automatically:

    python experiments/osm_sentinel/train.py

Use `--device` and `--dtype` for the hardware path; no gated model access is
required.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
THIS = Path(__file__).resolve().parent
for path in (ROOT, THIS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from osm_captions import LABEL_PROMPTS  # noqa: E402

CACHE = ROOT / "artifacts" / "embeddings" / "osm_sentinel"
ALIGN = ROOT / "artifacts" / "alignment"
BACKBONE = "dinov2-small"


class PILList:
    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from PIL import Image

        return self.transform(Image.open(self.paths[index]).convert("RGB"))


def image_transform(input_size: int = 224):
    from src.data import get_transforms

    return get_transforms(input_size, train=False)


def use_device_preprocess(args) -> bool:
    """`--preprocess auto` resizes on the accelerator for CUDA/XPU only; MPS/CPU keep the PIL path."""
    mode = getattr(args, "preprocess", "cpu")
    if mode == "auto":
        return args.device_resolved.split(":", 1)[0] in ("cuda", "xpu")
    return mode == "gpu"


def extract(model, paths: list[Path], args) -> np.ndarray:
    """Embed images with the configured preprocessing path (PIL on CPU, or uint8 + on-device resize)."""
    from src.encoders import GPUEvalTransform, UInt8ImageList, extract_image_embeddings

    if use_device_preprocess(args):
        dataset = UInt8ImageList(paths)
        device_transform = GPUEvalTransform(
            args.input_size, channels_last=args.device_resolved.startswith("cuda")
        )
    else:
        dataset, device_transform = PILList(paths, image_transform(args.input_size)), None
    return extract_image_embeddings(
        model,
        dataset,
        batch_size=args.batch_size,
        device=args.device_resolved,
        num_workers=args.num_workers,
        dtype=args.dtype,
        prefetch_factor=getattr(args, "prefetch_factor", 2),
        device_transform=device_transform,
        progress_every=max(1, 50_000 // args.batch_size),
    )


# A (a/osm-1m-fixes): all 50 Upper Arlington test tiles lie inside the us_columbus *train* region,
# so it is left out of every OSM fit by default; otherwise "UA" would not be out of distribution.
DEFAULT_EXCLUDE_FIT = ("us_columbus",)


def drop_fit_regions(rows: list[dict], exclude: list[str] | tuple[str, ...]) -> list[dict]:
    """Remove train rows from excluded regions (val/test rows are untouched)."""
    excluded = set(exclude or ())
    return [r for r in rows if not (r["split"] == "train" and r.get("region_id") in excluded)]


def load_rows(data_dir: Path) -> list[dict]:
    manifest = data_dir / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing {manifest}; run build_dataset.py first")
    rows = []
    captions = None
    caption_path = data_dir / "captions.jsonl"
    for source in csv.DictReader(manifest.open(newline="")):
        row = {
            key: source[key]
            for key in ("tile_id", "image", "caption", "label", "split", "country", "region_id")
        }
        if not row["caption"]:
            if captions is None:
                captions = {}
                if caption_path.exists():
                    for line in caption_path.read_text().splitlines():
                        if line.strip():
                            item = json.loads(line)
                            captions[item["tile_id"]] = item
            row["caption"] = captions.get(row["tile_id"], {}).get("caption", "")
        row["path"] = data_dir / row["image"]
        if not row["caption"]:
            print(f"skipping {row['tile_id']}: empty caption", flush=True)
            continue
        rows.append(row)
    if not rows:
        raise SystemExit(f"No captioned chips in {manifest}")
    return rows


def _cache_ok(path: Path, meta_path: Path, rows: list[dict], backbone: str, args) -> bool:
    if not (path.exists() and meta_path.exists()):
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    ids = [row["tile_id"] for row in rows]
    return (
        meta.get("backbone") == backbone
        and meta.get("tag") == args.tag
        and meta.get("input_size") == args.input_size
        and meta.get("tile_ids") == ids
    )


def embed(model, rows: list[dict], backbone: str, split: str, args) -> np.ndarray:
    path = CACHE / f"{backbone}_{args.tag}_{split}.npy"
    meta_path = CACHE / f"{backbone}_{args.tag}_{split}.json"
    if not args.force and _cache_ok(path, meta_path, rows, backbone, args):
        print(f"cache {path.relative_to(ROOT)} ({len(rows)} rows)", flush=True)
        return np.load(path)
    print(f"embedding {split}: {len(rows)} images", flush=True)
    start = time.perf_counter()
    embeddings = extract(model, [row["path"] for row in rows], args)
    elapsed = time.perf_counter() - start
    print(f"embedded {split}: {len(rows)} images in {elapsed:.1f}s ({len(rows) / max(elapsed, 1e-9):.0f} img/s)",
          flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)
    meta_path.write_text(json.dumps({
        "backbone": backbone,
        "split": split,
        "tag": args.tag,
        "input_size": args.input_size,
        "tile_ids": [row["tile_id"] for row in rows],
        "n": len(rows),
    }, indent=2) + "\n")
    return embeddings


def class_scores(images, text, caption_labels: list[str], classes: list[str]):
    # Use the one shared classification rule, so train.py and compare_eurosat.py
    # cannot drift on normalization or max-over-captions.
    from src.retrieval import class_scores as shared_class_scores

    return shared_class_scores(images, text, caption_labels, classes)


def cls_metrics(scores, labels: list[str], classes: list[str]) -> dict:
    from sklearn.metrics import f1_score

    y = np.array([classes.index(label) for label in labels])
    pred = scores.argmax(1)
    return {
        "accuracy": round(float((pred == y).mean()), 4),
        "macro_f1": round(float(f1_score(
            y, pred, labels=list(range(len(classes))), average="macro", zero_division=0
        )), 4),
        "n": int(len(y)),
    }


def text_batch_size(device) -> int:
    """MiniLM batch: large on CUDA (16 GB card has room), the old default elsewhere."""
    return 512 if str(device).startswith("cuda") else 64


def encode_captions(text_model, captions: list[str], model_name: str, args) -> tuple[np.ndarray, np.ndarray]:
    """(unique caption embeddings U x d, int64 index with captions[i] -> row index[i]).

    Templated OSM captions repeat a lot, so only the unique ones are encoded. The unique
    embeddings are cached under a hash of (model, unique captions), so train.py and
    compare_eurosat.py (and reruns) share one encoding; --force re-encodes.
    """
    import hashlib

    from src.alignment import unique_rows
    from src.encoders import encode_texts

    unique, index = unique_rows(captions)
    digest = hashlib.sha256(model_name.encode())
    for caption in unique:
        digest.update(b"\0" + caption.encode())
    path = CACHE / f"minilm_text_{digest.hexdigest()[:20]}.npy"
    if not getattr(args, "force", False) and path.exists():
        cached = np.load(path)
        if cached.shape[0] == len(unique):
            print(f"text cache {path.name} ({len(unique)} unique / {len(captions)} captions)", flush=True)
            return cached, index
    print(f"encoding {len(unique)} unique / {len(captions)} captions", flush=True)
    embeddings = encode_texts(
        text_model, unique, batch_size=text_batch_size(getattr(args, "device_resolved", "")), dedupe=False
    )
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)
    return embeddings, index


def pair_recall(images, text, ks=(1, 10), chunk_size: int = 512, text_index=None) -> dict:
    """Caption→image recall without an N×N similarity matrix.

    A full matrix is hundreds of GB at 1M-row scale. Score a chunk of captions
    against all images at a time and rank only the diagonal target.

    With `text_index`, `text` holds the unique caption rows and caption i is
    text[text_index[i]]. Each unique caption is scored and sorted once; every image
    sharing it is ranked by binary search (same strict ">" count as the dense path).
    """
    n = len(images)
    if n == 0:
        return {f"recall@{k}": float("nan") for k in ks} | {"n": 0}
    if text_index is not None and len(text) > n // 2:
        text, text_index = text[text_index], None  # mostly unique: the dense path is cheaper
    # Keep the caption chunk around ~256 MB even when n is large.
    chunk_size = max(1, min(chunk_size, (256 * 1024 * 1024) // (4 * n)))
    ranks = np.empty(n, dtype=np.int64)
    if text_index is None:
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            sims = text[start:stop] @ images.T
            diagonal = sims[np.arange(stop - start), np.arange(start, stop)]
            ranks[start:stop] = (sims > diagonal[:, None]).sum(axis=1)
    else:
        text_index = np.asarray(text_index, dtype=np.int64)
        by_caption = np.argsort(text_index, kind="stable")
        bounds = np.searchsorted(text_index[by_caption], np.arange(len(text) + 1))
        for start in range(0, len(text), chunk_size):
            stop = min(start + chunk_size, len(text))
            sims = text[start:stop] @ images.T
            ordered = np.sort(sims, axis=1)
            for row, u in enumerate(range(start, stop)):
                members = by_caption[bounds[u]:bounds[u + 1]]
                diagonal = sims[row, members]
                ranks[members] = n - np.searchsorted(ordered[row], diagonal, side="right")
    return {f"recall@{k}": round(float((ranks < k).mean()), 4) for k in ks} | {"n": int(n)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "raw" / "osm_sentinel_1m_640"))
    parser.add_argument("--tag", default="default", help="Namespace for cached embeddings and results")
    parser.add_argument("--device", default=None, help="Override autodetection (cuda/mps/xpu/cpu)")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16", "auto"])
    parser.add_argument("--batch-size", type=int, default=None, help="Default: 512 on CUDA, else 32")
    parser.add_argument("--num-workers", type=int, default=None, help="Default: min(8, CPUs) on CUDA, else 4")
    parser.add_argument("--prefetch-factor", type=int, default=None, help="Batches per worker; default 4 on CUDA, else 2")
    parser.add_argument("--preprocess", default="auto", choices=["auto", "cpu", "gpu"],
                        help="Resize/normalize with PIL in workers (cpu) or on the device (gpu); "
                             "auto = gpu on CUDA/XPU, cpu on MPS/CPU")
    parser.add_argument("--compile", action="store_true", help="torch.compile the image model (opt-in)")
    parser.add_argument("--force", action="store_true", help="Recompute cached embeddings")
    parser.add_argument("--eval-eurosat", action="store_true", help="Also score EuroSAT test (slower)")
    parser.add_argument("--exclude-fit-region", action="append", default=None,
                        help="Drop this region's train rows from the fit (default: us_columbus, which contains UA)")
    args = parser.parse_args()

    from src.alignment import fit_procrustes, orthogonality_error, save_procrustes, transform_images, transform_texts
    from src.config import load_config
    from src.encoders import device_summary, encode_texts, get_device, load_dinov2, load_minilm, resolve_dtype

    config = load_config(str(ROOT / "configs" / "default.yaml"))
    args.input_size = int(config["image"]["input_size"])
    args.device_resolved = get_device(args.device)
    args.dtype_effective = str(resolve_dtype(args.device_resolved, args.dtype)).replace("torch.", "")
    on_cuda = args.device_resolved.startswith("cuda")
    if args.batch_size is None:
        args.batch_size = 512 if on_cuda else 32
    if args.num_workers is None:
        args.num_workers = min(8, os.cpu_count() or 4) if on_cuda else 4
    if args.prefetch_factor is None:
        args.prefetch_factor = 4 if on_cuda else 2
    print(
        f"backbone={BACKBONE} device={device_summary(args.device_resolved)} "
        f"dtype={args.dtype}/{args.dtype_effective} batch={args.batch_size} workers={args.num_workers} "
        f"preprocess={'device' if use_device_preprocess(args) else 'pil'} compile={args.compile}",
        flush=True,
    )
    model_label = config["image"]["model_name"]
    model = load_dinov2(model_label, device=args.device_resolved, compile=args.compile)

    exclude = args.exclude_fit_region if args.exclude_fit_region is not None else list(DEFAULT_EXCLUDE_FIT)
    rows = drop_fit_regions(load_rows(Path(args.data)), exclude)
    print(f"excluded from the OSM fit: {exclude}", flush=True)
    by_split = {split: [row for row in rows if row["split"] == split] for split in ("train", "val", "test")}
    for split, items in by_split.items():
        if not items:
            raise SystemExit(f"Split {split!r} is empty; check regions.csv and dataset_stats.json")
        print(f"{split}: {len(items)} chips, {len({row['country'] for row in items})} countries", flush=True)

    X = {split: embed(model, items, BACKBONE, split, args) for split, items in by_split.items()}
    text_model = load_minilm(config["text"]["model_name"], device=args.device_resolved)
    # Unique caption embeddings + index per split (captions repeat; see encode_captions).
    Y = {
        split: encode_captions(text_model, [row["caption"] for row in items], config["text"]["model_name"], args)
        for split, items in by_split.items()
    }

    center = bool(config["alignment"]["center"])
    params = fit_procrustes(X["train"], Y["train"][0], center=center, y_index=Y["train"][1])
    random_params = fit_procrustes(
        X["train"], Y["train"][0], center=center, shuffle_y=True, seed=int(config["seed"]),
        y_index=Y["train"][1],
    )
    params.update(
        backbone=BACKBONE,
        model_name=model_label,
        data=str(Path(args.data)),
        tag=args.tag,
        input_size=args.input_size,
        train_n=len(by_split["train"]),
    )
    ALIGN.mkdir(parents=True, exist_ok=True)
    save_procrustes(ALIGN / f"osm_sentinel_{BACKBONE}_{args.tag}.npz", params)

    labels = sorted({row["label"] for row in rows})
    prompt_text = []
    for label in labels:
        prompt = LABEL_PROMPTS.get(label, label)
        prompt_text.append(f"{prompt[:1].upper()}{prompt[1:]}.")
    Y_classes = encode_texts(text_model, prompt_text)

    results = {
        "excluded_from_fit": exclude,
        "backbone": BACKBONE,
        "model_name": model_label,
        "device": device_summary(args.device_resolved),
        "dtype": args.dtype,
        "dtype_effective": args.dtype_effective,
        "image_preprocess": "device" if use_device_preprocess(args) else "pil",
        "n_train": len(by_split["train"]),
        "n_val": len(by_split["val"]),
        "n_test": len(by_split["test"]),
        "tag": args.tag,
        "data": str(Path(args.data)),
        "input_size": args.input_size,
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
            text = transform_texts(Y[split][0], fit)
            class_text = transform_texts(Y_classes, fit)
            fit_result[split] = {
                "caption_to_image": pair_recall(images, text, text_index=Y[split][1]),
                "image_to_label": cls_metrics(
                    class_scores(images, class_text, labels, labels),
                    [row["label"] for row in items],
                    labels,
                ),
            }
            # One canonical caption per class, searched against the whole split.
            fit_result[split]["label_search"] = {
                "top1": round(float((
                    np.array([row["label"] for row in items])[
                        (class_text @ images.T).argmax(1)
                    ] == np.array(labels)
                ).mean()), 4),
                "queries": len(labels),
            }
        results["fits"][name] = fit_result

    local_manifest = ROOT / "data" / "local_tiles" / "manifest.csv"
    euro_captions = ROOT / "data" / "captions.jsonl"
    euro_cache: dict = {}

    def euro_caption_side():
        # Read and encode the EuroSAT captions once, shared by Upper Arlington and --eval-eurosat.
        if not euro_cache:
            rows_ = [json.loads(line) for line in euro_captions.read_text().splitlines() if line.strip()]
            euro_cache.update(
                rows=rows_,
                classes=sorted({row["label"] for row in rows_}),
                text=encode_texts(text_model, [row["text"] for row in rows_]),
            )
        return euro_cache["rows"], euro_cache["classes"], euro_cache["text"]

    if local_manifest.exists() and euro_captions.exists():
        local_rows = list(csv.DictReader(local_manifest.open(newline="")))
        local_paths = [ROOT / row["path"] for row in local_rows]
        local_cache = CACHE / f"{BACKBONE}_{args.tag}_upper_arlington.npy"
        if not args.force and local_cache.exists():
            local_emb = np.load(local_cache)
        else:
            local_emb = extract(model, local_paths, args)
            CACHE.mkdir(parents=True, exist_ok=True)
            np.save(local_cache, local_emb)
        euro_rows, euro_classes, euro_text = euro_caption_side()
        local_labels = [row["label"] for row in local_rows]
        keep = [i for i, label in enumerate(local_labels) if label in euro_classes]
        if not keep:
            print("upper_arlington: no labels overlap EuroSAT; skipping", flush=True)
        else:
            local_emb = local_emb[keep]
            local_labels = [local_labels[i] for i in keep]
        for name, fit in (("osm_fit", params), ("random_rotation", random_params)):
            if not keep:
                break
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
            cache_path = CACHE / f"{BACKBONE}_{args.tag}_eurosat_test.npy"
            if not args.force and cache_path.exists():
                euro_emb = np.load(cache_path)
            else:
                euro_emb = extract(model, [euro_root / item["path"] for item in items], args)
                CACHE.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, euro_emb)
            euro_rows, euro_classes, euro_text = euro_caption_side()
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

    output = THIS / f"results_{BACKBONE}_{args.tag}.json"
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
