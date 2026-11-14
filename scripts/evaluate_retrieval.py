#!/usr/bin/env python
"""T07 — text→image retrieval on the test split: Procrustes vs random rotation vs unaligned.

Each of the 50 class captions is a query; top-k = a tile of the caption's class is among the k
most similar test tiles. N(queries) = 50, N(tiles searched) = test split size.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.alignment import (  # noqa: E402
    load_procrustes,
    orthogonality_error,
    random_rotation_params,
    training_pairs,
    transform_images,
    transform_texts,
)
from src.captions import load_captions  # noqa: E402
from src.config import load_config, load_json, refuse_subset_overwrite, resolve_path, save_json, set_seed  # noqa: E402
from src.data import load_split  # noqa: E402
from src.encoders import check_split_embeddings, load_embeddings  # noqa: E402
from src.evaluate import evaluate_retrieval, save_metrics_csv  # noqa: E402

N_RANDOM_SEEDS = 5  # random rotation is one shuffle per seed; report the mean (50 queries are noisy)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-split limit (first N items)")
    parser.add_argument("--force", action="store_true", help="Allow a --limit run to overwrite full-split retrieval.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    set_seed(seed)
    embedding_dir = resolve_path("artifacts/embeddings", config)
    metrics_dir = resolve_path("artifacts/metrics", config)

    # Caption vectors and labels both come from the T05 artifacts, so their order always matches.
    caption_embeddings = load_embeddings(embedding_dir / "minilm_captions.npy")
    index = load_json(embedding_dir / "minilm_caption_index.json")
    caption_labels = [index[str(i)]["label"] for i in range(len(index))]
    if len(caption_labels) != len(caption_embeddings):
        raise ValueError(f"caption index ({len(caption_labels)}) != embeddings {caption_embeddings.shape}")

    test_items = load_split(resolve_path(config["data"]["splits_dir"], config) / "test.json")["items"]
    full_test_n = len(test_items)
    test_items = test_items[: args.limit]
    test_images = load_embeddings(embedding_dir / "dinov2_test.npy")[: args.limit]
    meta_path = embedding_dir / "dinov2_meta.json"
    check_split_embeddings(test_images, test_items, "test", load_json(meta_path) if meta_path.exists() else None)
    test_labels = [item["label"] for item in test_items]

    params = load_procrustes(resolve_path("artifacts/alignment/procrustes.npz", config))
    procrustes = evaluate_retrieval(
        transform_images(test_images, params), transform_texts(caption_embeddings, params), test_labels, caption_labels
    )

    captions = load_captions(resolve_path(config["data"]["captions_path"], config))
    X_train, Y_train = training_pairs(config, caption_embeddings, captions, limit=args.limit)
    seeds = list(range(seed, seed + N_RANDOM_SEEDS))
    per_seed = [
        evaluate_retrieval(transform_images(test_images, p), transform_texts(caption_embeddings, p), test_labels, caption_labels)
        for p in random_rotation_params(X_train, Y_train, seeds, center=bool(config["alignment"]["center"]))
    ]
    random_metrics = {
        "top1": float(np.mean([r["top1"] for r in per_seed])),
        "top5": float(np.mean([r["top5"] for r in per_seed])),
        "top1_std": float(np.std([r["top1"] for r in per_seed])),
        "top5_std": float(np.std([r["top5"] for r in per_seed])),
        "per_seed": {str(s): {"top1": r["top1"], "top5": r["top5"]} for s, r in zip(seeds, per_seed)},
        "n_images": per_seed[0]["n_images"],
        "n_captions": per_seed[0]["n_captions"],
    }

    results = {"procrustes": procrustes, "random_rotation": random_metrics}
    if test_images.shape[1] == caption_embeddings.shape[1]:  # raw cosine needs equal dims (not ViT-B)
        results["unaligned"] = evaluate_retrieval(test_images, caption_embeddings, test_labels, caption_labels)

    subset = f"first {len(test_items)} of {full_test_n} test tiles" if len(test_items) < full_test_n else ""
    rows = []
    for method, r in results.items():
        notes = [subset] if subset else []
        if method == "random_rotation":
            notes.append(f"mean of {N_RANDOM_SEEDS} seeds {seeds[0]}-{seeds[-1]}; "
                         f"std top1 {r['top1_std']:.3f}, top5 {r['top5_std']:.3f}")
        rows.append({
            "method": method, "split": "test", "top1": round(r["top1"], 4), "top5": round(r["top5"], 4),
            "macro_f1": "", "n_images": r["n_images"], "test_n": r["n_images"], "n_captions": r["n_captions"],
            "notes": "; ".join(notes),
        })
    existing = metrics_dir / "retrieval.csv"
    existing_full = existing.exists() and any(
        int(float(r.get("n_images") or 0)) >= full_test_n for r in csv.DictReader(existing.open(newline=""))
    )
    refuse_subset_overwrite(args.limit, args.force, existing_full, "retrieval.csv")
    save_metrics_csv(metrics_dir / "retrieval.csv", rows)
    out_name = "retrieval_test_full.json" if args.limit is None else f"retrieval_test_{args.limit}.json"
    save_json(metrics_dir / out_name,
              {**{k: {**v, "method": k, "split": "test"} for k, v in results.items()},
               "orthogonality_error": orthogonality_error(params), "limit": args.limit})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
