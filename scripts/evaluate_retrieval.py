#!/usr/bin/env python
"""Evaluate test and baseline retrieval."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.alignment import (
    fit_procrustes,
    load_procrustes,
    orthogonality_error,
    transform_images,
    transform_texts,
)
from src.captions import captions_by_label, load_captions, sample_caption
from src.config import load_config, resolve_path, save_json, set_seed
from src.data import load_split
from src.encoders import encode_texts, load_embeddings, load_minilm
from src.evaluate import evaluate_retrieval, save_metrics_csv


def _sample_train_texts(config, captions, text_model, rng, limit=None):
    split_dir = resolve_path(config["data"]["splits_dir"], config)
    embedding_dir = resolve_path("artifacts/embeddings", config)
    train_payload = load_split(split_dir / "train.json")
    if limit is not None:
        train_payload = {**train_payload, "items": train_payload["items"][:limit]}
    grouped = captions_by_label(captions)
    sampled = [sample_caption(grouped, item["label"], rng) for item in train_payload["items"]]
    train_images = load_embeddings(embedding_dir / "dinov2_train.npy")
    if limit is not None:
        train_images = train_images[:limit]
    return train_images, encode_texts(text_model, sampled)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Optional per-split limit")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    rng = random.Random(int(config.get("seed", 42)))

    split_dir = resolve_path(config["data"]["splits_dir"], config)
    embedding_dir = resolve_path("artifacts/embeddings", config)
    alignment_dir = resolve_path("artifacts/alignment", config)
    metrics_dir = resolve_path("artifacts/metrics", config)

    captions = load_captions(resolve_path(config["data"]["captions_path"], config))
    caption_labels = [record["label"] for record in captions]
    text_model = load_minilm(config["text"]["model_name"])
    caption_embeddings = load_embeddings(embedding_dir / "minilm_captions.npy")

    test_payload = load_split(split_dir / "test.json")
    if args.limit is not None:
        test_payload = {**test_payload, "items": test_payload["items"][: args.limit]}
    test_images = load_embeddings(embedding_dir / "dinov2_test.npy")
    if args.limit is not None:
        test_images = test_images[: args.limit]
    test_labels = [item["label"] for item in test_payload["items"]]

    params = load_procrustes(alignment_dir / "procrustes.npz")
    rotated = transform_images(test_images, params)
    texts = transform_texts(caption_embeddings, params)
    procrustes = evaluate_retrieval(rotated, texts, test_labels, caption_labels)
    procrustes.update({"method": "procrustes", "split": "test"})

    train_images, train_texts = _sample_train_texts(
        config, captions, text_model, rng, limit=args.limit
    )
    random_params = fit_procrustes(train_images, train_texts, center=True, shuffle_y=True, seed=42)
    random_rotated = transform_images(test_images, random_params)
    random_texts = transform_texts(caption_embeddings, random_params)
    random_metrics = evaluate_retrieval(random_rotated, random_texts, test_labels, caption_labels)
    random_metrics.update({"method": "random_rotation", "split": "test"})

    unaligned = evaluate_retrieval(test_images, caption_embeddings, test_labels, caption_labels)
    unaligned.update({"method": "unaligned", "split": "test"})

    rows = []
    for result in (procrustes, random_metrics, unaligned):
        rows.append(
            {
                "method": result["method"],
                "split": "test",
                "top1": result["top1"],
                "top5": result["top5"],
                "macro_f1": "",
                "n_images": result["n_images"],
                "test_n": result["n_images"],
                "n_captions": result["n_captions"],
                "notes": f"{result['n_images']}-tile subset" if result["n_images"] < 4050 else "",
            }
        )
    save_metrics_csv(metrics_dir / "retrieval.csv", rows)
    save_json(
        metrics_dir / "retrieval_test.json",
        {
            "procrustes": procrustes,
            "random_rotation": random_metrics,
            "unaligned": unaligned,
            "orthogonality_error": orthogonality_error(params),
        },
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
