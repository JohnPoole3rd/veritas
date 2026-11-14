#!/usr/bin/env python
"""Fit Procrustes alignment and evaluate validation retrieval."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.alignment import fit_procrustes, orthogonality_error, save_procrustes
from src.captions import captions_by_label, load_captions, sample_caption
from src.config import load_config, resolve_path, save_json, set_seed
from src.data import load_split
from src.encoders import encode_texts, load_embeddings, load_minilm, save_embeddings
from src.evaluate import evaluate_retrieval, save_metrics_csv


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
    grouped = captions_by_label(captions)
    text_model = load_minilm(config["text"]["model_name"])
    caption_embeddings = encode_texts(text_model, [record["text"] for record in captions])
    save_embeddings(embedding_dir / "minilm_captions.npy", caption_embeddings)
    save_json(
        embedding_dir / "minilm_caption_index.json",
        {str(i): {"caption_id": r["caption_id"], "label": r["label"]} for i, r in enumerate(captions)},
    )

    train_payload = load_split(split_dir / "train.json")
    if args.limit is not None:
        train_payload = {**train_payload, "items": train_payload["items"][: args.limit]}
    train_images = load_embeddings(embedding_dir / "dinov2_train.npy")
    if args.limit is not None:
        train_images = train_images[: args.limit]
    if len(train_payload["items"]) != len(train_images):
        raise ValueError("Train split and embedding count do not match")

    sampled_captions = [
        sample_caption(grouped, item["label"], rng) for item in train_payload["items"]
    ]
    text_train = encode_texts(text_model, sampled_captions)
    params = fit_procrustes(train_images, text_train, center=True, shuffle_y=False)
    params["image_model"] = config["image"]["model_name"]
    params["text_model"] = config["text"]["model_name"]
    params["eurosat_root"] = config["data"]["eurosat_root"]
    params["limit"] = args.limit
    save_procrustes(alignment_dir / "procrustes.npz", params)
    print(f"W shape: {params['W'].shape}")
    print(f"orthogonality error: {orthogonality_error(params):.6f}")

    val_payload = load_split(split_dir / "val.json")
    if args.limit is not None:
        val_payload = {**val_payload, "items": val_payload["items"][: args.limit]}
    val_images = load_embeddings(embedding_dir / "dinov2_val.npy")
    if args.limit is not None:
        val_images = val_images[: args.limit]
    from src.alignment import transform_images, transform_texts

    rotated = transform_images(val_images, params)
    texts = transform_texts(caption_embeddings, params)
    metrics = evaluate_retrieval(
        rotated,
        texts,
        image_labels=[item["label"] for item in val_payload["items"]],
        caption_labels=[record["label"] for record in captions],
    )
    metrics["method"] = "procrustes"
    metrics["split"] = "val"
    metrics["orthogonality_error"] = orthogonality_error(params)
    save_json(metrics_dir / "retrieval_val.json", metrics)
    save_metrics_csv(
        metrics_dir / "retrieval_val.csv",
        [
            {
                "method": "procrustes",
                "split": "val",
                "top1": metrics["top1"],
                "top5": metrics["top5"],
                "n_images": metrics["n_images"],
                "n_captions": metrics["n_captions"],
            }
        ],
    )
    print(json.dumps({k: metrics[k] for k in ("top1", "top5")}, indent=2))


if __name__ == "__main__":
    main()
