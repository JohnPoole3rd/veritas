#!/usr/bin/env python
"""T05/T06 — encode captions, fit Procrustes on train, report validation retrieval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.alignment import (  # noqa: E402
    fit_procrustes,
    orthogonality_error,
    save_procrustes,
    training_pairs,
    transform_images,
    transform_texts,
)
from src.captions import load_captions  # noqa: E402
from src.config import load_config, load_json, resolve_path, save_json, set_seed  # noqa: E402
from src.data import load_split  # noqa: E402
from src.encoders import (  # noqa: E402
    check_split_embeddings,
    encode_texts,
    load_embeddings,
    load_minilm,
    save_embeddings,
)
from src.evaluate import evaluate_retrieval, save_metrics_csv  # noqa: E402

MAX_ORTHOGONALITY_ERROR = 1e-3  # INTERFACES §9: stop if W.T @ W is not close to I


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default=None, help="device for MiniLM caption encoding")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-split limit (first N items)")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    split_dir = resolve_path(config["data"]["splits_dir"], config)
    embedding_dir = resolve_path("artifacts/embeddings", config)
    metrics_dir = resolve_path("artifacts/metrics", config)

    # T05: caption embeddings + index (the order every later step relies on).
    captions = load_captions(resolve_path(config["data"]["captions_path"], config))
    text_model = load_minilm(config["text"]["model_name"], device=args.device)
    caption_embeddings = encode_texts(text_model, [record["text"] for record in captions])
    if caption_embeddings.shape[1] != int(config["text"]["embedding_dim"]):
        raise ValueError(f"MiniLM dim {caption_embeddings.shape} != config text.embedding_dim")
    save_embeddings(embedding_dir / "minilm_captions.npy", caption_embeddings)
    save_json(
        embedding_dir / "minilm_caption_index.json",
        {str(i): {"caption_id": r["caption_id"], "label": r["label"]} for i, r in enumerate(captions)},
    )

    # T06: fit on train pairs only.
    X_train, Y_train = training_pairs(config, caption_embeddings, captions, limit=args.limit)
    params = fit_procrustes(X_train, Y_train, center=bool(config["alignment"]["center"]), shuffle_y=False)
    params.update(
        image_model=config["image"]["model_name"],
        text_model=config["text"]["model_name"],
        eurosat_root=config["data"]["eurosat_root"],
        limit=args.limit,
        train_n=int(len(X_train)),
        seed=int(config.get("seed", 42)),
    )
    error = orthogonality_error(params)
    print(f"W shape: {params['W'].shape}; orthogonality error: {error:.2e}")
    if error > MAX_ORTHOGONALITY_ERROR:
        raise ValueError(f"W.T @ W is not close to identity (error {error:.2e}); not saving")
    save_procrustes(resolve_path("artifacts/alignment/procrustes.npz", config), params)

    # Validation retrieval (reporting only; nothing is tuned on it).
    val_items = load_split(split_dir / "val.json")["items"][: args.limit]
    val_images = load_embeddings(embedding_dir / "dinov2_val.npy")[: args.limit]
    meta_path = embedding_dir / "dinov2_meta.json"
    check_split_embeddings(val_images, val_items, "val", load_json(meta_path) if meta_path.exists() else None)
    metrics = evaluate_retrieval(
        transform_images(val_images, params),
        transform_texts(caption_embeddings, params),
        image_labels=[item["label"] for item in val_items],
        caption_labels=[record["label"] for record in captions],
    )
    metrics.update(method="procrustes", split="val", orthogonality_error=error, limit=args.limit)
    save_json(metrics_dir / "retrieval_val.json", metrics)
    save_metrics_csv(
        metrics_dir / "retrieval_val.csv",
        [{"method": "procrustes", "split": "val", "top1": metrics["top1"], "top5": metrics["top5"],
          "n_images": metrics["n_images"], "n_captions": metrics["n_captions"]}],
    )
    print(json.dumps({k: metrics[k] for k in ("top1", "top5", "n_images")}, indent=2))


if __name__ == "__main__":
    main()
