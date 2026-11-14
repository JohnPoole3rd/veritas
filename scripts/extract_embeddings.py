#!/usr/bin/env python
"""Extract frozen DINOv2 embeddings for EuroSAT splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config, resolve_path, set_seed
from src.data import dataset_from_split
from src.encoders import (
    extract_image_embeddings,
    load_dinov2,
    save_embeddings,
    save_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Optional per-split limit")
    args = parser.parse_args()

    import torch
    from torch.utils.data import Subset

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size or int(config["image"]["batch_size"])
    input_size = int(config["image"]["input_size"])

    model = load_dinov2(config["image"]["model_name"], device=device)
    embedding_dir = resolve_path("artifacts/embeddings", config)
    split_sizes = {}
    embedding_dim = None

    for split in ("train", "val", "test"):
        print(f"Extracting {split}...", flush=True)
        dataset = dataset_from_split(config, split, transform=None, return_id=True)
        if args.limit is not None:
            dataset = Subset(dataset, range(min(args.limit, len(dataset))))
        embeddings = extract_image_embeddings(
            model,
            dataset,
            batch_size=batch_size,
            device=device,
        )
        save_embeddings(embedding_dir / f"dinov2_{split}.npy", embeddings)
        split_sizes[split] = int(embeddings.shape[0])
        embedding_dim = int(embeddings.shape[1])
        print(f"{split}: {embeddings.shape}")

    save_metadata(
        embedding_dir / "dinov2_meta.json",
        {
            "model_name": config["image"]["model_name"],
            "input_size": input_size,
            "embedding_dim": embedding_dim,
            "split_sizes": split_sizes,
            "limit": args.limit,
        },
    )


if __name__ == "__main__":
    main()
