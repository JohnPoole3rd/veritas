#!/usr/bin/env python
"""Extract frozen DINOv2 embeddings for EuroSAT splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config, resolve_path, save_json, set_seed
from src.data import dataset_from_split
from src.encoders import (
    device_summary,
    extract_image_embeddings,
    get_device,
    load_dinov2,
    save_embeddings,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Optional per-split limit")
    parser.add_argument("--num-workers", type=int, default=4, help="Image decode workers (0 = main process)")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16", "auto"])
    args = parser.parse_args()

    from torch.utils.data import Subset

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    device = get_device(args.device)
    batch_size = args.batch_size or int(config["image"]["batch_size"])
    input_size = int(config["image"]["input_size"])
    print(f"device={device_summary(device)} dtype={args.dtype} batch={batch_size} workers={args.num_workers}", flush=True)

    model = load_dinov2(config["image"]["model_name"], device=device)
    embedding_dir = resolve_path("artifacts/embeddings", config)
    split_sizes = {}
    tile_ids = {}
    embedding_dim = None

    for split in ("train", "val", "test"):
        print(f"Extracting {split}...", flush=True)
        dataset = dataset_from_split(config, split, transform=None)
        tile_ids[split] = [item["tile_id"] for item in dataset.items][: args.limit]
        if args.limit is not None:  # first N items of the (shuffled) split, not a stratified subset
            dataset = Subset(dataset, range(min(args.limit, len(dataset))))
        embeddings = extract_image_embeddings(
            model,
            dataset,
            batch_size=batch_size,
            device=device,
            num_workers=args.num_workers,
            dtype=args.dtype,
        )
        save_embeddings(embedding_dir / f"dinov2_{split}.npy", embeddings)
        split_sizes[split] = int(embeddings.shape[0])
        embedding_dim = int(embeddings.shape[1])
        if embedding_dim != int(config["image"]["embedding_dim"]):
            raise ValueError(f"{config['image']['model_name']} gives dim {embedding_dim}, "
                             f"config image.embedding_dim is {config['image']['embedding_dim']}")
        print(f"{split}: {embeddings.shape}")

    save_json(
        embedding_dir / "dinov2_meta.json",
        {
            "model_name": config["image"]["model_name"],
            "input_size": input_size,
            "embedding_dim": embedding_dim,
            "split_sizes": split_sizes,
            "limit": args.limit,
            "device": device,
            "dtype": args.dtype,
            "batch_size": batch_size,
            "num_workers": args.num_workers,
            "tile_ids": tile_ids,  # checked by fit/eval (src.encoders.check_split_embeddings)
        },
    )


if __name__ == "__main__":
    main()
