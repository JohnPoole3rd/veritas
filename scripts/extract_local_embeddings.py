#!/usr/bin/env python
"""Extract DINOv2 embeddings for Upper Arlington local tiles (A: T10/T12)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config, resolve_path, save_json, set_seed
from src.data import get_transforms
from src.encoders import extract_image_embeddings, load_dinov2, save_embeddings
from src.local_tiles import LocalTileDataset, load_manifest, validate_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    import torch

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    manifest_path = resolve_path(config["data"]["local_manifest"], config)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Local manifest not found: {manifest_path}. Run T10/T11 first."
        )
    manifest = load_manifest(manifest_path)
    summary = validate_manifest(manifest)
    if not summary["ok"] and not args.allow_partial:
        raise ValueError("Local manifest is incomplete:\n- " + "\n- ".join(summary["problems"]))
    if not summary["ok"]:
        print("[warn] continuing with partial local manifest")
        for problem in summary["problems"]:
            print(f"  - {problem}")

    root = Path(config["_repo_root"])
    transform = get_transforms(int(config["image"]["input_size"]), train=False)
    dataset = LocalTileDataset(manifest, root=root, transform=transform)
    model = load_dinov2(config["image"]["model_name"], device=args.device)
    embeddings = extract_image_embeddings(
        model,
        dataset,
        batch_size=args.batch_size,
        device=args.device,
    )
    embedding_dir = resolve_path("artifacts/embeddings", config)
    save_embeddings(embedding_dir / "dinov2_local.npy", embeddings)
    save_json(
        embedding_dir / "dinov2_local_meta.json",
        {
            "model_name": config["image"]["model_name"],
            "input_size": int(config["image"]["input_size"]),
            "embedding_dim": int(embeddings.shape[1]),
            "n_tiles": int(embeddings.shape[0]),
            "tile_ids": manifest["tile_id"].tolist(),
            "manifest": str(manifest_path),
        },
    )
    print(f"local embeddings: {embeddings.shape}")


if __name__ == "__main__":
    main()
