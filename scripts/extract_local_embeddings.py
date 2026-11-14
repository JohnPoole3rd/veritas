#!/usr/bin/env python
"""T12 — DINOv2 embeddings for the Upper Arlington tiles (owner: A).

Same frozen encoder and eval transforms as the EuroSAT embeddings. Writes the one local cache
format (artifacts/embeddings/dinov2_local.npy + dinov2_local_meta.json with tile_ids, model_name,
input_size) that scripts/evaluate_local.py and app.py validate before reuse.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config, resolve_path  # noqa: E402
from src.encoders import get_device, load_dinov2  # noqa: E402
from src.local_tiles import _embed_images, load_local_manifest, save_local_embeddings, validate_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    manifest_rel = config["data"]["local_manifest"]
    manifest = load_local_manifest(resolve_path(manifest_rel, config))  # raises on bad rows / files
    summary = validate_manifest(manifest)
    if not summary["ok"]:
        if not args.allow_partial:
            raise ValueError("Local manifest is incomplete:\n- " + "\n- ".join(summary["problems"]))
        print("[warn] continuing with partial local manifest:\n  - " + "\n  - ".join(summary["problems"]))

    device = args.device or get_device()
    model = load_dinov2(config["image"]["model_name"], device=device)
    X = _embed_images(model, manifest, int(config["image"]["input_size"]), device)
    save_local_embeddings(X, manifest, config, manifest_rel)
    print(f"local embeddings: {X.shape}")


if __name__ == "__main__":
    main()
