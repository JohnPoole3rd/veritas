#!/usr/bin/env python
"""Evaluate Procrustes alignment on Upper Arlington local tiles (A: T12)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.alignment import load_procrustes
from src.config import load_config, resolve_path, save_json, set_seed
from src.encoders import load_embeddings
from src.local_tiles import evaluate_local, load_manifest, validate_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    manifest = load_manifest(resolve_path(config["data"]["local_manifest"], config))
    summary = validate_manifest(manifest)
    if not summary["ok"] and not args.allow_partial:
        raise ValueError("Local manifest is incomplete:\n- " + "\n- ".join(summary["problems"]))

    embedding_dir = resolve_path("artifacts/embeddings", config)
    alignment_dir = resolve_path("artifacts/alignment", config)
    metrics_dir = resolve_path("artifacts/metrics", config)

    local_embeddings = load_embeddings(embedding_dir / "dinov2_local.npy")
    if len(local_embeddings) != len(manifest):
        raise ValueError(
            f"Local embeddings have {len(local_embeddings)} rows but manifest has {len(manifest)}"
        )
    params = load_procrustes(alignment_dir / "procrustes.npz")
    captions = np.load(embedding_dir / "minilm_captions.npy")
    index = json.loads((embedding_dir / "minilm_caption_index.json").read_text())
    caption_labels = [index[str(i)]["label"] for i in range(len(captions))]

    result = evaluate_local(
        manifest,
        local_embeddings,
        params,
        captions,
        caption_labels,
    )
    result.to_csv(metrics_dir / "local.csv", index=False)

    per_category = (
        result.groupby("local_category")["correct"].agg(["count", "mean"]).reset_index()
    )
    per_category = per_category.rename(columns={"count": "n", "mean": "accuracy"})
    per_category.to_csv(metrics_dir / "local_by_category.csv", index=False)
    summary_payload = {
        "n": int(len(result)),
        "accuracy": float(result["correct"].mean()) if len(result) else 0.0,
        "per_category": per_category.to_dict("records"),
        "manifest": str(resolve_path(config["data"]["local_manifest"], config)),
    }
    save_json(metrics_dir / "local.json", summary_payload)
    print(json.dumps(summary_payload, indent=2))
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
