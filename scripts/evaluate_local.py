#!/usr/bin/env python
"""T12 — evaluate Procrustes (and ResNet-18, if trained) on the Upper Arlington tiles (owner: A).

Thin wrapper around `src.local_tiles.run_local_eval` (one implementation). Uses the validated
local-embedding cache from scripts/extract_local_embeddings.py, or embeds the tiles itself.

Writes (INTERFACES §4):
    artifacts/metrics/local.csv              tile_id,true_label,local_category,predicted_label,score,correct
    artifacts/metrics/local_by_category.csv  local_category,n,accuracy (+ resnet18/majority columns)
    artifacts/metrics/local.json             n, accuracy, per_category, manifest (+ methods)
plus A's extras: local_predictions.csv (top-3, ResNet-18) and local_summary.csv (method comparison,
including the EuroSAT train-majority baseline and the oracle "most common local label").
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config, resolve_path  # noqa: E402
from src.local_tiles import load_manifest, run_local_eval, validate_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Allow evaluating with a procrustes.npz fitted on a --limit subset")
    args = parser.parse_args()

    config = load_config(args.config)
    from src.alignment import load_procrustes

    limit = load_procrustes(resolve_path("artifacts/alignment/procrustes.npz", config)).get("limit")
    if limit is not None and not args.force:  # guard (J): don't overwrite full-split local metrics
        raise SystemExit(f"procrustes.npz was fitted with --limit {limit}; rerun the full split or pass --force.")
    manifest = load_manifest(resolve_path(config["data"]["local_manifest"], config))
    summary = validate_manifest(manifest)
    if not summary["ok"] and not args.allow_partial:
        raise ValueError("Local manifest is incomplete:\n- " + "\n- ".join(summary["problems"]))
    run_local_eval(args.config)


if __name__ == "__main__":
    main()
