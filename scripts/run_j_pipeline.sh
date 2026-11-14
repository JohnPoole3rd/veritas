#!/usr/bin/env bash
set -euo pipefail

# Full J pipeline. Run from the repository root.
# WARNING: this overwrites artifacts/alignment/procrustes.npz and
# artifacts/metrics/retrieval.csv. On main, use a scratch worktree unless you
# intend to rerun the full split and regenerate the committed metrics.
python scripts/prepare_data.py
python scripts/extract_embeddings.py
python scripts/fit_alignment.py
python scripts/evaluate_retrieval.py
python scripts/evaluate_majority.py

# A's supervised baseline:
# python scripts/train_resnet.py
