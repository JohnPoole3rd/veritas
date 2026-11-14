#!/usr/bin/env bash
# Full pipeline, in dependency order. Run from the repository root.
# Verified end to end on the full split in the code review (plan/learnings/general-code-review.md).
set -euo pipefail
PY=${PYTHON:-python}

$PY scripts/prepare_data.py                                   # T02/T03 splits + captions
$PY scripts/extract_embeddings.py                             # T04 DINOv2 (cuda > mps > cpu)
$PY scripts/fit_alignment.py                                  # T05/T06 captions + Procrustes, val retrieval
$PY scripts/evaluate_retrieval.py                             # T07 text→image search (50 queries)
$PY scripts/evaluate_majority.py                              # T09 majority baseline
$PY -m src.evaluate caption-classification --split val        # per-image, comparable with ResNet
$PY -m src.evaluate caption-classification --split test
[ "${SKIP_RESNET:-0}" = 1 ] || $PY scripts/train_resnet.py    # T08 (~4 min on MPS/GPU; SKIP_RESNET=1 to reuse)
$PY scripts/extract_local_embeddings.py                       # T12 Upper Arlington
$PY scripts/evaluate_local.py
$PY -m src.failures                                           # T13 figures + taxonomy
