#!/usr/bin/env python
"""Train the ResNet-18 baseline (A's T08). Thin wrapper around `python -m src.baselines resnet18`.

All flags are passed through, e.g. `python scripts/train_resnet.py --epochs 10 --device mps`.
Writes artifacts/checkpoints/resnet18.pt and artifacts/metrics/resnet18{.csv,_per_class.csv,_config.json}.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines import main  # noqa: E402

if __name__ == "__main__":
    main(["resnet18", *sys.argv[1:]])
