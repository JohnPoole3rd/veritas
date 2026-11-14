#!/usr/bin/env python
"""T09 — majority-class baseline (owner: A). Thin wrapper around `src.baselines.run_majority`.

Writes artifacts/metrics/majority.csv (INTERFACES §4) and majority.json. Ties between equally common
train classes are broken by `class_names` order (EuroSAT has a 5-way tie → AnnualCrop).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines import main  # noqa: E402

if __name__ == "__main__":
    main(["majority", *sys.argv[1:]])
