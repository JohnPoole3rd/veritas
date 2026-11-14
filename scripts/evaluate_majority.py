#!/usr/bin/env python
"""Majority baseline wrapper.

Delegates to A's single implementation in `src.baselines.run_majority` so the
tie-break and CSV schema match `results/T09.md` and `plan/INTERFACES.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines import run_majority
from src.config import load_config, resolve_path, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    split_dir = resolve_path(config["data"]["splits_dir"], config)
    out_dir = resolve_path("artifacts", config)
    metrics = run_majority(split_dir, out_dir)
    payload = {
        key: value
        for key, value in metrics.items()
        if key not in {"confusion_matrix"}
    }
    save_json(out_dir / "metrics" / "majority.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
