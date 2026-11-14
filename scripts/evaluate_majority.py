#!/usr/bin/env python
"""Majority baseline for A's T09."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config, resolve_path, save_json
from src.data import load_split, split_labels
from src.evaluate import majority_baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    split_dir = resolve_path(config["data"]["splits_dir"], config)
    train = load_split(split_dir / "train.json")
    test = load_split(split_dir / "test.json")
    metrics = majority_baseline(split_labels(train), split_labels(test))
    save_json(resolve_path("artifacts/metrics/majority.json", config), metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
