#!/usr/bin/env python
"""Prepare EuroSAT splits and class captions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.captions import build_captions, save_captions
from src.config import load_config, resolve_path, set_seed
from src.data import make_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    splits = make_splits(config, download=False if args.no_download else None)  # None: config data.download
    class_names = splits["train"]["class_names"]
    captions = build_captions(class_names)
    caption_path = resolve_path(config["data"]["captions_path"], config)
    save_captions(caption_path, captions)

    print("Created splits:")
    for name, payload in splits.items():
        print(f"  {name}: {len(payload['items'])}")
    print(f"Created {len(captions)} captions at {caption_path}")


if __name__ == "__main__":
    main()
