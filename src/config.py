"""Configuration loading and path helpers."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load the YAML config and resolve it relative to the repo root."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["_repo_root"] = str(REPO_ROOT)
    config["_config_path"] = str(config_path)
    return config


def resolve_path(path: str | os.PathLike[str], config: dict[str, Any] | None = None) -> Path:
    """Resolve a config path relative to the repo root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    root = Path((config or {}).get("_repo_root", REPO_ROOT))
    return root / candidate


def ensure_parent(path: str | os.PathLike[str]) -> Path:
    """Create the parent directory for a file path and return the path."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def set_seed(seed: int) -> None:
    """Set Python and NumPy seeds, plus Torch if available."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_json(path: str | os.PathLike[str], payload: Any) -> None:
    target = ensure_parent(path)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def load_json(path: str | os.PathLike[str]) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
