"""Closed-form Procrustes alignment."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.config import ensure_parent


def l2_normalize(array: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norms, eps)


def center_embeddings(
    X: np.ndarray,
    Y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Center image and text embeddings using their training means."""
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"Pair count mismatch: {X.shape[0]} images vs {Y.shape[0]} texts")
    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    return X - mu_x, Y - mu_y, mu_x, mu_y


def fit_procrustes(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    center: bool = True,
    shuffle_y: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit the orthogonal/semi-orthogonal map X -> Y.

    Returns a dict with W, mu_x, mu_y, scale, and method.
    """
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    if shuffle_y:
        rng = np.random.default_rng(seed)
        Y = Y[rng.permutation(Y.shape[0])]
    if center:
        Xc, Yc, mu_x, mu_y = center_embeddings(X, Y)
    else:
        Xc, Yc = X, Y
        mu_x = np.zeros(X.shape[1], dtype=np.float32)
        mu_y = np.zeros(Y.shape[1], dtype=np.float32)
    cross = Xc.T @ Yc
    U, singular_values, Vt = np.linalg.svd(cross, full_matrices=False)
    W = (U @ Vt).astype(np.float32)
    scale = float(singular_values.sum())
    return {
        "W": W,
        "mu_x": np.asarray(mu_x, dtype=np.float32),
        "mu_y": np.asarray(mu_y, dtype=np.float32),
        "scale": scale,
        "method": "orthogonal_procrustes",
        "shuffle_y": bool(shuffle_y),
        "center": bool(center),
    }


def transform_images(X: np.ndarray, params: dict[str, Any], normalize: bool = True) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32) - np.asarray(params["mu_x"], dtype=np.float32)
    transformed = X @ np.asarray(params["W"], dtype=np.float32)
    return l2_normalize(transformed) if normalize else transformed.astype(np.float32)


def transform_texts(Y: np.ndarray, params: dict[str, Any], normalize: bool = True) -> np.ndarray:
    Y = np.asarray(Y, dtype=np.float32) - np.asarray(params["mu_y"], dtype=np.float32)
    return l2_normalize(Y) if normalize else Y.astype(np.float32)


def orthogonality_error(params: dict[str, Any]) -> float:
    W = np.asarray(params["W"], dtype=np.float32)
    gram = W.T @ W
    identity = np.eye(gram.shape[0], dtype=np.float32)
    return float(np.linalg.norm(gram - identity, ord="fro"))


def save_procrustes(path: str | os.PathLike[str], params: dict[str, Any]) -> None:
    target = ensure_parent(path)
    metadata = {
        key: value
        for key, value in params.items()
        if key not in {"W", "mu_x", "mu_y"}
    }
    np.savez(
        target,
        W=np.asarray(params["W"], dtype=np.float32),
        mu_x=np.asarray(params["mu_x"], dtype=np.float32),
        mu_y=np.asarray(params["mu_y"], dtype=np.float32),
        metadata=json.dumps(metadata),
    )


def load_procrustes(path: str | os.PathLike[str]) -> dict[str, Any]:
    data = np.load(Path(path), allow_pickle=False)
    if "metadata" in data:
        metadata = json.loads(str(data["metadata"]))
    elif "config" in data:
        metadata = json.loads(str(data["config"]))
    else:
        metadata = {}
    params: dict[str, Any] = {
        "W": data["W"].astype(np.float32),
        "mu_x": data["mu_x"].astype(np.float32),
        "mu_y": data["mu_y"].astype(np.float32),
    }
    params.update(metadata)
    return params


def training_pairs(
    config: dict[str, Any],
    caption_embeddings: np.ndarray,
    captions: list[dict[str, Any]],
    limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(train image embeddings, caption embedding of each image's label): the Procrustes pairs.

    Shared by scripts/fit_alignment.py and both random-rotation baselines, so they all fit on
    exactly the same pairs (`src.captions.sample_caption_indices`, seeded by config["seed"]).
    """
    from src.captions import sample_caption_indices
    from src.config import load_json, resolve_path
    from src.data import load_split
    from src.encoders import check_split_embeddings, load_embeddings

    emb_dir = resolve_path("artifacts/embeddings", config)
    items = load_split(resolve_path(config["data"]["splits_dir"], config) / "train.json")["items"][:limit]
    X = load_embeddings(emb_dir / "dinov2_train.npy")[:limit]
    meta_path = emb_dir / "dinov2_meta.json"
    check_split_embeddings(X, items, "train", load_json(meta_path) if meta_path.exists() else None)
    idx = sample_caption_indices(items, captions, int(config.get("seed", 42)))
    return X, np.asarray(caption_embeddings, dtype=np.float32)[idx]


def random_rotation_params(X: np.ndarray, Y: np.ndarray, seeds: list[int], center: bool = True) -> list[dict[str, Any]]:
    """Null baseline: Procrustes on shuffled (image, caption) pairs, one fit per seed."""
    return [fit_procrustes(X, Y, center=center, shuffle_y=True, seed=s) for s in seeds]
