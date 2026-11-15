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


def unique_rows(texts_or_rows) -> tuple[list, np.ndarray]:
    """(unique items in first-seen order, int64 index so that items == unique[index]).

    Works on any hashable sequence (e.g. caption strings). Templated captions repeat a lot,
    so encoding / fitting on the unique set and indexing back saves most of the work.
    """
    position: dict = {}
    index = np.fromiter(
        (position.setdefault(item, len(position)) for item in texts_or_rows),
        dtype=np.int64,
    )
    return list(position), index


def _as_blocks(X) -> list[np.ndarray]:
    if isinstance(X, (list, tuple)):
        blocks = [np.asarray(b, dtype=np.float32) for b in X]
    else:
        blocks = [np.asarray(X, dtype=np.float32)]
    return [b for b in blocks if b.shape[0]] or blocks[:1]


def fit_procrustes(
    X,
    Y: np.ndarray,
    *,
    center: bool = True,
    shuffle_y: bool = False,
    seed: int = 42,
    chunk_size: int = 100_000,
    y_index: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit the orthogonal/semi-orthogonal map X -> Y.

    The cross-covariance is accumulated in chunks so 1M-row fits do not allocate
    full centered copies of both embedding matrices. Returns W, mu_x, mu_y,
    scale, and method.

    Scale options (all optional, defaults unchanged):
      * `X` may be a list/tuple of row blocks, treated as their vstack without copying it.
      * `y_index`: `Y` is then a table of unique rows and the paired targets are
        `Y[y_index]` (templated captions repeat). The cross-covariance is computed as
        (per-unique-row sums of X)^T @ Y, i.e. U x d instead of N x d in the GEMM.
        With shuffle_y the same seed draws the same permutation of the N targets, so the
        random-rotation control is the same statistic as without y_index (equal up to
        float32 summation order).
    """
    blocks = _as_blocks(X)
    n = sum(b.shape[0] for b in blocks)
    Y = np.asarray(Y, dtype=np.float32)
    if y_index is not None:
        y_index = np.asarray(y_index, dtype=np.int64).reshape(-1)
        n_targets = y_index.shape[0]
    else:
        n_targets = Y.shape[0]
    if n != n_targets:
        raise ValueError(f"X and Y have different row counts: {n} vs {n_targets}")
    if n == 0:
        raise ValueError("Cannot fit Procrustes alignment with zero rows")
    if any(b.ndim != 2 for b in blocks) or Y.ndim != 2:
        raise ValueError(f"Expected 2-D embeddings, got X{blocks[0].shape} and Y{Y.shape}")
    if len({b.shape[1] for b in blocks}) != 1:
        raise ValueError(f"X blocks have different widths: {[b.shape for b in blocks]}")
    d_x = blocks[0].shape[1]
    permutation = None
    if shuffle_y:
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(n)
    if center:
        if len(blocks) == 1:
            mu_x = blocks[0].mean(axis=0, dtype=np.float32)
        else:
            mu_x = (sum(b.sum(axis=0, dtype=np.float64) for b in blocks) / n).astype(np.float32)
        if y_index is None:
            mu_y = Y.mean(axis=0, dtype=np.float32)
        else:
            counts = np.bincount(y_index, minlength=Y.shape[0]).astype(np.float64)
            mu_y = ((counts @ Y.astype(np.float64)) / n).astype(np.float32)
    else:
        mu_x = np.zeros(d_x, dtype=np.float32)
        mu_y = np.zeros(Y.shape[1], dtype=np.float32)
    step = max(1, int(chunk_size))

    if y_index is None:
        cross = np.zeros((d_x, Y.shape[1]), dtype=np.float32)
        offset = 0
        for block in blocks:
            for start in range(0, block.shape[0], step):
                stop = min(start + step, block.shape[0])
                Xc = block[start:stop] - mu_x
                rows = slice(offset + start, offset + stop)
                Yc = (Y[rows] if permutation is None else Y[permutation[rows]]) - mu_y
                cross += Xc.T @ Yc
            offset += block.shape[0]
    else:
        from scipy import sparse

        target = y_index if permutation is None else y_index[permutation]
        n_unique = Y.shape[0]
        # group_sums[u] = sum of centered X rows whose (possibly shuffled) target is Y[u].
        group_sums = np.zeros((n_unique, d_x), dtype=np.float32)
        offset = 0
        for block in blocks:
            for start in range(0, block.shape[0], step):
                stop = min(start + step, block.shape[0])
                groups = target[offset + start: offset + stop]
                onehot = sparse.csr_matrix(
                    (np.ones(stop - start, dtype=np.float32), (groups, np.arange(stop - start))),
                    shape=(n_unique, stop - start),
                )
                group_sums += np.asarray(onehot @ (block[start:stop] - mu_x), dtype=np.float32)
            offset += block.shape[0]
        cross = (group_sums.T @ (Y - mu_y)).astype(np.float32)
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
