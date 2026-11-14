"""Retrieval and classification helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from src.alignment import l2_normalize


def retrieve_top_k(
    query_embedding: np.ndarray,
    gallery_embeddings: np.ndarray,
    k: int = 3,
) -> list[int]:
    """Return indices of the top-k most similar gallery items."""
    indices, _ = retrieve_top_k_with_scores(query_embedding, gallery_embeddings, k=k)
    return indices


def retrieve_top_k_with_scores(
    query_embedding: np.ndarray,
    gallery_embeddings: np.ndarray,
    k: int = 3,
) -> tuple[list[int], list[float]]:
    query = l2_normalize(np.asarray(query_embedding, dtype=np.float32).reshape(1, -1))[0]
    gallery = l2_normalize(np.asarray(gallery_embeddings, dtype=np.float32))
    scores = gallery @ query
    k = int(min(max(k, 0), len(gallery)))
    order = np.argsort(-scores)[:k]
    return [int(i) for i in order], [float(scores[i]) for i in order]


def class_scores(
    image_embeddings: np.ndarray,
    caption_embeddings: np.ndarray,
    caption_labels: Iterable[str],
    class_names: list[str],
    agg: str = "max",
) -> np.ndarray:
    """(N, C) image-to-class scores: cosine to each caption, reduced per class by max or mean.

    The one classification rule shared by evaluation (`src.evaluate`), the local tiles and the demo.
    """
    X = l2_normalize(np.atleast_2d(np.asarray(image_embeddings, dtype=np.float32)))
    Y = l2_normalize(np.asarray(caption_embeddings, dtype=np.float32))
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"embedding dims differ: images {X.shape} vs captions {Y.shape}")
    sims = X @ Y.T
    labels = np.asarray(list(caption_labels))
    reduce = {"max": np.max, "mean": np.mean}[agg]
    return np.stack([reduce(sims[:, labels == c], axis=1) for c in class_names], axis=1)
