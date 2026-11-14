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
    k = int(min(max(k, 1), len(gallery)))
    order = np.argsort(-scores)[:k]
    return [int(i) for i in order], [float(scores[i]) for i in order]


def classify_tile(
    image_embedding: np.ndarray,
    caption_embeddings: np.ndarray,
    caption_labels: Iterable[str],
) -> list[tuple[str, float]]:
    """Return labels and scores sorted by similarity."""
    labels = list(caption_labels)
    indices, scores = retrieve_top_k_with_scores(
        image_embedding,
        caption_embeddings,
        k=len(labels),
    )
    return [(labels[i], score) for i, score in zip(indices, scores)]


def class_centroids(
    embeddings: np.ndarray,
    labels: Iterable[str],
) -> tuple[np.ndarray, list[str]]:
    """Average embeddings per label, returning centroids and labels."""
    labels = list(labels)
    if len(labels) != len(embeddings):
        raise ValueError("Label count does not match embedding count")
    grouped: dict[str, list[np.ndarray]] = {}
    for embedding, label in zip(embeddings, labels):
        grouped.setdefault(label, []).append(np.asarray(embedding, dtype=np.float32))
    class_names = sorted(grouped)
    centroids = np.stack([np.mean(grouped[name], axis=0) for name in class_names], axis=0)
    return centroids.astype(np.float32), class_names
