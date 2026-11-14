"""Evaluation metrics and result saving."""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.alignment import l2_normalize


def _topk_hit(scores: np.ndarray, labels: list[str], correct_label: str, k: int) -> bool:
    order = np.argsort(-scores)[:k]
    return any(labels[int(i)] == correct_label for i in order)


def evaluate_retrieval(
    image_embeddings: np.ndarray,
    caption_embeddings: np.ndarray,
    image_labels: Iterable[str],
    caption_labels: Iterable[str],
    k_values: tuple[int, ...] = (1, 5),
) -> dict[str, Any]:
    """Evaluate text->image retrieval."""
    images = l2_normalize(np.asarray(image_embeddings, dtype=np.float32))
    captions = l2_normalize(np.asarray(caption_embeddings, dtype=np.float32))
    image_labels = list(image_labels)
    caption_labels = list(caption_labels)
    if len(image_labels) != len(images):
        raise ValueError("Image label count does not match embeddings")
    if len(caption_labels) != len(captions):
        raise ValueError("Caption label count does not match embeddings")

    similarity = captions @ images.T
    results: dict[str, Any] = {
        "n_images": len(images),
        "n_captions": len(captions),
    }
    per_class: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(list))
    for k in k_values:
        hits = []
        for row, label in enumerate(caption_labels):
            hit = _topk_hit(similarity[row], image_labels, label, k)
            hits.append(hit)
            per_class[label][f"top{k}"].append(float(hit))
        results[f"top{k}"] = float(np.mean(hits)) if hits else 0.0

    results["per_class"] = {
        label: {
            metric: float(np.mean(values)) if values else 0.0
            for metric, values in metrics.items()
        }
        for label, metrics in sorted(per_class.items())
    }
    return results


def evaluate_image_to_text(
    image_embeddings: np.ndarray,
    caption_embeddings: np.ndarray,
    image_labels: Iterable[str],
    caption_labels: Iterable[str],
    k_values: tuple[int, ...] = (1, 5),
) -> dict[str, Any]:
    """Evaluate image->text classification/retrieval."""
    images = l2_normalize(np.asarray(image_embeddings, dtype=np.float32))
    captions = l2_normalize(np.asarray(caption_embeddings, dtype=np.float32))
    image_labels = list(image_labels)
    caption_labels = list(caption_labels)
    similarity = images @ captions.T
    results: dict[str, Any] = {
        "n_images": len(images),
        "n_captions": len(captions),
    }
    for k in k_values:
        hits = [
            _topk_hit(similarity[row], caption_labels, label, k)
            for row, label in enumerate(image_labels)
        ]
        results[f"top{k}"] = float(np.mean(hits)) if hits else 0.0
    return results


def confusion_matrix(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    labels: Iterable[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

    y_true = list(y_true)
    y_pred = list(y_pred)
    label_list = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    matrix = sklearn_confusion_matrix(y_true, y_pred, labels=label_list)
    return matrix, label_list


def classification_metrics(
    y_true: Iterable[str],
    y_pred: Iterable[str],
) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    y_true = list(y_true)
    y_pred = list(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def majority_baseline(
    train_labels: Iterable[str],
    test_labels: Iterable[str],
) -> dict[str, Any]:
    from collections import Counter

    train_labels = list(train_labels)
    test_labels = list(test_labels)
    most_common = Counter(train_labels).most_common(1)[0][0]
    predictions = [most_common] * len(test_labels)
    metrics = classification_metrics(test_labels, predictions)
    metrics["majority_class"] = most_common
    metrics["test_n"] = len(test_labels)
    return metrics


def save_metrics_csv(path: str | os.PathLike[str], rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No metric rows to save")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix_figure(
    matrix: np.ndarray,
    labels: list[str],
    path: str | os.PathLike[str],
    title: str = "Confusion matrix",
) -> None:
    import matplotlib.pyplot as plt

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)
