"""Evaluation metrics and result saving."""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.alignment import l2_normalize
from src.retrieval import class_scores


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
    results: dict[str, Any] = {
        "n_images": len(images),
        "n_captions": len(captions),
    }
    per_class: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(list))
    # Rank each caption row once (same per-row argsort as _topk_hit, so ties break identically)
    # and reuse it for every k; caption rows go in chunks so no full (captions x images)
    # matrix is held at scale.
    image_label_arr = np.asarray(image_labels, dtype=object)
    caption_label_arr = np.asarray(caption_labels, dtype=object)
    k_max = max(k_values) if k_values else 0
    hits_by_k = {k: np.zeros(len(captions), dtype=bool) for k in k_values}
    step = max(1, (64 * 1024 * 1024) // (4 * max(1, len(images))))  # ~64 MB sims (+ argsort int64)
    for start in range(0, len(captions), step):
        similarity = captions[start:start + step] @ images.T
        order = np.argsort(-similarity, axis=1)[:, :k_max]
        match = image_label_arr[order] == caption_label_arr[start:start + step, None]
        for k in k_values:
            hits_by_k[k][start:start + step] = match[:, :k].any(axis=1)
    for k in k_values:
        hits = hits_by_k[k]
        for label, hit in zip(caption_labels, hits):
            per_class[label][f"top{k}"].append(float(hit))
        results[f"top{k}"] = float(np.mean(hits)) if len(hits) else 0.0

    results["per_class"] = {
        label: {
            metric: float(np.mean(values)) if values else 0.0
            for metric, values in metrics.items()
        }
        for label, metrics in sorted(per_class.items())
    }
    return results


def classification_metrics(
    y_true: Iterable,
    y_pred: Iterable,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Accuracy, macro-F1, per-class recall, and confusion matrix.

    y_true / y_pred are label strings or integer indices into class_names. With no class_names,
    the sorted union of observed labels is used (backward compatible with the 2-argument form).
    Confusion matrix rows are true classes, columns predicted, ordered as class_names.
    """
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix

    y_true, y_pred = list(y_true), list(y_pred)
    if class_names is None:
        class_names = sorted(set(y_true) | set(y_pred), key=str)
    to_idx = {c: i for i, c in enumerate(class_names)}

    def idx(v):
        if v in to_idx:
            return to_idx[v]
        if isinstance(v, str):
            raise ValueError(f"label {v!r} is not in class_names {list(class_names)}")
        return int(v)

    yt = np.array([idx(v) for v in y_true], dtype=int)
    yp = np.array([idx(v) for v in y_pred], dtype=int)
    labels = list(range(len(class_names)))
    recall = recall_score(yt, yp, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0)),
        "per_class": {str(c): float(r) for c, r in zip(class_names, recall)},
        "confusion_matrix": sk_confusion_matrix(yt, yp, labels=labels),
        "class_names": [str(c) for c in class_names],
        "n": int(len(yt)),
    }


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


def evaluate_classifier(model, loader, device=None, class_names: list[str] | None = None) -> dict:
    """Run a torch classifier over a loader yielding (images, int_labels).

    class_names defaults to loader.dataset.class_names. Also returns softmax `probs` (N, C)
    and integer `y_true` / `y_pred` arrays for downstream analysis.
    """
    import torch

    if class_names is None:
        class_names = loader.dataset.class_names
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    ys, probs = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            ys.append(np.asarray(y))
    probs = np.concatenate(probs)
    y_true = np.concatenate(ys)
    y_pred = probs.argmax(1)
    out = classification_metrics(y_true, y_pred, class_names)
    out.update(probs=probs, y_true=y_true, y_pred=y_pred)
    return out


def evaluate_caption_classification(
    image_embeddings: np.ndarray,
    caption_embeddings: np.ndarray,
    labels: list[str],
    caption_labels: list[str],
    class_names: list[str] | None = None,
    agg: str = "max",
) -> dict:
    """Image->class classification by caption similarity (A, T08/T12).

    This is the per-image metric that is comparable with ResNet-18 accuracy: every test image
    is classified, so N = number of images. (`evaluate_retrieval` above is J's T07
    text->image metric, where N = number of caption queries.)

    Each image is scored against every caption by cosine similarity (both sides are
    L2-normalized here). Caption scores are aggregated to one score per class with
    `agg` ("max" = best-matching caption for that class, the default; "mean" = average
    over that class's captions, i.e. a prompt ensemble). Classes are ranked by score;
    top1/top5 check whether the true label is in the first 1/5 classes.

    Returns top1, top5, per_class (top-1 recall per class), confusion_matrix (top-1),
    macro_f1, class_names, plus class_scores (N, C) and y_pred labels.
    """
    X = np.asarray(image_embeddings, dtype=np.float32)
    Y = np.asarray(caption_embeddings, dtype=np.float32)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"embedding dims differ: images {X.shape} vs captions {Y.shape}")
    if len(labels) != len(X):
        raise ValueError(f"labels ({len(labels)}) != image rows {X.shape}")
    if len(caption_labels) != len(Y):
        raise ValueError(f"caption_labels ({len(caption_labels)}) != caption rows {Y.shape}")
    if class_names is None:
        class_names = sorted(set(caption_labels))
    scores = class_scores(X, Y, caption_labels, class_names, agg=agg)  # (N, C), shared rule

    to_idx = {c: i for i, c in enumerate(class_names)}
    y_true = np.array([to_idx[y] for y in labels])
    ranking = np.argsort(-scores, axis=1)
    k = min(5, len(class_names))
    top5 = float(np.mean((ranking[:, :k] == y_true[:, None]).any(axis=1))) if len(y_true) else float("nan")

    out = classification_metrics(y_true, ranking[:, 0], class_names)
    out.update(
        top1=out["accuracy"],
        top5=top5,
        agg=agg,
        class_scores=scores,
        y_pred=[class_names[i] for i in ranking[:, 0]],
    )
    return out


def save_confusion_matrix_png(cm, class_names: list[str], path, title: str = "") -> Path:
    """Row-normalized confusion matrix heatmap with raw counts annotated."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)
    norm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(1.0 + 0.7 * n, 0.8 + 0.6 * n))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), class_names, rotation=45, ha="right")
    ax.set_yticks(range(n), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if title:
        ax.set_title(title)
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row fraction")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- A: per-image comparison CLI
def run_caption_classification(split: str = "test", agg: str = "max", n_random_seeds: int = 5) -> list[dict]:
    """Per-image accuracy / macro-F1 for Procrustes, random rotation and unaligned cosine on a
    EuroSAT split. This is the metric comparable with ResNet-18 (N = images, not caption queries).

    Uses the T04-T06 artifacts and the same training pairs as scripts/fit_alignment.py
    (`src.alignment.training_pairs`). Random rotation = mean over `n_random_seeds` shuffles,
    as in scripts/evaluate_retrieval.py. `agg` is fixed in advance, never chosen on test.
    """
    from src.alignment import load_procrustes, random_rotation_params, training_pairs, transform_images, transform_texts
    from src.captions import load_captions
    from src.config import load_config, load_json, resolve_path
    from src.data import load_split
    from src.encoders import check_split_embeddings, load_embeddings

    cfg = load_config()
    seed = int(cfg.get("seed", 42))
    emb = resolve_path("artifacts/embeddings", cfg)
    split_payload = load_split(resolve_path(cfg["data"]["splits_dir"], cfg) / f"{split}.json")
    class_names = split_payload["class_names"]
    X = load_embeddings(emb / f"dinov2_{split}.npy")
    meta_path = emb / "dinov2_meta.json"
    check_split_embeddings(X, split_payload["items"], split, load_json(meta_path) if meta_path.exists() else None)
    labels = [it["label"] for it in split_payload["items"]]
    Y = load_embeddings(emb / "minilm_captions.npy")
    index = load_json(emb / "minilm_caption_index.json")
    cap_labels = [index[str(i)]["label"] for i in range(len(index))]

    params = load_procrustes(resolve_path("artifacts/alignment/procrustes.npz", cfg))
    captions = load_captions(resolve_path(cfg["data"]["captions_path"], cfg))
    X_train, Y_train = training_pairs(cfg, Y, captions)
    seeds = list(range(seed, seed + n_random_seeds))

    def run(Xi, Yi):
        return evaluate_caption_classification(Xi, Yi, labels, cap_labels, class_names=class_names, agg=agg)

    results = {"procrustes": [run(transform_images(X, params), transform_texts(Y, params))]}
    results["random_rotation"] = [run(transform_images(X, p), transform_texts(Y, p))
                                  for p in random_rotation_params(X_train, Y_train, seeds, center=bool(cfg["alignment"]["center"]))]
    if X.shape[1] == Y.shape[1]:  # raw cosine needs equal dims (not ViT-B)
        results["unaligned"] = [run(X, Y)]

    rows = []
    for name, rs in results.items():
        note = f"image->class, agg={agg}"
        if len(rs) > 1:
            note += (f"; mean of {len(rs)} seeds {seeds[0]}-{seeds[-1]}, "
                     f"std acc {np.std([r['accuracy'] for r in rs]):.4f}")
        rows.append({"method": name, "split": split,
                     "accuracy": round(float(np.mean([r["accuracy"] for r in rs])), 4),
                     "top5": round(float(np.mean([r["top5"] for r in rs])), 4),
                     "macro_f1": round(float(np.mean([r["macro_f1"] for r in rs])), 4),
                     "n_images": rs[0]["n"], "n_captions": len(cap_labels), "notes": note})
    r = results["procrustes"][0]
    save_confusion_matrix_png(r["confusion_matrix"], class_names,
                              resolve_path(f"artifacts/figures/procrustes_{split}_confusion_matrix.png", cfg),
                              title=f"Procrustes image->class, {split} (acc {r['accuracy']:.3f})")
    save_metrics_csv(resolve_path(f"artifacts/metrics/procrustes_{split}_per_class.csv", cfg),
                     [{"class": c, "recall": round(r["per_class"][c], 4)} for c in class_names])
    cm = r["confusion_matrix"]  # rows = true class, columns = predicted class
    save_metrics_csv(resolve_path(f"artifacts/metrics/procrustes_{split}_confusion.csv", cfg),
                     [{"true": t, **{c: int(cm[i, j]) for j, c in enumerate(class_names)}}
                      for i, t in enumerate(class_names)])
    save_metrics_csv(resolve_path(f"artifacts/metrics/caption_classification_{split}.csv", cfg), rows)
    return rows

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(prog="python -m src.evaluate")
    ap.add_argument("command", choices=["caption-classification"])
    ap.add_argument("--split", default="test", choices=["val", "test"])
    args = ap.parse_args()
    for row in run_caption_classification(args.split):
        print(row)
