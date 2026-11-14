"""Upper Arlington local tile loading and evaluation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.alignment import transform_images, transform_texts
from src.retrieval import classify_tile

REQUIRED_COLUMNS = [
    "tile_id",
    "path",
    "label",
    "local_category",
    "lat",
    "lon",
    "date",
    "source",
    "notes",
]
LOCAL_CATEGORIES = {
    "residential",
    "commercial_industrial",
    "road_highway",
    "park_forest_herbaceous",
    "water_river",
}


def load_manifest(path: str | os.PathLike[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    return frame


def validate_manifest(
    frame: pd.DataFrame,
    expected_total: int = 50,
    expected_per_category: int = 10,
) -> dict[str, Any]:
    """Validate the local tile manifest and return a summary."""
    problems: list[str] = []
    if len(frame) != expected_total:
        problems.append(f"Expected {expected_total} tiles, found {len(frame)}")
    counts = frame["local_category"].value_counts().to_dict()
    for category in sorted(LOCAL_CATEGORIES):
        if counts.get(category, 0) != expected_per_category:
            problems.append(
                f"Category {category}: expected {expected_per_category}, found {counts.get(category, 0)}"
            )
    unknown = sorted(set(frame["local_category"]) - LOCAL_CATEGORIES)
    if unknown:
        problems.append(f"Unknown categories: {unknown}")
    if frame["tile_id"].duplicated().any():
        problems.append("Duplicate tile_id values")
    for column in ("lat", "lon", "date", "source"):
        if frame[column].isna().any():
            problems.append(f"Missing values in {column}")
    return {
        "ok": not problems,
        "problems": problems,
        "counts": counts,
        "total": len(frame),
    }


class LocalTileDataset:
    """Dataset for local tiles using the manifest."""

    def __init__(self, manifest: pd.DataFrame, root: str | os.PathLike[str], transform=None):
        self.manifest = manifest.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int):
        from PIL import Image

        row = self.manifest.iloc[index]
        image = Image.open(self.root / row["path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, row["tile_id"]


def evaluate_local(
    manifest: pd.DataFrame,
    image_embeddings: np.ndarray,
    procrustes_params: dict[str, Any],
    caption_embeddings: np.ndarray,
    caption_labels: list[str],
) -> pd.DataFrame:
    """Classify local tiles with the aligned image embeddings."""
    rotated = transform_images(image_embeddings, procrustes_params)
    texts = transform_texts(caption_embeddings, procrustes_params)
    rows: list[dict[str, Any]] = []
    for row, embedding in zip(manifest.to_dict("records"), rotated):
        predictions = classify_tile(embedding, texts, caption_labels)
        top_label, top_score = predictions[0]
        rows.append(
            {
                "tile_id": row["tile_id"],
                "true_label": row["label"],
                "local_category": row["local_category"],
                "predicted_label": top_label,
                "score": top_score,
                "correct": top_label == row["label"],
            }
        )
    result = pd.DataFrame(rows)
    return result
