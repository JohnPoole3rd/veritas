"""Frozen image and text encoders."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.config import ensure_parent


def get_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def load_dinov2(model_name: str, device: str | None = None):
    """Load a frozen DINOv2 model."""
    from transformers import AutoModel

    device = device or get_device()
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def extract_image_embeddings(
    model,
    dataset,
    batch_size: int = 32,
    device: str | None = None,
    num_workers: int = 0,
) -> np.ndarray:
    """Extract CLS embeddings for a dataset in order."""
    import torch
    from torch.utils.data import DataLoader

    device = device or get_device()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                images = batch[0]
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                images = batch[0]
            else:
                images = batch
            images = images.to(device)
            outputs = model(pixel_values=images)
            if getattr(outputs, "pooler_output", None) is not None:
                embeddings = outputs.pooler_output
            else:
                embeddings = outputs.last_hidden_state[:, 0, :]
            chunks.append(embeddings.detach().cpu().float().numpy())
    if not chunks:
        raise ValueError("No embeddings extracted; is the dataset empty?")
    return np.concatenate(chunks, axis=0).astype(np.float32)


def load_minilm(model_name: str):
    """Load a frozen sentence-transformer."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    model.eval()
    return model


def encode_texts(
    model,
    texts: Iterable[str],
    batch_size: int = 64,
    normalize: bool = False,
) -> np.ndarray:
    """Encode texts to a float32 matrix."""
    texts = list(texts)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def save_embeddings(path: str | os.PathLike[str], embeddings: np.ndarray) -> None:
    target = ensure_parent(path)
    np.save(target, np.asarray(embeddings, dtype=np.float32))


def load_embeddings(path: str | os.PathLike[str]) -> np.ndarray:
    return np.load(Path(path)).astype(np.float32)


def save_metadata(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    target = ensure_parent(path)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def load_metadata(path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
