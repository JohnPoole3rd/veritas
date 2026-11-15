"""Frozen image and text encoders.

Hardware support is intentionally small and explicit: CUDA (including ROCm),
Apple MPS, Intel XPU, then CPU. PyTorch's Vulkan backend is not a usable
training/inference path for these models (it has no official wheels and limited
operator coverage), so we do not pretend otherwise.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.config import ensure_parent


def _device_type(device: str) -> str:
    return device.split(":", 1)[0]


def get_device(preferred: str | None = None) -> str:
    """Return a usable torch device string.

    Order: CUDA/ROCm -> Apple MPS -> Intel XPU -> CPU. `PLATONIC_DEVICE` or an
    explicit `preferred` value overrides, and an unavailable device fails loudly
    instead of silently falling back.
    """
    import torch

    requested = preferred or os.environ.get("PLATONIC_DEVICE")
    if requested:
        kind = _device_type(requested)
        available = {
            "cuda": torch.cuda.is_available(),
            "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
            "xpu": bool(getattr(torch, "xpu", None) and torch.xpu.is_available()),
            "cpu": True,
        }.get(kind, False)
        if not available:
            raise RuntimeError(f"Requested device {requested!r} is not available on this machine")
        return requested

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if getattr(torch, "xpu", None) and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def device_summary(device: str) -> str:
    """Human-readable backend name for logs and result JSON."""
    import torch

    kind = _device_type(device)
    index = int(device.split(":", 1)[1]) if ":" in device else 0
    if kind == "cuda":
        return f"cuda ({torch.cuda.get_device_name(index)})"
    if kind == "mps":
        return "mps (Apple Silicon)"
    if kind == "xpu":
        return f"xpu ({torch.xpu.get_device_name(index)})"
    return "cpu"


def resolve_dtype(device: str, dtype: str = "fp32"):
    """Resolve a dtype name to a torch dtype.

    Default is fp32. `auto` uses bf16 on CUDA/XPU when supported. We avoid fp16
    for DINOv3 because its attention/softmax path is sensitive to fp16 ranges.
    """
    import torch

    if dtype in ("fp32", "float32"):
        return torch.float32
    if dtype in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype in ("fp16", "float16"):
        return torch.float16
    if dtype != "auto":
        raise ValueError(f"Unknown dtype {dtype!r}; use fp32, bf16, fp16, or auto")
    kind = _device_type(device)
    if kind == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if kind == "xpu":
        try:
            if torch.xpu.is_bf16_supported():
                return torch.bfloat16
        except AttributeError:
            pass
    return torch.float32


@contextlib.contextmanager
def autocast(device: str, dtype):
    """Use autocast only where PyTorch supports it well; otherwise run fp32."""
    import torch

    kind = _device_type(device)
    if kind == "cpu" and dtype == torch.float16:
        yield
        return
    if dtype == torch.float32 or kind not in ("cuda", "xpu", "cpu"):
        yield
        return
    with torch.autocast(device_type=kind, dtype=dtype):
        yield


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
    dtype: str = "fp32",
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int = 2,
) -> np.ndarray:
    """Extract CLS embeddings for a dataset in order.

    Optimizations are deliberately boring: inference mode, a worker-backed
    DataLoader, pinned host memory on GPU backends, and optional autocast. There
    is no model training here, so `torch.compile` is left to the caller.
    """
    import torch
    from torch.utils.data import DataLoader

    device = device or get_device()
    kind = _device_type(device)
    pin = kind in ("cuda", "xpu") if pin_memory is None else pin_memory
    persistent = num_workers > 0 if persistent_workers is None else persistent_workers
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    torch_dtype = resolve_dtype(device, dtype)
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode(), autocast(device, torch_dtype):
        for batch in loader:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(device, non_blocking=pin)
            outputs = model(pixel_values=images)
            if getattr(outputs, "pooler_output", None) is not None:
                embeddings = outputs.pooler_output
            else:
                embeddings = outputs.last_hidden_state[:, 0, :]
            chunks.append(embeddings.detach().cpu().float().numpy())
    if not chunks:
        raise ValueError("No embeddings extracted; is the dataset empty?")
    return np.concatenate(chunks, axis=0).astype(np.float32)


def load_minilm(model_name: str, device: str | None = None):
    """Load a frozen sentence-transformer."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
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


def check_split_embeddings(
    embeddings: np.ndarray,
    items: list[dict[str, Any]],
    split: str,
    meta: dict[str, Any] | None = None,
) -> None:
    """Fail loudly (INTERFACES §9) if embeddings do not line up with split items.

    Checks the row count and, when `dinov2_meta.json` recorded them, the tile_ids in order.
    """
    if len(embeddings) != len(items):
        raise ValueError(
            f"{split}: embeddings {tuple(embeddings.shape)} vs {len(items)} split items. "
            "Re-run scripts/extract_embeddings.py with the same --limit."
        )
    ids = ((meta or {}).get("tile_ids") or {}).get(split)
    if ids is not None and ids[: len(items)] != [item["tile_id"] for item in items]:  # --limit uses a prefix
        raise ValueError(
            f"{split}: dinov2_meta.json tile_ids do not match data/splits/{split}.json; "
            "the splits changed after extraction. Re-run scripts/extract_embeddings.py."
        )
