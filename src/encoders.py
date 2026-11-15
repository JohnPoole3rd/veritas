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
        if available and ":" in requested and kind in ("cuda", "xpu"):
            index = int(requested.split(":", 1)[1])
            count = torch.cuda.device_count() if kind == "cuda" else torch.xpu.device_count()
            available = 0 <= index < count
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
    for ViT attention/softmax paths because they are sensitive to fp16 ranges.
    """
    import torch

    kind = _device_type(device)
    if dtype in ("fp32", "float32"):
        return torch.float32
    if dtype in ("bf16", "bfloat16"):
        return torch.float32 if kind == "mps" else torch.bfloat16
    if dtype in ("fp16", "float16"):
        return torch.float32 if kind in ("mps", "cpu") else torch.float16
    if dtype != "auto":
        raise ValueError(f"Unknown dtype {dtype!r}; use fp32, bf16, fp16, or auto")
    if kind == "mps":
        # PyTorch MPS does not expose autocast; keep the actual compute dtype
        # in sync with what is logged instead of silently claiming bf16.
        return torch.float32
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


def configure_cuda_fast_paths() -> None:
    """Enable cuDNN autotuning and TF32 matmuls on CUDA (no-op elsewhere).

    TF32 only affects fp32 matmuls/convs; under bf16 autocast it changes nothing.
    """
    import torch

    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def load_dinov2(model_name: str, device: str | None = None, compile: bool = False):
    """Load a frozen DINOv2 model.

    On CUDA this also turns on cuDNN benchmark/TF32. `compile=True` wraps the model in
    `torch.compile` (opt-in only: Blackwell + a new torch can be flaky).
    """
    from transformers import AutoModel

    device = device or get_device()
    if _device_type(device) == "cuda":
        configure_cuda_fast_paths()
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    if compile:
        import torch

        model = torch.compile(model, dynamic=False)
    return model


class UInt8ImageList:
    """Decode images to uint8 CHW tensors in DataLoader workers (no resize/normalize).

    Pair with `GPUEvalTransform` so the resize, crop and normalization run on the
    accelerator. A 64x64 uint8 chip is ~50x fewer bytes than a 224x224 float32 tensor,
    and PIL's upscale to 256 px was most of the per-image CPU cost.
    """

    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        import torch
        from PIL import Image

        with Image.open(self.paths[index]) as image:
            array = np.array(image.convert("RGB"), dtype=np.uint8)
        return index, torch.from_numpy(array).permute(2, 0, 1)


def collate_by_shape(batch):
    """Collate (index, uint8 CHW) pairs into one stacked tensor per image shape.

    Returns a list of (indices, images) groups; all-64x64 data is a single group.
    """
    import torch

    groups: dict[tuple[int, ...], list] = {}
    for index, image in batch:
        groups.setdefault(tuple(image.shape), []).append((index, image))
    return [
        (torch.tensor([i for i, _ in items], dtype=torch.long), torch.stack([im for _, im in items]))
        for items in groups.values()
    ]


class GPUEvalTransform:
    """On-device equivalent of `src.data.get_transforms(input_size)`.

    torchvision Resize(int(size*256/224)) on a PIL image (bilinear, shorter side) ->
    CenterCrop(size) -> ToTensor -> Normalize(ImageNet). PIL resizes separably
    (horizontal pass, then vertical) and rounds to uint8 after *each* pass; doing the same
    here reproduces PIL bit-exactly for upscaling (64 -> 256 px chips). This matters:
    a single-pass resize differs by 1 grey level on ~17% of pixels, which already moves
    DINOv2 CLS embeddings to cos-sim ~0.95-0.99.
    """

    def __init__(self, input_size: int = 224, channels_last: bool = False):
        from src.data import IMAGENET_MEAN, IMAGENET_STD

        self.input_size = int(input_size)
        self.resize = int(input_size * 256 / 224)
        self.mean = IMAGENET_MEAN
        self.std = IMAGENET_STD
        self.channels_last = channels_last
        self._stats: dict = {}

    def _resized_hw(self, h: int, w: int) -> tuple[int, int]:
        # torchvision's _compute_resized_output_size for an int size (shorter side).
        short, long = (w, h) if w <= h else (h, w)
        new_short, new_long = self.resize, int(self.resize * long / short)
        return (new_long, new_short) if w <= h else (new_short, new_long)

    def __call__(self, images):
        import torch

        kind = images.device.type
        if kind in ("cuda", "xpu", "cpu"):
            # Preprocessing must stay fp32 even when called inside a bf16/fp16 autocast block.
            with torch.autocast(device_type=kind, enabled=False):
                return self._apply(images)
        return self._apply(images)

    def _apply(self, images):
        import torch
        import torch.nn.functional as F

        x = images.float()
        h, w = x.shape[-2:]
        out_h, out_w = self._resized_hw(h, w)
        # PIL order: horizontal pass, round to uint8 (half up), then vertical pass, round.
        if out_w != w:
            x = F.interpolate(x, size=(h, out_w), mode="bilinear", align_corners=False,
                              antialias=out_w < w)
            x = x.add_(0.5).floor_().clamp_(0, 255)
        if out_h != h:
            x = F.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=False,
                              antialias=out_h < h)
            x = x.add_(0.5).floor_().clamp_(0, 255)
        size = self.input_size
        top = int(round((out_h - size) / 2.0))
        left = int(round((out_w - size) / 2.0))
        x = x[..., top:top + size, left:left + size]
        if x.device not in self._stats:
            mean = torch.tensor(self.mean, device=x.device).view(1, 3, 1, 1)
            std = torch.tensor(self.std, device=x.device).view(1, 3, 1, 1)
            self._stats[x.device] = (mean, std)
        mean, std = self._stats[x.device]
        x = (x.div_(255.0) - mean) / std  # same op order as ToTensor + Normalize
        memory_format = torch.channels_last if self.channels_last else torch.contiguous_format
        return x.contiguous(memory_format=memory_format)


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
    device_transform=None,
    progress_every: int = 0,
) -> np.ndarray:
    """Extract CLS embeddings for a dataset in order.

    Two input modes:
    * default: the dataset yields normalized float tensors (CPU transform), as before;
    * `device_transform` set: the dataset yields (index, uint8 CHW) pairs (see
      `UInt8ImageList`) and `device_transform` resizes/normalizes on the device.

    Inference mode, worker-backed DataLoader, pinned memory + non_blocking copies on
    GPU backends, optional autocast. Embeddings stay on the device and are copied back
    in large chunks, so there is no host sync after every batch.
    """
    import time

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
        collate_fn=collate_by_shape if device_transform is not None else None,
    )
    torch_dtype = resolve_dtype(device, dtype)
    n = len(dataset)
    out: np.ndarray | None = None
    pending: list = []  # (indices, embeddings on device)
    filled = 0

    def flush():
        nonlocal out
        for indices, emb in pending:
            emb = emb.float().cpu().numpy()
            if out is None:
                out = np.empty((n, emb.shape[1]), dtype=np.float32)
            out[indices.numpy()] = emb
        pending.clear()

    def forward(images):
        outputs = model(pixel_values=images)
        if getattr(outputs, "pooler_output", None) is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0, :]

    model.eval()
    start = time.perf_counter()
    with torch.inference_mode(), autocast(device, torch_dtype):
        for step, batch in enumerate(loader):
            if device_transform is not None:
                groups = batch
            else:
                images = batch[0] if isinstance(batch, (list, tuple)) else batch
                groups = [(torch.arange(filled, filled + len(images)), images)]
            for indices, images in groups:
                images = images.to(device, non_blocking=pin)
                if device_transform is not None:
                    images = device_transform(images)
                pending.append((indices, forward(images)))
                filled += len(indices)
            if sum(len(i) for i, _ in pending) >= 16384:
                flush()
            if progress_every and (step + 1) % progress_every == 0:
                rate = filled / max(time.perf_counter() - start, 1e-9)
                print(f"  {filled}/{n} images ({rate:.0f} img/s)", flush=True)
        flush()
    if out is None:
        raise ValueError("No embeddings extracted; is the dataset empty?")
    if filled != n:
        raise ValueError(f"Extracted {filled} embeddings for {n} dataset items")
    if not np.isfinite(out).all():
        raise FloatingPointError("Non-finite image embeddings (fp16 overflow?); rerun with --dtype fp32")
    return out


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
    dedupe: bool = True,
) -> np.ndarray:
    """Encode texts to a float32 matrix (one row per input text, input order).

    With `dedupe` (default) each distinct text is encoded once and the rows are indexed
    back; templated captions repeat heavily, so this is most of the text-encoding cost at
    1M rows. Output equals the non-deduplicated encoding up to float tolerance (batch padding).
    """
    texts = list(texts)
    if dedupe:
        from src.alignment import unique_rows

        unique, index = unique_rows(texts)
        if len(unique) < len(texts):
            return encode_texts(model, unique, batch_size=batch_size, normalize=normalize, dedupe=False)[index]
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
