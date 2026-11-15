"""Frozen image backbones for the OSM-Sentinel experiment.

DINOv3 is loaded either from a local Meta source/checkpoint (the same method as
`d/exp-dinov3-skyscript`) or from a gated Hugging Face model. The small DINOv3
checkpoint is the default because it is the one the other experiments actually
ran. The satellite ViT-L is selectable when your account has access.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

EXPECTED_SHA256_PREFIX = "08c60483"
HF_MODELS = {
    "dinov2-small": "facebook/dinov2-small",
    "dinov3-small": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "dinov3-sat-large": "facebook/dinov3-vitl16-pretrain-sat493m",
}

# Official SAT-493M normalization. The LVD-1689M checkpoints use the project's
# existing ImageNet statistics.
SAT_MEAN = (0.430, 0.411, 0.296)
SAT_STD = (0.213, 0.156, 0.143)


def transform_stats(name: str) -> tuple[tuple[float, ...] | None, tuple[float, ...] | None]:
    if name == "dinov3-sat-large":
        return SAT_MEAN, SAT_STD
    return None, None


def _local_paths() -> tuple[Path, Path] | None:
    repo, weights = os.environ.get("DINOV3_REPO"), os.environ.get("DINOV3_WEIGHTS")
    if not repo or not weights:
        return None
    repo, weights = Path(repo).expanduser(), Path(weights).expanduser()
    for path in (repo / "dinov3", weights):
        if not path.exists():
            raise FileNotFoundError(f"DINOv3 path not found: {path}")
    return repo, weights


def _verify_checkpoint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    if not value.startswith(EXPECTED_SHA256_PREFIX):
        raise ValueError(f"DINOv3 checkpoint hash {value[:8]} != expected {EXPECTED_SHA256_PREFIX}")
    return value


class _Wrapper:
    """Expose Meta's DINOv3 CLS token through the HF `pooler_output` contract."""

    def __init__(self, vit):
        self.vit = vit

    def __call__(self, pixel_values=None, **_):
        return SimpleNamespace(pooler_output=self.vit(pixel_values))

    def eval(self):
        self.vit.eval()
        return self

    def to(self, device):
        self.vit.to(device)
        return self

    def parameters(self):
        return self.vit.parameters()


def load_dinov3_small_local(device: str = "cpu"):
    import torch

    paths = _local_paths()
    if paths is None:
        raise RuntimeError("Set DINOV3_REPO and DINOV3_WEIGHTS, or use the HF DINOv3 path")
    repo, weights = paths
    _verify_checkpoint(weights)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from dinov3.hub.backbones import dinov3_vits16

    vit = dinov3_vits16(pretrained=False)
    vit.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    vit.eval().to(device)
    for parameter in vit.parameters():
        parameter.requires_grad = False
    return _Wrapper(vit)


def load_dinov3_sat_local(device: str = "cpu"):
    """Load the official SAT-493M ViT-L checkpoint from Meta's repo."""
    repo = os.environ.get("DINOV3_REPO")
    weights = os.environ.get("DINOV3_SAT_WEIGHTS")
    if not repo or not weights:
        return None
    repo, weights = Path(repo).expanduser(), Path(weights).expanduser()
    if not (repo / "dinov3").exists() or not weights.exists():
        raise FileNotFoundError(f"DINOv3 SAT paths not found: {repo}, {weights}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from dinov3.hub.backbones import dinov3_vitl16

    vit = dinov3_vitl16(weights=str(weights), check_hash=True)
    vit.eval().to(device)
    for parameter in vit.parameters():
        parameter.requires_grad = False
    return _Wrapper(vit)


def load_backbone(name: str, device: str = "cpu"):
    """Return `(model, model_label)` for a frozen image encoder."""
    if name == "dinov3-small" and _local_paths() is not None:
        return load_dinov3_small_local(device), "dinov3_vits16_pretrain_lvd1689m (local Meta checkpoint)"
    if name == "dinov3-sat-large":
        local = load_dinov3_sat_local(device)
        if local is not None:
            return local, "dinov3_vitl16_pretrain_sat493m (local Meta checkpoint)"

    from src.encoders import load_dinov2

    if name not in HF_MODELS:
        raise ValueError(f"Unknown backbone {name!r}; choose {sorted(HF_MODELS)}")
    label = HF_MODELS[name]
    return load_dinov2(label, device=device), label
