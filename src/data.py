"""EuroSAT loading, split creation, and datasets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.model_selection import train_test_split

from src.config import load_json, resolve_path, save_json

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_eurosat(root: str | os.PathLike[str], download: bool = True):
    """Load torchvision's EuroSAT dataset."""
    from torchvision.datasets import EuroSAT

    return EuroSAT(root=str(root), download=download)


def _samples(dataset) -> list[tuple[str, int]]:
    """Return (path, label_index) pairs for torchvision EuroSAT versions."""
    if hasattr(dataset, "samples"):
        return list(dataset.samples)
    if hasattr(dataset, "data") and hasattr(dataset, "targets"):
        return [(str(path), int(label)) for path, label in zip(dataset.data, dataset.targets)]
    raise AttributeError("Unsupported EuroSAT dataset layout")


def _relative_path(path: str | os.PathLike[str], root: str | os.PathLike[str]) -> str:
    try:
        return os.path.relpath(str(path), str(root))
    except ValueError:
        return str(path)


def make_splits(config: dict[str, Any], download: bool | None = None) -> dict[str, Any]:
    """Create stratified train/val/test splits and save them as JSON."""
    root = resolve_path(config["data"]["eurosat_root"], config)
    if download is None:
        download = bool(config["data"].get("download", True))
    dataset = load_eurosat(root, download=download)
    samples = _samples(dataset)
    labels = np.asarray([label for _, label in samples], dtype=np.int64)
    indices = np.arange(len(samples))
    seed = int(config.get("seed", 42))

    train_share = float(config["split"]["train"])
    val_share = float(config["split"]["val"])
    test_share = float(config["split"]["test"])
    if not np.isclose(train_share + val_share + test_share, 1.0):
        raise ValueError("Split shares must sum to 1.0")

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(val_share + test_share),
        stratify=labels,
        random_state=seed,
    )
    # train_test_split returns (keep, held_out); held_out is the *test* part of temp.
    test_fraction = test_share / (val_share + test_share)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_fraction,
        stratify=labels[temp_idx],
        random_state=seed,
    )

    class_names = list(dataset.classes)
    split_map = {"train": train_idx, "val": val_idx, "test": test_idx}
    payloads: dict[str, Any] = {}
    for split_name, split_indices in split_map.items():
        items = []
        for idx in sorted(int(i) for i in split_indices):
            path, label_idx = samples[idx]
            items.append(
                {
                    "tile_id": f"eurosat_{idx:06d}",
                    "path": _relative_path(path, root),
                    "label": class_names[int(label_idx)],
                }
            )
        # Shuffle so small debug subsets are not class-ordered.
        np.random.default_rng(seed).shuffle(items)
        payload = {
            "seed": seed,
            "class_names": class_names,
            "items": items,
        }
        split_path = resolve_path(config["data"]["splits_dir"], config) / f"{split_name}.json"
        save_json(split_path, payload)
        payloads[split_name] = payload

    return payloads


def load_split(path: str | os.PathLike[str]) -> dict[str, Any]:
    return load_json(path)


def split_path(config: dict[str, Any], split: str) -> Path:
    return resolve_path(config["data"]["splits_dir"], config) / f"{split}.json"


def get_transforms(input_size: int = 224, train: bool = False):
    """Eval transforms for the frozen DINOv2 encoder: Resize(256/224 * size) + CenterCrop(size) +
    ImageNet normalisation. EuroSAT and local tiles both go through this, so they stay comparable
    (the centre crop drops ~12.5 % of the border of a 64x64 tile; changing it would invalidate the
    logged embeddings). Nothing is trained on these, so there is no train variant; ResNet-18 uses
    `src.baselines.make_transforms`. `train` is kept for call-site compatibility and must be False."""
    from torchvision import transforms

    if train:
        raise ValueError("get_transforms is eval-only (frozen encoder); use src.baselines.make_transforms to train")
    return transforms.Compose(
        [
            transforms.Resize(int(input_size * 256 / 224)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class SplitDataset:
    """Dataset backed by a saved split JSON and a EuroSAT root."""

    def __init__(
        self,
        items: Iterable[dict[str, Any]],
        root: str | os.PathLike[str],
        class_names: list[str],
        transform=None,
    ) -> None:
        self.items = list(items)
        self.root = Path(root)
        self.class_names = list(class_names)
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        from PIL import Image

        item = self.items[index]
        path = self.root / item["path"]
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path} (check data.eurosat_root / split paths)")
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self.class_to_idx[item["label"]]
        return image, label


def dataset_from_split(
    config: dict[str, Any],
    split: str,
    transform=None,
) -> SplitDataset:
    payload = load_split(split_path(config, split))
    root = resolve_path(config["data"]["eurosat_root"], config)
    if transform is None:
        transform = get_transforms(int(config["image"]["input_size"]), train=False)
    return SplitDataset(
        payload["items"],
        root=root,
        class_names=payload["class_names"],
        transform=transform,
    )
