"""T08 ResNet-18 and T09 majority-class baselines on the EuroSAT splits.

CLI (run from the repo root):
    python -m src.baselines majority
    python -m src.baselines resnet18 [--epochs N] [--limit N] [--input-size 64]

Model selection uses val only; test is evaluated exactly once, with the best-val weights.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.evaluate import classification_metrics, evaluate_classifier, save_confusion_matrix_png

REPO_ROOT = Path(__file__).resolve().parents[1]

# Fallback when configs/default.yaml (owned by T01) does not exist yet. Mirrors INTERFACES §2.
DEFAULT_CONFIG = {
    "seed": 42,
    "data": {"eurosat_root": "data/raw/eurosat", "splits_dir": "data/splits"},
    "baseline": {"resnet18_epochs": 10, "resnet18_batch_size": 64, "resnet18_lr": 0.001},
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# torchvision.datasets.EuroSAT extracts to <root>/eurosat/2750/<Class>/<Class>_N.jpg
KNOWN_SUBDIRS = ("", "eurosat/2750", "2750", "EuroSAT", "EuroSAT_RGB")


# ---------------------------------------------------------------- config / paths
def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path | None = None) -> dict:
    """Read configs/default.yaml if present, merged over the INTERFACES §2 defaults."""
    path = Path(path) if path else REPO_ROOT / "configs" / "default.yaml"
    if not path.is_absolute():
        path = REPO_ROOT / path
    if path.exists():
        import yaml

        with open(path) as f:
            return _merge(DEFAULT_CONFIG, yaml.safe_load(f) or {})
    return copy.deepcopy(DEFAULT_CONFIG)


def repo_path(p: str | Path) -> Path:
    """Resolve a config path relative to the repo root (absolute paths pass through)."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def load_split(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path} (run T02 make_splits first)")
    with open(path) as f:
        split = json.load(f)
    for key in ("class_names", "items"):
        if key not in split:
            raise ValueError(f"{path} is missing key '{key}' (see INTERFACES §3)")
    return split


def limit_split(split: dict, limit: int | None, seed: int) -> dict:
    """Seeded random subsample of a split's items (order preserved). For smoke tests only."""
    if not limit or limit >= len(split["items"]):
        return split
    idx = sorted(random.Random(seed).sample(range(len(split["items"])), limit))
    return {**split, "items": [split["items"][i] for i in idx]}


def resolve_image_root(image_root: str | Path, rel_path: str) -> Path:
    """Find the directory under image_root that contains rel_path (tries KNOWN_SUBDIRS)."""
    image_root = Path(image_root)
    for sub in KNOWN_SUBDIRS:
        base = image_root / sub
        if (base / rel_path).exists():
            return base
    tried = ", ".join(str(image_root / s / rel_path) for s in KNOWN_SUBDIRS)
    raise FileNotFoundError(f"Image not found: {rel_path}. Tried: {tried}")


# ---------------------------------------------------------------- dataset
class SplitDataset:
    """torch Dataset over a split JSON: yields (transformed image, int label)."""

    def __init__(self, split: dict | str | Path, image_root: str | Path, transform=None):
        self.split = split if isinstance(split, dict) else load_split(split)
        self.class_names = list(self.split["class_names"])
        self.items = self.split["items"]
        self.transform = transform
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.targets = [self.class_to_idx[it["label"]] for it in self.items]
        self.root = resolve_image_root(image_root, self.items[0]["path"]) if self.items else Path(image_root)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image

        path = self.root / self.items[i]["path"]
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        img = Image.open(path).convert("RGB")
        return (self.transform(img) if self.transform else img), self.targets[i]


def make_transforms(input_size: int = 64, train: bool = False):
    from torchvision import transforms as T

    t = [] if input_size == 64 else [T.Resize((input_size, input_size), antialias=True)]
    if train:
        # Overhead imagery has no canonical orientation, so both flips are label-preserving.
        t += [T.RandomHorizontalFlip(), T.RandomVerticalFlip()]
    return T.Compose(t + [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


# ---------------------------------------------------------------- helpers
def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed: int):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _write_csv(path: Path, rows: list[dict], columns: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})


def _per_class_rows(metrics: dict) -> list[dict]:
    cm, names = np.asarray(metrics["confusion_matrix"]), metrics["class_names"]
    rows = []
    for i, c in enumerate(names):
        off = cm[i].copy()
        off[i] = 0
        rows.append({
            "class": c,
            "recall": round(metrics["per_class"][c], 4),
            "n": int(cm[i].sum()),
            "most_confused_with": names[int(off.argmax())] if off.sum() else "",
            "n_confused": int(off.max()),
        })
    return rows


# ---------------------------------------------------------------- T09 majority
def majority_baseline(train_split: dict, test_split: dict) -> dict:
    """Predict the most common train label for every test item.

    Ties are broken by class_names order, so the result is deterministic.
    """
    class_names = train_split["class_names"]
    counts = Counter(it["label"] for it in train_split["items"])
    most_common = max(class_names, key=lambda c: (counts[c], -class_names.index(c)))
    y_true = [it["label"] for it in test_split["items"]]
    m = classification_metrics(y_true, [most_common] * len(y_true), class_names)
    m.update(most_common_class=most_common, train_count=counts[most_common],
             train_n=len(train_split["items"]))
    return m


def run_majority(splits_dir: Path, out_dir: Path) -> dict:
    train = load_split(splits_dir / "train.json")
    test = load_split(splits_dir / "test.json")
    m = majority_baseline(train, test)
    row = {
        "method": "majority",
        "accuracy": round(m["accuracy"], 4),
        "macro_f1": round(m["macro_f1"], 4),
        "test_n": m["n"],
        "most_common_class": m["most_common_class"],
        "notes": f"train count {m['train_count']}/{m['train_n']}; seed {train.get('seed')}",
    }
    path = out_dir / "metrics" / "majority.csv"
    _write_csv(path, [row], ["method", "accuracy", "macro_f1", "test_n", "most_common_class", "notes"])
    print(f"majority: {row}\nwrote {_rel(path)}")
    return m


# ---------------------------------------------------------------- T08 ResNet-18
def build_resnet18(num_classes: int, pretrained: bool = True):
    import torch.nn as nn
    from torchvision.models import ResNet18_Weights, resnet18

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_resnet18(
    splits_dir: Path,
    image_root: Path,
    out_dir: Path,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
    input_size: int = 64,
    limit: int | None = None,
    pretrained: bool = True,
    num_workers: int = 0,
    device: str | None = None,
) -> dict:
    """Fine-tune ImageNet ResNet-18 on train, pick the best epoch on val, evaluate test once.

    input_size=64 keeps EuroSAT's native 64x64 tiles (fast on MPS; ResNet's stride-32 stem
    leaves a 2x2 final feature map, which still works well). Use 224 to match ImageNet scale.
    """
    import torch
    import torchvision
    from torch.utils.data import DataLoader

    seed_everything(seed)
    device = device or pick_device()
    splits = {s: limit_split(load_split(splits_dir / f"{s}.json"), limit, seed)
              for s in ("train", "val", "test")}
    class_names = splits["train"]["class_names"]
    for s in ("val", "test"):
        if splits[s]["class_names"] != class_names:
            raise ValueError(f"class_names differ between train and {s} splits")

    ds = {s: SplitDataset(splits[s], image_root, make_transforms(input_size, train=(s == "train")))
          for s in splits}
    g = torch.Generator().manual_seed(seed)
    loaders = {s: DataLoader(ds[s], batch_size=batch_size, shuffle=(s == "train"), generator=g,
                             num_workers=num_workers, persistent_workers=num_workers > 0)
               for s in ds}

    model = build_resnet18(len(class_names), pretrained).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    loss_fn = torch.nn.CrossEntropyLoss()

    history, best = [], {"val_accuracy": -1.0}
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)
        sched.step()
        val = evaluate_classifier(model, loaders["val"], device, class_names)
        rec = {"epoch": epoch, "train_loss": round(total / max(n, 1), 4),
               "val_accuracy": round(val["accuracy"], 4), "val_macro_f1": round(val["macro_f1"], 4),
               "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        print(rec, flush=True)
        if val["accuracy"] > best["val_accuracy"]:
            best = {"epoch": epoch, "val_accuracy": val["accuracy"],
                    "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

    # Test: once, with the best-val weights.
    model.load_state_dict(best["state_dict"])
    test = evaluate_classifier(model, loaders["test"], device, class_names)

    ckpt_path = out_dir / "checkpoints" / "resnet18.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best["state_dict"], "class_names": class_names, "arch": "resnet18",
                "input_size": input_size, "best_epoch": best["epoch"]}, ckpt_path)

    notes = (f"best_epoch={best['epoch']} val_acc={best['val_accuracy']:.4f} "
             f"input={input_size} device={device}" + (f" LIMIT={limit} (smoke test)" if limit else ""))
    metrics_dir = out_dir / "metrics"
    _write_csv(metrics_dir / "resnet18.csv",
               [{"method": "resnet18", "accuracy": round(test["accuracy"], 4),
                 "macro_f1": round(test["macro_f1"], 4), "test_n": test["n"],
                 "epochs": epochs, "notes": notes}],
               ["method", "accuracy", "macro_f1", "test_n", "epochs", "notes"])
    _write_csv(metrics_dir / "resnet18_per_class.csv", _per_class_rows(test),
               ["class", "recall", "n", "most_confused_with", "n_confused"])
    fig = save_confusion_matrix_png(test["confusion_matrix"], class_names,
                                    out_dir / "figures" / "resnet18_confusion_matrix.png",
                                    title=f"ResNet-18 test (acc {test['accuracy']:.3f})")
    config = {
        "model": "resnet18",
        "weights": "ResNet18_Weights.IMAGENET1K_V1" if pretrained else None,
        "epochs": epochs, "lr": lr, "batch_size": batch_size, "seed": seed,
        "optimizer": "Adam", "scheduler": "CosineAnnealingLR", "augmentation": "hflip+vflip (train only)",
        "normalization": "ImageNet mean/std", "input_size": input_size, "limit": limit,
        "split_sizes": {s: len(ds[s]) for s in ds}, "splits_dir": _rel(splits_dir),
        "split_seed": splits["train"].get("seed"), "class_names": class_names,
        "selection": "best val accuracy", "best_epoch": best["epoch"],
        "best_val_accuracy": round(best["val_accuracy"], 4),
        "test_accuracy": round(test["accuracy"], 4), "test_macro_f1": round(test["macro_f1"], 4),
        "history": history, "device": device,
        "torch": torch.__version__, "torchvision": torchvision.__version__,
    }
    with open(metrics_dir / "resnet18_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"test accuracy={test['accuracy']:.4f} macro_f1={test['macro_f1']:.4f} n={test['n']}")
    print("wrote", _rel(ckpt_path), _rel(metrics_dir / "resnet18.csv"), _rel(fig))
    return {**test, "best_epoch": best["epoch"], "history": history, "config": config}


@lru_cache(maxsize=2)
def _load_predictor(checkpoint_path: str):
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_resnet18(len(ckpt["class_names"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    device = pick_device()
    return model.to(device).eval(), ckpt["class_names"], ckpt["input_size"], device


def predict_resnet18(checkpoint_path: str | Path, images: list) -> list[tuple[str, float]]:
    """Top-1 (label, softmax prob) for each PIL image (e.g. local tiles in T12).

    Images are resized to the checkpoint's input size (64 = EuroSAT native, ~10 m/px).
    """
    import torch
    from torchvision import transforms as T

    path = repo_path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path} (run T08 first)")
    model, class_names, size, device = _load_predictor(str(path))
    tf = T.Compose([T.Resize((size, size), antialias=True), T.ToTensor(),
                    T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    x = torch.stack([tf(im.convert("RGB")) for im in images]).to(device)
    with torch.no_grad():
        p = torch.softmax(model(x).float(), dim=1).cpu()
    conf, idx = p.max(1)
    return [(class_names[i], float(c)) for i, c in zip(idx.tolist(), conf.tolist())]


# ---------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.baselines", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("method", choices=["majority", "resnet18"])
    ap.add_argument("--config", default=None, help="default: configs/default.yaml if present")
    ap.add_argument("--splits-dir", default=None, help="override data.splits_dir")
    ap.add_argument("--image-root", default=None, help="override data.eurosat_root")
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--input-size", type=int, default=64, help="64 = native EuroSAT; 224 = ImageNet scale")
    ap.add_argument("--limit", type=int, default=None, help="subsample each split (smoke tests only)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default=None, help="default: cuda > mps > cpu")
    ap.add_argument("--no-pretrained", action="store_true", help="random init (tests only)")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    splits_dir = repo_path(a.splits_dir or cfg["data"]["splits_dir"])
    out_dir = repo_path(a.out_dir)
    if a.method == "majority":
        return run_majority(splits_dir, out_dir)
    b = cfg["baseline"]
    return train_resnet18(
        splits_dir=splits_dir,
        image_root=repo_path(a.image_root or cfg["data"]["eurosat_root"]),
        out_dir=out_dir,
        epochs=a.epochs if a.epochs is not None else b["resnet18_epochs"],
        batch_size=a.batch_size or b["resnet18_batch_size"],
        lr=a.lr or b["resnet18_lr"],
        seed=cfg["seed"],
        input_size=a.input_size,
        limit=a.limit,
        pretrained=not a.no_pretrained,
        num_workers=a.workers,
        device=a.device,
    )



# ---------------------------------------------------------------- compatibility (J's T01 helpers)
def save_checkpoint(model, path) -> None:
    import torch

    path = repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, device: str | None = None):
    import torch

    model.load_state_dict(torch.load(repo_path(path), map_location=device or "cpu"))
    return model


if __name__ == "__main__":
    main()
