"""T08 ResNet-18 and T09 majority-class baselines on the EuroSAT splits.

CLI (run from the repo root):
    python -m src.baselines majority
    python -m src.baselines resnet18 [--epochs N] [--limit N] [--input-size 64]

Model selection uses val only; test is evaluated exactly once, with the best-val weights.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.config import load_config, resolve_path, save_json, set_seed
from src.data import IMAGENET_MEAN, IMAGENET_STD, SplitDataset, load_split
from src.encoders import get_device
from src.evaluate import classification_metrics, evaluate_classifier, save_confusion_matrix_png

REPO_ROOT = Path(__file__).resolve().parents[1]


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _load_split(path: Path) -> dict:
    """J's split JSON (INTERFACES §3), with a clear error naming the path (INTERFACES §9)."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Split file not found: {path} (run scripts/prepare_data.py first)")
    split = load_split(path)
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


def make_transforms(input_size: int = 64, train: bool = False):
    from torchvision import transforms as T

    t = [] if input_size == 64 else [T.Resize((input_size, input_size), antialias=True)]
    if train:
        # Overhead imagery has no canonical orientation, so both flips are label-preserving.
        t += [T.RandomHorizontalFlip(), T.RandomVerticalFlip()]
    return T.Compose(t + [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


# ---------------------------------------------------------------- helpers
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
    """T09. The only majority implementation; writes majority.csv (INTERFACES §4) and majority.json."""
    train = _load_split(splits_dir / "train.json")
    test = _load_split(splits_dir / "test.json")
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
    save_json(out_dir / "metrics" / "majority.json",
              {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "majority_class": m["most_common_class"],
               "test_n": m["n"], "train_count": m["train_count"], "train_n": m["train_n"],
               "tie_break": "class_names order"})
    print(f"majority: {row}\nwrote {_rel(path)} and majority.json")
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

    if epochs < 1:
        raise ValueError("epochs must be >= 1 (model selection needs at least one val epoch)")
    set_seed(seed)
    device = device or get_device()
    splits = {s: limit_split(_load_split(splits_dir / f"{s}.json"), limit, seed)
              for s in ("train", "val", "test")}
    class_names = splits["train"]["class_names"]
    for s in ("val", "test"):
        if splits[s]["class_names"] != class_names:
            raise ValueError(f"class_names differ between train and {s} splits")

    ds = {s: SplitDataset(splits[s]["items"], image_root, class_names,
                          transform=make_transforms(input_size, train=(s == "train")))
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
def _load_predictor(checkpoint_path: str, mtime: float):  # mtime: retrained file -> new cache entry
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_resnet18(len(ckpt["class_names"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    device = get_device()
    return model.to(device).eval(), ckpt["class_names"], ckpt["input_size"], device


def predict_resnet18(checkpoint_path: str | Path, images: list) -> list[tuple[str, float]]:
    """Top-1 (label, softmax prob) for each PIL image (e.g. local tiles in T12).

    Images are resized to the checkpoint's input size (64 = EuroSAT native, ~10 m/px).
    """
    import torch
    from torchvision import transforms as T

    path = resolve_path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path} (run T08 first)")
    model, class_names, size, device = _load_predictor(str(path), path.stat().st_mtime)
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
    ap.add_argument("--config", default=None, help="default: configs/default.yaml")
    ap.add_argument("--splits-dir", default=None, help="override data.splits_dir")
    ap.add_argument("--image-root", default=None, help="override data.eurosat_root")
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--input-size", type=int, default=64, help="64 = native EuroSAT; 224 = ImageNet scale")
    ap.add_argument("--limit", type=int, default=None, help="subsample each split (smoke tests only)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default=None, help="default: cuda > mps > xpu > cpu")
    ap.add_argument("--no-pretrained", action="store_true", help="random init (tests only)")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    splits_dir = resolve_path(a.splits_dir or cfg["data"]["splits_dir"], cfg)
    out_dir = resolve_path(a.out_dir, cfg)
    if a.method == "majority":
        return run_majority(splits_dir, out_dir)
    b = cfg["baseline"]
    return train_resnet18(
        splits_dir=splits_dir,
        image_root=resolve_path(a.image_root or cfg["data"]["eurosat_root"], cfg),
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


if __name__ == "__main__":
    main()
