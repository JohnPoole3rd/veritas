#!/usr/bin/env python
"""Train the ResNet-18 baseline for A's T08."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torch.utils.data import DataLoader

from src.baselines import evaluate_classifier, save_checkpoint, train_resnet18
from src.config import load_config, resolve_path, save_json, set_seed
from src.data import dataset_from_split, get_transforms, load_split
from src.evaluate import save_metrics_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    input_size = int(config["image"]["input_size"])
    batch_size = args.batch_size or int(config["baseline"]["resnet18_batch_size"])
    epochs = args.epochs or int(config["baseline"]["resnet18_epochs"])

    root = resolve_path(config["data"]["eurosat_root"], config)
    class_names = load_split(resolve_path(config["data"]["splits_dir"], config) / "train.json")["class_names"]
    train_ds = dataset_from_split(config, "train", transform=get_transforms(input_size, train=True))
    val_ds = dataset_from_split(config, "val", transform=get_transforms(input_size, train=False))
    test_ds = dataset_from_split(config, "test", transform=get_transforms(input_size, train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model, history = train_resnet18(
        train_loader,
        val_loader,
        num_classes=len(class_names),
        epochs=epochs,
        learning_rate=float(config["baseline"]["resnet18_lr"]),
        device=args.device,
    )
    test_metrics = evaluate_classifier(model, test_loader, device=args.device)
    metrics_dir = resolve_path("artifacts/metrics", config)
    checkpoint_dir = resolve_path("artifacts/checkpoints", config)
    save_checkpoint(model, checkpoint_dir / "resnet18.pt")
    save_json(
        metrics_dir / "resnet18.json",
        {
            "history": history,
            "test": {k: v for k, v in test_metrics.items() if k not in {"predictions", "targets"}},
            "epochs": epochs,
            "batch_size": batch_size,
        },
    )
    save_metrics_csv(
        metrics_dir / "resnet18.csv",
        [
            {
                "method": "resnet18",
                "accuracy": test_metrics["accuracy"],
                "macro_f1": test_metrics["macro_f1"],
                "test_n": test_metrics["test_n"],
                "notes": f"epochs={epochs}; batch_size={batch_size}",
            }
        ],
    )
    print(json.dumps({k: v for k, v in test_metrics.items() if k not in {"predictions", "targets"}}, indent=2))


if __name__ == "__main__":
    main()
