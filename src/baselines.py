"""Supervised baselines for comparison."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluate import classification_metrics


def build_resnet18(num_classes: int):
    import torch
    from torchvision import models

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def train_resnet18(
    train_loader,
    val_loader,
    num_classes: int,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    device: str | None = None,
) -> tuple[Any, list[dict[str, float]]]:
    """Train ResNet-18 and return the model plus per-epoch history."""
    import torch
    import torch.nn as nn

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * images.size(0)
            seen += images.size(0)
        val_metrics = evaluate_classifier(model, val_loader, device=device)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": running_loss / max(seen, 1),
                **val_metrics,
            }
        )
    return model, history


def evaluate_classifier(model, loader, device: str | None = None) -> dict[str, Any]:
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1).detach().cpu().tolist()
            y_pred.extend(int(value) for value in predictions)
            y_true.extend(int(value) for value in labels.tolist())
    metrics = classification_metrics(y_true, y_pred)
    metrics["test_n"] = len(y_true)
    metrics["predictions"] = y_pred
    metrics["targets"] = y_true
    return metrics


def save_checkpoint(model, path: str | os.PathLike[str]) -> None:
    import torch

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), target)


def load_checkpoint(model, path: str | os.PathLike[str], device: str | None = None):
    import torch

    state = torch.load(Path(path), map_location=device or "cpu")
    model.load_state_dict(state)
    return model
