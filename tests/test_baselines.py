"""Smoke tests for src.baselines / src.evaluate on a tiny synthetic EuroSAT-like folder.

Run either way from the repo root (no downloads):
    python tests/test_baselines.py
    python -m pytest tests/test_baselines.py
"""
import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines import main, majority_baseline, predict_resnet18  # noqa: E402
from src.evaluate import classification_metrics, evaluate_caption_classification  # noqa: E402

CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
           "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]


def make_fake_eurosat(tmp: Path, per_class: int = 6):
    """Images under <tmp>/raw/eurosat/2750/<Class>/ (torchvision layout) + split JSONs."""
    rng = np.random.default_rng(0)
    img_dir = tmp / "raw" / "eurosat" / "2750"
    splits = {"train": [], "val": [], "test": []}
    n = 0
    for ci, c in enumerate(CLASSES):
        (img_dir / c).mkdir(parents=True)
        for k in range(per_class):
            arr = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
            arr[..., ci % 3] = 25 * ci  # weak class signal
            rel = f"{c}/{c}_{k + 1}.jpg"
            Image.fromarray(arr).save(img_dir / rel)
            n += 1
            s = "train" if k < per_class - 2 else ("val" if k == per_class - 2 else "test")
            splits[s].append({"tile_id": f"eurosat_{n:06d}", "path": rel, "label": c})
    # Make AnnualCrop the unique majority in train.
    splits["train"].append(dict(splits["train"][0], tile_id="eurosat_999999"))
    sdir = tmp / "splits"
    sdir.mkdir()
    for s, items in splits.items():
        (sdir / f"{s}.json").write_text(json.dumps({"seed": 42, "class_names": CLASSES, "items": items}))
    # image root deliberately points at the parent: SplitDataset must find eurosat/2750.
    return sdir, tmp / "raw"


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def test_classification_metrics():
    m = classification_metrics(["a", "b", "b"], ["a", "b", "a"], ["a", "b"])
    assert abs(m["accuracy"] - 2 / 3) < 1e-9
    assert m["confusion_matrix"].tolist() == [[1, 0], [1, 1]]
    assert m["per_class"] == {"a": 1.0, "b": 0.5}


def test_evaluate_caption_classification():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(10, 16))
    cap = np.repeat(centers, 3, axis=0) + 0.05 * rng.normal(size=(30, 16))
    cap_labels = [c for c in CLASSES for _ in range(3)]
    labels = [CLASSES[i % 10] for i in range(50)]
    img = centers[[i % 10 for i in range(50)]] + 0.05 * rng.normal(size=(50, 16))
    for agg in ("max", "mean"):
        r = evaluate_caption_classification(img, cap, labels, cap_labels, class_names=CLASSES, agg=agg)
        assert r["top1"] == 1.0 and r["top5"] == 1.0, r["top1"]
        assert r["confusion_matrix"].shape == (10, 10)
        assert set(r["per_class"]) == set(CLASSES)


def test_majority_and_resnet18_cli():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        sdir, root = make_fake_eurosat(tmp)
        out = tmp / "artifacts"

        train = json.loads((sdir / "train.json").read_text())
        test = json.loads((sdir / "test.json").read_text())
        m = majority_baseline(train, test)
        assert m["most_common_class"] == "AnnualCrop"
        assert abs(m["accuracy"] - 0.1) < 1e-9

        main(["majority", "--splits-dir", str(sdir), "--out-dir", str(out)])
        rows = read_csv(out / "metrics" / "majority.csv")
        assert list(rows[0]) == ["method", "accuracy", "macro_f1", "test_n", "most_common_class", "notes"]
        assert rows[0]["test_n"] == "10" and rows[0]["most_common_class"] == "AnnualCrop"

        main(["resnet18", "--splits-dir", str(sdir), "--image-root", str(root), "--out-dir", str(out),
              "--epochs", "1", "--batch-size", "16", "--no-pretrained", "--device", "cpu"])
        rows = read_csv(out / "metrics" / "resnet18.csv")
        assert list(rows[0]) == ["method", "accuracy", "macro_f1", "test_n", "epochs", "notes"]
        assert rows[0]["test_n"] == "10" and rows[0]["epochs"] == "1"
        per_class = read_csv(out / "metrics" / "resnet18_per_class.csv")
        assert [r["class"] for r in per_class] == CLASSES
        cfg = json.loads((out / "metrics" / "resnet18_config.json").read_text())
        assert cfg["split_sizes"] == {"train": 41, "val": 10, "test": 10}
        assert (out / "figures" / "resnet18_confusion_matrix.png").stat().st_size > 0

        imgs = [Image.open(root / "eurosat" / "2750" / "Forest" / "Forest_1.jpg"),
                Image.new("RGB", (128, 128))]  # non-64px input is resized
        preds = predict_resnet18(out / "checkpoints" / "resnet18.pt", imgs)
        assert len(preds) == 2 and all(p[0] in CLASSES and 0 <= p[1] <= 1 for p in preds)


def test_missing_split_message():
    try:
        main(["majority", "--splits-dir", "/nonexistent_splits"])
    except FileNotFoundError as e:
        assert "/nonexistent_splits/train.json" in str(e)
    else:
        raise AssertionError("expected FileNotFoundError")


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} passed")
