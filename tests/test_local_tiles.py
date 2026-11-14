"""T11/T12 tests: manifest validation and local evaluation with synthetic stand-in encoders.
Runs with plain python or pytest; no downloads, no J artifacts needed."""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.local_tiles import (EUROSAT_CLASSES, ManifestError, evaluate_local,  # noqa: E402
                             load_local_manifest, local_metrics)


def test_manifest_valid():
    df = load_local_manifest()
    assert len(df) == 50 and df.tile_id.is_unique
    assert df.local_category.value_counts().eq(10).all()


def test_manifest_rejects_bad_rows():
    m = pd.read_csv("data/local_tiles/manifest.csv")
    m.loc[0, "lat"] = None
    m.loc[1, "label"] = "Forrest"
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        m.to_csv(f.name, index=False)
    try:
        load_local_manifest(f.name)
        raise AssertionError("expected ManifestError")
    except ManifestError as e:
        assert "missing lat" in str(e) and "Forrest" in str(e)
    finally:
        os.unlink(f.name)


def test_evaluate_local_synthetic():
    rng = np.random.default_rng(0)
    D = 16
    centers = {c: rng.normal(size=D) for c in EUROSAT_CLASSES}
    cap_labels = [c for c in EUROSAT_CLASSES for _ in range(3)]
    caps = np.stack([centers[c] + 0.05 * rng.normal(size=D) for c in cap_labels])
    df = load_local_manifest()
    # Fake encoder: first 40 tiles embed near their true class, last 10 near "Forest" (wrong for River).
    targets = [lab if i < 40 else "Forest" for i, lab in enumerate(df.label)]
    fake = lambda images: np.stack([centers[t] + 0.05 * rng.normal(size=D) for t in targets])  # noqa: E731
    params = {"W": np.eye(D, dtype=np.float32), "mu_x": np.zeros(D, np.float32), "mu_y": np.zeros(D, np.float32)}
    preds = evaluate_local(df, fake, None, params, caption_embeddings=caps, caption_labels=cap_labels)
    assert len(preds) == 50
    assert preds.procrustes_correct.sum() == 40, preds.procrustes_correct.sum()
    summary, per_cat = local_metrics(preds)
    row = summary.set_index("method").loc["procrustes"]
    assert row.accuracy == 0.8 and row.n == 50
    maj = summary.set_index("method").loc["majority_local"]
    assert maj.accuracy == 0.62, maj.accuracy  # 31/50 Residential
    assert per_cat.set_index("local_category").loc["water_river", "procrustes_acc"] == 0.0


if __name__ == "__main__":
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name); n += 1
    print(f"{n} passed")
