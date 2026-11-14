import unittest

import numpy as np
import pandas as pd

from src.alignment import fit_procrustes
import os
import tempfile

from src.local_tiles import (DEFAULT_MANIFEST, EUROSAT_CLASSES, LOCAL_CATEGORIES, ROOT, ManifestError,
                             evaluate_local, load_local_manifest, local_metrics, validate_manifest)


def _manifest(n_per_category=10):
    rows = []
    categories = sorted(LOCAL_CATEGORIES)
    tile_id = 1
    for category in categories:
        for _ in range(n_per_category):
            rows.append(
                {
                    "tile_id": f"ua_{tile_id:03d}",
                    "path": f"data/raw/local_tiles/ua_{tile_id:03d}.png",
                    "label": "Residential",
                    "local_category": category,
                    "lat": 40.02,
                    "lon": -83.06,
                    "date": "2026-06-01",
                    "source": "Copernicus",
                    "notes": "test",
                }
            )
            tile_id += 1
    return pd.DataFrame(rows)


class LocalTileTests(unittest.TestCase):
    def test_validate_manifest_ok(self):
        manifest = _manifest()
        summary = validate_manifest(manifest)
        self.assertTrue(summary["ok"], summary["problems"])
        self.assertEqual(summary["total"], 50)

    def test_validate_manifest_flags_missing_category(self):
        manifest = _manifest().iloc[:-1].copy()
        summary = validate_manifest(manifest)
        self.assertFalse(summary["ok"])
        self.assertTrue(any("Expected 50" in problem for problem in summary["problems"]))

    def test_evaluate_local_runs(self):
        manifest = _manifest(n_per_category=1)
        rng = np.random.default_rng(0)
        X = rng.normal(size=(5, 16)).astype(np.float32)
        Y = rng.normal(size=(5, 16)).astype(np.float32)
        params = fit_procrustes(X, Y)
        captions = rng.normal(size=(10, 16)).astype(np.float32)
        labels = [
            "Residential",
            "Forest",
            "River",
            "Industrial",
            "Pasture",
            "AnnualCrop",
            "HerbaceousVegetation",
            "Highway",
            "PermanentCrop",
            "SeaLake",
        ]
        result = evaluate_local(manifest, X, params, captions, labels)
        self.assertEqual(len(result), 5)
        self.assertIn("predicted_label", result.columns)
        self.assertIn("correct", result.columns)

    # ---- A's tests on the real 50-tile manifest (T11/T12)
    def test_real_manifest_valid(self):
        df = load_local_manifest()
        self.assertEqual(len(df), 50)
        self.assertTrue(df.tile_id.is_unique)
        self.assertTrue(df.local_category.value_counts().eq(10).all())
        self.assertTrue(validate_manifest(df)["ok"])

    def test_manifest_rejects_bad_rows(self):
        m = pd.read_csv(ROOT / DEFAULT_MANIFEST)
        m.loc[0, "lat"] = None
        m.loc[1, "label"] = "Forrest"
        m.loc[2, "path"] = None
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            m.to_csv(f.name, index=False)
        try:
            with self.assertRaises(ManifestError) as ctx:
                load_local_manifest(f.name)
            self.assertIn("missing lat", str(ctx.exception))
            self.assertIn("Forrest", str(ctx.exception))
            self.assertIn("missing path", str(ctx.exception))
        finally:
            os.unlink(f.name)

    def test_evaluate_local_synthetic_metrics(self):
        rng = np.random.default_rng(0)
        D = 16
        centers = {c: rng.normal(size=D) for c in EUROSAT_CLASSES}
        cap_labels = [c for c in EUROSAT_CLASSES for _ in range(3)]
        caps = np.stack([centers[c] + 0.05 * rng.normal(size=D) for c in cap_labels])
        df = load_local_manifest()
        # First 40 tiles embed near their true class, last 10 (River) near Forest.
        targets = [lab if i < 40 else "Forest" for i, lab in enumerate(df.label)]
        X = np.stack([centers[t] + 0.05 * rng.normal(size=D) for t in targets]).astype(np.float32)
        params = {"W": np.eye(D, dtype=np.float32), "mu_x": np.zeros(D, np.float32),
                  "mu_y": np.zeros(D, np.float32)}
        preds = evaluate_local(df, X, params, caps, cap_labels)
        self.assertEqual(list(preds.columns[:6]),
                         ["tile_id", "true_label", "local_category", "predicted_label", "score", "correct"])
        self.assertEqual(int(preds.correct.sum()), 40)
        summary, per_cat = local_metrics(preds, train_majority_class="AnnualCrop")
        s = summary.set_index("method")
        self.assertAlmostEqual(s.loc["procrustes", "accuracy"], 0.8)
        self.assertAlmostEqual(s.loc["majority_train", "accuracy"], 0.0)  # no AnnualCrop tiles locally
        self.assertAlmostEqual(s.loc["oracle_majority_local", "accuracy"], 0.62)  # 31/50 Residential
        self.assertEqual(per_cat.set_index("local_category").loc["water_river", "procrustes_acc"], 0.0)

    def test_other_tiles_are_excluded_not_wrong(self):
        df = load_local_manifest().copy()
        df.loc[:1, "label"] = "Other"
        rng = np.random.default_rng(1)
        X = rng.normal(size=(len(df), 8)).astype(np.float32)
        caps = rng.normal(size=(10, 8)).astype(np.float32)
        params = {"W": np.eye(8, dtype=np.float32), "mu_x": np.zeros(8, np.float32), "mu_y": np.zeros(8, np.float32)}
        preds = evaluate_local(df, X, params, caps, EUROSAT_CLASSES)
        self.assertFalse(preds.loc[:1, "correct"].any())
        summary, _ = local_metrics(preds)
        row = summary.set_index("method").loc["procrustes"]
        self.assertEqual(row["n"], 48)
        self.assertEqual(row["n_other_excluded"], 2)


if __name__ == "__main__":
    unittest.main()
