import unittest

import numpy as np
import pandas as pd

from src.alignment import fit_procrustes
from src.local_tiles import LOCAL_CATEGORIES, evaluate_local, validate_manifest


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


if __name__ == "__main__":
    unittest.main()
