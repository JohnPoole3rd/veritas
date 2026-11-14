"""Smoke tests for app.py that need no model download and no gitignored artifacts."""
import unittest
from pathlib import Path

import app


class AppTests(unittest.TestCase):
    def test_mock_ui_builds(self):
        backend = app.load_backend(force_mock=True)
        self.assertEqual(backend.name, "MOCK")
        self.assertIsNotNone(app.build_ui(backend))
        hits = backend.search_text("dense forest")
        self.assertEqual(len(hits), app.TOP_K)

    def test_results_tables_read_committed_metrics(self):
        titles = [t["title"] for t in app.results_tables()]
        self.assertEqual(len(titles), 3, titles)
        local = app.results_tables()[2]
        methods = [r[0] for r in local["rows"]]
        self.assertTrue(any("oracle" in m for m in methods), methods)
        self.assertTrue(any("EuroSAT majority" in m for m in methods), methods)

    def test_examples_use_logged_predictions(self):
        ex = app.local_examples()
        self.assertEqual([e["tile_id"] for e in ex], ["ua_011", "ua_049"])
        for e in ex:
            self.assertIn("predicted_label", e["pred"])
            self.assertIn("top3", e["pred"])

    def test_missing_tile_is_neutral_not_mock(self):
        img = app.tile_image(Path("does/not/exist.png"))
        self.assertEqual(img.size, (192, 192))
        self.assertEqual(img.getpixel((2, 2)), (128, 128, 128))


if __name__ == "__main__":
    unittest.main()
