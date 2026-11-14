import unittest

import numpy as np

from src.alignment import (
    center_embeddings,
    fit_procrustes,
    l2_normalize,
    orthogonality_error,
    transform_images,
    transform_texts,
)
from src.captions import build_captions, captions_by_label
from src.evaluate import evaluate_retrieval
from src.retrieval import retrieve_top_k


class AlignmentTests(unittest.TestCase):
    def test_procrustes_recovers_rotation(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 16)).astype(np.float32)
        true_w = np.linalg.qr(rng.normal(size=(16, 16)))[0].astype(np.float32)
        Y = X @ true_w
        params = fit_procrustes(X, Y)
        self.assertEqual(params["W"].shape, (16, 16))
        self.assertLess(orthogonality_error(params), 1e-4)
        rotated = transform_images(X, params)
        texts = transform_texts(Y, params)
        similarity = np.sum(rotated * texts, axis=1)
        self.assertGreater(float(similarity.mean()), 0.99)

    def test_centering_uses_training_means(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        Y = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        Xc, Yc, mu_x, mu_y = center_embeddings(X, Y)
        np.testing.assert_allclose(Xc.mean(axis=0), [0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(Yc.mean(axis=0), [0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(mu_x, [2.0, 3.0])
        np.testing.assert_allclose(mu_y, [1.0, 2.0])

    def test_rectangular_alignment(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(100, 12)).astype(np.float32)
        Y = rng.normal(size=(100, 8)).astype(np.float32)
        params = fit_procrustes(X, Y)
        self.assertEqual(params["W"].shape, (12, 8))
        self.assertLess(orthogonality_error(params), 1e-4)

    def test_l2_normalize(self):
        array = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
        normalized = l2_normalize(array)
        np.testing.assert_allclose(np.linalg.norm(normalized[0]), 1.0)
        self.assertTrue(np.all(np.isfinite(normalized)))


class RetrievalTests(unittest.TestCase):
    def test_top_k(self):
        query = np.array([1.0, 0.0], dtype=np.float32)
        gallery = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
        self.assertEqual(retrieve_top_k(query, gallery, k=2), [0, 1])

    def test_evaluate_retrieval(self):
        images = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        captions = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        metrics = evaluate_retrieval(images, captions, ["a", "b"], ["a", "b"])
        self.assertAlmostEqual(metrics["top1"], 1.0)
        self.assertAlmostEqual(metrics["top5"], 1.0)


class CaptionTests(unittest.TestCase):
    def test_every_class_has_captions(self):
        captions = build_captions()
        grouped = captions_by_label(captions)
        self.assertEqual(len(grouped), 10)
        for label, texts in grouped.items():
            self.assertGreaterEqual(len(texts), 5, label)


if __name__ == "__main__":
    unittest.main()
