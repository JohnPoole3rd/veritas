import numpy as np

from experiments.osm_sentinel.train import pair_recall


def test_pair_recall_matches_full_matrix():
    rng = np.random.default_rng(0)
    images = rng.normal(size=(37, 8)).astype(np.float32)
    text = rng.normal(size=(37, 8)).astype(np.float32)
    full = text @ images.T
    rank = (full > full[np.arange(37), np.arange(37)][:, None]).sum(1)
    expected = {
        "recall@1": round(float((rank < 1).mean()), 4),
        "recall@10": round(float((rank < 10).mean()), 4),
        "n": 37,
    }
    assert pair_recall(images, text, chunk_size=5) == expected
