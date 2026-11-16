import csv
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.osm_sentinel import benchmark


def _write_dataset(root: Path):
    fields = ["tile_id", "image", "region_id", "country", "label", "split", "scene_id"]
    rows = [
        # train scene shared with one test scene
        ("t1", "images/t1.jpg", "r_train", "United States", "residential", "train", "S1"),
        ("t2", "images/t2.jpg", "r_train", "United States", "forest", "train", "S1"),
        ("v1", "images/v1.jpg", "r_val", "France", "water", "val", "S2"),
        # test: two continents and three labels
        ("a1", "images/a1.jpg", "r_a", "United States", "residential", "test", "S1"),
        ("a2", "images/a2.jpg", "r_a", "United States", "residential", "test", "S1"),
        ("a3", "images/a3.jpg", "r_a", "United States", "residential", "test", "S3"),
        ("b1", "images/b1.jpg", "r_b", "Brazil", "forest", "test", "S4"),
        ("b2", "images/b2.jpg", "r_b", "Brazil", "forest", "test", "S4"),
        ("c1", "images/c1.jpg", "r_c", "France", "water", "test", "S5"),
        ("c2", "images/c2.jpg", "r_c", "France", "water", "test", "S5"),
    ]
    with (root / "manifest.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)


def test_b4_mapping_and_prompts_cover_all_labels():
    assert benchmark.b4_label("green", "osm") == "grass/pasture"
    assert benchmark.b4_label("Highway", "eurosat") == "other"
    assert benchmark.b4_label("HerbaceousVegetation", "eurosat") == "grass/pasture"
    assert set(benchmark.B4_FROM_OSM.values()) <= set(benchmark.B4_CLASSES)
    assert set(benchmark.B4_FROM_EUROSAT.values()) <= set(benchmark.B4_CLASSES)


def test_sample_is_test_only_deterministic_and_scene_disjoint_flag(tmp_path):
    _write_dataset(tmp_path)
    out = tmp_path / "b3.csv"
    info = benchmark.sample_b3(tmp_path, out, seed=7, cell_cap=2, region_cap=3)
    rows = list(csv.DictReader(out.open(newline="")))
    assert info["n"] == len(rows) == 6
    assert all(r["tile_id"][0] not in {"t", "v"} for r in rows)
    assert len({r["tile_id"] for r in rows}) == len(rows)
    assert {r["region_id"] for r in rows} == {"r_a", "r_b", "r_c"}
    assert all(sum(1 for r in rows if r["region_id"] == region) == 2
               for region in ("r_a", "r_b", "r_c"))
    assert (out.parent / "b4_mapping.json").exists()
    assert (out.with_suffix(".csv.sha256")).exists()
    assert {r["scene_shared_with_train"] for r in rows} == {"true", "false"}

    out2 = tmp_path / "b3-again.csv"
    benchmark.sample_b3(tmp_path, out2, seed=7, cell_cap=2, region_cap=3)
    assert out.read_text() == out2.read_text()


def test_metrics_and_wilson():
    y = np.array([0, 0, 1, 1])
    p = np.array([0, 0, 1, 0])
    m = benchmark._metrics(y, p, ["a", "b"])
    assert m["n"] == 4
    assert m["accuracy"] == pytest.approx(0.75)
    assert 0.0 <= benchmark._wilson(2, 4)[0] <= benchmark._wilson(2, 4)[1] <= 1.0
