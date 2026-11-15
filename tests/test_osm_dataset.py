import csv
import json

from experiments.osm_sentinel.build_dataset import append_chip, load_existing


def _row(tile_id: str):
    return {
        "tile_id": tile_id,
        "image": f"images/{tile_id}.jpg",
        "region_id": "test_region",
        "country": "Test",
        "biome": "test",
        "split": "train",
        "lat": "0",
        "lon": "0",
        "scene_id": "scene",
        "datetime": "2025-01-01",
        "cloud_cover": "0",
        "label": "built",
        "caption": "Buildings with major road.",
        "n_features": "2",
        "building_count": "5",
        "feature_counts": json.dumps({"buildings": 5, "major road": 1}),
    }


def test_append_and_resume_repairs_captions(tmp_path):
    manifest = tmp_path / "manifest.csv"
    captions = tmp_path / "captions.jsonl"
    append_chip(manifest, captions, _row("t1"))
    append_chip(manifest, captions, _row("t2"))

    assert len(captions.read_text().splitlines()) == 2
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["tile_id"] for row in rows] == ["t1", "t2"]

    # Simulate a crash after the manifest append but before the caption append.
    captions.write_text("")
    done, stats = load_existing(manifest, captions)
    assert done == {"t1", "t2"}
    assert stats["n"] == 2
    assert len(captions.read_text().splitlines()) == 2

    append_chip(manifest, captions, _row("t3"))
    done, stats = load_existing(manifest, captions)
    assert done == {"t1", "t2", "t3"}
    assert stats["n"] == 3
    assert len(captions.read_text().splitlines()) == 3
