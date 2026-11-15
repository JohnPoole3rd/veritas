import argparse
import csv
import json

from experiments.osm_sentinel.build_dataset import append_chip, check_build_meta, load_existing


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


def test_grid_candidates_are_unique_and_inside_radius():
    from experiments.osm_sentinel.build_dataset import grid_candidates

    cells = list(grid_candidates("test", 40.0, -83.0, 3.0, 1280.0))
    ids = [tid for tid, _, _ in cells]
    assert len(ids) == len(set(ids))
    assert len(cells) > 10
    for _, lat, lon in cells:
        dy = (lat - 40.0) * 111.32
        dx = (lon - -83.0) * 111.32 * 0.766
        assert (dx * dx + dy * dy) <= 3.05**2


def test_resume_drops_a_torn_final_manifest_row(tmp_path):
    manifest = tmp_path / "manifest.csv"
    captions = tmp_path / "captions.jsonl"
    append_chip(manifest, captions, _row("t1"))
    with manifest.open("a") as handle:
        handle.write("torn,row")

    done, stats = load_existing(manifest, captions)
    assert done == {"t1"}
    assert stats["n"] == 1
    assert not manifest.read_text().endswith("torn,row")


def test_build_meta_refuses_mixed_chip_configs(tmp_path):
    regions = tmp_path / "regions.csv"
    regions.write_text("region_id,lat,lon,radius_km,country,biome,split\n")
    base = argparse.Namespace(
        regions_csv=str(regions),
        chip_m=640.0,
        chip_px=64,
        sample_mode="grid",
        radius_km=None,
    )
    check_build_meta(tmp_path, base, overwrite=False)
    check_build_meta(tmp_path, base, overwrite=False)
    changed = argparse.Namespace(**vars(base))
    changed.chip_px = 128
    try:
        check_build_meta(tmp_path, changed, overwrite=False)
    except SystemExit as exc:
        assert "chip_px" in str(exc)
    else:
        raise AssertionError("mismatched chip size must be refused")


def test_read_chip_rejects_a_window_outside_the_region_array():
    import numpy as np
    from rasterio.transform import from_origin

    from experiments.osm_sentinel.build_dataset import read_chip

    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    transform = from_origin(0.0, 0.01, 0.0001, 0.0001)
    assert read_chip(rgb, transform, "EPSG:4326", 0.02, 0.02, 640.0, 64) is None


def test_read_region_keeps_georeference_when_bbox_exceeds_the_scene(tmp_path):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    from experiments.osm_sentinel.build_dataset import read_region

    path = tmp_path / "scene.tif"
    with rasterio.open(
        path, "w", driver="GTiff", width=100, height=100, count=3,
        dtype="uint8", crs="EPSG:4326", transform=from_origin(-1.0, 1.0, 0.01, 0.01),
    ) as dst:
        dst.write(np.full((3, 100, 100), 30, dtype=np.uint8))
    with rasterio.open(path) as scene:
        rgb, transform, _ = read_region(scene, (-1.2, -0.2, 0.1, 1.2))
    assert rgb.shape == (140, 130, 3)
    assert abs(transform.c - -1.2) < 1e-9
    assert abs(transform.f - 1.2) < 1e-9
