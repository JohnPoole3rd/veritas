import pytest

gpd = pytest.importorskip("geopandas")
from shapely.geometry import box

from experiments.osm_sentinel.osm_source import RegionFeatures


def test_region_features_clips_by_chip_bbox():
    gdf = gpd.GeoDataFrame(
        {
            "landuse": ["forest", None],
            "building": [None, "yes"],
        },
        geometry=[box(0.0, 0.0, 0.02, 0.02), box(0.03, 0.03, 0.031, 0.031)],
        crs="EPSG:4326",
    )
    features = RegionFeatures(gdf)
    pairs, buildings = features.for_bbox((0.0, 0.0, 0.01, 0.01))
    assert len(pairs) == 1
    assert pairs[0][0]["landuse"] == "forest"
    assert buildings == []

    pairs, buildings = features.for_bbox((0.029, 0.029, 0.032, 0.032))
    assert len(pairs) == 0
    assert len(buildings) == 1
