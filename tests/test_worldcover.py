from experiments.osm_sentinel.worldcover import CLASSES, tile_code, tile_codes


def test_worldcover_tile_codes():
    assert tile_code(40.0, -83.0) == "N39W084"
    assert tile_code(-3.5, -62.2) == "S06W063"
    codes = tile_codes((-83.1, 39.9, -82.9, 40.1))
    assert codes == ["N39W084"]


def test_worldcover_classes_cover_visible_land_cover():
    assert CLASSES[10][0] == "forest"
    assert CLASSES[40][0] == "farmland"
    assert CLASSES[50][0] == "built-up area"
    assert CLASSES[80][0] == "water"


def test_worldcover_fraction_uses_the_full_chip_window():
    import numpy as np
    from rasterio.transform import from_origin

    from experiments.osm_sentinel.worldcover import RegionWorldCover

    wc = object.__new__(RegionWorldCover)
    wc.array = np.array([[0, 10], [10, 10]], dtype=np.uint8)
    wc.transform = from_origin(0.0, 2.0, 1.0, 1.0)
    phrase, fraction, code, label = wc.for_bbox((0.0, 0.0, 2.0, 2.0))
    assert (phrase, code, label) == ("forest", 10, "forest")
    assert fraction == 0.75


def test_worldcover_returns_none_when_most_of_the_window_is_nodata():
    import numpy as np
    from rasterio.transform import from_origin

    from experiments.osm_sentinel.worldcover import RegionWorldCover

    wc = object.__new__(RegionWorldCover)
    wc.array = np.array([[0, 0], [0, 10]], dtype=np.uint8)
    wc.transform = from_origin(0.0, 2.0, 1.0, 1.0)
    assert wc.for_bbox((0.0, 0.0, 2.0, 2.0)) is None
