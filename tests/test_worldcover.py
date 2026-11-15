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
