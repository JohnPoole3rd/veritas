from shapely.geometry import LineString, Point, box

from experiments.osm_sentinel.osm_captions import (
    LABEL_BY_PHRASE,
    LABEL_PROMPTS,
    build_caption_from_features,
    visible_phrases,
)


def test_only_visible_tags_are_used():
    tags = {
        "landuse": "residential",
        "building": "house",
        "highway": "primary",
        "name": "Secret Estate",
        "brand": "ACME",
        "opening_hours": "24/7",
        "addr:street": "Hidden Lane",
        "phone": "555-0100",
    }
    phrases = visible_phrases(tags)
    assert "residential area" in phrases
    assert "houses" in phrases
    assert "major road" in phrases
    assert not any(x in " ".join(phrases) for x in ("Secret", "ACME", "Hidden", "555"))


def test_non_visible_tags_do_not_create_a_caption():
    caption, label, counts, buildings = build_caption_from_features(
        [({"name": "A place", "opening_hours": "24/7"}, box(0, 0, 1, 1))],
        [],
        (-1, -1, 1, 1),
    )
    assert caption is None
    assert label is None
    assert counts == {}
    assert buildings == 0


def test_caption_combines_land_use_roads_and_buildings():
    bbox = (0.0, 0.0, 0.02, 0.02)
    features = [
        ({"landuse": "residential"}, box(0.001, 0.001, 0.019, 0.019)),
        ({"highway": "primary"}, LineString([(0.0, 0.010), (0.02, 0.010)])),
    ]
    buildings = [Point(0.005, 0.005), Point(0.006, 0.006)]
    caption, label, counts, building_count = build_caption_from_features(features, buildings, bbox)
    assert label == "residential"
    assert building_count == 2
    assert caption.startswith("Residential area")
    assert "major road" in caption
    assert "scattered buildings" in caption
    assert "residential area" in counts


def test_backend_neutral_caption_path():
    features = [
        ({"landuse": "farmland"}, box(0.001, 0.001, 0.019, 0.019)),
        ({"highway": "track"}, LineString([(0.0, 0.010), (0.020, 0.010)])),
    ]
    buildings = [Point(0.005, 0.005)]
    caption, label, counts, building_count = build_caption_from_features(
        features, buildings, (0.0, 0.0, 0.02, 0.02)
    )
    assert label == "farmland"
    assert building_count == 1
    assert "farmland" in caption.lower()
    assert "farm track" in caption
    assert "scattered buildings" in counts


def test_tiny_water_does_not_outrank_the_surrounding_land_use():
    features = [
        ({"natural": "water"}, box(0.001, 0.001, 0.0012, 0.0012)),
        ({"landuse": "residential"}, box(0.002, 0.002, 0.019, 0.019)),
    ]
    caption, label, counts, _ = build_caption_from_features(
        features, [], (0.0, 0.0, 0.02, 0.02)
    )
    assert label == "residential"
    assert caption.startswith("Residential area")
    assert "water" in counts


def test_river_line_does_not_outrank_residential_polygon():
    bbox = (0.0, 0.0, 0.0115, 0.0115)
    features = [
        ({"landuse": "residential"}, box(0.0005, 0.0005, 0.0105, 0.0105)),
        ({"waterway": "river"}, LineString([(0.0, 0.005), (0.0115, 0.005)])),
    ]
    caption, label, counts, _ = build_caption_from_features(features, [], bbox)
    assert label == "residential"
    assert caption.startswith("Residential area")
    assert "river" in counts


def test_caption_prefers_major_infrastructure_over_minor_paths():
    features = [
        ({"landuse": "residential"}, box(0.001, 0.001, 0.019, 0.019)),
        ({"highway": "primary"}, LineString([(0.0, 0.010), (0.02, 0.010)])),
        ({"highway": "footway"}, LineString([(0.0, 0.005), (0.02, 0.005)])),
    ]
    caption, _, _, _ = build_caption_from_features(features, [], (0.0, 0.0, 0.02, 0.02))
    assert "major road" in caption
    assert "path" not in caption


def test_every_mapped_label_has_a_prompt():
    assert set(LABEL_BY_PHRASE.values()) <= set(LABEL_PROMPTS)
