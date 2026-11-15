from experiments.osm_sentinel.osm_captions import (
    LABEL_PROMPTS,
    build_caption,
    build_caption_from_features,
    overpass_geometry,
    visible_phrases,
)


def _way(tags, coords):
    return {"type": "way", "tags": tags, "geometry": [{"lon": x, "lat": y} for x, y in coords]}


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
    caption, label, counts, buildings = build_caption(
        [{"tags": {"name": "A place", "opening_hours": "24/7"}}], [], (-1, -1, 1, 1)
    )
    assert caption is None
    assert label is None
    assert counts == {}
    assert buildings == 0


def test_caption_combines_land_use_roads_and_buildings():
    bbox = (0.0, 0.0, 0.02, 0.02)
    residential = _way(
        {"landuse": "residential"},
        [(0.001, 0.001), (0.019, 0.001), (0.019, 0.019), (0.001, 0.019), (0.001, 0.001)],
    )
    road = _way({"highway": "primary"}, [(0.0, 0.010), (0.02, 0.010)])
    buildings = [
        {"type": "way", "tags": {"building": "house"}, "center": {"lon": 0.005, "lat": 0.005}},
        {"type": "way", "tags": {"building": "yes"}, "center": {"lon": 0.006, "lat": 0.006}},
    ]
    caption, label, counts, building_count = build_caption([residential, road], buildings, bbox)
    assert label == "residential"
    assert building_count == 2
    assert caption.startswith("A 10 m Sentinel-2 satellite image of residential area")
    assert "major road" in caption
    assert "scattered buildings" in caption
    assert "residential area" in counts


def test_closed_way_geometry_is_a_polygon():
    geom = overpass_geometry(
        _way(
            {"landuse": "forest"},
            [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)],
        ),
        {"landuse": "forest"},
    )
    assert geom.geom_type == "Polygon"


def test_backend_neutral_caption_path():
    from shapely.geometry import Point, box

    features = [
        ({"landuse": "farmland"}, box(0.001, 0.001, 0.019, 0.019)),
        ({"highway": "track"}, box(0.0, 0.010, 0.020, 0.0101)),
    ]
    buildings = [Point(0.005, 0.005)]
    caption, label, counts, building_count = build_caption_from_features(
        features, buildings, (0.0, 0.0, 0.02, 0.02)
    )
    assert label == "farmland"
    assert building_count == 1
    assert "farmland" in caption
    assert "farm track" in caption
    assert "scattered buildings" in counts


def test_every_mapped_label_has_a_prompt():
    from experiments.osm_sentinel.osm_captions import LABEL_BY_PHRASE

    assert set(LABEL_BY_PHRASE.values()) <= set(LABEL_PROMPTS)
