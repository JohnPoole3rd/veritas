"""Turn visible OpenStreetMap tags into short Sentinel-2 captions.

Only physical, visible properties are used: land use/cover, water, vegetation,
buildings, roads, railways, parking, airfields, and similar infrastructure.
Names, brands, addresses, opening hours, contact details, religion, and other
non-visible metadata are deliberately excluded.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable

# Ordered tag maps. A tag value that is not listed contributes nothing.
TAG_MAPS: tuple[tuple[str, dict[str, str]], ...] = (
    ("landuse", {
        "residential": "residential area",
        "commercial": "commercial area",
        "retail": "commercial area",
        "industrial": "industrial area",
        "farmland": "farmland",
        "farmyard": "farm buildings",
        "orchard": "orchard",
        "vineyard": "vineyard",
        "forest": "forest",
        "meadow": "meadow",
        "grass": "grassland",
        "recreation_ground": "park",
        "cemetery": "cemetery",
        "quarry": "quarry",
        "landfill": "landfill",
        "railway": "railway",
        "construction": "construction site",
        "port": "port",
        "aquaculture": "aquaculture ponds",
        "greenhouse_horticulture": "greenhouses",
    }),
    ("natural", {
        "water": "water",
        "wood": "forest",
        "scrub": "scrubland",
        "heath": "heathland",
        "grassland": "grassland",
        "wetland": "wetland",
        "beach": "beach",
        "sand": "sand",
        "bare_rock": "bare rock",
        "scree": "scree",
        "glacier": "glacier",
        "coastline": "coastline",
        "tree_row": "tree row",
        "bay": "bay",
        "strait": "strait",
    }),
    ("leisure", {
        "park": "park",
        "golf_course": "golf course",
        "pitch": "sports field",
        "nature_reserve": "nature reserve",
        "marina": "marina",
        "garden": "garden",
        "playground": "playground",
    }),
    ("building", {
        "yes": "buildings",
        "house": "houses",
        "detached": "houses",
        "semidetached_house": "houses",
        "bungalow": "houses",
        "apartments": "apartment buildings",
        "residential": "apartment buildings",
        "retail": "shops",
        "commercial": "commercial buildings",
        "industrial": "industrial buildings",
        "warehouse": "warehouses",
        "school": "school",
        "university": "campus buildings",
        "college": "campus buildings",
        "hospital": "hospital",
        "hotel": "hotel",
        "church": "place of worship",
        "cathedral": "place of worship",
        "mosque": "place of worship",
        "synagogue": "place of worship",
        "temple": "place of worship",
        "farm": "farm buildings",
        "barn": "farm buildings",
        "garage": "garages",
        "garages": "garages",
        "hangar": "hangar",
        "terminal": "airport terminal",
    }),
    ("highway", {
        "motorway": "major road",
        "trunk": "major road",
        "primary": "major road",
        "secondary": "major road",
        "tertiary": "major road",
        "motorway_link": "major road",
        "trunk_link": "major road",
        "primary_link": "major road",
        "secondary_link": "major road",
        "residential": "residential street",
        "unclassified": "residential street",
        "service": "service road",
        "living_street": "residential street",
        "track": "farm track",
        "footway": "path",
        "path": "path",
        "cycleway": "cycle path",
    }),
    ("waterway", {
        "river": "river",
        "stream": "stream",
        "canal": "canal",
        "drain": "drainage channel",
        "riverbank": "river bank",
        "dock": "dock",
    }),
    ("railway", {
        "rail": "railway",
        "light_rail": "railway",
        "subway": "railway",
        "tram": "tram line",
        "station": "rail station",
        "halt": "rail station",
    }),
    ("man_made", {
        "pier": "pier",
        "bridge": "bridge",
        "breakwater": "breakwater",
        "wastewater_plant": "water treatment plant",
        "water_works": "waterworks",
        "lighthouse": "lighthouse",
        "tower": "tower",
        "silo": "silo",
        "storage_tank": "storage tanks",
    }),
    ("amenity", {
        "parking": "parking lot",
        "school": "school",
        "university": "campus",
        "college": "campus",
        "hospital": "hospital",
        "place_of_worship": "place of worship",
        "marketplace": "market",
        "bus_station": "bus station",
        "fuel": "fuel station",
    }),
    ("aeroway", {
        "runway": "airport runway",
        "taxiway": "airport taxiway",
        "apron": "airport apron",
        "terminal": "airport terminal",
    }),
)

# Primary-phrase priority. Used only to order extras after the subject is chosen.
PRIORITY: tuple[str, ...] = (
    "water", "bay", "strait", "coastline", "beach", "sand", "wetland", "river",
    "stream", "canal", "drainage channel", "dock", "marina", "glacier",
    "forest", "scrubland", "heathland", "tree row", "grassland", "meadow",
    "farmland", "orchard", "vineyard", "greenhouses", "aquaculture ponds",
    "quarry", "landfill", "bare rock", "scree",
    "park", "golf course", "sports field", "nature reserve", "garden",
    "residential area", "commercial area", "industrial area", "port",
    "construction site", "railway", "rail station", "tram line",
    "airport runway", "airport taxiway", "airport apron", "airport terminal",
    "major road", "residential street", "service road", "farm track",
    "parking lot", "bridge", "pier", "breakwater", "lighthouse", "tower",
    "storage tanks", "silo", "water treatment plant", "waterworks",
    "apartment buildings", "houses", "shops", "commercial buildings",
    "industrial buildings", "warehouses", "school", "campus", "hospital",
    "hotel", "place of worship", "garages", "hangar", "farm buildings",
    "buildings", "dense buildings",
)

# Polygonal land cover/use can be the subject only if it covers a meaningful
# fraction of the chip. A tiny pond or a single building must not outrank the
# surrounding neighbourhood.
AREA_SUBJECTS: tuple[str, ...] = (
    "water", "bay", "strait", "coastline", "beach", "sand", "wetland", "river",
    "glacier", "forest", "scrubland", "heathland", "grassland", "meadow",
    "farmland", "orchard", "vineyard", "greenhouses", "aquaculture ponds",
    "quarry", "landfill", "bare rock", "scree", "snow", "bare ground", "moss",
    "mangrove", "park", "golf course", "sports field", "nature reserve", "garden",
    "residential area", "commercial area", "industrial area", "port",
    "construction site", "built-up area", "airport runway", "airport apron",
    "airport terminal", "cemetery",
)

BUILT_PHRASES: tuple[str, ...] = (
    "apartment buildings", "houses", "shops", "commercial buildings",
    "industrial buildings", "warehouses", "school", "campus", "hospital",
    "hotel", "place of worship", "garages", "hangar", "farm buildings",
)

MAJOR_INFRA: tuple[str, ...] = (
    "major road", "railway", "rail station", "tram line", "parking lot",
    "bridge", "pier", "breakwater", "port", "airport runway",
    "airport taxiway", "airport apron", "airport terminal", "bus station",
    "lighthouse", "storage tanks", "silo", "water treatment plant",
    "waterworks", "farm track",
)

# Nominal on-the-ground widths for linear OSM features. A line's weight is
# converted to an area so a river crossing a chip cannot outweigh a polygon
# covering the whole chip.
METRES_PER_DEG = 111_320.0
LINE_WIDTH_M: dict[str, float] = {
    "river": 30.0,
    "canal": 15.0,
    "stream": 4.0,
    "drainage channel": 3.0,
    "major road": 15.0,
    "railway": 8.0,
    "residential street": 8.0,
    "service road": 6.0,
    "farm track": 4.0,
    "path": 2.0,
    "cycle path": 2.0,
    "tram line": 3.0,
    "bridge": 10.0,
    "pier": 5.0,
    "breakwater": 10.0,
    "coastline": 10.0,
}
DEFAULT_LINE_WIDTH_M = 5.0

LABEL_BY_PHRASE: dict[str, str] = {
    "water": "water", "bay": "water", "strait": "water", "coastline": "coast",
    "beach": "coast", "sand": "coast", "wetland": "water", "river": "water",
    "stream": "water", "canal": "water", "drainage channel": "water",
    "dock": "water", "marina": "water", "glacier": "ice",
    "forest": "forest", "scrubland": "forest", "heathland": "forest",
    "tree row": "forest", "grassland": "green", "meadow": "green",
    "farmland": "farmland", "orchard": "farmland", "vineyard": "farmland",
    "greenhouses": "farmland", "aquaculture ponds": "water",
    "quarry": "industrial", "landfill": "industrial", "bare rock": "barren",
    "scree": "barren", "snow": "ice", "bare ground": "barren", "moss": "green",
    "mangrove": "forest", "built-up area": "built", "park": "green",
    "golf course": "green", "sports field": "green", "nature reserve": "green",
    "garden": "green",
    "residential area": "residential", "commercial area": "commercial",
    "industrial area": "industrial", "port": "industrial",
    "construction site": "built", "railway": "transport",
    "rail station": "transport", "tram line": "transport",
    "airport runway": "airport", "airport taxiway": "airport",
    "airport apron": "airport", "airport terminal": "airport",
    "major road": "transport", "residential street": "transport",
    "service road": "transport", "farm track": "transport",
    "parking lot": "transport", "bridge": "transport", "pier": "transport",
    "breakwater": "transport", "lighthouse": "built", "tower": "built",
    "storage tanks": "industrial", "silo": "farmland",
    "water treatment plant": "industrial", "waterworks": "industrial",
    "apartment buildings": "residential", "houses": "residential",
    "shops": "commercial", "commercial buildings": "commercial",
    "industrial buildings": "industrial", "warehouses": "industrial",
    "school": "built", "campus": "built", "hospital": "built",
    "hotel": "built", "place of worship": "built", "garages": "built",
    "hangar": "airport", "farm buildings": "farmland", "buildings": "built",
    "dense buildings": "built",
}

LABEL_PROMPTS: dict[str, str] = {
    "water": "water, a river, coast, or lake",
    "coast": "a coastline or beach",
    "forest": "forest or dense trees",
    "green": "grassland, a park, or open green space",
    "farmland": "farm fields or orchards",
    "residential": "a residential area with houses and streets",
    "commercial": "commercial buildings, shops, or offices",
    "industrial": "industrial buildings, a port, or a quarry",
    "transport": "a major road, railway, or parking area",
    "airport": "an airport runway or terminal",
    "built": "buildings and built-up land",
    "barren": "bare rock, sand, or barren land",
    "ice": "glacier, permanent ice, or snow",
    "other": "land with no dominant visible feature",
}


def visible_phrases(tags: dict[str, Any]) -> list[str]:
    """Return only physical phrases implied by visible OSM tags."""
    phrases: list[str] = []
    for key, mapping in TAG_MAPS:
        value = tags.get(key)
        if isinstance(value, str) and value in mapping:
            phrase = mapping[value]
            if phrase not in phrases:
                phrases.append(phrase)
    return phrases


def _weight(geom, chip, phrase: str) -> float:
    """Return the feature's share of the chip area (0..1).

    Polygons use area directly. Lines use length × nominal width, so a river
    line cannot outweigh a land-cover polygon. Invalid OSM geometry is repaired
    once with `make_valid`; if that fails, the feature is ignored.
    """
    try:
        inter = geom.intersection(chip)
    except Exception:
        try:
            from shapely import make_valid

            inter = make_valid(geom).intersection(chip)
        except Exception:
            return 0.0
    if inter.is_empty:
        return 0.0
    if inter.geom_type in ("Polygon", "MultiPolygon"):
        return float(inter.area / chip.area)
    if inter.geom_type in ("LineString", "MultiLineString"):
        lat = math.radians(chip.centroid.y)
        width_deg = LINE_WIDTH_M.get(phrase, DEFAULT_LINE_WIDTH_M) / (
            METRES_PER_DEG * max(math.cos(lat), 1e-6)
        )
        return float(inter.length * width_deg / chip.area)
    if inter.geom_type == "GeometryCollection":
        return sum(_weight(part, chip, phrase) for part in inter.geoms)
    return 0.0


def _finish_caption(
    counts: Counter[str], weights: Counter[str], building_count: int
) -> tuple[str | None, str | None, dict[str, int], int]:
    if not counts and building_count == 0:
        return None, None, {}, 0

    if building_count >= 25:
        density = "dense buildings"
    elif building_count >= 5:
        density = "buildings"
    elif building_count > 0:
        density = "scattered buildings"
    else:
        density = ""
    if density:
        counts[density] += building_count
        weights[density] += building_count

    present = set(counts)
    area_candidates = [
        p for p in present
        if p in AREA_SUBJECTS and weights[p] >= 0.01
    ]
    if area_candidates:
        primary = max(area_candidates, key=lambda p: (weights[p], counts[p]))
    elif "dense buildings" in present:
        primary = "dense buildings"
    else:
        primary = max(counts, key=lambda p: (weights[p], counts[p]))
    if primary == "buildings" and building_count >= 25:
        primary = "dense buildings"

    def rank(phrase: str):
        return (weights[phrase], counts[phrase])

    extras: list[str] = []
    area_extras = sorted(
        (p for p in present if p in AREA_SUBJECTS and p != primary),
        key=rank,
        reverse=True,
    )
    extras.extend(area_extras[:2])
    density = "dense buildings" if building_count >= 25 else "buildings" if building_count >= 5 else "scattered buildings"
    if building_count > 0 and density in present and primary != density:
        extras.append(density)
    else:
        built_extras = sorted(
            (p for p in present if p in BUILT_PHRASES and p != primary),
            key=rank,
            reverse=True,
        )
        if built_extras:
            extras.append(built_extras[0])
    infra_extras = sorted(
        (p for p in present if p in MAJOR_INFRA and p != primary),
        key=rank,
        reverse=True,
    )
    extras.extend(infra_extras[: max(0, 3 - len(extras))])
    extras = list(dict.fromkeys(extras))[:3]
    caption = primary[:1].upper() + primary[1:]
    if extras:
        caption += " with " + ", ".join(extras)
    caption += "."
    label = LABEL_BY_PHRASE.get(primary, "other")
    return caption, label, dict(counts), building_count


def build_caption_from_features(
    features: Iterable[tuple[dict[str, Any], Any]],
    building_geometries: Iterable[Any],
    bbox: tuple[float, float, float, float],
    worldcover: tuple[str, float, int, str] | None = None,
) -> tuple[str | None, str | None, dict[str, int], int]:
    """Build a caption from OSM features plus an optional WorldCover result."""
    from shapely.geometry import box

    chip = box(*bbox)
    counts: Counter[str] = Counter()
    weights: Counter[str] = Counter()
    for tags, geom in features:
        phrases = visible_phrases(tags)
        if not phrases or geom is None or geom.is_empty:
            continue
        for phrase in phrases:
            weight = _weight(geom, chip, phrase)
            if weight <= 0:
                continue
            counts[phrase] += 1
            weights[phrase] += weight

    if worldcover is not None:
        phrase, fraction, _code, _label = worldcover
        if phrase and fraction > 0:
            counts[phrase] += 1
            weights[phrase] += float(fraction)

    building_count = 0
    for geom in building_geometries:
        if geom is None or geom.is_empty:
            continue
        if chip.contains(geom.centroid):
            building_count += 1
    return _finish_caption(counts, weights, building_count)
