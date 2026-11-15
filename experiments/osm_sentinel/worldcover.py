"""Small ESA WorldCover 2021 helper for OSM-Sentinel.

WorldCover is a 10 m global land-cover map. We read a window around each region
from the public COGs, then use the dominant class as a fallback caption subject
where OSM has no polygon. It is not a replacement for OSM: OSM still supplies
buildings, roads, and detailed land use.

WorldCover 2021: © ESA, CC BY 4.0.
https://esa-worldcover.org/
"""

from __future__ import annotations

import math

WORLDCOVER_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
)

# code -> (caption phrase, coarse label)
CLASSES: dict[int, tuple[str, str]] = {
    10: ("forest", "forest"),
    20: ("scrubland", "forest"),
    30: ("grassland", "green"),
    40: ("farmland", "farmland"),
    50: ("built-up area", "built"),
    60: ("bare ground", "barren"),
    70: ("snow", "ice"),
    80: ("water", "water"),
    90: ("wetland", "water"),
    95: ("mangrove", "forest"),
    100: ("moss", "green"),
}


def tile_code(lat: float, lon: float) -> str:
    """WorldCover 3-degree tile code containing a point."""
    lat0 = int(math.floor(lat / 3.0) * 3)
    lon0 = int(math.floor(lon / 3.0) * 3)
    ns = "N" if lat0 >= 0 else "S"
    ew = "E" if lon0 >= 0 else "W"
    return f"{ns}{abs(lat0):02d}{ew}{abs(lon0):03d}"


def tile_codes(bbox: tuple[float, float, float, float]) -> list[str]:
    """All 3-degree WorldCover tiles overlapping (west, south, east, north)."""
    west, south, east, north = bbox
    lat0 = int(math.floor(south / 3.0) * 3)
    lat1 = int(math.floor(north / 3.0) * 3)
    lon0 = int(math.floor(west / 3.0) * 3)
    lon1 = int(math.floor(east / 3.0) * 3)
    codes = []
    for lat in range(lat0, lat1 + 1, 3):
        for lon in range(lon0, lon1 + 1, 3):
            codes.append(tile_code(float(lat), float(lon)))
    return sorted(set(codes))


class RegionWorldCover:
    """A region-sized WorldCover array, read once from the public COGs."""

    def __init__(self, bbox: tuple[float, float, float, float]):
        import rasterio
        from rasterio.merge import merge

        urls = [WORLDCOVER_URL.format(tile=code) for code in tile_codes(bbox)]
        datasets = []
        try:
            for url in urls:
                datasets.append(rasterio.open(url))
            data, transform = merge(datasets, bounds=bbox, res=1.0 / 12000.0, nodata=0)
        finally:
            for dataset in datasets:
                dataset.close()
        self.array = data[0]
        self.transform = transform

    def for_bbox(self, bbox: tuple[float, float, float, float]):
        """Return `(phrase, fraction, code)` for the dominant WorldCover class."""
        import numpy as np
        from rasterio.windows import from_bounds

        window = from_bounds(*bbox, self.transform).round_offsets().round_lengths()
        row0 = max(0, int(window.row_off))
        col0 = max(0, int(window.col_off))
        row1 = min(self.array.shape[0], row0 + max(1, int(window.height)))
        col1 = min(self.array.shape[1], col0 + max(1, int(window.width)))
        if row1 <= row0 or col1 <= col0:
            return None
        values, counts = np.unique(self.array[row0:row1, col0:col1], return_counts=True)
        valid = [(int(v), int(n)) for v, n in zip(values, counts) if int(v) in CLASSES]
        if not valid:
            return None
        total = sum(n for _, n in valid)
        code, count = max(valid, key=lambda item: item[1])
        phrase, label = CLASSES[code]
        return phrase, count / total, code, label
