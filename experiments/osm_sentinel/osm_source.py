"""Local OSM source for the OSM-Sentinel experiment.

The public Overpass API is not used. This module reads a cropped local OSM PBF
with `pyrosm`: one download per region, then in-memory spatial lookups for every
chip. `pyrosm` is a required dependency.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

CUSTOM_FILTER: dict[str, Any] = {
    "landuse": True,
    "natural": True,
    "leisure": True,
    "highway": True,
    "waterway": True,
    "railway": True,
    "man_made": True,
    "aeroway": True,
    "building": True,
    "amenity": [
        "parking", "school", "university", "college", "hospital",
        "place_of_worship", "marketplace", "bus_station", "fuel", "stadium",
    ],
}


FILTER_VERSION = 2


def cache_key(region_id: str, bbox: tuple[float, float, float, float]) -> str:
    """Stable cache key for a region bbox + current OSM filter schema."""
    payload = json.dumps({"bbox": bbox, "filter": CUSTOM_FILTER, "v": FILTER_VERSION}, sort_keys=True)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    return f"{region_id}_{digest}"


def provider_for_url(url: str) -> str:
    """Return pyrosm's provider name for a provider extract URL."""
    host = (urlparse(url).hostname or "").lower()
    if "geofabrik" in host:
        return "geofabrik"
    if "bbbike" in host:
        return "bbbike"
    if "movisda" in host:
        return "movisda"
    raise ValueError(f"Unrecognized OSM provider URL: {url}")


def provider_cache_name(url: str) -> str:
    """Filename pyrosm uses for this provider extract inside its download directory."""
    from pyrosm.data.geofabrik_index import extract_file_name

    return f"{provider_for_url(url)}_{extract_file_name(url)}"


def provider_cache_path(cache_dir: Path, url: str) -> Path:
    """Where pyrosm expects the complete provider extract."""
    return cache_dir / "pbf" / provider_cache_name(url)


def resolve_provider_urls(
    bbox: tuple[float, float, float, float], cache_dir: Path | None = None
) -> list[str]:
    """Resolve the provider URL(s) pyrosm would use for a bbox, without downloading."""
    from pyrosm import get_data_by_bbox

    directory = str((cache_dir or Path(".")) / "pbf")
    result = get_data_by_bbox(
        list(bbox),
        download=False,
        directory=directory,
        strategy="smallest_total",
    )
    if isinstance(result, str):
        return [result]
    return [str(url) for url in result]


def require_provider_cache(
    bbox: tuple[float, float, float, float], cache_dir: Path
) -> list[Path]:
    """Fail unless every source PBF pyrosm needs is already in the local cache."""
    urls = resolve_provider_urls(bbox, cache_dir)
    paths = [provider_cache_path(cache_dir, url) for url in urls]
    missing = [path for path in paths if not path.exists() or path.stat().st_size == 0]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Required OSM provider PBF is not in the prefetched cache: "
            f"{names}. Run prefetch_osm.py without --require-osm-cache."
        )
    return paths


def ensure_region_pbf(
    region_id: str,
    bbox: tuple[float, float, float, float],
    cache_dir: Path,
    *,
    require_cache: bool = False,
) -> Path:
    """Download the smallest covering OSM extract once and cache it."""
    from pyrosm import get_data_by_bbox

    target = cache_dir / "pbf" / f"{region_id}.osm.pbf"
    if target.exists():
        return target
    if require_cache:
        require_provider_cache(bbox, cache_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    downloaded = get_data_by_bbox(
        list(bbox),
        directory=str(target.parent),
        crop=True,
        strategy="smallest_total",
    )
    downloaded = Path(downloaded)
    if downloaded != target:
        shutil.copyfile(downloaded, target)
    return target


def load_region_gdf(pbf: Path, bbox: tuple[float, float, float, float]):
    """Read only the requested OSM keys inside the region bbox."""
    from pyrosm import OSM

    osm = OSM(
        str(pbf),
        bounding_box=list(bbox),
        complete_relations=True,
        keep_metadata=False,
    )
    gdf = osm.get_data_by_custom_criteria(
        custom_filter=CUSTOM_FILTER,
        tags_as_columns=list(CUSTOM_FILTER),
        keep_nodes=False,
        keep_ways=True,
        keep_relations=True,
        keep_other_tags=True,
    )
    if gdf is None:
        import geopandas as gpd
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return gdf


def _tags_of(row) -> dict[str, Any]:
    raw = row.get("tags")
    return _tags_from(raw, [row.get(key) for key in CUSTOM_FILTER])


def _tags_from(raw, values) -> dict[str, Any]:
    """Shared by `_tags_of` (one row) and the column-wise path in RegionFeatures."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = {}
    tags = dict(raw) if isinstance(raw, dict) else {}
    for key, value in zip(CUSTOM_FILTER, values):
        if isinstance(value, str) and value:
            tags[key] = value
    return tags


class RegionFeatures:
    """A region GeoDataFrame with a spatial index for fast chip queries."""

    def __init__(self, gdf):
        import geopandas as gpd

        if not isinstance(gdf, gpd.GeoDataFrame):
            raise TypeError("RegionFeatures expects a GeoDataFrame")
        if gdf.crs is not None and str(gdf.crs).upper() not in {"EPSG:4326", "OGC:CRS84"}:
            gdf = gdf.to_crs("EPSG:4326")
        self.gdf = gdf[gdf.geometry.notna()].reset_index(drop=True).copy()
        self.sindex = self.gdf.sindex
        if "building" in self.gdf.columns:
            values = self.gdf["building"].astype("string").fillna("").str.lower()
            self._is_building = (~values.isin(["", "no", "entrance", "window"])).to_numpy()
            # Generic building polygons are summarized by building_count. Keep
            # specific building types as features so hospital/stadium/school
            # tags are not lost, without emitting "buildings" twice per chip.
            detail_values = {
                "school", "university", "college", "hospital", "hotel",
                "church", "cathedral", "mosque", "synagogue", "temple",
                "stadium", "sports_hall", "terminal", "hangar",
            }
            self._building_detail = values.isin(detail_values).to_numpy()
        else:
            self._is_building = self.gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).to_numpy()
            self._building_detail = self._is_building.copy()
        self._geoms = list(self.gdf.geometry)
        # Column-wise equivalent of `[_tags_of(row) for _, row in gdf.iterrows()]`:
        # iterrows built a Series per row and dominated region load (~25 s for 150k rows).
        n = len(self.gdf)

        def column(name):
            return self.gdf[name].tolist() if name in self.gdf.columns else [None] * n

        raw_tags = column("tags")
        key_columns = [column(key) for key in CUSTOM_FILTER]
        self._tags = [_tags_from(raw, values) for raw, values in zip(raw_tags, zip(*key_columns))]

    def for_bbox(self, bbox: tuple[float, float, float, float]):
        """Return `(feature_pairs, building_geometries)` intersecting a chip."""
        from shapely.geometry import box

        indices = self.sindex.query(box(*bbox), predicate="intersects")
        features = [
            (self._tags[i], self._geoms[i])
            for i in indices
            if not self._is_building[i] or self._building_detail[i]
        ]
        buildings = [self._geoms[i] for i in indices if self._is_building[i]]
        return features, buildings


def load_region_features(
    region_id: str,
    bbox: tuple[float, float, float, float],
    cache_dir: Path,
    *,
    require_cache: bool = False,
) -> RegionFeatures:
    """Load/cache a region's OSM features as GeoParquet."""
    import geopandas as gpd

    key = cache_key(region_id, bbox)
    parquet = cache_dir / "osm" / f"{key}.geoparquet"
    if parquet.exists():
        return RegionFeatures(gpd.read_parquet(parquet))
    pbf = ensure_region_pbf(key, bbox, cache_dir, require_cache=require_cache)
    gdf = load_region_gdf(pbf, bbox)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    tmp = parquet.with_suffix(parquet.suffix + ".tmp")
    try:
        gdf.to_parquet(tmp)
        tmp.replace(parquet)
    except Exception as exc:
        print(f"[osm] GeoParquet cache write failed for {region_id}: {type(exc).__name__}: {exc}", flush=True)
        if tmp.exists():
            tmp.unlink()
    return RegionFeatures(gdf)
