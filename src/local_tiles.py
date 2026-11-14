"""Upper Arlington local tiles: manifest loading/validation (T11) and OOD evaluation (T12).

Owner: A. These tiles are an out-of-distribution TEST set. Never fit or tune on them.

CLI:
    python -m src.local_tiles validate [--manifest data/local_tiles/manifest.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = "data/local_tiles/manifest.csv"

REQUIRED_COLUMNS = ["tile_id", "path", "label", "local_category", "lat", "lon", "date", "source", "notes"]
EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]
LABELS = EUROSAT_CLASSES + ["Other"]
LOCAL_CATEGORIES = [
    "residential", "commercial_industrial", "road_highway", "park_forest_herbaceous", "water_river",
]
# Loose bounding box around Upper Arlington + the adjacent Scioto corridor.
LAT_RANGE, LON_RANGE = (39.97, 40.08), (-83.12, -83.02)


class ManifestError(ValueError):
    pass


def load_local_manifest(path: str | Path = DEFAULT_MANIFEST, check_files: bool = True) -> pd.DataFrame:
    """Load and validate the local tile manifest (INTERFACES §3). Raises ManifestError listing
    every problem found; rows with missing coordinates are rejected (INTERFACES §9)."""
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Local manifest not found: {path}")
    df = pd.read_csv(path, dtype={"tile_id": str, "date": str}, keep_default_na=False, na_values=[""])

    errors: list[str] = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ManifestError(f"{path}: missing columns {missing}")
    if df.tile_id.duplicated().any():
        errors.append(f"duplicate tile_id: {sorted(df.tile_id[df.tile_id.duplicated()])}")
    bad_id = df[~df.tile_id.str.fullmatch(r"ua_\d{3}")]
    if len(bad_id):
        errors.append(f"tile_id not ua_NNN: {list(bad_id.tile_id)}")
    for col, allowed in (("label", LABELS), ("local_category", LOCAL_CATEGORIES)):
        bad = df[~df[col].isin(allowed)]
        if len(bad):
            errors.append(f"bad {col}: {dict(zip(bad.tile_id, bad[col]))}")
    for col in ("lat", "lon", "date", "source"):
        bad = df[df[col].isna()]
        if len(bad):
            errors.append(f"missing {col}: {list(bad.tile_id)}")
    lat, lon = pd.to_numeric(df.lat, errors="coerce"), pd.to_numeric(df.lon, errors="coerce")
    out = df[~lat.between(*LAT_RANGE) | ~lon.between(*LON_RANGE)]
    if len(out):
        errors.append(f"coordinates outside Upper Arlington area: {list(out.tile_id)}")
    if (pd.to_datetime(df.date, errors="coerce").isna()).any():
        errors.append("unparseable date values")
    if check_files:
        absent = [p for p in df.path if not (ROOT / p).exists()]
        if absent:
            errors.append(f"missing image files: {absent}")
    if errors:
        raise ManifestError(f"{path}:\n  - " + "\n  - ".join(errors))
    df["lat"], df["lon"] = lat, lon
    return df


def summarize(df: pd.DataFrame) -> str:
    lines = [f"{len(df)} tiles", "", pd.crosstab(df.local_category, df.label, margins=True).to_string()]
    return "\n".join(lines)


# --------------------------------------------------------------------------- compatibility (J's T01 API)
def load_manifest(path) -> pd.DataFrame:
    """J's T01 name; same as load_local_manifest without the file-existence check."""
    return load_local_manifest(path, check_files=False)


def validate_manifest(frame: pd.DataFrame, expected_total: int = 50, expected_per_category: int = 10) -> dict:
    """J's T01 summary-style validator (counts per local category = sampling stratum)."""
    problems: list[str] = []
    if len(frame) != expected_total:
        problems.append(f"Expected {expected_total} tiles, found {len(frame)}")
    counts = frame["local_category"].value_counts().to_dict()
    for category in LOCAL_CATEGORIES:
        if counts.get(category, 0) != expected_per_category:
            problems.append(f"Category {category}: expected {expected_per_category}, found {counts.get(category, 0)}")
    unknown = sorted(set(frame["local_category"]) - set(LOCAL_CATEGORIES))
    if unknown:
        problems.append(f"Unknown categories: {unknown}")
    if frame["tile_id"].duplicated().any():
        problems.append("Duplicate tile_id values")
    for column in ("lat", "lon", "date", "source"):
        if frame[column].isna().any():
            problems.append(f"Missing values in {column}")
    return {"ok": not problems, "problems": problems, "counts": counts, "total": len(frame)}


class LocalTileDataset:
    """Torch-style dataset over the manifest: yields (image, tile_id)."""

    def __init__(self, manifest: pd.DataFrame, root=ROOT, transform=None):
        self.manifest = manifest.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int):
        from PIL import Image

        row = self.manifest.iloc[index]
        image = Image.open(self.root / row["path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, row["tile_id"]


def _main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--manifest", default=DEFAULT_MANIFEST)
    args = ap.parse_args(argv)
    if args.cmd == "validate":
        df = load_local_manifest(args.manifest)
        print("manifest OK")
        print(summarize(df))


if __name__ == "__main__":
    _main()
