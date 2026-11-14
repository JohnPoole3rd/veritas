"""Caption templates for EuroSAT classes."""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from src.config import ensure_parent

CAPTION_TEMPLATES: dict[str, list[str]] = {
    "AnnualCrop": [
        "a satellite image of annual cropland",
        "farmland with seasonal crops",
        "agricultural fields with bare soil between rows",
        "a patchwork of crop fields",
        "cropland seen from above",
    ],
    "Forest": [
        "a satellite image of dense forest",
        "tree canopy seen from above",
        "a wooded area with closed canopy",
        "forest land",
        "dark green tree cover",
    ],
    "HerbaceousVegetation": [
        "a satellite image of grassland",
        "low vegetation and shrubs",
        "herbaceous vegetation",
        "open grassy land",
        "sparse natural vegetation",
    ],
    "Highway": [
        "a satellite image of a highway",
        "a major road with vehicles",
        "a multi-lane highway",
        "transport corridor",
        "road network",
    ],
    "Industrial": [
        "a satellite image of an industrial site",
        "factories and warehouses",
        "large commercial buildings",
        "industrial buildings and pavement",
        "manufacturing area",
    ],
    "Pasture": [
        "a satellite image of pasture",
        "grazing land",
        "open grass for livestock",
        "meadow",
        "managed grassland",
    ],
    "PermanentCrop": [
        "a satellite image of permanent crops",
        "orchards and vineyards",
        "tree crops in rows",
        "long-lived agricultural plants",
        "fruit tree plantation",
    ],
    "Residential": [
        "a satellite image of a residential area",
        "houses and streets",
        "a neighborhood",
        "suburban housing",
        "dense residential buildings",
    ],
    "River": [
        "a satellite image of a river",
        "a winding waterway",
        "a river through land",
        "flowing water",
        "river channel",
    ],
    "SeaLake": [
        "a satellite image of a lake",
        "open water",
        "a sea or lake",
        "large body of water",
        "blue water surface",
    ],
}


def build_captions(class_names: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """Build caption records for every requested class."""
    names = list(class_names) if class_names is not None else list(CAPTION_TEMPLATES)
    records: list[dict[str, Any]] = []
    for label in names:
        if label not in CAPTION_TEMPLATES:
            raise KeyError(f"No caption templates for class: {label}")
        for template_id, text in enumerate(CAPTION_TEMPLATES[label], start=1):
            records.append(
                {
                    "caption_id": f"{label.lower()}_{template_id:02d}",
                    "label": label,
                    "text": text,
                    "template_id": template_id,
                }
            )
    return records


def save_captions(path: str | os.PathLike[str], records: list[dict[str, Any]]) -> None:
    target = ensure_parent(path)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def load_captions(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def captions_by_label(records: Iterable[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        grouped[record["label"]].append(record["text"])
    return dict(grouped)


def sample_caption(grouped: dict[str, list[str]], label: str, rng: random.Random) -> str:
    if label not in grouped or not grouped[label]:
        raise KeyError(f"No captions for label: {label}")
    return rng.choice(grouped[label])
