"""Bake the static data the scroll story needs (owner: D). Run once from the repo root:

    .venv/bin/python -m web.bake_story

Writes web/static/data/story.json and web/static/img/ua-*.{jpg,png}. Every number comes from
committed artifacts or the cached Upper Arlington Sentinel-2 / Landsat pass; nothing is typed in.

story.json "embedding" block: a 2-D picture of the real 384-D vectors. One linear projection (PCA of
the rotated image vectors + caption vectors) is applied to three sets:
  - images before the rotation (centred, unrotated DINOv2 vectors),
  - images after the rotation (src.alignment.transform_images),
  - the 50 caption vectors (src.alignment.transform_texts).
It is a projection for illustration; the accuracy numbers come from the full 384-D space.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.alignment import l2_normalize, load_procrustes, transform_images, transform_texts
from src.local_tiles import load_caption_artifacts
from web import geo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "static"
UA_BBOX = (-83.110, 39.985, -83.037, 40.070)
PER_CLASS = 36


def _pca2(fit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = fit.mean(axis=0)
    _, _, vt = np.linalg.svd(fit - mu, full_matrices=False)
    return mu, vt[:2].T


def embedding_block() -> dict:
    params = load_procrustes(ROOT / "artifacts" / "alignment" / "procrustes.npz")
    X = np.load(ROOT / "artifacts" / "embeddings" / "dinov2_test.npy")
    items = json.loads((ROOT / "data" / "splits" / "test.json").read_text())["items"][: len(X)]
    labels = np.array([it["label"] for it in items])
    classes = sorted(set(labels))
    rng = np.random.default_rng(7)
    pick = np.concatenate([rng.choice(np.flatnonzero(labels == c), PER_CLASS, replace=False) for c in classes])

    caps, cap_labels = load_caption_artifacts()
    after = transform_images(X[pick], params)
    before = l2_normalize(np.asarray(X[pick], np.float32) - params["mu_x"])
    texts = transform_texts(caps, params)

    mu, P = _pca2(np.concatenate([after, texts]))
    proj = lambda a: (a - mu) @ P  # noqa: E731
    a2, b2, t2 = proj(after), proj(before), proj(texts)
    s = float(np.abs(np.concatenate([a2, t2])).max()) or 1.0
    # the unrotated vectors are nearly orthogonal to this plane; scale them up so they are visible
    sb = float(np.abs(b2).max()) or 1.0

    def pts(a, scale):
        return [[round(float(x) / scale, 4), round(float(y) / scale, 4)] for x, y in a]

    cos_before = float(np.mean(np.sum(before * texts[[list(cap_labels).index(l) for l in labels[pick]]], axis=1)))
    cos_after = float(np.mean(np.sum(after * texts[[list(cap_labels).index(l) for l in labels[pick]]], axis=1)))
    return {
        "classes": classes,
        "image_class": [classes.index(l) for l in labels[pick]],
        "before": pts(b2, sb), "after": pts(a2, s),
        "text": pts(t2, s), "text_class": [classes.index(l) for l in cap_labels],
        "dims": int(params["W"].shape[0]),
        "n_images": int(len(pick)), "n_captions": int(len(texts)),
        "mean_cos_to_own_caption": {"before": round(cos_before, 3), "after": round(cos_after, 3)},
        "note": "2-D PCA projection of real 384-D embeddings (EuroSAT test tiles + the 50 class captions).",
    }


def ua_block() -> dict:
    region = geo.cached_region_for_bbox(UA_BBOX)
    if region is None:
        raise SystemExit("Upper Arlington region is not cached; run the site once online first.")
    img = OUT / "img"
    img.mkdir(parents=True, exist_ok=True)
    Image.fromarray(region.rgb).save(img / "ua-truecolor.jpg", quality=90)
    ind, lst = geo.fetch_indices(region), geo.fetch_lst(region)
    (img / "ua-ndvi.png").write_bytes(geo.colorize(ind["ndvi"], 0.0, 0.9, "ndvi"))
    t = lst["lst_c"]
    lo, hi = (round(float(v)) for v in np.nanpercentile(t, [2, 98]))
    (img / "ua-lst.png").write_bytes(geo.colorize(t, lo, hi, "heat"))
    clean = lambda a, d: [None if not np.isfinite(v) else round(float(v), d) for v in a]  # noqa: E731
    return {
        "bbox": list(UA_BBOX), "corners": region.corners, "rows": region.rows, "cols": region.cols,
        "scene": region.scene, "lst_scene": lst["scene"], "days_from_s2": lst["days_from_s2"],
        "tile_ndvi": clean(ind["tile_ndvi"], 3), "tile_water": clean(ind["tile_water_frac"], 3),
        "tile_lst": clean(lst["tile_lst_c"], 2),
        "lst_range": [lo, hi], "ndvi_range": [0.0, 0.9],
        "ndvi_palette": geo.palette_stops("ndvi"), "lst_palette": geo.palette_stops("heat"),
    }


def main() -> None:
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    story = {"embedding": embedding_block(), "ua": ua_block()}
    (OUT / "data" / "story.json").write_text(json.dumps(story, separators=(",", ":")))
    e = story["embedding"]
    print(f"[bake] {e['n_images']} images, {e['n_captions']} captions; mean cos to own caption "
          f"{e['mean_cos_to_own_caption']['before']} -> {e['mean_cos_to_own_caption']['after']}")
    print(f"[bake] Upper Arlington {story['ua']['rows']}x{story['ua']['cols']} tiles -> web/static/img/")


if __name__ == "__main__":
    main()
