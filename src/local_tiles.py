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
    for col in ("path", "lat", "lon", "date", "source"):
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
        absent = [p for p in df.path.dropna() if not (ROOT / p).exists()]
        if absent:
            errors.append(f"missing image files: {absent}")
    if errors:
        raise ManifestError(f"{path}:\n  - " + "\n  - ".join(errors))
    df["lat"], df["lon"] = lat, lon
    return df


# --------------------------------------------------------------------------- T12 evaluation
QA_PATH = "data/local_tiles/labels_qa.csv"
LOCAL_EMB = "artifacts/embeddings/dinov2_local.npy"
LOCAL_META = "artifacts/embeddings/dinov2_local_meta.json"


def load_tile_images(df: pd.DataFrame) -> list:
    from PIL import Image

    return [Image.open(ROOT / p).convert("RGB") for p in df.path]


def _embed_images(image_model, manifest: pd.DataFrame, input_size: int = 224, device=None):
    """Return (N, D) DINOv2 embeddings for the manifest tiles. image_model may be:
    - an (N, D) array of precomputed embeddings (INTERFACES §5 `image_embeddings`),
    - a callable list[PIL] -> (N, D) array (tests / custom encoders),
    - the frozen DINOv2 model from `src.encoders.load_dinov2`. This uses J's eval transforms
      (`get_transforms(224)`: Resize(256) + CenterCrop(224), so the outer ~12.5 % border is cropped),
      exactly as the EuroSAT tiles were embedded, so local and EuroSAT embeddings are comparable."""
    import numpy as np

    if isinstance(image_model, np.ndarray):
        return image_model
    if callable(image_model) and not hasattr(image_model, "parameters"):
        return image_model(load_tile_images(manifest))
    from src.data import get_transforms
    from src.encoders import extract_image_embeddings

    ds = LocalTileDataset(manifest, ROOT, transform=get_transforms(input_size, train=False))
    return extract_image_embeddings(image_model, ds, batch_size=32, device=device)


def _cache_key(manifest: pd.DataFrame, config: dict) -> dict:
    return {"model_name": config["image"]["model_name"], "input_size": int(config["image"]["input_size"]),
            "tile_ids": manifest.tile_id.tolist()}


def load_cached_local_embeddings(manifest: pd.DataFrame, config: dict):
    """The cached (N, D) local embeddings, or None if absent or stale. Valid only if
    dinov2_local_meta.json records the same tile_ids (in order), image model and input size."""
    import json

    import numpy as np

    from src.config import resolve_path

    emb, meta = resolve_path(LOCAL_EMB, config), resolve_path(LOCAL_META, config)
    if not (emb.exists() and meta.exists()):
        return None
    m = json.loads(meta.read_text())
    if any(m.get(k) != v for k, v in _cache_key(manifest, config).items()):
        return None
    X = np.load(emb)
    return X if len(X) == len(manifest) else None


def save_local_embeddings(X, manifest: pd.DataFrame, config: dict, manifest_path: str = DEFAULT_MANIFEST) -> None:
    """Write dinov2_local.npy + dinov2_local_meta.json (the one cache format; INTERFACES §4)."""
    import json

    import numpy as np

    from src.config import resolve_path

    emb = resolve_path(LOCAL_EMB, config)
    emb.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb, np.asarray(X, dtype=np.float32))
    meta = {**_cache_key(manifest, config), "embedding_dim": int(np.asarray(X).shape[1]),
            "n_tiles": int(len(X)), "manifest": str(manifest_path)}
    resolve_path(LOCAL_META, config).write_text(json.dumps(meta, indent=2) + "\n")


def local_embeddings(manifest: pd.DataFrame, config: dict, model=None, device=None):
    """Cached local embeddings if valid, else embed with DINOv2 and refresh the cache."""
    X = load_cached_local_embeddings(manifest, config)
    if X is None:
        if model is None:
            from src.encoders import get_device, load_dinov2

            device = device or get_device()
            model = load_dinov2(config["image"]["model_name"], device=device)
        X = _embed_images(model, manifest, int(config["image"]["input_size"]), device)
        save_local_embeddings(X, manifest, config)
    return X


def load_caption_artifacts(emb_path="artifacts/embeddings/minilm_captions.npy",
                           index_path="artifacts/embeddings/minilm_caption_index.json"):
    """Caption embeddings (T05) and their labels, in index order."""
    import json

    import numpy as np

    emb_path, index_path = ROOT / emb_path, ROOT / index_path
    for p in (emb_path, index_path):
        if not p.exists():
            raise FileNotFoundError(f"Caption artifact not found: {p} (run T05 first)")
    Y = np.load(emb_path)
    idx = json.loads(index_path.read_text())
    labels = [idx[str(i)]["label"] for i in range(len(idx))]
    if len(labels) != len(Y):
        raise ValueError(f"caption index ({len(labels)}) != embeddings {Y.shape}")
    return Y, labels


def evaluate_local(manifest: pd.DataFrame, image_embeddings, procrustes_params: dict,
                   caption_embeddings=None, caption_labels=None, *, resnet_checkpoint=None,
                   agg: str = "max", device=None) -> pd.DataFrame:
    """Predict every local tile with Procrustes caption retrieval (and ResNet-18 if a checkpoint
    is given). Returns one row per tile. Nothing here is fitted on the local tiles.

    Signature follows INTERFACES §5 (manifest, image_embeddings, procrustes_params,
    caption_embeddings, caption_labels). `image_embeddings` may also be a DINOv2 model or a
    callable (see `_embed_images`). Captions default to the T05 artifacts, so they match T06.

    Columns: the INTERFACES local.csv contract (tile_id, true_label, local_category,
    predicted_label, score, correct) for Procrustes, then `scorable`, `top3` and, with a
    checkpoint, resnet18_pred / resnet18_prob / resnet18_correct. `Other` tiles are kept but
    `scorable=False` and never counted as correct.
    """
    import numpy as np

    from src.alignment import transform_images, transform_texts
    from src.evaluate import evaluate_caption_classification

    if caption_embeddings is None:
        caption_embeddings, caption_labels = load_caption_artifacts()
    X = np.asarray(_embed_images(image_embeddings, manifest, device=device), np.float32)
    if len(X) != len(manifest):
        raise ValueError(f"image embeddings {X.shape} != manifest rows {len(manifest)}")
    Xa = transform_images(X, procrustes_params)
    Ya = transform_texts(caption_embeddings, procrustes_params)

    scorable = manifest.label.isin(EUROSAT_CLASSES).to_numpy()
    true = manifest.label.where(scorable, EUROSAT_CLASSES[0]).tolist()  # placeholder; masked below
    r = evaluate_caption_classification(Xa, Ya, true, caption_labels, class_names=EUROSAT_CLASSES, agg=agg)
    scores = r["class_scores"]
    order = np.argsort(-scores, axis=1)
    pred = [EUROSAT_CLASSES[i] for i in order[:, 0]]

    out = pd.DataFrame({
        "tile_id": manifest.tile_id.to_numpy(), "true_label": manifest.label.to_numpy(),
        "local_category": manifest.local_category.to_numpy(), "predicted_label": pred,
        "score": scores[np.arange(len(manifest)), order[:, 0]].round(4),
    })
    out["correct"] = (out.predicted_label == out.true_label) & scorable
    out["scorable"] = scorable
    out["top3"] = ["|".join(EUROSAT_CLASSES[j] for j in row[:3]) for row in order]
    if resnet_checkpoint is not None:
        from src.baselines import predict_resnet18

        preds = predict_resnet18(resnet_checkpoint, load_tile_images(manifest))
        out["resnet18_pred"] = [p for p, _ in preds]
        out["resnet18_prob"] = [round(c, 4) for _, c in preds]
        out["resnet18_correct"] = (out.resnet18_pred == out.true_label) & scorable
    return out


def _as_bool(series: pd.Series) -> pd.Series:
    return series.map(lambda v: str(v).strip().lower() in ("true", "1", "yes")).astype(bool)


def local_metrics(preds: pd.DataFrame, train_majority_class: str | None = None,
                  qa_path: str | Path = QA_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Method comparison (local_summary.csv) and per-category table, on scorable tiles only.

    Baselines:
    - `majority_train`: the EuroSAT train majority class (T09), a real baseline fitted on train.
    - `oracle_majority_local`: the most common *local* label. It reads the test labels, so it is an
      oracle, not a model; it is the bar any model must clear to be better than a constant guess.
    Also reports accuracy on tiles not flagged `ambiguous` in labels_qa.csv (fixed in T11, before T12).
    """
    p = preds[preds.scorable].copy()
    n_other = int((~preds.scorable).sum())
    qa = ROOT / qa_path if not Path(qa_path).is_absolute() else Path(qa_path)
    if qa.exists():
        amb = pd.read_csv(qa, dtype={"ambiguous": str}).set_index("tile_id")["ambiguous"]
        p["ambiguous"] = _as_bool(p.tile_id.map(amb).fillna("False"))
    else:
        p["ambiguous"] = False
    oracle = p.true_label.mode().iloc[0]
    p["procrustes_correct"] = p["correct"]
    p["oracle_majority_local_correct"] = p.true_label == oracle
    methods = ["procrustes"] + (["resnet18"] if "resnet18_correct" in p else [])
    notes = {"procrustes": "Upper Arlington OOD", "resnet18": "Upper Arlington OOD",
             "oracle_majority_local": f"constant '{oracle}' = most common local label (uses test labels; oracle)"}
    if train_majority_class:
        p["majority_train_correct"] = p.true_label == train_majority_class
        methods.append("majority_train")
        notes["majority_train"] = f"constant '{train_majority_class}' = EuroSAT train majority (T09)"
    methods.append("oracle_majority_local")
    rows = []
    for m in methods:
        c = p[f"{m}_correct"]
        clear = c[~p.ambiguous]
        rows.append({"method": m, "accuracy": round(float(c.mean()), 4), "n": int(len(c)),
                     "accuracy_unambiguous": round(float(clear.mean()), 4) if len(clear) else None,
                     "n_unambiguous": int(len(clear)), "n_other_excluded": n_other, "notes": notes[m]})
    summary = pd.DataFrame(rows)
    per_cat = p.groupby("local_category").agg(
        n=("tile_id", "size"),
        **{f"{m}_acc": (f"{m}_correct", "mean") for m in methods},
    ).round(3).reset_index()
    return summary, per_cat


def run_local_eval(config_path: str | Path | None = None, resnet_checkpoint="artifacts/checkpoints/resnet18.pt",
                   out_dir: str | Path = "artifacts/metrics"):
    """T12 entry point (scripts/evaluate_local.py, `python -m src.local_tiles evaluate`).

    Needs J's T05/T06 artifacts and, optionally, T08's ResNet-18 checkpoint (its columns are
    omitted with a warning if it is missing). Local embeddings come from the validated cache
    (scripts/extract_local_embeddings.py) or are computed with DINOv2 and cached.
    """
    import json

    from src.alignment import load_procrustes
    from src.config import load_config, resolve_path

    cfg = load_config(config_path)
    manifest_path = cfg["data"]["local_manifest"]
    pz = resolve_path("artifacts/alignment/procrustes.npz", cfg)
    if not pz.exists():
        raise FileNotFoundError(f"Procrustes artifact not found: {pz} (run T06 first)")
    params = load_procrustes(pz)
    df = load_local_manifest(resolve_path(manifest_path, cfg))
    X = local_embeddings(df, cfg)
    ck = resolve_path(resnet_checkpoint, cfg)
    if not ck.exists():
        print(f"[warn] {ck} not found; ResNet-18 columns omitted (run T08)")
    preds = evaluate_local(df, X, params, resnet_checkpoint=ck if ck.exists() else None)

    maj_json = resolve_path("artifacts/metrics/majority.json", cfg)
    train_majority = json.loads(maj_json.read_text()).get("majority_class") if maj_json.exists() else None
    summary, per_cat = local_metrics(preds, train_majority_class=train_majority)

    m = resolve_path(out_dir, cfg)
    m.mkdir(parents=True, exist_ok=True)
    contract_cols = ["tile_id", "true_label", "local_category", "predicted_label", "score", "correct"]
    preds[contract_cols].to_csv(m / "local.csv", index=False)               # INTERFACES §4 contract
    preds.to_csv(m / "local_predictions.csv", index=False)                  # + scorable, top-3, ResNet-18
    by_cat = per_cat.rename(columns={"procrustes_acc": "accuracy"})
    by_cat.to_csv(m / "local_by_category.csv", index=False)
    summary.to_csv(m / "local_summary.csv", index=False)                    # method comparison
    p = summary.set_index("method")
    payload = {
        "n": int(p.loc["procrustes", "n"]),
        "accuracy": float(p.loc["procrustes", "accuracy"]),
        "n_other_excluded": int(p.loc["procrustes", "n_other_excluded"]),
        "per_category": by_cat[["local_category", "n", "accuracy"]].to_dict("records"),
        "manifest": str(manifest_path),
        "methods": summary.to_dict("records"),
    }
    (m / "local.json").write_text(json.dumps(payload, indent=2))
    print(summary.to_string(index=False))
    print(by_cat.to_string(index=False))
    return preds, summary, per_cat


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
    sub.add_parser("evaluate", help="T12: Procrustes + ResNet-18 on the 50 local tiles")
    args = ap.parse_args(argv)
    if args.cmd == "evaluate":
        run_local_eval()
        return
    if args.cmd == "validate":
        df = load_local_manifest(args.manifest)
        print("manifest OK")
        print(summarize(df))


if __name__ == "__main__":
    _main()
