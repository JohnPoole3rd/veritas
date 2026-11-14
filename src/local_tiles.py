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


# --------------------------------------------------------------------------- T12 evaluation
QA_PATH = "data/local_tiles/labels_qa.csv"


def load_tile_images(df: pd.DataFrame) -> list:
    from PIL import Image

    return [Image.open(ROOT / p).convert("RGB") for p in df.path]


def _embed_images(image_model, manifest: pd.DataFrame, input_size: int = 224, device=None):
    """Return (N, D) DINOv2 embeddings for the manifest tiles. image_model may be:
    - an (N, D) array of precomputed embeddings (J's original evaluate_local API),
    - a callable list[PIL] -> (N, D) array (tests / custom encoders),
    - the frozen DINOv2 model from J's `src.encoders.load_dinov2` (uses J's eval transforms,
      i.e. the same 64->224 upscale as EuroSAT, so local and EuroSAT embeddings match)."""
    import numpy as np

    if isinstance(image_model, np.ndarray):
        return image_model
    if callable(image_model) and not hasattr(image_model, "parameters"):
        return image_model(load_tile_images(manifest))
    from src.data import get_transforms
    from src.encoders import extract_image_embeddings

    ds = LocalTileDataset(manifest, ROOT, transform=get_transforms(input_size, train=False))
    return extract_image_embeddings(image_model, ds, batch_size=32, device=device)


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
    predicted_label, score, correct) first, then A's extras (top-3, ResNet-18, scorable).
    """
    import numpy as np

    from src.alignment import transform_images, transform_texts
    from src.evaluate import evaluate_caption_classification

    if caption_embeddings is None:
        caption_embeddings, caption_labels = load_caption_artifacts()
    params = procrustes_params
    X = np.asarray(_embed_images(image_embeddings, manifest, device=device), np.float32)
    if len(X) != len(manifest):
        raise ValueError(f"image embeddings {X.shape} != manifest rows {len(manifest)}")
    Xa = transform_images(X, params)
    Ya = transform_texts(caption_embeddings, params)

    # 'Other' tiles cannot be scored against EuroSAT classes; keep them, mark as not scorable.
    scorable = manifest.label.isin(EUROSAT_CLASSES).to_numpy()
    true = manifest.label.where(scorable, EUROSAT_CLASSES[0]).tolist()
    r = evaluate_caption_classification(Xa, Ya, true, caption_labels, class_names=EUROSAT_CLASSES, agg=agg)
    scores = r["class_scores"]
    order = np.argsort(-scores, axis=1)

    out = manifest[["tile_id", "label", "local_category"]].copy()
    out["scorable"] = scorable
    out["procrustes_pred"] = [EUROSAT_CLASSES[i] for i in order[:, 0]]
    out["procrustes_score"] = scores[np.arange(len(out)), order[:, 0]].round(4)
    out["procrustes_top3"] = ["|".join(EUROSAT_CLASSES[j] for j in row[:3]) for row in order]
    out["procrustes_correct"] = (out.procrustes_pred == out.label) & out.scorable
    if resnet_checkpoint is not None:
        from src.baselines import predict_resnet18

        preds = predict_resnet18(resnet_checkpoint, load_tile_images(manifest))
        out["resnet18_pred"] = [p for p, _ in preds]
        out["resnet18_prob"] = [round(c, 4) for _, c in preds]
        out["resnet18_correct"] = (out.resnet18_pred == out.label) & out.scorable
    contract = pd.DataFrame({
        "tile_id": out.tile_id, "true_label": out.label, "local_category": out.local_category,
        "predicted_label": out.procrustes_pred, "score": out.procrustes_score,
        "correct": out.procrustes_correct,
    })
    return pd.concat([contract, out.drop(columns=["tile_id", "local_category"])], axis=1)


def local_metrics(preds: pd.DataFrame, qa_path: str | Path = QA_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Method comparison (artifacts/metrics/local_summary.csv) plus a per-category table.

    Includes the LOCAL majority baseline (most common local label; T11: Residential, 31/50) and
    accuracy on non-ambiguous tiles (labels_qa.csv `ambiguous` flag)."""
    p = preds[preds.scorable].copy()
    qa = ROOT / qa_path if not Path(qa_path).is_absolute() else Path(qa_path)
    if qa.exists():
        amb = pd.read_csv(qa).set_index("tile_id")["ambiguous"]
        p["ambiguous"] = p.tile_id.map(amb).fillna(False).astype(bool)
    else:
        p["ambiguous"] = False
    maj = p.label.mode().iloc[0]
    p["majority_correct"] = p.label == maj
    methods = [m for m in ("procrustes", "resnet18") if f"{m}_correct" in p] + ["majority"]
    rows = []
    for m in methods:
        c = p[f"{m}_correct"]
        clear = c[~p.ambiguous]
        note = f"Upper Arlington OOD; constant '{maj}'" if m == "majority" else "Upper Arlington OOD"
        rows.append({"method": m if m != "majority" else "majority_local", "accuracy": round(c.mean(), 4),
                     "n": int(len(c)), "accuracy_unambiguous": round(clear.mean(), 4) if len(clear) else None,
                     "n_unambiguous": int(len(clear)), "notes": note})
    summary = pd.DataFrame(rows)
    per_cat = p.groupby("local_category").agg(
        n=("tile_id", "size"),
        **{f"{m}_acc": (f"{m}_correct", "mean") for m in methods},
    ).round(3).reset_index()
    return summary, per_cat


def run_local_eval(out_dir: str | Path = "artifacts", resnet_checkpoint="artifacts/checkpoints/resnet18.pt",
                   procrustes_path="artifacts/alignment/procrustes.npz", local_embeddings="auto",
                   manifest_path: str | Path | None = None, config: dict | None = None):
    """T12 entry point: needs J's T05/T06 artifacts (caption embeddings, procrustes.npz) and
    T08's ResNet-18 checkpoint (optional). Uses artifacts/embeddings/dinov2_local.npy if present,
    else embeds the tiles with DINOv2 directly (identical transforms)."""
    import json

    import numpy as np

    out_dir = ROOT / out_dir
    pz = ROOT / procrustes_path
    if not pz.exists():
        raise FileNotFoundError(f"Procrustes artifact not found: {pz} (run T06 first)")
    from src.alignment import load_procrustes
    from src.config import load_config
    from src.encoders import get_device, load_dinov2

    params = load_procrustes(pz)
    cfg = config or load_config()
    device = get_device()
    ck = ROOT / resnet_checkpoint
    df = load_local_manifest(manifest_path or DEFAULT_MANIFEST)
    emb_path = ROOT / "artifacts/embeddings/dinov2_local.npy"
    if local_embeddings == "auto":
        local_embeddings = emb_path.exists()
    if local_embeddings:  # from scripts/extract_local_embeddings.py (same transforms)
        image_input = np.load(emb_path)
    else:
        image_input = load_dinov2(cfg["image"]["model_name"], device=device)
    preds = evaluate_local(df, image_input, params, resnet_checkpoint=ck if ck.exists() else None,
                           device=device)
    summary, per_cat = local_metrics(preds)
    m = out_dir / "metrics"
    m.mkdir(parents=True, exist_ok=True)
    contract_cols = ["tile_id", "true_label", "local_category", "predicted_label", "score", "correct"]
    preds[contract_cols].to_csv(m / "local.csv", index=False)               # INTERFACES contract
    preds.to_csv(m / "local_predictions.csv", index=False)                  # + top-3, ResNet-18
    by_cat = per_cat.rename(columns={"procrustes_acc": "accuracy"})
    by_cat.to_csv(m / "local_by_category.csv", index=False)                 # INTERFACES name
    summary.to_csv(m / "local_summary.csv", index=False)                    # method comparison
    payload = {
        "n": int(len(preds)),
        "accuracy": float(preds["correct"].mean()),
        "per_category": by_cat[["local_category", "n", "accuracy"]].to_dict("records"),
        "manifest": str(manifest_path or DEFAULT_MANIFEST),
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
