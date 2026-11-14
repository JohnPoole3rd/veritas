"""T13 — Failure taxonomy figures (owner: A).

    python -m src.failures

Reads artifacts/metrics/local_predictions.csv (T12) and the full-split EuroSAT artifacts, writes:
    artifacts/figures/failure_gallery.png       success + failure tiles with both models' outputs
    artifacts/figures/local_confusion.png        Procrustes and ResNet-18 on the 50 local tiles
    artifacts/metrics/failure_taxonomy.csv       failure types with counts, examples, likely causes
    artifacts/metrics/procrustes_test_top_confusions.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "artifacts/figures"
MET = ROOT / "artifacts/metrics"

# (tile_id, headline) — chosen from T12 predictions; see results/T13.md
GALLERY = [
    ("ua_011", "SUCCESS: big-box retail"),
    ("ua_049", "Scioto River -> 'Highway'"),
    ("ua_036", "Golf turf -> ResNet 'Forest' p=1.00"),
    ("ua_034", "Houses under canopy (labellers disagreed)"),
    ("ua_013", "Strip mall -> Procrustes 'Residential'"),
]


def eurosat_top_confusions(k: int = 8) -> pd.DataFrame:
    from src.alignment import load_procrustes, transform_images, transform_texts
    from src.data import load_split
    from src.evaluate import evaluate_caption_classification

    emb = ROOT / "artifacts/embeddings"
    p = load_procrustes(ROOT / "artifacts/alignment/procrustes.npz")
    X, Y = np.load(emb / "dinov2_test.npy"), np.load(emb / "minilm_captions.npy")
    idx = json.loads((emb / "minilm_caption_index.json").read_text())
    cap_labels = [idx[str(i)]["label"] for i in range(len(idx))]
    split = load_split(ROOT / "data/splits/test.json")
    names, labels = split["class_names"], [it["label"] for it in split["items"]]
    r = evaluate_caption_classification(transform_images(X, p), transform_texts(Y, p), labels, cap_labels,
                                        class_names=names)
    cm = r["confusion_matrix"]
    rows = [{"true": names[i], "predicted": names[j], "count": int(cm[i, j]),
             "share_of_true": round(cm[i, j] / cm[i].sum(), 3)}
            for i in range(len(names)) for j in range(len(names)) if i != j]
    return pd.DataFrame(rows).sort_values("count", ascending=False).head(k).reset_index(drop=True)


def failure_gallery(preds: pd.DataFrame, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    p = preds.set_index("tile_id")
    man = pd.read_csv(ROOT / "data/local_tiles/manifest.csv").set_index("tile_id")
    fig, axes = plt.subplots(1, len(GALLERY), figsize=(3.1 * len(GALLERY), 4.6))
    for ax, (tid, head) in zip(axes, GALLERY):
        ax.imshow(Image.open(ROOT / man.loc[tid, "path"]), interpolation="nearest")
        r = p.loc[tid]
        ok = lambda c: "✓" if c else "✗"  # noqa: E731
        ax.set_title(f"{tid}: {head}", fontsize=8.5, wrap=True)
        ax.set_xlabel(
            f"true: {r.label}\n"
            f"Procrustes: {r.procrustes_pred} ({r.procrustes_score:.2f}) {ok(r.procrustes_correct)}\n"
            f"ResNet-18: {r.resnet18_pred} (p={r.resnet18_prob:.2f}) {ok(r.resnet18_correct)}",
            fontsize=8)
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle("Upper Arlington, Sentinel-2 2025-08-01, 640 m tiles (10 m/px)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def local_confusion(preds: pd.DataFrame, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, m, title in zip(axes, ("procrustes", "resnet18"), ("Procrustes", "ResNet-18")):
        ct = pd.crosstab(preds.label, preds[f"{m}_pred"])
        ax.imshow(ct.to_numpy(), cmap="Blues")
        ax.set_xticks(range(ct.shape[1]), ct.columns, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(ct.shape[0]), ct.index, fontsize=8)
        for i in range(ct.shape[0]):
            for j in range(ct.shape[1]):
                if ct.iat[i, j]:
                    ax.text(j, i, ct.iat[i, j], ha="center", va="center", fontsize=8,
                            color="white" if ct.iat[i, j] > ct.to_numpy().max() / 2 else "black")
        acc = preds[f"{m}_correct"].mean()
        ax.set_title(f"{title}: local acc {acc:.2f} (N=50; always-Residential 0.62)", fontsize=9)
        ax.set_xlabel("Predicted"), ax.set_ylabel("True (visual label)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def taxonomy(preds: pd.DataFrame, conf: pd.DataFrame) -> pd.DataFrame:
    def n(mask):
        return int(mask.sum())

    P = preds
    river_hw_eu = conf[(conf.true == "River") & (conf.predicted == "Highway")]
    rows = [
        {"failure_type": "Water read as road (linear feature)",
         "example": "ua_049, ua_041",
         "count": f"local: Procrustes {n((P.label == 'River') & (P.procrustes_pred == 'Highway'))}/10 water tiles; "
                  f"EuroSAT test: River->Highway {int(river_hw_eu['count'].sum())} "
                  f"({float(river_hw_eu['share_of_true'].sum()):.0%} of River)",
         "likely_cause": "a 60–150 m channel is a thin dark line at 10 m; caption space puts 'river' and 'road' close",
         "fix_idea": "NIR/NDWI band (water is unambiguous in NIR); shape-agnostic water captions"},
        {"failure_type": "Managed turf mistaken for forest/pasture",
         "example": "ua_036–ua_038 (golf course)",
         "count": f"ResNet Forest on {n((P.label == 'HerbaceousVegetation') & (P.resnet18_pred == 'Forest'))}/4 "
                  f"Herbaceous tiles, mean p={P[(P.label == 'HerbaceousVegetation') & (P.resnet18_pred == 'Forest')].resnet18_prob.mean():.3f}",
         "likely_cause": "irrigated, saturated-green turf matches no EuroSAT class; ResNet extrapolates confidently",
         "fix_idea": "add local labelled examples; calibrate (temperature) and show uncertainty"},
        {"failure_type": "Canopy vs residential",
         "example": "ua_032–ua_035",
         "count": f"ResNet Forest on {n((P.label == 'Residential') & (P.resnet18_pred == 'Forest'))}/31 Residential tiles; "
                  "primary and blind second labeller disagreed on 2/2 such QA tiles",
         "likely_cause": "34 % city canopy hides houses at 10 m; the label itself is ambiguous",
         "fix_idea": "multi-label / fractional cover (canopy %), which is what the forestry user needs anyway"},
        {"failure_type": "Commercial strip vs residential",
         "example": "ua_013",
         "count": f"Procrustes missed {n((P.label == 'Industrial') & ~P.procrustes_correct)}/3 Industrial; "
                  f"EuroSAT Residential->Industrial {int(conf[(conf.true == 'Residential') & (conf.predicted == 'Industrial')]['count'].sum())}",
         "likely_cause": "US strip centres are small and mixed with houses; EuroSAT Industrial is large European sites",
         "fix_idea": "more context (larger tiles) or local examples"},
        {"failure_type": "Crop-type confusion (in distribution)",
         "example": "EuroSAT test",
         "count": f"PermanentCrop->AnnualCrop {int(conf[(conf.true == 'PermanentCrop') & (conf.predicted == 'AnnualCrop')]['count'].sum())}",
         "likely_cause": "orchards vs fields differ by phenology, not one-date RGB texture",
         "fix_idea": "multi-date imagery"},
        {"failure_type": "Uninformative confidence (cross-cutting)",
         "example": "all local tiles",
         "count": f"Procrustes score correct {P[P.procrustes_correct].procrustes_score.mean():.3f} vs wrong "
                  f"{P[~P.procrustes_correct].procrustes_score.mean():.3f}; ResNet p when wrong "
                  f"{P[~P.resnet18_correct].resnet18_prob.mean():.2f}",
         "likely_cause": "out-of-distribution inputs; no calibration",
         "fix_idea": "do not threshold on these scores; show top-3 and a 'prototype' warning"},
    ]
    return pd.DataFrame(rows)


def main():
    preds = pd.read_csv(MET / "local_predictions.csv")
    FIG.mkdir(parents=True, exist_ok=True)
    conf = eurosat_top_confusions()
    conf.to_csv(MET / "procrustes_test_top_confusions.csv", index=False)
    failure_gallery(preds, FIG / "failure_gallery.png")
    local_confusion(preds, FIG / "local_confusion.png")
    tax = taxonomy(preds, conf)
    tax.to_csv(MET / "failure_taxonomy.csv", index=False)
    print(conf.to_string(index=False))
    print(tax[["failure_type", "count"]].to_string(index=False))


if __name__ == "__main__":
    main()
