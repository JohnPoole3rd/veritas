"""T13 — Failure taxonomy figures (owner: A).

    python -m src.failures

Reads T12's artifacts/metrics/local_predictions.csv and the full EuroSAT confusion matrix written by
`python -m src.evaluate caption-classification` (procrustes_test_confusion.csv). Writes:
    artifacts/figures/failure_gallery.png       success + failure tiles with both models' outputs
    artifacts/figures/local_confusion.png        Procrustes (and ResNet-18) on the local tiles
    artifacts/metrics/failure_taxonomy.csv       failure types with counts, examples, likely causes
    artifacts/metrics/procrustes_test_top_confusions.csv
Every count is computed from those files; nothing is typed in.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "artifacts/figures"
MET = ROOT / "artifacts/metrics"

# (tile_id, short description). Chosen in T13 from T12 predictions; outcomes are read from data.
GALLERY = [
    ("ua_011", "big-box retail"),
    ("ua_049", "Scioto River channel"),
    ("ua_036", "golf-course turf"),
    ("ua_034", "houses under canopy (labellers disagreed)"),
    ("ua_013", "strip mall"),
]


def eurosat_confusions(split: str = "test") -> pd.DataFrame:
    """All off-diagonal (true, predicted) pairs of the Procrustes EuroSAT confusion matrix."""
    path = MET / f"procrustes_{split}_confusion.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m src.evaluate caption-classification --split {split}`")
    cm = pd.read_csv(path).set_index("true")
    rows = [{"true": t, "predicted": c, "count": int(cm.loc[t, c]),
             "share_of_true": round(cm.loc[t, c] / cm.loc[t].sum(), 3)}
            for t in cm.index for c in cm.columns if t != c]
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def _pair(conf: pd.DataFrame, true: str, pred: str) -> tuple[int, float]:
    row = conf[(conf.true == true) & (conf.predicted == pred)]
    return (int(row["count"].iloc[0]), float(row["share_of_true"].iloc[0])) if len(row) else (0, 0.0)


def _outcome(r, has_resnet: bool) -> str:
    ok = lambda c: "✓" if c else "✗"  # noqa: E731
    lines = [f"true: {r.true_label}", f"Procrustes: {r.predicted_label} ({r.score:.2f}) {ok(r.correct)}"]
    if has_resnet:
        lines.append(f"ResNet-18: {r.resnet18_pred} (p={r.resnet18_prob:.2f}) {ok(r.resnet18_correct)}")
    return "\n".join(lines)


def failure_gallery(preds: pd.DataFrame, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    p = preds.set_index("tile_id")
    man = pd.read_csv(ROOT / "data/local_tiles/manifest.csv").set_index("tile_id")
    picks = [(t, d) for t, d in GALLERY if t in p.index]
    has_resnet = "resnet18_pred" in preds
    fig, axes = plt.subplots(1, len(picks), figsize=(3.1 * len(picks), 4.6), squeeze=False)
    for ax, (tid, desc) in zip(axes[0], picks):
        r = p.loc[tid]
        ax.imshow(Image.open(ROOT / man.loc[tid, "path"]), interpolation="nearest")
        status = "(Procrustes ✓)" if r.correct else f"→ Procrustes '{r.predicted_label}'"
        ax.set_title(f"{tid}: {desc} {status}", fontsize=8.5, wrap=True)
        ax.set_xlabel(_outcome(r, has_resnet), fontsize=8)
        ax.set_xticks([]), ax.set_yticks([])
    dates = ", ".join(sorted(man["date"].astype(str).unique()))
    fig.suptitle(f"Upper Arlington, Sentinel-2 {dates}, 640 m tiles (10 m/px)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def local_confusion(preds: pd.DataFrame, path: Path, oracle_acc: float | None = None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = preds[preds.scorable]
    methods = [("predicted_label", "correct", "Procrustes")]
    if "resnet18_pred" in p:
        methods.append(("resnet18_pred", "resnet18_correct", "ResNet-18"))
    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 4.8), squeeze=False)
    for ax, (pred_col, ok_col, title) in zip(axes[0], methods):
        labels = sorted(set(p.true_label) | set(p[pred_col]))  # same order on both axes: diagonal = correct
        ct = pd.crosstab(p.true_label, p[pred_col]).reindex(index=labels, columns=labels, fill_value=0)
        ax.imshow(ct.to_numpy(), cmap="Blues")
        ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        vmax = ct.to_numpy().max()
        for i in range(len(labels)):
            for j in range(len(labels)):
                if ct.iat[i, j]:
                    ax.text(j, i, ct.iat[i, j], ha="center", va="center", fontsize=8,
                            color="white" if ct.iat[i, j] > vmax / 2 else "black")
        baseline = f"; most-common-label oracle {oracle_acc:.2f}" if oracle_acc is not None else ""
        ax.set_title(f"{title}: local acc {p[ok_col].mean():.2f} (N={len(p)}{baseline})", fontsize=9)
        ax.set_xlabel("Predicted"), ax.set_ylabel("True (visual label)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def taxonomy(preds: pd.DataFrame, conf: pd.DataFrame) -> pd.DataFrame:
    P = preds[preds.scorable]
    R = "resnet18_pred" in P

    def frac(mask, of):
        return f"{int(mask.sum())}/{int(of.sum())}"

    river, herb, resid, ind = (P.true_label == c for c in ("River", "HerbaceousVegetation", "Residential", "Industrial"))
    rh_n, rh_s = _pair(conf, "River", "Highway")
    ri_n, _ = _pair(conf, "Residential", "Industrial")
    pa_n, pa_s = _pair(conf, "PermanentCrop", "AnnualCrop")
    rows = [
        {"failure_type": "Water read as road (linear feature)", "example": "ua_049, ua_041",
         "count": f"local: Procrustes {frac(river & (P.predicted_label == 'Highway'), river)} water tiles → Highway; "
                  f"EuroSAT test: River→Highway {rh_n} ({rh_s:.0%} of River)",
         "likely_cause": "a 60–150 m channel is a thin dark line at 10 m; caption space puts 'river' and 'road' close",
         "fix_idea": "NIR/NDWI band (water is unambiguous in NIR); shape-agnostic water captions"},
        {"failure_type": "Managed turf mistaken for forest/pasture", "example": "ua_036–ua_038 (golf course)",
         "count": (f"ResNet Forest on {frac(herb & (P.resnet18_pred == 'Forest'), herb)} Herbaceous tiles, mean p="
                   f"{P[herb & (P.resnet18_pred == 'Forest')].resnet18_prob.mean():.3f}; " if R else "")
                  + f"Procrustes Pasture on {frac(herb & (P.predicted_label == 'Pasture'), herb)}",
         "likely_cause": "irrigated, saturated-green turf matches no EuroSAT class; ResNet extrapolates confidently",
         "fix_idea": "add local labelled examples; calibrate (temperature) and show uncertainty"},
        {"failure_type": "Canopy vs residential", "example": "ua_032–ua_035",
         "count": (f"ResNet Forest on {frac(resid & (P.resnet18_pred == 'Forest'), resid)} Residential tiles; " if R else "")
                  + "primary and blind second labeller disagreed on ua_034/ua_035 (labels_qa.csv)",
         "likely_cause": "34 % city canopy hides houses at 10 m; the label itself is ambiguous",
         "fix_idea": "multi-label / fractional cover (canopy %), which is what the forestry user needs anyway"},
        {"failure_type": "Commercial strip vs residential", "example": "ua_013",
         "count": f"Procrustes missed {frac(ind & ~P.correct, ind)} Industrial; EuroSAT Residential→Industrial {ri_n}",
         "likely_cause": "US strip centres are small and mixed with houses; EuroSAT Industrial is large European sites",
         "fix_idea": "more context (larger tiles) or local examples"},
        {"failure_type": "Crop-type confusion (in distribution)", "example": "EuroSAT test",
         "count": f"PermanentCrop→AnnualCrop {pa_n} ({pa_s:.0%} of PermanentCrop)",
         "likely_cause": "orchards vs fields differ by phenology, not one-date RGB texture",
         "fix_idea": "multi-date imagery"},
        {"failure_type": "Uninformative confidence (cross-cutting)", "example": "all local tiles",
         "count": f"Procrustes score correct {P[P.correct].score.mean():.3f} vs wrong {P[~P.correct].score.mean():.3f}"
                  + (f"; ResNet p when wrong {P[~P.resnet18_correct].resnet18_prob.mean():.2f}" if R else ""),
         "likely_cause": "out-of-distribution inputs; no calibration",
         "fix_idea": "do not threshold on these scores; show top-3 and a 'prototype' warning"},
    ]
    return pd.DataFrame(rows)


def main():
    preds = pd.read_csv(MET / "local_predictions.csv")
    FIG.mkdir(parents=True, exist_ok=True)
    conf = eurosat_confusions()
    conf.head(8).to_csv(MET / "procrustes_test_top_confusions.csv", index=False)
    oracle = None
    summary = MET / "local_summary.csv"
    if summary.exists():
        s = pd.read_csv(summary).set_index("method")
        oracle = float(s.loc["oracle_majority_local", "accuracy"]) if "oracle_majority_local" in s.index else None
    failure_gallery(preds, FIG / "failure_gallery.png")
    local_confusion(preds, FIG / "local_confusion.png", oracle)
    tax = taxonomy(preds, conf)
    tax.to_csv(MET / "failure_taxonomy.csv", index=False)
    print(conf.head(8).to_string(index=False))
    print(tax[["failure_type", "count"]].to_string(index=False))


if __name__ == "__main__":
    main()
