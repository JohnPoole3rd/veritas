#!/usr/bin/env python
"""Does OSM-Sentinel training beat the original EuroSAT training? (A, for J)

Same frozen DINOv2-small + frozen MiniLM + one orthogonal Procrustes rotation; only the *fitting
pairs* differ:
  eurosat   — the main pipeline's rotation (artifacts/alignment/procrustes.npz, 18,900 EuroSAT pairs)
  osm       — fitted on the OSM-Sentinel train split (OSM-derived captions)
  combined  — fitted on EuroSAT train pairs + OSM train pairs together

Every fit is scored with the same rule (src.retrieval.class_scores, max over a class's captions) on:
  1. EuroSAT test (4,050 images, 10 classes, EuroSAT caption templates)       — old method's home turf
  2. Upper Arlington (50 tiles, EuroSAT classes/captions; also 37 unambiguous) — out of distribution for both
  3. OSM-Sentinel test split (held-out regions, OSM labels + LABEL_PROMPTS)  — new method's home turf
plus EuroSAT text→image search (50 caption queries), and a random-rotation control (mean of 5 seeds)
for every fit.

    python experiments/osm_sentinel/compare_eurosat.py \
        --data data/raw/osm_sentinel_1m_640 --tag full_1m_640
Writes experiments/osm_sentinel/compare_eurosat_<tag>.json and prints a markdown table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
THIS = Path(__file__).resolve().parent
for path in (ROOT, THIS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

SEEDS = [42, 43, 44, 45, 46]


def main() -> None:
    import pandas as pd

    import train as osm_train
    from osm_captions import LABEL_PROMPTS
    from src.alignment import fit_procrustes, load_procrustes, training_pairs, transform_images, transform_texts
    from src.captions import load_captions
    from src.config import load_config, resolve_path
    from src.data import load_split
    from src.encoders import encode_texts, get_device, load_dinov2, load_minilm
    from src.evaluate import classification_metrics, evaluate_retrieval
    from src.local_tiles import load_local_manifest, local_embeddings
    from src.retrieval import class_scores

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/raw/osm_sentinel_1m_640"))
    ap.add_argument("--tag", default="full_1m_640", help="label for this caption variant")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--exclude-fit-region", action="append", default=None,
                    help="Drop this region's train rows from the OSM fit (default: us_columbus, contains UA)")
    args = ap.parse_args()
    args.force = False
    args.device_resolved = get_device(None)

    cfg = load_config()
    args.input_size = int(cfg["image"]["input_size"])
    emb = resolve_path("artifacts/embeddings", cfg)

    # ---- EuroSAT side (main pipeline artifacts, full split) ----
    caps_eu = np.load(emb / "minilm_captions.npy")
    idx = json.loads((emb / "minilm_caption_index.json").read_text())
    caplab_eu = [idx[str(i)]["label"] for i in range(len(idx))]
    test = load_split(resolve_path(cfg["data"]["splits_dir"], cfg) / "test.json")
    classes_eu = test["class_names"]
    X_eu_test = np.load(emb / "dinov2_test.npy")
    y_eu_test = [it["label"] for it in test["items"]]
    X_eu_train, Y_eu_train = training_pairs(cfg, caps_eu, load_captions(resolve_path(cfg["data"]["captions_path"], cfg)))

    manifest = load_local_manifest()
    X_ua = local_embeddings(manifest, cfg)
    y_ua = manifest.label.tolist()
    qa = pd.read_csv(ROOT / "data/local_tiles/labels_qa.csv").set_index("tile_id")["ambiguous"].astype(str)
    clear = ~manifest.tile_id.map(qa).str.lower().eq("true").to_numpy()

    # ---- OSM side ----
    exclude = args.exclude_fit_region if args.exclude_fit_region is not None else list(osm_train.DEFAULT_EXCLUDE_FIT)
    rows = [r for r in osm_train.drop_fit_regions(osm_train.load_rows(Path(args.data)), exclude)
            if Path(r["path"]).exists() and r["caption"]]
    by = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    model = load_dinov2(cfg["image"]["model_name"], device=args.device_resolved)
    # Image embeddings depend only on the images, so share train.py's cache (validated by tile_ids):
    # after train.py on the same data this is a cache hit instead of re-embedding every chip.
    X_osm = {s: osm_train.embed(model, by[s], osm_train.BACKBONE, s, args) for s in ("train", "test")}
    text_model = load_minilm(cfg["text"]["model_name"], device=args.device_resolved)
    Y_osm_train = encode_texts(text_model, [r["caption"] for r in by["train"]])
    osm_classes = sorted({r["label"] for r in rows})
    prompts = [f"{LABEL_PROMPTS.get(c, c)[:1].upper()}{LABEL_PROMPTS.get(c, c)[1:]}." for c in osm_classes]
    Y_osm_prompts = encode_texts(text_model, prompts)
    y_osm_test = [r["label"] for r in by["test"]]

    center = bool(cfg["alignment"]["center"])
    fits = {
        "eurosat": load_procrustes(resolve_path("artifacts/alignment/procrustes.npz", cfg)),
        "osm": fit_procrustes(X_osm["train"], Y_osm_train, center=center),
        "combined": None,  # filled below from the single stacked copy
    }
    # A: build the combined pairs once (was stacked twice: ~5 GB transient at 1M)
    X_comb, Y_comb = np.vstack([X_eu_train, X_osm["train"]]), np.vstack([Y_eu_train, Y_osm_train])
    fits["combined"] = fit_procrustes(X_comb, Y_comb, center=center)
    train_pairs = {"eurosat": (X_eu_train, Y_eu_train), "osm": (X_osm["train"], Y_osm_train),
                   "combined": (X_comb, Y_comb)}

    def score(fit) -> dict:
        def clf(X, Y, caplab, classes, y, mask=None):
            pred = np.array(classes)[class_scores(transform_images(X, fit), transform_texts(Y, fit), caplab, classes).argmax(1)]
            y = np.array(y)
            if mask is not None:
                pred, y = pred[mask], y[mask]
            m = classification_metrics(list(y), list(pred), classes)
            return {"accuracy": round(m["accuracy"], 4), "macro_f1": round(m["macro_f1"], 4), "n": m["n"]}

        out = {
            "eurosat_test": clf(X_eu_test, caps_eu, caplab_eu, classes_eu, y_eu_test),
            "upper_arlington": clf(X_ua, caps_eu, caplab_eu, classes_eu, y_ua),
            "upper_arlington_unambiguous": clf(X_ua, caps_eu, caplab_eu, classes_eu, y_ua, clear),
            "osm_test": clf(X_osm["test"], Y_osm_prompts, osm_classes, osm_classes, y_osm_test),
        }
        r = evaluate_retrieval(transform_images(X_eu_test, fit), transform_texts(caps_eu, fit), y_eu_test, caplab_eu)
        out["eurosat_text_search"] = {"top1": round(r["top1"], 4), "top5": round(r["top5"], 4), "queries": r["n_captions"]}
        return out

    results = {"tag": args.tag, "data": str(args.data), "excluded_from_fit": exclude,
               "n": {"osm_train": len(by["train"]), "osm_val": len(by["val"]), "osm_test": len(by["test"]),
                     "eurosat_train": len(X_eu_train), "eurosat_test": len(X_eu_test), "upper_arlington": len(X_ua)},
               "osm_test_regions": sorted({r["region_id"] for r in by["test"]}),
               "osm_classes": osm_classes, "fits": {}}
    for name, fit in fits.items():
        X, Y = train_pairs[name]
        controls = [score(fit_procrustes(X, Y, center=center, shuffle_y=True, seed=s)) for s in SEEDS]
        ctrl = {k: {m: round(float(np.mean([c[k][m] for c in controls])), 4)
                    for m in controls[0][k] if m not in ("n", "queries")} for k in controls[0]}
        results["fits"][name] = {"score": score(fit), "random_rotation_mean5": ctrl}

    out = THIS / f"compare_eurosat_{args.tag}.json"
    out.write_text(json.dumps(results, indent=2) + "\n")

    def cell(f, k, m="accuracy"):
        return f"{results['fits'][f]['score'][k][m]:.3f} ({results['fits'][f]['random_rotation_mean5'][k][m]:.3f})"

    print(f"\n### {args.tag}: OSM train n={results['n']['osm_train']}, OSM test n={results['n']['osm_test']} "
          f"({len(results['osm_test_regions'])} held-out regions)\n")
    print("| Fit | EuroSAT test acc | EuroSAT macro-F1 | EuroSAT search top-1 | Upper Arlington acc (all 50) | UA unambiguous (37) | OSM test acc | OSM macro-F1 |")
    print("|---|---|---|---|---|---|---|---|")
    for f in fits:
        print(f"| {f} | {cell(f,'eurosat_test')} | {cell(f,'eurosat_test','macro_f1')} | {cell(f,'eurosat_text_search','top1')} | "
              f"{cell(f,'upper_arlington')} | {cell(f,'upper_arlington_unambiguous')} | {cell(f,'osm_test')} | {cell(f,'osm_test','macro_f1')} |")
    print("\nCells: score (random-rotation control, mean of 5 seeds). Upper Arlington reference: always 'Residential' = 0.62 (oracle), EuroSAT majority = 0.00.")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
