#!/usr/bin/env python
"""Search benchmark: does a plain-English query find matching places in regions no model was trained on?

This measures the product (text -> places inside an area), not image -> label.

- Places: every test-split region of the final 1M build (held out of every rotation fit).
- Queries: bench/search_queries_v1.json, frozen before results existed. "open" queries name things a
  10-class land-cover map cannot express; "vocab" queries name land cover a map also has.
- Relevance (automatic proxy): a square is relevant if its OpenStreetMap feature_counts contain one of the
  query's phrases, i.e. that thing is mapped there. Mapped != visible, and OSM-trained rotations learned
  from the same source, so this proxy favours them; the EuroSAT rotation is shown for reference.
- For each (query, region) with >= MIN_REL relevant squares: rank all squares of the region by cosine
  similarity (as the website does) and take Precision@10; chance = relevant / squares in the region.
- 95% CIs: bootstrap over regions (squares within a region are correlated), 2000 resamples, seed 0.

    python experiments/osm_sentinel/search_benchmark.py --data data/raw/osm_sentinel_1m_640 \
        --rotation eurosat=artifacts/alignment/procrustes.npz --rotation osm_1m=... --out bench_out/
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
THIS = Path(__file__).resolve().parent
for p in (ROOT, THIS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

K, MIN_REL, N_BOOT = 10, 10, 2000


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    from src.alignment import load_procrustes, transform_images, transform_texts
    from src.encoders import encode_texts, get_device, load_dinov2, load_minilm
    import train as osm_train

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--queries", default=str(THIS / "bench" / "search_queries_v1.json"))
    ap.add_argument("--rotation", action="append", required=True, help="name=path.npz (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--dtype", default="fp16")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data)

    rows = [r for r in csv.DictReader((data / "manifest.csv").open(newline="")) if r["split"] == "test"]
    rows.sort(key=lambda r: r["tile_id"])
    qspec = json.loads(Path(args.queries).read_text())
    print(f"test squares {len(rows):,} in {len({r['region_id'] for r in rows})} held-out regions", flush=True)

    # image embeddings of every test square (cached by tile list)
    key = hashlib.sha256("\n".join(r["tile_id"] for r in rows).encode()).hexdigest()[:16]
    cache = out / f"dinov2_test_{key}.npy"
    if cache.exists():
        X = np.load(cache)
    else:
        dev = get_device(None)
        ea = SimpleNamespace(device_resolved=dev, input_size=224, batch_size=args.batch_size, num_workers=args.num_workers,
                             dtype=args.dtype, preprocess="auto", prefetch_factor=4)
        model = load_dinov2("facebook/dinov2-small", device=dev)
        X = osm_train.extract(model, [data / r["image"] for r in rows], ea)
        np.save(cache, X)
    assert len(X) == len(rows)

    tm = load_minilm("sentence-transformers/all-MiniLM-L6-v2")
    Q = encode_texts(tm, [q["query"] for q in qspec["queries"]])

    feats = [set(json.loads(r["feature_counts"] or "{}")) for r in rows]
    by_region: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_region[r["region_id"]].append(i)
    regions = sorted(by_region)

    rots = {}
    for spec in args.rotation:
        name, path = spec.split("=", 1)
        rots[name] = (load_procrustes(path), sha(Path(path))[:12])
    # controls: no rotation (identity, same centring as the first OSM fit) and a random orthogonal rotation
    base = next(p for n, (p, _) in rots.items() if n != "eurosat")
    rng = np.random.default_rng(0)
    Rr, _ = np.linalg.qr(rng.standard_normal((X.shape[1], X.shape[1])))
    controls = {"identity (no rotation)": dict(base, W=np.eye(X.shape[1], dtype=np.float32)),
                "random rotation": dict(base, W=Rr.astype(np.float32))}

    results = {"version": qspec["version"], "data": str(data), "n_squares": len(rows), "n_regions": len(regions),
               "k": K, "min_relevant": MIN_REL, "queries_sha256": sha(Path(args.queries))[:12], "models": {}}
    pairs_meta = None
    for name, (params, h) in list(rots.items()) + [(n, (p, "control")) for n, p in controls.items()]:
        Xi, Qt = transform_images(X, params), transform_texts(Q, params)
        S = Xi @ Qt.T  # cosine (both normalised)
        pairs = []
        for qi, q in enumerate(qspec["queries"]):
            want = set(q["phrases"])
            for reg in regions:
                idx = np.array(by_region[reg])
                rel = np.array([bool(feats[i] & want) for i in idx])
                if rel.sum() < MIN_REL:
                    continue
                top = idx[np.argsort(-S[idx, qi], kind="stable")[:K]]
                hit = np.mean([bool(feats[i] & want) for i in top])
                pairs.append({"query": q["query"], "group": q["group"], "region": reg, "p_at_10": float(hit),
                              "chance": float(rel.mean()), "n": int(len(idx)), "n_rel": int(rel.sum())})
        pairs_meta = pairs_meta or [(p["query"], p["region"]) for p in pairs]
        results["models"][name] = {"rotation_sha256": h, "pairs": pairs}
        print(f"{name}: {len(pairs)} (query, region) pairs", flush=True)

    # aggregate with region-cluster bootstrap
    def agg(pairs, group=None):
        ps = [p for p in pairs if group is None or p["group"] == group]
        regs = sorted({p["region"] for p in ps})
        byr = defaultdict(list)
        for p in ps:
            byr[p["region"]].append(p)
        def stat(sel):
            xs = [p for r in sel for p in byr[r]]
            return (np.mean([p["p_at_10"] for p in xs]), np.mean([p["chance"] for p in xs]))
        m, c = stat(regs)
        brng = np.random.default_rng(0)
        boots = [stat(list(brng.choice(regs, size=len(regs), replace=True))) for _ in range(N_BOOT)]
        lo, hi = np.percentile([b[0] for b in boots], [2.5, 97.5])
        return {"p_at_10": round(float(m), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "chance": round(float(c), 4),
                "lift": round(float(m / c), 2) if c else None, "pairs": len(ps), "regions": len(regs)}

    for name, mres in results["models"].items():
        mres["summary"] = {g or "all": agg(mres["pairs"], g) for g in (None, "open", "vocab")}
        perq = defaultdict(list)
        for p in mres["pairs"]:
            perq[p["query"]].append(p)
        mres["per_query"] = {q: {"p_at_10": round(float(np.mean([p["p_at_10"] for p in v])), 3),
                                 "chance": round(float(np.mean([p["chance"] for p in v])), 3), "regions": len(v)}
                             for q, v in perq.items()}
    (out / "search_benchmark_v1.json").write_text(json.dumps(results, indent=1) + "\n")

    # markdown table for the write-up
    lines = [f"Search benchmark v1: {len(rows):,} squares in {len(regions)} held-out regions; "
             f"{len(results['models'][next(iter(results['models']))]['pairs'])} (query, region) pairs; "
             f"relevance = mapped in OpenStreetMap; 95% CI by region bootstrap.", "",
             "| Model | All queries P@10 [95% CI] | Open-vocabulary P@10 | Land-cover P@10 | Chance (all) |",
             "|---|---|---|---|---|"]
    for name, mres in results["models"].items():
        s = mres["summary"]
        lines.append(f"| {name} | {s['all']['p_at_10']:.3f} [{s['all']['ci95'][0]:.3f}, {s['all']['ci95'][1]:.3f}] | "
                     f"{s['open']['p_at_10']:.3f} | {s['vocab']['p_at_10']:.3f} | {s['all']['chance']:.3f} |")
    lines += ["", "| Query | " + " | ".join(results["models"]) + " | chance |", "|---|" + "---|" * (len(results["models"]) + 1)]
    for q in qspec["queries"]:
        cells = [results["models"][n]["per_query"].get(q["query"], {}) for n in results["models"]]
        if not cells[0]:
            continue
        lines.append(f"| {q['query']} ({q['group']}, {cells[0]['regions']} regions) | " +
                     " | ".join(f"{c.get('p_at_10', float('nan')):.2f}" for c in cells) + f" | {cells[0]['chance']:.2f} |")
    (out / "search_benchmark_v1.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
