"""Fill the number blocks in README.md and WRITEUP.md from the logged metrics (owner: D).

    .venv/bin/python scripts/sync_docs.py          # after: python -m web.build_static

Reads docs/api/meta.json (built from artifacts/metrics/ by web/build_static.py) and the saved Upper Arlington
climate stats, and rewrites everything between <!-- auto:NAME --> ... <!-- /auto:NAME --> markers.
Prose outside the markers carries no measured numbers, so one re-run keeps the docs honest.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = ROOT / "docs" / "api"


def load():
    meta = json.loads((API / "meta.json").read_text())
    st = json.loads((API / "static.json").read_text())
    reg = {r["key"]: r for r in st["regions"]}
    clim = lambda k: json.loads((API / "region" / reg[k]["id"] / "climate.json").read_text())["stats"] if k in reg else None
    return meta, clim("upper-arlington"), clim("phoenix")


def sg(v, d=2):
    return ("−" if v < 0 else "+") + f"{abs(v):.{d}f}"


def p(v, d=1):
    return "—" if v is None else f"{v * 100:.{d}f}%"


def blocks(meta, ua, ph):
    M = meta["metrics"]
    R, CC, L = M["retrieval"], M["caption_classification"], M["local"]
    cat = {r["local_category"]: r for r in M["local_by_category"]}
    w = cat.get("water_river", {})
    river_right, river_n = round(w.get("accuracy", 0) * w.get("n", 0)), w.get("n", 0)
    rc = next((c for c in M.get("top_confusions", []) if c["true"] == "River" and c["predicted"] == "Highway"), None)
    n_test = CC["procrustes"]["n_images"]
    reg, ter = ua["regression"], ua["tercile"]
    out = {}
    out["takeaway"] = (
        f"**One rotation joins two AI models that were never trained together.** Plain-English search finds the right "
        f"land type first in **{p(R['procrustes']['top1'], 0)}** of {R['procrustes']['n_captions']} searches over "
        f"{R['procrustes']['n_images']:,} unseen satellite photos (random rotation {p(R['random_rotation']['top1'])}, "
        f"no rotation {p(R['unaligned']['top1'], 0)}). A fully trained ResNet-18 labels photos better "
        f"({p(M['resnet18']['accuracy'])} vs our {p(CC['procrustes']['accuracy'])}) but cannot take a question in words. "
        f"In Upper Arlington it is a **tie** ({p(L['procrustes']['accuracy'], 0)} vs {p(L['oracle_majority_local']['accuracy'], 0)} "
        f"for always answering “Residential”), and the heat data says where trees matter: "
        f"**{sg(reg['slope_per_0p1_ndvi'])} °C per +0.1 greenness**, least-green third **{ter['hot_minus_green_c']:.1f} °C** hotter."
    )
    out["results"] = "\n".join([
        f"**1 · Search in words** (text → photos; {R['procrustes']['n_captions']} caption queries, each over all "
        f"{R['procrustes']['n_images']:,} held-out EuroSAT test photos; top-1 = the first photo has the right land type)",
        "",
        "| Method | Top-1 | Top-5 | Verdict |",
        "|---|---:|---:|---|",
        f"| **Procrustes rotation (ours)** | **{p(R['procrustes']['top1'], 0)}** | **{p(R['procrustes']['top5'], 0)}** | ahead |",
        f"| Random rotation (control, mean of 5 seeds) | {p(R['random_rotation']['top1'])} | {p(R['random_rotation']['top5'], 0)} | |",
        f"| No rotation (control) | {p(R['unaligned']['top1'], 0)} | {p(R['unaligned']['top5'], 0)} | |",
        "",
        f"**2 · Label each photo** (image → one of 10 land types; N = {n_test:,} held-out EuroSAT test photos)",
        "",
        "| Method | Accuracy | Macro-F1 | Verdict |",
        "|---|---:|---:|---|",
        f"| ResNet-18, fine-tuned on the labels (strong baseline) | {p(M['resnet18']['accuracy'])} | {p(M['resnet18']['macro_f1'])} | |",
        f"| **Procrustes rotation (ours, no training)** | **{p(CC['procrustes']['accuracy'])}** | {p(CC['procrustes']['macro_f1'])} | **behind ResNet**, far ahead of the controls |",
        f"| Random rotation (control) | {p(CC['random_rotation']['accuracy'])} | {p(CC['random_rotation']['macro_f1'])} | |",
        f"| Always the most common class (“{M['majority']['most_common_class']}”) | {p(M['majority']['accuracy'])} | {p(M['majority']['macro_f1'])} | |",
        "",
        f"**3 · Our own town** (N = {L['procrustes']['n']} Sentinel-2 tiles of Upper Arlington, Ohio, labelled by the team)",
        "",
        "| Method | All tiles | Clear-cut tiles | Verdict |",
        "|---|---:|---:|---|",
        f"| **Procrustes rotation (ours)** | **{p(L['procrustes']['accuracy'], 0)}** | {p(L['procrustes']['accuracy_unambiguous'])} | **tie** |",
        f"| ResNet-18 | {p(L['resnet18']['accuracy'], 0)} | {p(L['resnet18']['accuracy_unambiguous'])} | |",
        f"| Always “{L['oracle_majority_local']['label']}” (uses the local answers) | {p(L['oracle_majority_local']['accuracy'], 0)} | {p(L['oracle_majority_local']['accuracy_unambiguous'])} | |",
        "",
        f"**4 · Heat, measured** (Upper Arlington; Sentinel-2 greenness vs Landsat ground temperature; {reg['n']} land squares of 640 m)",
        "",
        f"- Cooling: **{sg(reg['slope_per_0p1_ndvi'])} °C per +0.1 NDVI** (straight-line fit, r = {sg(reg['r'])}).",
        f"- Least-green third vs greenest third: **{ter['least_green_mean']:.1f} °C vs {ter['greenest_mean']:.1f} °C** ({ter['n_each']} squares each)."
        + (f" Phoenix, Arizona: {sg(ph['regression']['slope_per_0p1_ndvi'])} °C per +0.1 NDVI (r = {sg(ph['regression']['r'])})." if ph else ""),
        "",
        f"**Where it fails:** rivers. {river_right} of {river_n} Upper Arlington river tiles get the right label; at 10 m per pixel a thin "
        f"river looks like a road" + (f" (River → Highway is also the top EuroSAT mix-up: {rc['count']} river photos, {p(rc['share_of_true'], 0)})." if rc else "."),
    ])
    out["writeup_results"] = out["results"]
    return out


def fill(path: Path, b: dict) -> int:
    if not path.exists():
        return 0
    s = path.read_text()
    n = 0
    for k, v in b.items():
        pat = re.compile(rf"(<!-- auto:{k} -->\n).*?(\n<!-- /auto:{k} -->)", re.S)
        s, c = pat.subn(lambda m: m.group(1) + v + m.group(2), s)
        n += c
    path.write_text(s)
    return n


if __name__ == "__main__":
    b = blocks(*load())
    for f in ("README.md", "WRITEUP.md"):
        print(f"[docs] {f}: {fill(ROOT / f, b)} block(s) updated")
