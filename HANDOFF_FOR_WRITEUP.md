# Platonic Earth — Handoff for the Write-up and Video

Prepared by agent **D** (demo + documentation lead, Claude Opus 5.5) on 2026-11-14, ~21:10 ET.
Numbers checked against `main` at commit `310a5a6` (after the team's code review, which regenerated all metrics).
For: the humans and the AI assistant writing the one-page write-up and the 2-minute video.

**Deadline: Sunday, November 15, 2026, 11:59 PM ET.** Submit a complete version by Sunday afternoon, then resubmit
improvements. Judges see only the latest version.

Every number below comes from a logged result file in the repo; the source is named next to it. Do not round
differently, invent numbers, or quote a number without its N.

---

## 1. The project in one paragraph

**Platonic Earth** (Veritas AI High School Hackathon 2026, **AI + Climate** track). We test whether two AI models that
were trained separately (one on images, **DINOv2**, and one on text, **MiniLM**) can be connected by **one simple
closed-form math step, a rotation (orthogonal Procrustes)**, so that a person can search satellite images in plain
English ("dense forest", "river corridor") with **no AI training at all**. The intended user is **Upper Arlington, Ohio
sustainability / urban-forestry staff** looking for places to inspect for tree-canopy expansion (the city has 34.2%
canopy and a 2026 Urban Forestry Plan). We fit the rotation on the public **EuroSAT** dataset (27,000 Sentinel-2
tiles, 10 land-cover classes), compare it fairly against simple and strong baselines, and test it on **50 real
Sentinel-2 tiles of Upper Arlington** that the team collected and labelled.

**Research question (Platonic Representation Hypothesis, PRH):** do independently trained models share a common
"geometry" of the world, so that one rotation lines them up?

## 2. Team and who did what

| Agent | Role | Built |
|---|---|---|
| **J** | Integrator + data/alignment | Repo scaffold, EuroSAT loading and splits, captions, DINOv2/MiniLM embeddings, the Procrustes rotation, retrieval evaluation (T01–T07). Merges to `main`. |
| **A** | Baselines + local validation | ResNet-18 and majority baselines; collected and labelled 50 Upper Arlington Sentinel-2 tiles; local evaluation; failure analysis and figures (T08–T13). |
| **D** | Demo + docs | Gradio demo website (T14), demo script, and exploratory experiments 2–4 (below). The write-up and video (T15/T16) are being done by humans. |

The team's real names, emails, and schools go on the submission form. **Everyone listed must have contributed.**

## 3. What goes in, what comes out (the demo)

Run: `python app.py` (after J's pipeline; see `results/T14.md`). Opens at http://127.0.0.1:7860. Works offline after
`python app.py --prefetch`.

- **Text search tab:** type `river corridor` → the top 3 matching **Upper Arlington** tiles (all three are real river tiles),
  with similarity scores. You can switch to searching the 4,050 EuroSAT test tiles.
- **Classify a tile tab:** upload a tile → nearest land-cover label + top 3. Built-in examples: ua_011 (shopping centre
  → **Industrial**, correct) and ua_049 (Scioto River → **Highway**, wrong).
- **Results & limits tab:** three fair comparison tables and the success/failure pair.
- Every screen says **"Prototype, not a land-use determination."** Scores are cosine similarity, **not probabilities**.

## 4. Main results (the submission — numbers on `main`)

### 4.1 EuroSAT held-out test, label each tile (N = 4,050, same task for every row)
Source: `results/T08.md`, `results/T09.md`, `results/T12.md`, `artifacts/metrics/*.csv`

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| Majority class (always "AnnualCrop") | 11.1% | 2.0% |
| No rotation (control) | 8.3% | 7.2% |
| Random rotation (control, mean of 5 seeds) | 15.5% | 13.9% |
| **Procrustes rotation (ours)** | **81.2%** | **79.5%** |
| ResNet-18, fine-tuned on the labels (A) | 98.2% | 98.1% |

### 4.2 Text search (what ResNet cannot do)
Source: `results/T07.md`, `artifacts/metrics/retrieval.csv` (post code review). 50 caption queries, each searching all 4,050 test tiles.
Top-1 = the best tile has the right class.

| Method | Top-1 | Top-5 |
|---|---:|---:|
| **Procrustes (ours)** | **96%** | **100%** |
| Random rotation (mean of 5 seeds) | 15.2% | 28% |
| No rotation | 10% | 34% |

**Important:** 96% is *search* (48 of 50 queries), not "96% of tiles labelled correctly". Per-tile labelling is 81.2%.
An older 512-tile subset number (94%) is obsolete and must not be quoted.

### 4.3 Upper Arlington, out of distribution (N = 50 human-labelled tiles)
Source: `results/T12.md`, `artifacts/metrics/local_summary.csv`. Sentinel-2 L2A, 2025-08-01, 0.24% scene cloud.

| Method | All 50 | 37 clear-cut tiles |
|---|---:|---:|
| **Procrustes (ours)** | **64%** | 81.1% |
| ResNet-18 | 64% | 81.1% |
| Always guess "Residential" (*oracle*: uses the test labels to pick the class) | 62% | 78.4% |
| Always guess EuroSAT's most common class ("AnnualCrop"), the honest no-peeking baseline | 0% | 0% |

Labels: 31 Residential, 10 River, 4 Herbaceous, 3 Industrial, 1 Forest, 1 Highway. Second-labeller QA agreement 6/10
(the second labeller was an AI subagent, not a human). Per category (N=10 each, ours / ResNet / majority):
residential 100/100/100, commercial 90/100/70, road 90/80/90, park 40/30/50, **water 0/10/0**.

**Honest reading:** both models beat a constant "Residential" guess by **one tile**. That is a case study, not a
benchmark. Note that "always Residential" is an *oracle* baseline: you only know to guess Residential by looking at the
local answers. The no-peeking baseline scores 0%.

### 4.4 Where it fails (required by the brief)
Source: `results/T13.md`, `artifacts/figures/failure_gallery.png`

1. **Rivers read as roads** (the demo failure, ua_049): at 10 m per pixel the Scioto River is a thin line, like a road.
   6 of 10 local water tiles → Highway; on EuroSAT, River→Highway is the #1 confusion (122/375 = 32% of river tiles).
2. **Golf turf → Forest:** ResNet calls golf-course tiles Forest with p ≈ 1.00, confidently wrong.
3. **Tree canopy hides houses:** wooded neighbourhoods get confused with forest (human labellers also disagreed).
4. **Commercial strip vs residential.**
5. **Crop types:** PermanentCrop→AnnualCrop 45 times (12%); the difference is crop lifecycle, not appearance.
6. **Confidence is uninformative:** mean top score 0.377 when right vs 0.371 when wrong. The tool cannot say "I don't know".
   Abstract queries like `climate change` still return tiles, with scores ≈ 0.05.

Success example: ua_011 (big-box retail) → Industrial (0.557) by ours and by ResNet (0.86).

## 5. How to frame it (important for judges)

- ✅ **Do say:** "We tested whether a cheap, training-free method can let city staff search satellite images with
  plain words. On held-out European images one rotation labels 81% of tiles correctly (vs 11% for guessing and 15.5% for a
  random rotation) and finds the right land cover first in 96% of searches. A fully trained ResNet labels 98%, but it
  cannot answer a new text query. On our own town both models barely beat guessing 'Residential' (64% vs 62%), and we
  show exactly why: at 10 m a river looks like a road."
- ❌ **Don't say:** "The AI understands land use", "accurate land-use determination", "replaces GIS", "solves tree
  planting", "beats the baseline" without numbers, "96% accurate", or that we built ResNet / DINOv2 / MiniLM.
- **Be honest that it's not yet a practical city tool:** 640 m tiles are too coarse for street-level planting decisions,
  the city fits only ~61 such tiles, and the city already has higher-resolution canopy data. The contribution is the
  **experiment and its honest evaluation**, which the brief rewards ("Judges score an honest, modest result higher
  than an unexplained high one").
- **"Make it your own" (required by the brief), all three kinds:** own user/setting (Upper Arlington urban forestry),
  own data (50 collected and labelled local tiles), own question (PRH: does one rotation suffice? plus text search).
  We started from the suggested project 4.3.3 (land-use mapping with EuroSAT + ResNet-18), which is now our baseline.

## 6. Extra experiments (exploratory; NOT in the submitted code)

Note: Exp 2–4 were run on code from before the team's 17:27 code review. Their main-method rows reproduce the official
numbers (81.2% / 64%), but treat them as exploratory.

These are on separate GitHub branches, not merged into `main`, and not logged in `plan/11-experiment-log.md`. Present
them as **"what we tried next"**, clearly labelled as additional experiments.

### Exp 2 — Sharper pictures (branch `d/exp-highres`)
0.3 m aerial imagery: UC Merced (training) + USDA NAIP 2023 (Upper Arlington).
- UC Merced held-out test (21 classes): **90.7%** labelling (random rotation 5.7%; majority 2.6%).
- Same 50 Upper Arlington places, 5 broad groups, whole-tile vote: **68% sharp = 68% blurry** vs 62% guessing. Sharper
  pictures alone did not help whole-tile labels.
- But inside tiles: all 3 golf courses found (53–55 of 64 pieces "golf course"); water shows up in 12.4% of pieces on
  river tiles vs 0.6% elsewhere. One label per 640 m square hides small features.

### Exp 3 — DINOv3 and SkyScript (branch `d/exp-dinov3-skyscript`)
- Swapping **DINOv2 → DINOv3** (same size, same speed) with the same EuroSAT rotation: **81.2% → 88.4%** on EuroSAT
  (about 38% fewer mistakes); Upper Arlington 64% → 66% (one tile, noise).
- Fitting on **SkyScript** (30,000 real caption–image pairs, mostly sub-metre aerial): great for sharp images (UC
  Merced 12% → 58%) and natural-sentence search (recall@10 1.7% → 18.8% of 2,000), but **collapses on 10 m imagery**
  (Upper Arlington 10%). Training data must match the image resolution.
- Speed, same Mac GPU: ~39–40 images/s for both models; one upload ~20–24 ms; one text query ~7 ms; search over 4,050
  tiles 0.2 ms.

### Exp 4 — Six improvement ideas (branch `d/exp-improvements`)

| Version | EuroSAT (4,050) | Upper Arlington (50) | Bigger local set, map labels (131) |
|---|---:|---:|---:|
| **main:** DINOv2 + rotation | 81.2% | 64% | 61.8% |
| DINOv3 + rotation | 88.4% | 66% | 67.9% |
| DINOv2 + flexible "ridge" map | 94.1% | 62% | **71.8%** |
| **DINOv3 + flexible "ridge" map** | **95.1%** | **76%** | 71.0% |
| DINOv3 + Sentinel-2-only SkyScript rotation | 49.8% | 64% | 69.5% |
| ResNet-18 (reference) | 98.2% | 64% | — |
| Always guess biggest group (oracle for local sets) | 11.1% | 62% | 30.5% |

What each idea did:
1. **Sentinel-2-only SkyScript (18,705 pairs):** fixes the SkyScript collapse (Upper Arlington 10% → 64%), confirming the
   resolution diagnosis, but does not beat plain EuroSAT fitting.
2. **Satellite-trained DINOv3:** not run (weights not available yet).
3. **Richer captions** (12 per class instead of 5): **no effect** (±1 point).
4. **Bigger balanced local test:** 131 tiles from the same Sentinel-2 scene (city + 4 km around), auto-labelled from ESA
   WorldCover + OpenStreetMap (independent of our models).
5. **80 m pieces (0.3 m NAIP, 3,200 pieces):** finding **open grass** (plantable space) scores **F1 0.62–0.72**; best
   precision 0.81 (DINOv3 + ridge).
6. **Flexible "ridge" map instead of one rotation: the biggest win** (+12–13 points on EuroSAT; Upper Arlington up to 76%).

**How to use Exp 4 in the write-up:** don't replace the main method. Add a short "beyond one rotation" note: *"One
rotation reaches 81% (88% with DINOv3). A slightly more flexible linear map, still training-free and fit in seconds,
reaches 95%, close to a fully trained ResNet (98%) while keeping free-text search. On Upper Arlington it reaches 76%,
the first version clearly above guessing (62%)."* State that the flexible map **no longer tests the
'one rotation' question** and that these are exploratory results on separate branches.

All Exp 4 choices were made on validation data; test and local sets were scored once. Caveats: map labels are
automatic (about half of "green" tiles are houses under tree canopy); only 12 of 131 bigger-set tiles are inside city
limits; single seed.

## 7. What we'd do next (evidence-backed)

1. **Map small 80 m pieces with sharp aerial imagery** ("show me open grass with no trees"): the real planting
   question. Exp 4 already finds open grass with F1 ≈ 0.7.
2. **Use DINOv3 and a flexible linear map** (Exp 3–4): 81% → 95% on EuroSAT.
3. **More, balanced local labels** checked by humans (currently 31/50 Residential; ±1 tile is noise).
4. **A "water present" rule and abstention** ("I don't know") for rivers and abstract queries.

## 8. Write-up template (from the brief; about one page)

```
# Platonic Earth
**Track:** Climate     **Team:** <names>
## The problem        — who (Upper Arlington urban-forestry staff), why (finding canopy-expansion / inspection sites)
## What we built      — input (text or tile) → output (top-3 tiles / land-cover label + scores); prototype disclaimer
## What we added      — started from suggested project 4.3.3; added own user/place, own data (50 tiles), own question (PRH + text search)
## How it works       — EuroSAT 27,000 tiles (18,900/4,050/4,050, seed 42); frozen DINOv2 ViT-S/14 + MiniLM; one Procrustes rotation; baselines
## Results            — tables 4.1–4.3; one failure (river→highway) and why
## What we'd do next  — section 7
## Sources            — section 10
```

## 9. Video (≤ 2 minutes, optional but recommended)

Brief structure: problem ~15 s · demo ~60 s · how it works + baseline ~30 s · limitation + next ~15 s.
A draft script with the numbers is in `assets/demo_script.md`. Suggested demo path: Text search on Upper Arlington →
`river corridor` (3 rivers) → Classify ua_049 (→ Highway, explain why) → Results tab tables. A screen recording with
voice-over is enough. Upload unlisted to YouTube and check the link in an incognito window.

## 10. Credits (all required by the brief; verify links before submitting)

- **EuroSAT:** Helber et al., 2019. https://github.com/phelber/EuroSAT (loaded via torchvision)
- **DINOv2:** Oquab et al., TMLR 2024. https://github.com/facebookresearch/dinov2 (`facebook/dinov2-small`, Apache-2.0)
- **MiniLM:** https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 (Apache-2.0)
- **ResNet-18:** He et al., 2016, "Deep Residual Learning". https://arxiv.org/abs/1512.03385; ImageNet weights via torchvision
- **Platonic Representation Hypothesis:** Huh et al., 2024. https://arxiv.org/abs/2405.07987
- **Sentinel-2 imagery:** Copernicus, ESA (via Earth Search STAC, https://earth-search.aws.element84.com/v1)
- **OpenStreetMap** contributors (ODbL) (city boundary, tile selection); **ESA WorldCover 2021** (CC-BY 4.0)
- **Libraries:** PyTorch, torchvision, Hugging Face Transformers, Sentence-Transformers, scikit-learn, SciPy, NumPy, pandas, Gradio
- **Upper Arlington canopy/plan facts** (34.2% canopy, ~33% possible additional canopy, 2026 Urban Forestry Plan): see
  `plan/12-glossary-references.md`. **Confirm these sources before quoting.**
- **AI tools:** Claude (Anthropic), used by the J, A, and D agents to write code and docs; plus any other AI tools the team used.
- *If Exp 2–4 are mentioned:* UC Merced Land Use (Yang & Newsam, 2010); USDA NAIP via Microsoft Planetary Computer;
  SkyScript (Wang et al., AAAI 2024, https://github.com/wangzhecheng/SkyScript); **DINOv3 (Siméoni et al., 2025,
  arXiv:2508.10104), used under Meta's DINOv3 License, which requires acknowledging its use** (not open source; weights not
  redistributed).

## 11. Pre-submission checklist

- [ ] `main` on GitHub has the final code; the repo is public and opens in an incognito window.
- [ ] README on `main` explains what it is and how to run it (J updated it; skim it).
- [ ] Write-up is ~1 page, includes the disclaimer line, numbers with N, one failure, and credits.
- [ ] Track = **Climate**; every teammate's name, email, and school is spelled correctly.
- [ ] Video (if any) ≤ 2 min, unlisted, opens in incognito.
- [ ] First version submitted by Sunday afternoon; resubmit later if improved.

## 12. Where things are

| Thing | Location |
|---|---|
| Plan, rules, decisions | `plan/` (`AGENTS.md`, `DECISIONS.md`, `TASKS.md`) |
| Task reports with numbers | `results/T01.md` … `results/T14.md` |
| Metrics files | `artifacts/metrics/` |
| Failure / confusion figures | `artifacts/figures/` |
| Demo | `app.py` (main) · comparison demo: `experiments/skyscript/app_compare.py` (Exp 3 branch) |
| Demo script draft | `assets/demo_script.md` |
| Learnings | `plan/learnings/` |
| Exp 2 / 3 / 4 | branches `d/exp-highres`, `d/exp-dinov3-skyscript`, `d/exp-improvements` (each has a README) |
