# Benchmark plan: show what the product actually does

Owner: A (proposal) · for J and the team to review · status: **proposal, not implemented**
Builds on `benchmark.py` (J's fixed-data harness: B1–B4, controls, hashed eval files). That harness stays; this plan changes *what we measure* and *what we headline*.

## 1. Why the current numbers don't show merit

Our product is **text → places**: a forestry officer types "parking lots and big roofs" and gets the matching 640 m squares inside a radius. Every current benchmark measures something else: **image → one label**, scored against labels that OpenStreetMap generated automatically.

From `~/agent_chat/share/bench/*.json` on john-pc:

| Problem | Evidence |
|---|---|
| We grade against noisy labels, not against what is visible. OSM labels describe what is *mapped*: an unmapped forest with a mapped stream is labelled "water". | OSM held-out accuracy is 0.32 (14 classes) for the 1M rotation, against 0.07 for a random rotation. It beats the control by a wide margin, but nobody can tell whether 0.32 is good. |
| More data looks worse, with no way to say whether that is noise. | Same 830 test chips: 6-class macro-F1 is **0.338** for the 91k rotation and **0.284** for the 1M one. EuroSAT 6-class macro-F1 is 0.588 vs 0.480. There are no confidence intervals, and the test is only 830 chips. |
| The Upper Arlington score is mostly "always say Residential". | 1M rotation: accuracy 0.68, but **balanced accuracy 0.20** and **macro-F1 0.13**. Answering "Residential" every time scores 0.62. |
| The real advantage is never tested. | Every suite uses EuroSAT's 10 classes or a 6-class merge of them. Nothing asks for "parking lots", "sports fields", "golf course" or "solar farm", which is the reason to use words at all. |
| There is no baseline a user would actually compare us to. | Random rotation and identity are sanity controls, not competitors. A forestry officer's real alternative is an existing **land-cover map** (ESA WorldCover) or scrolling the imagery by eye. |

## 2. The proposal: three numbers that map to the product

### Headline 1: "Does it find what you ask for?" (retrieval, human-checked)
- **Queries.** We freeze ~30 plain-English queries, written *before* we look at any results, in two groups:
  - **In-vocabulary (12):** things a land-cover map also has, e.g. "dense forest", "river", "farm fields", "residential streets", "industrial area", "lake".
  - **Open-vocabulary (18):** things no 10-class map has, e.g. "parking lots", "big flat roofs", "sports fields", "golf course", "solar farm", "rail yard", "airport runway", "construction site", "orchard", "cemetery", "quarry", "marina", "highway interchange".
- **Places.** 6 fixed study areas that **no rotation was trained on**: the test-split regions in `regions.csv` (one per continent), plus Upper Arlington. Each is a fixed 2 km radius around a fixed point. The list and its SHA-256 are committed under `bench/`.
- **Run.** For each query, rotation and place, take the **top 10** squares.
- **Judge.** Two teammates mark each returned square **yes / no / unsure** from the RGB chip plus the basemap, *blind to which model returned it*. Squares are shuffled and pooled across models, as in TREC pooling.
  - Scale: 30 queries × 6 places × 10 squares × 3 models ≈ 5,400 judgments before pooling, ~2,000 after de-duplication. That is about 1 h each with a keyboard-only labelling page.
  - **Fallback if time is short:** 10 queries × 3 places.
- **Metric.** **Precision@10**: the share of the top 10 a human says matches. Report it separately for in-vocabulary and open-vocabulary queries, with a 95% bootstrap CI clustered by place.
- **Also report** inter-annotator agreement (Cohen's κ) on the doubly-judged squares.
- **Models compared** (same places, same queries):
  1. **Chance:** the share of matching squares in the area. A random top-10 equals the base rate, which we estimate from the judged pool.
  2. **EuroSAT rotation:** our original model (`artifacts/alignment/procrustes.npz`).
  3. **OSM-Sentinel 1M rotation:** the product (`colab/artifacts/osm_1m_rotation.npz`, sha `a47275412f7a`).
  4. *(optional)* the 91k rotation, to show whether scale helped.
- **Land-cover-map baseline** (in-vocabulary queries only): rank squares by the ESA WorldCover share of the matching class. For open-vocabulary queries WorldCover **cannot answer at all**. We show that as "n/a" on purpose, because it is our merit.

**What a judge sees:** "When you ask for *parking lots* near you, X of the top 10 really are parking lots (chance: Y). A land-cover map can't answer that question."

### Headline 2: "Does it generalise to new places?" (classification, cleaned up)
- Keep J's **B3**, the fixed held-out test regions, but report it in the harmonised **6 classes**, with **macro-F1 + balanced accuracy only** (no raw accuracy).
- Add **95% CIs clustered by region**.
- Score on a **WorldCover-agreed subset**: only chips where the OSM label and the ESA WorldCover majority class agree. This removes most "mapped ≠ visible" noise, and we report how many chips it keeps.
- Keep the three controls (identity, random rotation, shuffled labels) in the same table, since they prove the rotation is doing the work.

### Headline 3: "Is it practical?" (operations)
Measured on the live site with the 1M rotation:
- **Time from "Analyse" to answers** for a 2 km radius, as a median over the 6 places. The Nairobi test took 15.4 s end to end, including Sentinel-2 download, embedding and Landsat heat.
- **Search latency** per query.
- Hardware: Mac M1 Pro, plus the Colab GPU/CPU.
- Squares per km² and the maximum area: a 6 km radius stays within the 400-square limit.

## 3. Supporting results (in the write-up, not headlines)
- **Scale curve:** Headline 2 macro-F1 for the 91k, 137k, 578k and 1M rotations on the **same** B3 chips, with CIs. If the curve is flat or falling, we say so and give the likely cause (label noise, ~35k unique captions out of 829k).
- **Upper Arlington case study:** a per-class confusion table plus Headline 1's queries in Upper Arlington. No single accuracy number: at n=50 it is ±0.13 and dominated by Residential.
- **Failure gallery:** for each model, the 3 worst queries with their top-5 squares, e.g. rivers returned as roads.

## 4. Rules that keep it consistent across development
1. **Frozen evaluation inputs.** The query list, place list, B3 chips and WorldCover-agreed subset are committed with SHA-256 hashes. Any change creates a new version (`v2`), never an edit.
2. **Write queries before looking at results.** This prevents picking queries that make us look good.
3. **One result JSON per model.** Fields: git hash, rotation sha, eval-file hashes, metrics + CI. A script builds the leaderboard table and plots from those files, so no number is typed by hand.
4. **Blind judging.** Judges never see which model returned a square.
5. **Every table has a baseline next to our score.** Chance, the land-cover map (where it applies) and the EuroSAT rotation.

## 5. Work and order
| # | Task | Owner (proposed) | Size |
|---|---|---|---|
| 1 | Freeze queries + places (`bench/queries_v1.json`, `bench/places_v1.json`) | team (humans write the queries) | 20 min |
| 2 | `bench_retrieval.py`: run each rotation on each place via the web backend (`/api/region` + `/api/region/{id}/search`) and dump the top-10 per query | A | 1 h |
| 3 | Judging page (static HTML, keyboard y/n/u, shuffled pooled squares) + judgments CSV | A | 1 h |
| 4 | Judging | two teammates | ~1 h each |
| 5 | WorldCover-agreed B3 subset + clustered CIs in `benchmark.py` | J | 1 h |
| 6 | Leaderboard/plot generator from result JSONs | J | 45 min |
| 7 | Write-up: three headline numbers, scale curve, failure gallery | A + team | 45 min |

**Minimum version, if only one thing gets done:** steps 1–4 for 10 queries × 3 places. That gives the one number that proves the product works: open-vocabulary Precision@10 against chance, with EuroSAT shown for comparison.
