# Evaluation Plan

**Owner:** A
**Tasks:** T08, T09, T12, T13; reviews T07.

## 1. Research questions

**RQ1.** Does a closed-form Procrustes rotation between frozen DINOv2 and MiniLM embeddings support natural-language text-to-satellite land-cover retrieval after fitting on labelled image–caption pairs?

**RQ2.** How does Procrustes retrieval compare with a supervised ResNet-18 classifier and a majority baseline?

**RQ3.** Does the alignment generalize to 50 Upper Arlington Sentinel-2 tiles outside the EuroSAT distribution?

**RQ4.** Which classes, local categories, and query types fail, and why?

## 2. Hypotheses

- **H1:** Procrustes retrieval beats a random rotation on top-1 and top-5.
- **H2:** ResNet-18 outperforms Procrustes on fixed-class classification.
- **H3:** Procrustes supports text queries that ResNet-18 cannot, which is the unique capability.
- **H4:** Upper Arlington tiles show a domain-shift drop.
- **H5:** Water/linear features (River→Highway) and similar vegetation classes (annual vs. permanent crop, forest vs. herbaceous) are the main confusions; water is the largest.

H2 is not a failure. The value is text search, not beating a supervised classifier at its own task.

## 3. Baselines

### Required

1. **Majority class** — predict the most common EuroSAT class.
2. **ResNet-18** — supervised classifier trained on the training split.

### For the alignment claim

3. **Random rotation** — fit Procrustes on shuffled image–text pairs.
4. **Unaligned cosine** — raw DINOv2 vs. MiniLM similarity.

### Optional

5. **CLIP zero-shot** — strong external baseline.
6. **Procrustes + class centroids** — stability ablation.
7. **DINOv2 ViT-B vs. ViT-S** — model-size ablation.

## 4. Metrics

### Retrieval

- Top-1 accuracy.
- Top-5 accuracy.
- Per-class top-1.
- Mean reciprocal rank (optional).

### Classification

- Accuracy.
- Macro-F1.
- Per-class recall.
- Confusion matrix.

### Upper Arlington local

- Accuracy on 50 tiles.
- Per-local-category accuracy:
  - residential,
  - commercial/industrial,
  - road/highway,
  - park/forest/herbaceous,
  - water/river.
- Per-mapped-EuroSAT-class accuracy.
- Qualitative success and failure examples.

### Calibration / abstention

- Optional: top-1 similarity threshold vs. accuracy.
- Choose threshold on validation, not test.

## 5. Splits

| Split | Share | Use |
|---|---:|---|
| Train | 70% | Fit Procrustes; train ResNet-18 |
| Validation | 15% | Captions, model size, threshold |
| Test | 15% | Final numbers only |
| Upper Arlington local | 50 tiles | OOD case study only |

Rules:

- Never fit on test.
- Never tune on test.
- Never fit on local.
- Deduplicate before splitting.
- Report exact split sizes.

## 6. Experiment matrix

| ID | Experiment | Data | Method | Metric | Status |
|---|---|---|---|---|---|
| E0 | Data sanity | EuroSAT | counts, duplicates | counts | DONE |
| E1 | Embedding sanity | 1k tiles | DINOv2 + MiniLM | shapes, norms | DONE |
| E2 | Alignment sanity | train/val | Procrustes | top-1 vs random | DONE |
| E3 | Full retrieval | test | Procrustes | top-1, top-5, per-class | DONE |
| E4 | Majority baseline | test | majority | accuracy, macro-F1 | DONE |
| E5 | ResNet-18 baseline | test | supervised | accuracy, macro-F1 | DONE |
| E6 | Random rotation | test | shuffled Procrustes | top-1, top-5 | DONE |
| E7 | Unaligned cosine | test | raw cosine | top-1, top-5 | DONE |
| E8 | CLIP zero-shot | test | CLIP | top-1, top-5 | OPTIONAL — not run |
| E9 | Upper Arlington local | local | Procrustes + ResNet | accuracy, examples | DONE |
| E10 | Failure taxonomy | test/local | manual review | table | DONE |

## 7. Metric file contract

Exact columns the demo reads:

`artifacts/metrics/majority.csv`:

```csv
method,accuracy,macro_f1,test_n,most_common_class,notes
majority,0.1111,0.0200,4050,AnnualCrop,train count 2100/18900; seed 42
```

`artifacts/metrics/resnet18.csv`:

```csv
method,accuracy,macro_f1,test_n,epochs,notes
resnet18,0.9817,0.9812,4050,10,best_epoch=10
```

`artifacts/metrics/retrieval.csv`:

```csv
method,split,top1,top5,macro_f1,n_images,test_n,n_captions,notes
procrustes,test,0.96,1.0,,4050,4050,50,full test; A MPS rerun of J scripts
random_rotation,test,0.18,0.34,,4050,4050,50,full test
unaligned,test,0.10,0.34,,4050,4050,50,full test
```

`artifacts/metrics/caption_classification_test.csv`:

```csv
method,split,accuracy,top5,macro_f1,n_images,n_captions,notes
procrustes,test,0.8121,0.9901,0.7948,4050,50,"image->class, agg=max"
random_rotation,test,0.1590,0.6588,0.1559,4050,50,"image->class, agg=max"
unaligned,test,0.0827,0.5326,0.0720,4050,50,"image->class, agg=max"
```

`artifacts/metrics/local.csv`:

```csv
tile_id,true_label,local_category,predicted_label,score,correct
ua_001,Residential,residential,Residential,0.3834,True
```

`artifacts/metrics/local_summary.csv` (demo table):

```csv
method,accuracy,n,accuracy_unambiguous,n_unambiguous,notes
procrustes,0.64,50,0.8108,37,Upper Arlington OOD
resnet18,0.64,50,0.8108,37,Upper Arlington OOD
majority_local,0.62,50,0.7838,37,constant Residential
```

`artifacts/metrics/local_predictions.csv` supplies the demo's success/failure tiles.

If a file is missing, the demo shows `—`; do not invent a number.

## 8. Results tables

### Table 1a — Text→image search (50 caption queries)

| Method | Top-1 | Top-5 | Queries |
|---|---:|---:|---:|
| Procrustes | 0.96 | 1.00 | 50 |
| Random rotation | 0.18 | 0.34 | 50 |
| Unaligned cosine | 0.10 | 0.34 | 50 |

### Table 1b — Image→class (4,050 test images)

| Method | Accuracy | Top-5 | Macro-F1 | Test N |
|---|---:|---:|---:|---:|
| Majority class | 0.1111 | — | 0.0200 | 4050 |
| ResNet-18 | 0.9817 | — | 0.9812 | 4050 |
| Unaligned cosine | 0.0827 | 0.5326 | 0.0720 | 4050 |
| Random rotation | 0.1590 | 0.6588 | 0.1559 | 4050 |
| Procrustes (ours) | 0.8121 | 0.9901 | 0.7948 | 4050 |
| CLIP zero-shot | — | — | — | optional |

Do not compare the 50-query search table with the per-image classification table.

### Table 2 — Per-class Procrustes image→class recall

| Class | Recall | Most confused with |
|---|---:|---|
| AnnualCrop | 0.9200 | |
| Forest | 0.9689 | |
| HerbaceousVegetation | 0.9289 | |
| Highway | 0.7840 | |
| Industrial | 0.9653 | |
| Pasture | 0.5867 | |
| PermanentCrop | 0.5973 | |
| Residential | 0.8489 | |
| River | 0.4613 | Highway |
| SeaLake | 0.9111 | |

Detailed per-class files: `procrustes_test_per_class.csv`, `procrustes_test_top_confusions.csv`.

### Table 3 — Upper Arlington local results

| Local category | N | Procrustes acc. | ResNet-18 acc. | Local majority |
|---|---:|---:|---:|---:|
| Residential | 10 | 1.00 | 1.00 | 1.00 |
| Commercial/industrial | 10 | 0.90 | 1.00 | 0.70 |
| Road/highway | 10 | 0.90 | 0.80 | 0.90 |
| Park/forest/herbaceous | 10 | 0.40 | 0.30 | 0.50 |
| Water/river | 10 | 0.00 | 0.10 | 0.00 |
| **Total** | **50** | **0.64** | **0.64** | **0.62** |

On the 37 unambiguous tiles, both Procrustes and ResNet-18 reach 0.8108; local majority reaches 0.7838.

### Table 4 — Local tile detail

| Tile ID | True label | Local category | Procrustes | ResNet-18 | Notes |
|---|---|---|---|---|---|
| ua_001 | | | | | |
| ... | | | | | |

### Table 5 — Failure taxonomy

Filled from real predictions (T13, `artifacts/metrics/failure_taxonomy.csv`; figures `failure_gallery.png`, `local_confusion.png`).

| Failure type | Example | Count | Likely cause | Fix idea |
|---|---|---:|---|---|
| Water read as road (linear feature) | ua_049, ua_041 | local 6/10 water → Highway (Procrustes); EuroSAT River→Highway 122 (32 % of River) | thin dark channel at 10 m; "river"/"road" captions close | NIR/NDWI band; water captions not tied to shape |
| Managed turf → forest/pasture | ua_036–ua_038 (golf) | ResNet Forest on 3/4 Herbaceous, mean p = 1.00 | saturated irrigated turf matches no EuroSAT class | local labels; calibration |
| Canopy vs residential | ua_030, ua_032–ua_034 | ResNet Forest on 4/31 Residential; labellers split on ua_034 and ua_035 | 34 % canopy hides houses at 10 m | fractional canopy cover output |
| Commercial strip vs residential | ua_013 | Procrustes 1/3 Industrial missed; EuroSAT Residential→Industrial 40 | small US strip centres vs large EU industrial | larger tiles / local examples |
| Annual vs permanent crop (in-distribution) | EuroSAT test | PermanentCrop→AnnualCrop 45 (12 %) | phenology, not texture | multi-date imagery |
| Uninformative confidence | all local | Procrustes score 0.377 right vs 0.371 wrong; ResNet p 0.82 when wrong | OOD, uncalibrated | no score threshold; show top-3 + disclaimer |

## 9. Statistical honesty

- Report test N.
- Report local N = 50 as a case study, not a benchmark.
- Do not overclaim small differences.
- Do not compare methods on different splits.
- Do not report only the best run.
- If results are weak, say so.

## 10. Success thresholds

| Result | Interpretation |
|---|---|
| Procrustes top-1 ≤ random + 5 points | Alignment likely not working; debug |
| Procrustes top-1 30–50% | Plausible for frozen, label-fitted alignment; report honestly |
| Procrustes top-1 > 50% | Strong for this setup |
| Procrustes close to ResNet-18 | Surprising; check leakage |
| Procrustes beats ResNet-18 | Check leakage before celebrating |
| Local accuracy much lower | Expected domain shift; good failure story |

## 11. Negative-result framing

If Procrustes fails:

> “We tested whether PRH holds strongly enough between DINOv2 and MiniLM for satellite land-cover retrieval. It does not at this resolution: top-1 was X, close to the random-rotation baseline. A supervised ResNet-18 reaches Y. This suggests a single global rotation is insufficient for fine-grained land-cover classes, even if it helps for coarse categories.”

This is a legitimate finding and satisfies the brief’s honest-results rule.

## 12. Evaluation checklist

- [x] Split created before fitting.
- [x] Test split untouched until final evaluation.
- [x] Majority baseline reported.
- [x] ResNet-18 baseline reported.
- [x] Random rotation reported.
- [x] Unaligned cosine reported.
- [x] Top-1 and top-5 reported.
- [x] Per-class results reported.
- [x] Confusion matrix saved.
- [x] Upper Arlington N = 50 reported.
- [x] Local category table reported.
- [x] One failure example explained.
- [x] Results CSV saved.
- [x] Model names and versions recorded.
