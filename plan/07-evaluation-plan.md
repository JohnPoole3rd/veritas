# Evaluation Plan

**Owner:** A
**Tasks:** T08, T09, T12, T13; reviews T07.

## 1. Research questions

**RQ1.** Does a closed-form Procrustes rotation between frozen DINOv2 and MiniLM embeddings support zero-shot text-to-satellite land-cover retrieval?

**RQ2.** How does Procrustes retrieval compare with a supervised ResNet-18 classifier and a majority baseline?

**RQ3.** Does the alignment generalize to 50 Upper Arlington Sentinel-2 tiles outside the EuroSAT distribution?

**RQ4.** Which classes, local categories, and query types fail, and why?

## 2. Hypotheses

- **H1:** Procrustes retrieval beats a random rotation on top-1 and top-5.
- **H2:** ResNet-18 outperforms Procrustes on fixed-class classification.
- **H3:** Procrustes supports text queries that ResNet-18 cannot, which is the unique capability.
- **H4:** Upper Arlington tiles show a domain-shift drop.
- **H5:** Similar classes (annual vs. permanent crop, forest vs. herbaceous vegetation) are the main confusions.

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
| E0 | Data sanity | EuroSAT | counts, duplicates | counts | TODO |
| E1 | Embedding sanity | 1k tiles | DINOv2 + MiniLM | shapes, norms | TODO |
| E2 | Alignment sanity | train/val | Procrustes | top-1 vs random | TODO |
| E3 | Full retrieval | test | Procrustes | top-1, top-5, per-class | TODO |
| E4 | Majority baseline | test | majority | accuracy, macro-F1 | TODO |
| E5 | ResNet-18 baseline | test | supervised | accuracy, macro-F1 | TODO |
| E6 | Random rotation | test | shuffled Procrustes | top-1, top-5 | TODO |
| E7 | Unaligned cosine | test | raw cosine | top-1, top-5 | TODO |
| E8 | CLIP zero-shot | test | CLIP | top-1, top-5 | optional |
| E9 | Upper Arlington local | local | Procrustes + ResNet | accuracy, examples | TODO |
| E10 | Failure taxonomy | test/local | manual review | table | TODO |

## 7. Metric file contract

Exact columns the demo reads:

`artifacts/metrics/majority.csv`:

```csv
method,accuracy,macro_f1,test_n,notes
majority,0.1111,0.0200,4050,majority_class=SeaLake
```

`artifacts/metrics/resnet18.csv`:

```csv
method,accuracy,macro_f1,test_n,notes
resnet18,0.96,0.95,4050,epochs=10
```

`artifacts/metrics/retrieval.csv`:

```csv
method,split,top1,top5,macro_f1,n_images,test_n,n_captions,notes
procrustes,test,0.94,0.96,,512,512,50,512-tile subset
```

`artifacts/metrics/local.csv`:

```csv
tile_id,true_label,local_category,predicted_label,score,correct
ua_001,Residential,residential,Residential,0.74,True
```

If a file is missing, the demo shows `—`; do not invent a number.

## 8. Results tables

### Table 1 — Main results

| Method | Top-1 | Top-5 | Macro-F1 | Test N | Notes |
|---|---:|---:|---:|---:|---|
| Majority class | | | | | |
| ResNet-18 | | | | | |
| Unaligned cosine | | | | | |
| Random rotation | | | | | |
| Procrustes (ours) | | | | | |
| CLIP zero-shot | | | | | optional |

### Table 2 — Per-class retrieval

| Class | Top-1 | Top-5 | Most confused with |
|---|---:|---:|---|
| AnnualCrop | | | |
| Forest | | | |
| HerbaceousVegetation | | | |
| Highway | | | |
| Industrial | | | |
| Pasture | | | |
| PermanentCrop | | | |
| Residential | | | |
| River | | | |
| SeaLake | | | |

### Table 3 — Upper Arlington local results

| Local category | N | Procrustes acc. | ResNet-18 acc. | Notes |
|---|---:|---:|---:|---|
| Residential | 10 | | | |
| Commercial/industrial | 10 | | | |
| Road/highway | 10 | | | |
| Park/forest/herbaceous | 10 | | | |
| Water/river | 10 | | | |
| **Total** | **50** | | | |

### Table 4 — Local tile detail

| Tile ID | True label | Local category | Procrustes | ResNet-18 | Notes |
|---|---|---|---|---|---|
| ua_001 | | | | | |
| ... | | | | | |

### Table 5 — Failure taxonomy

| Failure type | Example | Count | Likely cause | Fix idea |
|---|---|---:|---|---|
| Annual vs. permanent crop | | | similar texture | higher resolution |
| Forest vs. herbaceous | | | similar color | multi-band input |
| Industrial vs. residential | | | built-up similarity | more context |
| Cloud/snow | | | not in training | local fine-tune |
| Abstract query | | | not a land-cover class | query rewriting |
| Domain shift | | | Upper Arlington vs. EuroSAT | local labels |

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
| Procrustes top-1 30–50% | Plausible for frozen zero-shot; report honestly |
| Procrustes top-1 > 50% | Strong for this setup |
| Procrustes close to ResNet-18 | Surprising; check leakage |
| Procrustes beats ResNet-18 | Check leakage before celebrating |
| Local accuracy much lower | Expected domain shift; good failure story |

## 11. Negative-result framing

If Procrustes fails:

> “We tested whether PRH holds strongly enough between DINOv2 and MiniLM for satellite land-cover retrieval. It does not at this resolution: top-1 was X, close to the random-rotation baseline. A supervised ResNet-18 reaches Y. This suggests a single global rotation is insufficient for fine-grained land-cover classes, even if it helps for coarse categories.”

This is a legitimate finding and satisfies the brief’s honest-results rule.

## 12. Evaluation checklist

- [ ] Split created before fitting.
- [ ] Test split untouched until final evaluation.
- [ ] Majority baseline reported.
- [ ] ResNet-18 baseline reported.
- [ ] Random rotation reported.
- [ ] Unaligned cosine reported.
- [ ] Top-1 and top-5 reported.
- [ ] Per-class results reported.
- [ ] Confusion matrix saved.
- [ ] Upper Arlington N = 50 reported.
- [ ] Local category table reported.
- [ ] One failure example explained.
- [ ] Results CSV saved.
- [ ] Model names and versions recorded.
