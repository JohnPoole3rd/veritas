# Platonic Earth — Full Project Spec

**Track:** AI + Climate
**Test site:** Upper Arlington, Ohio, USA
**Core method:** Frozen DINOv2 + frozen MiniLM + closed-form Procrustes
**Team:** J (integrator + data/alignment), A (baselines + local validation), D (demo + docs)

## 1. Problem statement

Upper Arlington is a built-out inner-ring suburb of Columbus with a mature urban forest. About **34.2%** of its land is covered by tree canopy, and a 2011–2021 assessment found that roughly **33% of land** could support additional tree canopy. The city is preparing an **Urban Forestry Plan** and updating its Master Plan and Parks & Recreation Plan in 2026.

A sustainability or urban-forestry staff member who wants to find candidate areas for canopy expansion or land-cover inspection cannot easily search satellite imagery in plain language. Existing supervised classifiers answer a fixed set of classes; they do not let the user ask a new question without retraining.

This project tests whether a **closed-form rotation** between frozen image and text encoders can support natural-language land-cover search without multimodal training.

## 2. User and moment

**Primary user:** City of Upper Arlington sustainability coordinator or urban-forestry staff member.

**Secondary user:** a community volunteer or student helping with a local tree-canopy or climate project.

**Moment:**

> “I need to find candidate areas in Upper Arlington for tree-canopy expansion or land-cover inspection, but I cannot browse every Sentinel-2 tile.”

**Decision supported:**

- Which neighborhoods or parks to inspect first.
- Where to prioritize a tree-planting site visit.
- Where to verify a land-cover change.
- Where to focus a community climate project.

**What the tool is not:**

- Not a canopy measurement system.
- Not a parcel-level land-use determination.
- Not a legal or regulatory tool.
- Not a replacement for GIS or a site visit.

## 3. Product requirements

### MVP

- [ ] Load EuroSAT and split train/val/test.
- [ ] Extract DINOv2 embeddings.
- [ ] Extract MiniLM caption embeddings.
- [ ] Fit Procrustes rotation.
- [ ] Retrieve top-3 tiles for a text query.
- [ ] Classify an uploaded tile by nearest caption.
- [ ] Report top-1/top-5 retrieval.
- [ ] Report majority and ResNet-18 baselines.
- [ ] Report random rotation.
- [ ] Collect 50 Upper Arlington tiles.
- [ ] Evaluate on local tiles.
- [ ] Show one failure case.
- [ ] Gradio demo.

### Stretch

- [ ] CLIP zero-shot baseline.
- [ ] Local tile map.
- [ ] Confidence threshold / abstention.
- [ ] Per-class failure gallery.
- [ ] Feedback from a real Upper Arlington contact.
- [ ] Compare DINOv2 ViT-S vs. ViT-B.

### Out of scope

- Audio.
- Multilingual.
- Three modalities.
- Object detection/segmentation.
- Temporal change detection.
- Real-time ingestion.
- Parcel-level canopy measurement.
- Fine-tuning encoders.

## 4. Input/output examples

### Example A — text search

**Input:** `dense forest`

**Output:**

| Rank | Tile | Label | Score |
|---|---|---|---|
| 1 | ua_031 | Forest | 0.82 |
| 2 | ua_034 | Forest | 0.79 |
| 3 | ua_028 | HerbaceousVegetation | 0.71 |

**UI:** image grid with labels and scores.

### Example B — image upload

**Input:** a residential Upper Arlington tile.

**Output:**

> Nearest label: Residential (0.74)
> Next: Industrial (0.69), Highway (0.61)
> Note: outside the EuroSAT distribution; treat as a hint.

### Example C — failure

**Input:** a EuroSAT `PermanentCrop` test tile.

**Output:** image→class prediction is `AnnualCrop` (45/375 PermanentCrop tiles are confused this way).

**Explanation:** at 64×64, orchards and seasonal fields are visually similar; the difference depends on crop lifecycle, not a single image. A text query for `permanent crop` often retrieves the class correctly; this is an image→class confusion, not a text-search failure.

## 5. Data schema

### Tile record

```text
tile_id        string
image_path     string
split          train | val | test | local
label          EuroSAT class or "Other"
local_category residential | commercial_industrial | road_highway | park_forest_herbaceous | water_river
source         EuroSAT | Copernicus
lat            float or null
lon            float or null
imagery_date   string or null
notes          string
```

### Caption record

```text
caption_id     string
label          EuroSAT class
text           string
template_id    int
```

### Result record

```text
run_id
method
encoder_image
encoder_text
alignment
split
top1
top5
macro_f1
test_n
notes
```

## 6. Model pipeline

```text
EuroSAT tiles ──► DINOv2 ──► image embeddings ─┐
                                               ├──► center ──► Procrustes ──► W
Class captions ──► MiniLM ──► text embeddings ─┘

Query text ──► MiniLM ──► cosine search over rotated image embeddings
Uploaded tile ──► DINOv2 ──► rotate ──► nearest class caption
```

### Why this tests PRH

If independently trained image and text encoders share a geometry of land cover, one linear rotation should align them. If they do not, retrieval collapses to chance. The project measures that directly.

### Why Procrustes

- Closed form.
- No training.
- One interpretable matrix.
- Fast enough for 48 hours.
- Standard representation-alignment tool.

## 7. Retrieval logic

1. Normalize embeddings.
2. Center training image and text embeddings with their training means.
3. Fit rotation \(W\).
4. At query time, embed text, subtract the training text mean, normalize.
5. Rotate test image embeddings with \(W\).
6. Score with cosine similarity.
7. Return top-k.

Use the **same training means** at fit and inference. Do not center a query with its own mean.

## 8. Confidence and abstention

MVP: show cosine similarity, not a calibrated probability.

Stretch:

- Low-confidence label if top-1 similarity is below a validation-tuned threshold.
- “Outside known classes” if the uploaded tile is far from all class centroids.
- Calibrate on validation, not test.

Do not claim certainty.

## 9. Demo screens

### Screen 1 — Text search

- Search box.
- Example queries: `dense forest`, `residential neighborhood`, `park or grassland`, `commercial area`, `river corridor`.
- Top-3 image grid.
- Similarity scores.

### Screen 2 — Image upload

- Drag-and-drop tile.
- Predicted label.
- Top-3 labels with scores.
- “Prototype, not a land-use determination.”

### Screen 3 — Results

- Baseline comparison: majority, ResNet-18, Procrustes.
- Failure gallery: one image + explanation.
- One-sentence takeaway.

## 10. Evaluation summary

Full details in `07-evaluation-plan.md`.

- **Primary:** text→image top-1 on the held-out EuroSAT test split.
- **Secondary:** top-5, image→text per-class accuracy, macro-F1.
- **Local:** 50 Upper Arlington tiles, out-of-distribution accuracy.
- **Required comparison:** majority and ResNet-18.
- **Alignment control:** random rotation.

## 11. Failure cases

| Failure | Why | How to show |
|---|---|---|
| Annual vs. permanent crop | Similar texture/color at 64×64 | Confusion matrix |
| Forest vs. herbaceous vegetation | Both green; density differs | Example pair |
| Industrial vs. residential | Both built-up | Example pair |
| Clouds | EuroSAT mostly cloud-free; local may not be | Local failure |
| Seasonal snow | Not in training distribution | Local failure |
| Abstract query | “Climate change” is not a land-cover class | Text query failure |
| Local “Other” | EuroSAT has no unknown class | Abstention example |

## 12. What to tell the judges

**Problem:** Upper Arlington sustainability staff cannot search satellite imagery in plain language.

**What we built:** text-to-satellite land-cover search using frozen DINOv2 and MiniLM aligned by Procrustes.

**What we added:** local Upper Arlington tiles, a PRH research question, and a comparison against a supervised classifier.

**Result:** retrieval beats random rotation by X; ResNet-18 is stronger for fixed-class classification; text search is the unique capability.

**Failure:** river channel → Highway; abstract queries; local domain shift; annual vs. permanent crop as a secondary in-distribution confusion.

**Next:** more local labels, temporal change detection, a remote-sensing-specific text encoder.

## 13. Novelty statement

> We apply the Platonic Representation Hypothesis to Upper Arlington land-cover search: a single closed-form rotation fitted on labelled EuroSAT image–caption pairs enables natural-language text-to-image retrieval, evaluated on EuroSAT and on 50 local Sentinel-2 tiles. Unlike the suggested supervised land-use project, the user can ask new questions in natural language without retraining.

## 14. Ethical and accuracy notes

- Prototype, not a land-use determination.
- Do not use for permitting, taxation, or legal decisions.
- Report local sample size honestly.
- Do not claim the model “understands” land use.
- Do not collect personal data.
- Cite EuroSAT, DINOv2, MiniLM, Copernicus, and any AI tools used.
