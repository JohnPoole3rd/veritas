# Demo, Write-up, and Submission

**Owner:** D for T14–T16; J for T17.
**Tasks:** T14, T15, T16, T17
**Inputs:** T06/T07 from J, T12/T13 from A.

## 1. Gradio demo

### Layout

```text
┌─────────────────────────────────────────────────────┐
│  Platonic Earth — Upper Arlington                   │
│  Search Sentinel-2 land cover in plain language.    │
│  Prototype: not a land-use determination.           │
├─────────────────────────────────────────────────────┤
│  [ dense forest                           ] [Search]│
│  Examples: residential | park | commercial | river  │
├─────────────────────────────────────────────────────┤
│  Top results                                        │
│  ┌────────┐  ┌────────┐  ┌────────┐                │
│  │ tile 1 │  │ tile 2 │  │ tile 3 │                │
│  │ forest │  │ forest │  │ herb.  │                │
│  │ 0.82   │  │ 0.79   │  │ 0.71   │                │
│  └────────┘  └────────┘  └────────┘                │
├─────────────────────────────────────────────────────┤
│  Upload a local tile:                               │
│  [ drop image ] → Predicted: Residential (0.74)     │
├─────────────────────────────────────────────────────┤
│  Baseline comparison                                │
│  Majority: X% | ResNet-18: Y% | Ours: Z% top-1     │
│  Failure: annual vs. permanent crop                 │
└─────────────────────────────────────────────────────┘
```

### Demo rules

- Precompute the gallery embeddings, class captions, and \(W\).
- The text query still goes through MiniLM at runtime, and an uploaded tile still goes through DINOv2.
- Pre-warm both models before the demo; the demo must work with Wi-Fi off.
- Show one success and one failure.
- Label the prototype.
- Use real EuroSAT or Upper Arlington tiles.
- Keep under two minutes.
- Never screenshot `--mock` output for the write-up.

### Exact demo commands

Mock rehearsal:

```sh
python app.py --mock
```

Real mode after J's artifacts exist:

```sh
python app.py
```

Expected real-mode log:

```text
[demo] REAL mode — gallery N=512 of 4050, transforms from src.alignment
[demo] NOTE — using 512-tile stratified subset
```

If it says `MOCK mode`, the missing artifact list names the blocker.

Pre-warm the runtime encoders and test offline:

```sh
python app.py --prefetch
HF_HUB_OFFLINE=1 python app.py
```

### Baseline table source

The demo reads:

- `artifacts/metrics/majority.csv` (fallback: `majority.json`)
- `artifacts/metrics/resnet18.csv`
- `artifacts/metrics/caption_classification_test.csv` — image→class rows, comparable with ResNet
- `artifacts/metrics/retrieval.csv` — text search row, 50 caption queries

If a file is missing, the table shows `—`; do not replace it with a made-up number. Do not compare the 50-query search row with per-image ResNet accuracy.

### Example queries

- `dense forest`
- `residential neighborhood`
- `park or grassland`
- `commercial area`
- `river corridor`
- `open water`

### Example failure

**Query:** `permanent crop`
**Top result:** `annual crop`
**Explanation:** “At 64×64, orchards and seasonal fields look similar. The difference depends on crop lifecycle, which a single tile does not show.”

### Upper Arlington example

**Input:** a residential tile from Old Arlington.
**Output:** “Residential (0.74); next: Industrial (0.69). Outside the EuroSAT distribution; treat as a hint.”

## 2. Two-minute video script

### 0:00–0:15 — Problem

> “Upper Arlington is a built-out suburb with a 34% tree canopy and a new Urban Forestry Plan. Staff need to find candidate areas for canopy expansion, but browsing thousands of satellite tiles is slow.”

### 0:15–1:15 — Demo

> “Here is the input: ‘dense forest.’ The tool returns the top three matching tiles with confidence scores.”
>
> “Here is an Upper Arlington tile. It predicts residential, with a warning that local imagery is outside the training distribution.”
>
> “We compare with a supervised ResNet-18 classifier. ResNet is stronger at fixed classes, but it cannot answer a new text query without retraining. Our rotation can.”

### 1:15–1:45 — How it works

> “We use frozen DINOv2 for images and frozen MiniLM for text. We fit a closed-form Procrustes rotation on EuroSAT image-caption pairs. No multimodal training. This tests the Platonic Representation Hypothesis: do independently trained models share a geometry of land cover?”

### 1:45–2:00 — Failure and next steps

> “It fails on annual versus permanent crop, because the difference is crop lifecycle, not appearance. Next, we would add local labels and temporal change detection.”

## 3. Write-up template

```markdown
# Platonic Earth

**Track:** AI + Climate
**Team:** J, A, D

## The problem
Upper Arlington sustainability and urban-forestry staff need to find candidate areas for tree-canopy expansion or land-cover inspection, but browsing thousands of Sentinel-2 tiles is slow. Existing classifiers answer a fixed set of classes and cannot be queried in plain language.

## What we built
A Gradio app that takes a text query or a satellite tile and returns matching land-cover tiles with confidence scores. It uses frozen DINOv2 and MiniLM embeddings aligned by a closed-form Procrustes rotation. No multimodal training.

## What we added
We started from the suggested land-use mapping project and added:
- a natural-language search question,
- a PRH alignment test,
- 50 local Sentinel-2 tiles from Upper Arlington,
- a comparison against a supervised classifier.

## How it works
Data: EuroSAT (27,000 tiles, 10 classes). We used [N] training, [N] validation, [N] test. Local tiles: 50 from Copernicus Browser, clipped to the Upper Arlington boundary.
Model: DINOv2 ViT-S/14 (frozen), all-MiniLM-L6-v2 (frozen), orthogonal Procrustes. Baselines: majority class and ResNet-18.

## Results

### Image→class (4,050 test images)

| Method | Accuracy | Top-5 | Macro-F1 |
|---|---:|---:|---:|
| Majority | 0.1111 | — | 0.0200 |
| ResNet-18 | 0.9817 | — | 0.9812 |
| Random rotation | 0.1590 | 0.6588 | 0.1559 |
| Unaligned cosine | 0.0827 | 0.5326 | 0.0720 |
| Procrustes | 0.8121 | 0.9901 | 0.7948 |

### Text→image search (50 caption queries)

| Method | Top-1 | Top-5 |
|---|---:|---:|
| Procrustes | 0.96 | 1.00 |
| Random rotation | 0.18 | 0.34 |
| Unaligned cosine | 0.10 | 0.34 |

Procrustes is well above the null baselines but below supervised ResNet-18. The 50-query search numbers are not comparable with per-image ResNet accuracy.

On 50 Upper Arlington tiles, Procrustes and ResNet-18 both reached 0.64 overall and 0.8108 on the 37 unambiguous tiles; local majority was 0.62.

Where it fails: water read as road, managed turf vs forest/pasture, canopy vs residential, commercial strip vs residential, crop-type confusion, uninformative confidence.
Why: at 10 m, thin water channels and mixed suburban land cover are ambiguous; EuroSAT classes do not match Upper Arlington exactly.

## What we'd do next
1. Add more local labels.
2. Add temporal change detection.
3. Try a remote-sensing-specific text encoder.

## Sources
- EuroSAT: [link]
- DINOv2: [link]
- MiniLM: [link]
- Copernicus Browser: [link]
- Upper Arlington GIS/canopy context: [link]
- Gradio: [link]
- AI tools used: [list]
```

## 4. Submission checklist

### Code

- [ ] GitHub repo or Colab notebook.
- [ ] Link opens in incognito.
- [ ] Notebook outputs visible.
- [ ] `README.md` with run instructions.
- [ ] `requirements.txt` or install cells.
- [ ] No secrets or API keys.

### Write-up

- [ ] One page.
- [ ] Track = AI + Climate.
- [ ] Team names.
- [ ] Problem, user, input/output.
- [ ] Data and model.
- [ ] Baseline table.
- [ ] Failure case.
- [ ] What we added.
- [ ] Next steps.
- [ ] Sources.

### Video

- [ ] ≤2 minutes.
- [ ] Problem, demo, how it works, next steps.
- [ ] Unlisted link.
- [ ] Opens in incognito.

### Team

- [ ] Track.
- [ ] Every teammate’s name.
- [ ] Email.
- [ ] School.
- [ ] Everyone contributed.

### Credits

- [ ] EuroSAT.
- [ ] Sentinel-2 / Copernicus.
- [ ] DINOv2.
- [ ] MiniLM.
- [ ] PyTorch.
- [ ] Hugging Face.
- [ ] scikit-learn.
- [ ] SciPy.
- [ ] Gradio.
- [ ] Colab.
- [ ] AI coding assistant.
- [ ] Tutorial code.
- [ ] Upper Arlington canopy/GIS sources.

## 5. Judge Q&A

### “Why not just use CLIP?”

> “CLIP is jointly trained on image-text pairs. We test whether a closed-form rotation between independently trained models is enough. That is the PRH question.”

### “Why is ResNet-18 better?”

> “ResNet is trained on the exact labels and optimized for fixed-class classification. Our method fits one rotation on labelled EuroSAT image–caption pairs, but it can answer a new text query without retraining. ResNet cannot.”

### “Is this just a tutorial?”

> “The suggested project is supervised land-use classification. We added a text-search question, a PRH alignment test, Upper Arlington tiles, and a cross-modal baseline comparison.”

### “How do you know it is not overfitting?”

> “We split before fitting, never tuned on test, compared against a random rotation, and tested on local tiles outside the EuroSAT distribution.”

### “What is PRH?”

> “The idea that independently trained models converge toward a shared representation of reality. If true, a simple rotation should align their spaces. We test that.”

### “What would you do with more time?”

> “More local labels, temporal change detection, and a remote-sensing-specific text encoder.”

## 6. Link hygiene

Open every link in incognito:

- code,
- notebook,
- write-up,
- video,
- dataset credits.

Check:

- no sign-in,
- no permission request,
- no broken image,
- no missing output.

## 7. If results are weak

Say:

> “Our top-1 retrieval was X, close to the random-rotation baseline. This suggests a single global rotation is not enough for fine-grained land-cover classes. The supervised baseline reached Y. We report this as a negative result for PRH at this resolution.”

The brief rewards honest, modest results over unexplained high ones.

## 8. Final takeaway

> “A closed-form rotation can turn frozen satellite and text models into a natural-language land-cover search tool — but it is not a replacement for supervised classification, and it fails on classes that differ by lifecycle rather than appearance.”
