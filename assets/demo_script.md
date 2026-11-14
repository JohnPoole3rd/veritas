# Demo Script — Platonic Earth (≤ 2 minutes)

Owner: D · Source: `plan/09-demo-writeup-submission.md` §2 · Status: READY (full metrics)

Every number must come from `results/T07.md` and `artifacts/metrics/`. Do not invent a number.

Numbers below use A's full-split rerun of J's scripts:
- text→image search: 50 caption queries;
- image→class: 4,050 test images.

## Setup (before recording)

- [ ] `python app.py` loads with Wi-Fi off (no internet needed).
- [ ] Mock banner is gone (real artifacts are wired in).
- [ ] "Results & limits" tab shows real numbers, not `—`.
- [ ] Have one Upper Arlington tile ready to upload (from A's T12 success pick).
- [ ] Have a fallback screen capture recorded.

## 0:00–0:15 — Problem

> "Upper Arlington is a built-out suburb with a 34% tree canopy and a new Urban Forestry Plan.
> Staff need to find candidate areas for canopy expansion, but browsing thousands of satellite
> tiles is slow."

**Screen:** app title + disclaimer.

## 0:15–1:15 — Demo

1. **Upper Arlington gallery (default).** Type `commercial area` → Search.
   > "The default gallery is Upper Arlington. Here is the input: 'commercial area.' The tool returns
   > the top three matching local tiles with similarity scores. These are similarity scores, not
   > probabilities."
2. Switch to the **EuroSAT test tiles** gallery and type `dense forest`.
   > "The EuroSAT gallery shows the model's broader land-cover vocabulary: dense forest returns
   > Forest tiles."
3. Back on the Upper Arlington gallery, click the `river corridor` example.
   > "River corridor returns river tiles; any plain-language query works without retraining."
4. **Classify a tile tab.** Upload the Upper Arlington tile.
   > "Here is an Upper Arlington tile. It predicts [label], with a warning that local imagery is
   > outside the training distribution."
5. **Results & limits tab.**
   > "ResNet-18 is the supervised baseline: 98.2% per-image accuracy. Our rotation is 81.2% on the
   > same 4,050 test images, with 99.0% top-5 and 79.5% macro-F1. That is the fair comparison.
   > Separately, when we search with each of our 50 class captions, the correct class is top-1
   > 96% of the time (48 of 50), versus 18% for a random rotation. Those are 50 queries, not
   > 4,050 images. ResNet is stronger at fixed classes, but it cannot answer a new text query
   > without retraining. Our method can."

## 1:15–1:45 — How it works

> "We use frozen DINOv2 for images and frozen MiniLM for text. We fit one closed-form
> Procrustes rotation on labelled EuroSAT image–caption pairs, with no multimodal training.
> This tests the Platonic Representation Hypothesis: do independently trained models share a
> geometry of land cover?"

## 1:45–2:00 — Failure and next steps

**Screen:** "Where it fails" on the Results tab (computed live; read what it actually shows).

> "Ask it something abstract, like 'climate change', and it still returns tiles, with very low
> scores and unrelated classes. It has no way to say 'I don't know.' It also mixes up permanent
> and annual crops, because the difference is the crop lifecycle, not how one image looks.
> On 50 local Upper Arlington tiles, accuracy dropped to 64% for both Procrustes and ResNet,
> only 2 points above the local majority of 62%. Next, we would add local labels and temporal
> change detection."

## Claims to avoid

- "The AI understands land use."
- "Accurate land-use determination." / "Replaces GIS."
- "Beats the baseline" without a number.
- Comparing the 50-query text search 96% directly with per-image ResNet 98.2%.
- Calling the rotation "zero-shot."

## If results are weak

> "Our image→class accuracy was 81.2%, below ResNet's 98.2%. A single global rotation is not a
> replacement for supervised classification. On local tiles both models dropped to 64%.
> We report this honestly."
