# Demo Script — Platonic Earth (≤ 2 minutes)

Owner: D · Source: `plan/09-demo-writeup-submission.md` §2 · Status: DRAFT (mock data)

Every number in brackets must come from a logged run in `plan/11-experiment-log.md`
before recording. Do not say a number that is not in `artifacts/metrics/`.

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

1. **Text search tab.** Type `dense forest` → Search.
   > "Here is the input: 'dense forest.' The tool returns the top three matching tiles with
   > similarity scores. These are similarity scores, not probabilities."
2. Click the `river corridor` example.
   > "Any plain-language query works without retraining."
3. **Classify a tile tab.** Upload the Upper Arlington tile.
   > "Here is an Upper Arlington tile. It predicts [label], with a warning that local imagery is
   > outside the training distribution."
4. **Results & limits tab.**
   > "We compare with a supervised ResNet-18 classifier, which scored [Y] on the held-out test
   > split. Our rotation scored [X] top-1, against [R] for a random rotation. ResNet is stronger
   > at fixed classes, but it cannot answer a new text query without retraining. Our method can."

## 1:15–1:45 — How it works

> "We use frozen DINOv2 for images and frozen MiniLM for text. We fit one closed-form
> Procrustes rotation on EuroSAT image–caption pairs, with no multimodal training. This tests
> the Platonic Representation Hypothesis: do independently trained models share a geometry of
> land cover?"

## 1:45–2:00 — Failure and next steps

**Screen:** failure example on the Results tab.

> "It fails on annual versus permanent crop, because the difference is crop lifecycle, not
> appearance. On local Upper Arlington tiles, accuracy dropped to [L] on N=50. Next, we would
> add local labels and temporal change detection."

## Claims to avoid

- "The AI understands land use."
- "Accurate land-use determination." / "Replaces GIS."
- "Beats the baseline" without a number.

## If results are weak

> "Our top-1 retrieval was [X], close to the random-rotation baseline of [R]. A single global
> rotation is not enough for fine-grained land-cover classes at this resolution. We report this
> as a negative result for PRH."
