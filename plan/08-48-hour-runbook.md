# 48-Hour Runbook — Platonic Earth

**Start:** Saturday, November 14, 2026, 12:00 AM ET.
**Deadline:** Sunday, November 15, 2026, 11:59 PM ET.
**Project:** Platonic Earth, Upper Arlington, Ohio.
**Rule:** submit a complete first version at T+45.

Use `T+0` to `T+48` so the plan works in any timezone.

## 1. Decision gates

| Gate | When | Question | If no |
|---|---|---|---|
| G0 | T+0:30 | Track and project locked? | Default: AI + Climate, Platonic Earth |
| G1 | T+2:00 | Does EuroSAT load? | Use HF mirror; switch after 30 min |
| G2 | T+6:00 | Do embeddings extract? | Use 5k subset or CPU |
| G3 | T+10:00 | Does Procrustes beat random rotation? | Keep classifier demo; alignment as ablation |
| G4 | T+16:00 | Is the ResNet-18 baseline done? | Use logistic regression or majority only |
| G5 | T+24:00 | Are 50 Upper Arlington tiles collected? | Use 20–30 tiles; report N |
| G6 | T+30:00 | Does the Gradio demo run end-to-end? | Cut stretch features |
| G7 | T+42:00 | Is the write-up done? | Stop coding; write |
| G8 | T+45:00 | Is the video recorded? | Skip video; submit code + write-up |
| G9 | T+47:00 | Is the submission complete? | Submit what works |

## 2. Team roles

### 1-person team

- Data + embeddings first.
- Procrustes + evaluation second.
- Local tiles third.
- Simplest Gradio app.
- Write-up while final metrics run.
- Cut all stretch goals.

### 2-person team

- **Agent A:** data, embeddings, Procrustes.
- **Agent B:** ResNet-18, local tiles, Gradio.
- Both: write-up, video, failure analysis.

### 3-person team

- **Agent A:** data, embeddings, Procrustes, retrieval eval.
- **Agent B:** baselines, local tiles, metrics, figures.
- **Agent C:** Gradio, visuals, write-up, video.
- All: review results and claims.

## 3. Hour-by-hour plan

### T+0 to T+2 — Setup and data gate

- [ ] Create shared Colab/GitHub.
- [ ] Confirm track: AI + Climate.
- [ ] Confirm test site: Upper Arlington.
- [ ] Assign roles in `TASKS.md`.
- [ ] Load EuroSAT via torchvision.
- [ ] Print class counts, image sizes, sample tiles.
- [ ] Create stratified train/val/test split.
- [ ] Save split JSON.
- [ ] Open Copernicus Browser and locate Upper Arlington.

**Deliverable:** `01_data.ipynb` with visible outputs.

**If blocked:** HF mirror or TFDS. If still blocked after 30 min, switch dataset or project.

### T+2 to T+6 — Embeddings

- [ ] Load DINOv2 ViT-S/14.
- [ ] Extract CLS embeddings for a 5k subset.
- [ ] Load MiniLM.
- [ ] Encode 5 captions per class.
- [ ] Check shapes and norms.
- [ ] Save embeddings to `artifacts/embeddings/`.

**Deliverable:** cached embeddings + sanity check.

**If blocked:** CPU on 2k tiles.

### T+6 to T+10 — Alignment sanity

- [ ] Center embeddings.
- [ ] Fit Procrustes.
- [ ] Evaluate top-1 on validation.
- [ ] Compare with random rotation.
- [ ] Compare with unaligned cosine.
- [ ] Inspect 5 retrieved tiles visually.

**Deliverable:** first table.

**Decision:** if at chance, debug centering/normalization/caption mapping. If still at chance, keep classifier demo and present alignment as a negative result.

### T+10 to T+16 — Full run and ResNet-18

- [ ] Run all 27k tiles if time allows.
- [ ] Train ResNet-18.
- [ ] Evaluate ResNet-18 on test.
- [ ] Evaluate majority baseline.
- [ ] Save metrics CSV.
- [ ] Save confusion matrix.

**Deliverable:** main results table.

### T+16 to T+24 — Upper Arlington tiles

- [ ] Use `14-upper-arlington-data-collection.md`.
- [ ] Download 50 Sentinel-2 tiles.
- [ ] Clip to city boundary.
- [ ] Label 10 per category.
- [ ] QA 10 tiles with a second person.
- [ ] Save manifest.
- [ ] Run Procrustes and ResNet on local tiles.

**Deliverable:** local tile manifest + local results.

### T+24 to T+30 — Demo

- [ ] Build Gradio app:
  - text box,
  - top-3 image grid,
  - image upload,
  - label output,
  - baseline comparison,
  - failure note.
- [ ] Add example query buttons.
- [ ] Add prototype disclaimer.
- [ ] Test with 3 queries and 2 uploads.

**Deliverable:** demo runs end-to-end.

### T+30 to T+36 — Failure analysis and figures

- [ ] Pick one clear failure.
- [ ] Explain why it fails.
- [ ] Build confusion matrix figure.
- [ ] Build 3–5 example gallery.
- [ ] Save figures.
- [ ] Update experiment log.

**Deliverable:** failure table + figures.

### T+36 to T+42 — Write-up

- [ ] Fill the brief’s template.
- [ ] Problem, user, input/output.
- [ ] Data and model.
- [ ] Results table.
- [ ] Failure case.
- [ ] What we added.
- [ ] Next steps.
- [ ] Sources and credits.
- [ ] One page.

**Deliverable:** write-up link opens in incognito.

### T+42 to T+45 — Video

- [ ] Record 2 minutes.
- [ ] Problem: 15 seconds.
- [ ] Demo: 60 seconds.
- [ ] How it works: 30 seconds.
- [ ] Next steps: 15 seconds.
- [ ] Upload unlisted.
- [ ] Check link.

**Deliverable:** video link.

### T+45 to T+47 — Submission

- [ ] Code link opens.
- [ ] Write-up link opens.
- [ ] Video link opens.
- [ ] Team details correct.
- [ ] Track = AI + Climate.
- [ ] Credits complete.
- [ ] Submit.

### T+47 to T+48 — Buffer

- [ ] Fix links.
- [ ] Resubmit if needed.
- [ ] No new features.

## 4. Scope cuts

Cut in this order:

1. CLIP baseline.
2. Map view.
3. Confidence threshold.
4. Failure gallery.
5. Video.
6. ViT-B comparison.
7. Local tile count from 50 to 30.

Never cut:

1. Held-out evaluation.
2. Majority baseline.
3. ResNet-18 baseline.
4. Random rotation.
5. One failure case.
6. Working demo.
7. Write-up and credits.
8. Link check.

## 5. Do-not-do list

- Do not train a multimodal model.
- Do not add audio.
- Do not add multilingual.
- Do not add a third modality.
- Do not use private data.
- Do not claim parcel-level canopy measurement.
- Do not tune on test.
- Do not compare methods on different splits.
- Do not hide weak results.
- Do not spend >30 min on a download.
- Do not start a new idea after T+12.

## 6. Sync questions

At T+12 and T+24:

1. What works now?
2. What is blocked?
3. What is the next smallest deliverable?
4. What will we cut if behind?

## 7. Submission buffer

The brief allows resubmission. Submit at T+45. Improve only if time remains.
