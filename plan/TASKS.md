# TASKS.md — Task Board

Status values: `TODO`, `IN PROGRESS`, `BLOCKED`, `DONE`.
Owner values: `Agent A`, `Agent B`, `Agent C`, `Human`, or a name.

## Critical path

```text
T01 → T02 → T04 → T06 → T07 → T10 → T12 → T14 → T15 → T17
        └→ T03 → T05 ┘
        └→ T08 → T09 ┘
```

Do not start a downstream task until the upstream artifact exists.

## Task table

| ID | Task | Owner | Status | Depends on | Output |
|---|---|---|---|---|---|
| T01 | Repo scaffold and config | Agent A | TODO | — | `configs/default.yaml`, `src/`, `requirements.txt` |
| T02 | EuroSAT loader and splits | Agent A | TODO | T01 | `data/splits/*.json` |
| T03 | Caption templates | Agent A | TODO | T01 | `data/captions.jsonl` |
| T04 | DINOv2 embedding extraction | Agent A | TODO | T02 | `artifacts/embeddings/dinov2_*.npy` |
| T05 | MiniLM caption embeddings | Agent A | TODO | T03 | `artifacts/embeddings/minilm_captions.npy` |
| T06 | Procrustes fit and save | Agent A | TODO | T04, T05 | `artifacts/alignment/procrustes.npz` |
| T07 | Retrieval evaluation + random baseline | Agent B | TODO | T06 | `artifacts/metrics/retrieval.csv` |
| T08 | ResNet-18 baseline | Agent B | TODO | T02 | `artifacts/metrics/resnet18.csv` |
| T09 | Majority baseline | Agent B | TODO | T02 | `artifacts/metrics/majority.csv` |
| T10 | Upper Arlington tile collection | Agent C | TODO | T01 | `data/raw/local_tiles/` |
| T11 | Local tile labeling + QA | Agent C | TODO | T10 | `data/local_tiles/manifest.csv` |
| T12 | Local OOD evaluation | Agent B | TODO | T11, T06 | `artifacts/metrics/local.csv` |
| T13 | Failure taxonomy + figures | Agent B | TODO | T07, T12 | `artifacts/figures/`, failure table |
| T14 | Gradio demo | Agent C | TODO | T06, T07 | `app.py` |
| T15 | Write-up | Human + Agent C | TODO | T07, T12, T13 | one-page write-up |
| T16 | Demo video | Human | TODO | T14, T15 | unlisted video link |
| T17 | Submission checklist | Human | TODO | T15, T16 | submitted form |
| T18 | Final agent review | Human | TODO | T17 | final check |

## Task details

### T01 — Repo scaffold and config

**Inputs:** plan docs.
**Outputs:**

- `configs/default.yaml`
- `src/` modules
- `requirements.txt`
- `.gitignore`
- `README.md` for the implementation repo

**Acceptance:**

- `python -c "import src"` works.
- Config has paths, model names, seed, split ratios.
- No absolute paths.
- `.gitignore` excludes `data/` and `artifacts/embeddings/`.

### T02 — EuroSAT loader and splits

**Inputs:** `configs/default.yaml`.
**Outputs:**

- `data/splits/train.json`
- `data/splits/val.json`
- `data/splits/test.json`
- class count report

**Acceptance:**

- 27k tiles loaded.
- Class counts printed.
- Split is stratified and seeded.
- No duplicate tile IDs across splits.
- Test split untouched.

**Gate:** if this takes more than 30 minutes, use the Hugging Face mirror or switch dataset.

### T03 — Caption templates

**Inputs:** EuroSAT classes.
**Outputs:** `data/captions.jsonl` with 5–10 captions per class.
**Acceptance:**

- Every class has at least 5 captions.
- Captions are class-consistent.
- No caption mentions a class it does not belong to.

### T04 — DINOv2 embeddings

**Inputs:** EuroSAT splits, DINOv2 model.
**Outputs:**

- `artifacts/embeddings/dinov2_train.npy`
- `artifacts/embeddings/dinov2_val.npy`
- `artifacts/embeddings/dinov2_test.npy`
- `artifacts/embeddings/dinov2_meta.json`

**Acceptance:**

- Shapes: `(N, 384)`.
- No NaNs.
- Norms are finite.
- Same order as split IDs.
- Saved model name and input size.

### T05 — MiniLM caption embeddings

**Inputs:** `data/captions.jsonl`.
**Outputs:** `artifacts/embeddings/minilm_captions.npy`, `minilm_caption_index.json`.
**Acceptance:**

- Shape: `(num_captions, 384)`.
- Caption order matches index.
- No NaNs.

### T06 — Procrustes fit

**Inputs:** T04, T05.
**Outputs:** `artifacts/alignment/procrustes.npz` with `W`, `mu_x`, `mu_y`, config.
**Acceptance:**

- `W.shape == (384, 384)`.
- `W.T @ W` is close to identity.
- Train retrieval top-1 beats random rotation.
- Artifact loads and reproduces the score.

### T07 — Retrieval evaluation

**Inputs:** T06, test split.
**Outputs:** `artifacts/metrics/retrieval.csv`, confusion matrix.
**Acceptance:**

- Top-1 and top-5 for Procrustes, random rotation, unaligned cosine.
- Per-class table.
- Test N reported.
- No tuning on test.

### T08 — ResNet-18 baseline

**Inputs:** EuroSAT splits.
**Outputs:** `artifacts/metrics/resnet18.csv`, model checkpoint.
**Acceptance:**

- Trained only on train, validated on val.
- Test accuracy and macro-F1 reported.
- Confusion matrix saved.
- Training config logged.

### T09 — Majority baseline

**Inputs:** train labels, test labels.
**Outputs:** `artifacts/metrics/majority.csv`.
**Acceptance:**

- Accuracy and macro-F1 reported.
- Most common class stated.

### T10 — Upper Arlington tile collection

**Inputs:** `14-upper-arlington-data-collection.md`.
**Outputs:** raw tiles in `data/raw/local_tiles/`.
**Acceptance:**

- 50 tiles, cloud-free, clipped to city boundary.
- Coordinates and date recorded.
- File naming follows `INTERFACES.md`.
- Source is Copernicus/Sentinel-2.

### T11 — Local tile labeling

**Inputs:** T10.
**Outputs:** `data/local_tiles/manifest.csv`.
**Acceptance:**

- 10 tiles per category.
- Label rubric followed.
- `other` used when necessary.
- Second-person QA on 10 tiles.
- Disagreements noted.

### T12 — Local OOD evaluation

**Inputs:** T06, T11.
**Outputs:** `artifacts/metrics/local.csv`, example images.
**Acceptance:**

- Procrustes and ResNet predictions on all 50.
- Accuracy reported with N.
- One success and one failure selected.
- Domain-shift limitation stated.

### T13 — Failure taxonomy

**Inputs:** T07, T12.
**Outputs:** failure table, confusion matrix figure, 3–5 example images.
**Acceptance:**

- At least 3 failure types.
- Each has an example and a likely cause.
- One failure is shown in the demo.

### T14 — Gradio demo

**Inputs:** T06, T07.
**Outputs:** `app.py`.
**Acceptance:**

- Text query returns top-3 tiles.
- Image upload returns nearest label.
- Baseline comparison visible.
- One failure example.
- Prototype disclaimer.
- Loads without internet.
- Runs in under 2 minutes.

### T15 — Write-up

**Inputs:** all metrics.
**Outputs:** one-page document.
**Acceptance:**

- Brief template filled.
- User, problem, input/output.
- Baseline table.
- Failure case.
- What we added.
- Sources.
- Honest limitations.

### T16 — Demo video

**Inputs:** T14, T15.
**Outputs:** unlisted video link, ≤2 minutes.
**Acceptance:**

- Problem, demo, how it works, next steps.
- Real input/output.
- Link opens in incognito.

### T17 — Submission

**Inputs:** T15, T16.
**Outputs:** submitted form.
**Acceptance:**

- Code link opens.
- Write-up link opens.
- Video link opens.
- Team details correct.
- Credits complete.
- Track = AI + Climate.

## Handoff notes

Add a short note under the task when changing status:

```text
Task: T__
Status:
What works:
Artifacts:
How to run:
Known issues:
Next task:
```
