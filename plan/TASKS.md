# TASKS.md — J / A / D Task Board

Status values: `TODO`, `IN PROGRESS`, `BLOCKED`, `DONE`.
Owner values: `J`, `A`, `D`, `J1`, `J2`, … , `A1`, `A2`, … , `D1`, `D2`, … , `Human`.

**J is the only agent who merges to `main` and pushes `main`.**

## Critical path

```text
T01 → T02 → T04 → T06 → T07 → T12 → T14 → T15 → T17
        └→ T03 → T05 ┘
        └→ T08 → T09 ┘
T10 → T11 → T12
```

Do not start a downstream task until the upstream artifact exists.

## Task table

| ID | Task | Owner | Subagent of | Status | Depends on | Output |
|---|---|---|---|---|---|---|
| T01 | Repo scaffold and config | J | — | DONE | — | `configs/default.yaml`, `src/`, `requirements.txt`, `results/T01.md` |
| T02 | EuroSAT loader and splits | J1 | J | DONE | T01 | `data/splits/*.json`, `results/T02.md` |
| T03 | Caption templates | J | — | DONE | T01 | `data/captions.jsonl`, `results/T03.md` |
| T04 | DINOv2 embedding extraction | J2 | J | IN PROGRESS | T02 | `artifacts/embeddings/dinov2_*.npy`, `results/T04.md` |
| T05 | MiniLM caption embeddings | J2 | J | TODO | T03 | `artifacts/embeddings/minilm_captions.npy`, `results/T05.md` |
| T06 | Procrustes fit and save | J | — | TODO | T04, T05 | `artifacts/alignment/procrustes.npz`, `results/T06.md` |
| T07 | Retrieval evaluation + random baseline | J | — | TODO | T06 | `artifacts/metrics/retrieval.csv`, `results/T07.md` |
| T08 | ResNet-18 baseline | A1 | A | TODO | T02 | `artifacts/metrics/resnet18.csv`, `results/T08.md` |
| T09 | Majority baseline | A | — | TODO | T02 | `artifacts/metrics/majority.csv`, `results/T09.md` |
| T10 | Upper Arlington tile collection | A2 | A | TODO | T01 | `data/raw/local_tiles/`, `results/T10.md` |
| T11 | Local tile labeling + QA | A | — | TODO | T10 | `data/local_tiles/manifest.csv`, `results/T11.md` |
| T12 | Local OOD evaluation | A | — | TODO | T11, T06 | `artifacts/metrics/local.csv`, `results/T12.md` |
| T13 | Failure taxonomy + figures | A | — | TODO | T07, T12 | `artifacts/figures/`, failure table, `results/T13.md` |
| T14 | Gradio demo | D1 | D | TODO | T06, T07 | `app.py`, `results/T14.md` |
| T15 | Write-up | D | — | TODO | T07, T12, T13 | one-page write-up, `results/T15.md` |
| T16 | Demo video | D2 | D | TODO | T14, T15 | unlisted video link, `results/T16.md` |
| T17 | Submission checklist | J | — | TODO | T15, T16 | submitted form, `results/T17.md` |
| T18 | Final review | J + Human | — | TODO | T17 | final check, `results/T18.md` |

The `Subagent of` column shows the suggested owner. Create more subagents as they want; see the subagent rules below.

## Task details

### T01 — Repo scaffold and config

**Owner:** J
**Inputs:** plan docs.
**Outputs:** `configs/default.yaml`, `src/`, `requirements.txt`, `.gitignore`, implementation `README.md`, `results/T01.md`.

**Acceptance:**

- `python -c "import src"` works.
- Config has paths, model names, seed, split ratios.
- No absolute paths.
- `.gitignore` excludes `data/` and large artifacts.

### T02 — EuroSAT loader and splits

**Owner:** J1 (subagent of J)
**Inputs:** `configs/default.yaml`.
**Outputs:** `data/splits/train.json`, `val.json`, `test.json`; class count report; `results/T02.md`.

**Acceptance:**

- 27k tiles loaded.
- Class counts printed.
- Split stratified and seeded.
- No duplicate tile IDs across splits.
- Test split untouched.

**Gate:** if loading fails, use the HF mirror. If still blocked, switch dataset.

### T03 — Caption templates

**Owner:** J
**Inputs:** EuroSAT classes.
**Outputs:** `data/captions.jsonl`, `results/T03.md`.

**Acceptance:**

- Every class has at least 5 captions.
- Captions are class-consistent.
- No caption mentions another class.

### T04 — DINOv2 embeddings

**Owner:** J2 (subagent of J)
**Inputs:** EuroSAT splits, DINOv2 model.
**Outputs:** `artifacts/embeddings/dinov2_{split}.npy`, `dinov2_meta.json`, `results/T04.md`.

**Acceptance:**

- Shapes `(N, 384)`.
- No NaNs.
- Same order as split IDs.
- Model name and input size saved.

### T05 — MiniLM caption embeddings

**Owner:** J2 (subagent of J)
**Inputs:** `data/captions.jsonl`.
**Outputs:** `artifacts/embeddings/minilm_captions.npy`, `minilm_caption_index.json`, `results/T05.md`.

**Acceptance:**

- Shape `(num_captions, 384)`.
- Order matches index.
- No NaNs.

### T06 — Procrustes fit

**Owner:** J
**Inputs:** T04, T05.
**Outputs:** `artifacts/alignment/procrustes.npz` with `W`, `mu_x`, `mu_y`, config; `results/T06.md`.

**Acceptance:**

- `W.shape == (384, 384)`.
- `W.T @ W` close to identity.
- Validation top-1 beats random rotation.
- Artifact reload reproduces the score.

### T07 — Retrieval evaluation

**Owner:** J
**Inputs:** T06, test split.
**Outputs:** `artifacts/metrics/retrieval.csv`, confusion matrix, `results/T07.md`.

**Acceptance:**

- Top-1/top-5 for Procrustes, random rotation, unaligned cosine.
- Per-class table.
- Test N reported.
- No tuning on test.

### T08 — ResNet-18 baseline

**Owner:** A1 (subagent of A)
**Inputs:** EuroSAT splits.
**Outputs:** `artifacts/metrics/resnet18.csv`, checkpoint, `results/T08.md`.

**Acceptance:**

- Trained on train, validated on val.
- Test accuracy and macro-F1.
- Confusion matrix saved.
- Training config logged.

### T09 — Majority baseline

**Owner:** A
**Inputs:** train labels, test labels.
**Outputs:** `artifacts/metrics/majority.csv`, `results/T09.md`.

**Acceptance:**

- Accuracy and macro-F1.
- Most common class stated.

### T10 — Upper Arlington tile collection

**Owner:** A2 (subagent of A)
**Inputs:** `14-upper-arlington-data-collection.md`.
**Outputs:** raw tiles in `data/raw/local_tiles/`, `results/T10.md`.

**Acceptance:**

- 50 tiles, cloud-free, clipped to city boundary.
- Coordinates and date recorded.
- File naming follows `INTERFACES.md`.
- Source is Copernicus/Sentinel-2.

### T11 — Local tile labeling

**Owner:** A
**Inputs:** T10.
**Outputs:** `data/local_tiles/manifest.csv`, `results/T11.md`.

**Acceptance:**

- 10 tiles per category.
- Label rubric followed.
- `Other` used when necessary.
- Second-person QA on 10 tiles.
- Disagreements noted.

### T12 — Local OOD evaluation

**Owner:** A
**Inputs:** T06, T11.
**Outputs:** `artifacts/metrics/local.csv`, example images, `results/T12.md`.

**Acceptance:**

- Procrustes and ResNet predictions on all 50.
- Accuracy reported with N.
- One success and one failure selected.
- Domain-shift limitation stated.

### T13 — Failure taxonomy

**Owner:** A
**Inputs:** T07, T12.
**Outputs:** failure table, confusion matrix figure, 3–5 example images, `results/T13.md`.

**Acceptance:**

- At least 3 failure types.
- Each has an example and likely cause.
- One failure shown in the demo.

### T14 — Gradio demo

**Owner:** D1 (subagent of D)
**Inputs:** T06, T07.
**Outputs:** `app.py`, `results/T14.md`.

**Acceptance:**

- Text query returns top-3 tiles.
- Image upload returns nearest label.
- Baseline comparison visible.
- One failure example.
- Prototype disclaimer.
- Loads without internet.
- Runs under two minutes.

### T15 — Write-up

**Owner:** D
**Inputs:** all metrics.
**Outputs:** one-page document, `results/T15.md`.

**Acceptance:**

- Brief template filled.
- User, problem, input/output.
- Baseline table.
- Failure case.
- What we added.
- Sources.
- Honest limitations.

### T16 — Demo video

**Owner:** D2 (subagent of D)
**Inputs:** T14, T15.
**Outputs:** unlisted video link, `results/T16.md`.

**Acceptance:**

- Problem, demo, how it works, next steps.
- Real input/output.
- Link opens in incognito.

### T17 — Submission

**Owner:** J
**Inputs:** T15, T16.
**Outputs:** submitted form, `results/T17.md`.

**Acceptance:**

- Code link opens.
- Write-up link opens.
- Video link opens.
- Team details correct.
- Credits complete.
- Track = AI + Climate.

### T18 — Final review

**Owner:** J + Human
**Inputs:** T17.
**Outputs:** `results/T18.md`.

**Acceptance:**

- Links checked in incognito.
- Metrics match logged runs.
- No leakage.
- Submission complete.

## Subagent rules

- No fixed cap: create as many as they want.
- Names: J1, J2, J3, …; A1, A2, A3, …; D1, D2, D3, …
- Prefer one level of delegation. If a subagent needs more help, it reports to its parent, and the parent creates another subagent.
- Each subagent gets one bounded task and one file scope.
- Subagent branch: `j/sub-j1-t02`, `a/sub-a1-t08`, etc.
- Subagent cannot merge to `main`.
- Parent writes the `results/Txx.md` report.
- No two subagents edit the same file.
- If blocked for more than 30 minutes, parent stops and reassigns.
- Parent is accountable for every subagent it creates.

## Continuous planning

When a finding changes future work:

- Update the relevant plan file if the change is clear and local to your task.
- Create `plan/learnings/Txx-topic.md` for a new important finding.
- Use `plan/learnings/_template.md`.
- J owns `plan/TASKS.md` and `plan/DECISIONS.md`.
- If it changes a locked decision, write the learning and ask J/human.
- J reviews and merges learnings with the task branch.

Learning files are a durable knowledge base, not a chat board.

## Handoff notes

Each task report uses this format:

```text
Task: T__
Status:
Owner:
Branch:
What works:
Artifacts:
Metrics:
How to run:
Known issues:
Next task:
```
