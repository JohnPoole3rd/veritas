# Agent Work Plan

This file describes how multiple agents or teammates can work in parallel without blocking each other.

## 1. Phase overview

| Phase | Goal | Main output | Gate |
|---|---|---|---|
| P0 | Lock decisions | `DECISIONS.md` | Human approval |
| P1 | Scaffold + data | EuroSAT loads, splits saved | G1: data loads by T+2 |
| P2 | Embeddings | DINOv2 + MiniLM `.npy` | G2: shapes/norms pass |
| P3 | Alignment | Procrustes + retrieval | G3: beats random |
| P4 | Baselines | Majority + ResNet-18 | G4: metrics saved |
| P5 | Local tiles | 50 Upper Arlington tiles | G5: manifest validates |
| P6 | Demo | Gradio app | G6: end-to-end |
| P7 | Write-up | One-page report + video | G7/G8 |
| P8 | Submission | Form submitted | G9 |

## 2. Phase details

### P0 — Lock decisions

**Owner:** Human.

**Tasks:**

- Confirm track, project, site.
- Confirm team size and roles.
- Confirm local imagery date.
- Confirm whether CLIP and video are in scope.

**Output:** updated `13-open-questions.md` and `TASKS.md`.

**Gate:** no code until track/project/site are locked. They are already locked, so proceed.

### P1 — Scaffold + data

**Owner:** Agent A.

**Tasks:**

- T01 repo scaffold.
- T02 EuroSAT loader and splits.
- T03 caption templates.

**Outputs:**

- `configs/default.yaml`
- `data/splits/*.json`
- `data/captions.jsonl`

**Acceptance:**

- 27k tiles loaded.
- Class counts printed.
- Split stratified and seeded.
- No duplicates across splits.
- Captions exist for all 10 classes.

**Gate G1:** if data does not load by T+2, use HF mirror; if still blocked after 30 min, switch.

### P2 — Embeddings

**Owner:** Agent A.

**Tasks:**

- T04 DINOv2 embeddings.
- T05 MiniLM caption embeddings.

**Outputs:**

- `artifacts/embeddings/dinov2_{split}.npy`
- `artifacts/embeddings/minilm_captions.npy`
- metadata JSON

**Acceptance:**

- Shapes `(N, 384)`.
- No NaNs.
- Order matches split.
- Model names saved.

**Gate G2:** if embeddings take too long, use 5k subset.

### P3 — Alignment

**Owner:** Agent A.

**Tasks:**

- T06 Procrustes fit.
- T07 retrieval evaluation.

**Outputs:**

- `artifacts/alignment/procrustes.npz`
- `artifacts/metrics/retrieval.csv`
- confusion matrix

**Acceptance:**

- `W.T @ W ≈ I`.
- Top-1 beats random rotation.
- Test split untouched.
- Run logged.

**Gate G3:** if at chance, debug centering/normalization/caption mapping. If still at chance, pivot to classifier demo + negative result.

### P4 — Baselines

**Owner:** Agent B.

**Tasks:**

- T08 ResNet-18.
- T09 majority.
- Optional CLIP.

**Outputs:**

- `artifacts/metrics/resnet18.csv`
- `artifacts/metrics/majority.csv`

**Acceptance:**

- Trained on train, validated on val, tested on test.
- Accuracy and macro-F1.
- Confusion matrix.
- Config logged.

**Gate G4:** if ResNet-18 fails, use logistic regression or majority only.

### P5 — Upper Arlington tiles

**Owner:** Agent C.

**Tasks:**

- T10 collect 50 tiles.
- T11 label + QA.
- T12 local OOD evaluation (Agent B).

**Outputs:**

- `data/raw/local_tiles/ua_*.png`
- `data/local_tiles/manifest.csv`
- `artifacts/metrics/local.csv`

**Acceptance:**

- 10 per category.
- Within boundary.
- Coordinates/date/source recorded.
- QA done.
- Never used for fitting.

**Gate G5:** if 50 is too many, use 30; report N.

### P6 — Demo

**Owner:** Agent C.

**Tasks:**

- T14 Gradio app.

**Outputs:**

- `app.py`

**Acceptance:**

- Text query → top-3.
- Upload → label.
- Baseline comparison.
- Failure example.
- Prototype disclaimer.
- Loads without internet.

**Gate G6:** if demo breaks, use a static screenshot + notebook output for the video.

### P7 — Write-up

**Owner:** Human + Agent C.

**Tasks:**

- T15 write-up.
- T16 video.

**Outputs:**

- one-page doc
- unlisted video

**Acceptance:**

- Brief template filled.
- Baseline table.
- Failure case.
- Sources.
- Links open in incognito.

**Gate G7/G8:** write-up is mandatory; video optional.

### P8 — Submission

**Owner:** Human.

**Tasks:**

- T17 checklist.
- T18 final review.

**Output:** submitted form.

**Gate G9:** submit at T+45; resubmit only for fixes.

## 3. Parallelization

### 3 agents

```text
Agent A: P1 → P2 → P3
Agent B: P4 (after P1) → P5 evaluation (after P3/P5 data)
Agent C: P5 collection (after P1) → P6 demo (after P3)
Human: P0 → P7 → P8
```

### 2 agents

```text
Agent A: P1 → P2 → P3
Agent B: P4 after P1; P5 collection after P1; P6 after P3
Human: P7 → P8
```

### 1 agent

```text
P1 → P2 → P3 → P4 → P5 → P6 → P7 → P8
```

Cut CLIP, map, and video if solo.

## 4. Handoff artifacts

Each phase hands off:

| Phase | Artifact | Consumer |
|---|---|---|
| P1 | split JSON, captions | P2, P4 |
| P2 | embeddings `.npy` | P3 |
| P3 | `W`, retrieval metrics | P4, P6 |
| P4 | baseline metrics | P7 |
| P5 | local manifest, local metrics | P6, P7 |
| P6 | working demo | P7 |
| P7 | write-up, video | P8 |

## 5. Sync points

At T+6, T+12, T+24, T+36:

1. What is done?
2. What is blocked?
3. What is the next smallest deliverable?
4. What are we cutting?

Update `TASKS.md` and `11-experiment-log.md` at each sync.

## 6. Definition of done

A phase is done when:

- The output exists.
- The acceptance check passes.
- The result is logged.
- A teammate can reproduce it.
- The next phase is unblocked.

## 7. Critical path

```text
P1 → P2 → P3 → P6 → P7 → P8
```

P4 and P5 can run in parallel but must not delay P3/P6.

## 8. If a phase blocks

- P1: switch dataset.
- P2: use 5k subset.
- P3: debug centering; pivot to classifier demo.
- P4: use logistic regression.
- P5: use 30 tiles.
- P6: use notebook output + screenshots.
- P7: write a shorter report.
- P8: submit what works.

## 9. Agent communication template

```text
Phase:
Task:
Status:
What works:
Artifacts:
How to run:
Known issues:
Next:
```
