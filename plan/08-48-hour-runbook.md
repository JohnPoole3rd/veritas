# Phase Runbook — J / A / D

**Project:** Platonic Earth, Upper Arlington, Ohio
**Deadline:** Sunday, November 15, 2026, 11:59 PM ET
**Working method:** plain Git, normal task branches, no message board.
**Start:** when the human gives the signal.

There are no clock-based gates. Work moves by phase and by artifact readiness.

## Roles

| Agent | Role | Tasks |
|---|---|---|
| **J** | Integrator + data/alignment lead | T01–T07, integration, submission |
| **A** | Baselines + local validation lead | T08–T13 |
| **D** | Demo + documentation lead | T14–T17 |

J merges to `main`. A and D push task branches. J reviews and merges at phase boundaries.

## Subagents

Each primary agent can create as many subagents as they want:

```text
J1, J2, J3, ...
A1, A2, A3, ...
D1, D2, D3, ...
```

Rules:

- No fixed cap. Create as many as they want, provided each has a bounded task and file scope.
- Prefer one level of delegation. If a subagent needs more help, it reports to its parent, and the parent creates another subagent.
- Each subagent gets one bounded task and one file scope.
- Subagent branch: `j/sub-j1-t02`, `a/sub-a1-t08`, `d/sub-d1-t14`.
- Subagent cannot merge to `main`.
- Parent writes the `results/Txx.md` report.
- Parent is accountable for every subagent it creates.

## Phase 0 — Setup

### J

- [ ] T01: repo scaffold, `configs/default.yaml`, `requirements.txt`, `.gitignore`.
- [ ] Post `results/T01.md`.
- [ ] Start T02 EuroSAT loader.

### A

- [ ] Read plan docs.
- [ ] Start T10: collect 50 Upper Arlington tiles.
- [ ] Prepare T08 ResNet-18 script against the expected split format.
- [ ] Post `results/T10.md` when collection is done.

### D

- [ ] Read plan docs.
- [ ] Build mock Gradio shell with placeholder tiles and scores.
- [ ] Write the two-minute demo script.
- [ ] Start `results/T14.md` with status `IN PROGRESS`; do not mark T14 done until real data is wired.

**Phase exit:** J has scaffold and EuroSAT loading; A has tiles underway; D has a clickable mock UI.

## Phase 1 — Data foundation

### J

- [ ] T02: EuroSAT splits.
- [ ] T03: caption templates.
- [ ] Post `results/T02.md`, `results/T03.md`.

### A

- [ ] T10: finish tile collection.
- [ ] T11: label and QA 50 tiles.
- [ ] T08: prepare ResNet-18, wait for T02 if needed.
- [ ] Post `results/T10.md`, `results/T11.md`.

### D

- [ ] Refine mock UI.
- [ ] Prepare example queries and screenshot placeholders.
- [ ] Keep `results/T14.md` current.

**Phase exit:** splits and captions exist; local tiles are collected; mock UI is ready for real data.

## Phase 2 — Embeddings and alignment

### J

- [ ] T04: DINOv2 embeddings.
- [ ] T05: MiniLM caption embeddings.
- [ ] T06: Procrustes fit and save `W`, `mu_x`, `mu_y`.
- [ ] Post `results/T04.md`, `results/T05.md`, `results/T06.md`.

### A

- [ ] T08: ResNet-18 baseline.
- [ ] T09: majority baseline.
- [ ] Post `results/T08.md`, `results/T09.md`.

### D

- [ ] Prepare assets.
- [ ] Do not block J.
- [ ] Update mock UI with the final layout.

**Phase exit:** Procrustes beats random rotation on validation; A’s baselines run; D’s UI is ready to wire.

## Phase 3 — Retrieval and local validation

### J

- [ ] T07: retrieval evaluation + random baseline.
- [ ] Save metrics and confusion matrix.
- [ ] Post `results/T07.md`.

### A

- [ ] T12: local out-of-distribution evaluation.
- [ ] T13: failure taxonomy and figures.
- [ ] Post `results/T12.md`, `results/T13.md`.

### D

- [ ] T14: wire real artifacts into Gradio.
- [ ] Add baseline comparison and failure example.
- [ ] Post `results/T14.md`.

**Phase exit:** retrieval metrics exist; local N=50 metrics exist; failure example selected; demo uses real data.

## Phase 4 — Integration and demo

### J

- [ ] Merge J, A, and D branches.
- [ ] Run final evaluation on `main`.
- [ ] Update `plan/TASKS.md`.
- [ ] Check for leakage and split consistency.

### A

- [ ] Finish metrics and figures.
- [ ] Review claims against logged runs.

### D

- [ ] Polish demo.
- [ ] Start T15 write-up.
- [ ] Prepare screenshots.

**Phase exit:** `main` contains working code; final metrics are stable; demo runs end-to-end; write-up draft exists.

## Phase 5 — Write-up and video

### J

- [ ] Review results and claims.
- [ ] Final integration fixes.
- [ ] Prepare submission links.

### A

- [ ] Review metrics and failure examples.
- [ ] Confirm local sample size and labels.

### D

- [ ] T15: one-page write-up.
- [ ] T16: two-minute video.
- [ ] Post `results/T15.md`, `results/T16.md`.

**Phase exit:** write-up complete; video recorded; links open in incognito.

## Phase 6 — Submission

### J

- [ ] T17: submission checklist.
- [ ] Confirm code, write-up, video, team details, credits.
- [ ] Submit.

### A

- [ ] Stand by for metric questions.

### D

- [ ] Stand by for link or video questions.

**Phase exit:** submission complete.

## Git rules

- One `main`.
- Normal task branches only:
  - `j/t01-scaffold`
  - `a/t08-resnet18`
  - `d/t14-gradio`
- No message board.
- No board branch.
- No orphan branch.
- No GitHub CLI.
- J merges with plain `git merge --no-ff`.
- A and D never push to `main`.

## Handoff rules

Each task writes one report:

```text
results/T01.md
results/T02.md
...
results/T17.md
```

Format:

```markdown
# T06 — Procrustes

Status: DONE
Owner: J
Branch: j/t06-procrustes
Artifacts:
- artifacts/alignment/procrustes.npz
Metrics:
- top1=0.42
- random=0.10
How to run: notebooks/03_alignment.ipynb
Known issues: ViT-S only
Next: T07
```

This is a task report, not a message board.

At phase start, each agent reads:

```sh
git fetch --all
git log --all --oneline --decorate -20
git show origin/main:plan/TASKS.md
```

## Continuous planning

Plans are living documents. When a finding changes future work:

- Update the relevant plan file if the change is clear and local to your task.
- Create `plan/learnings/Txx-topic.md` if it is a new important finding.
- Use `plan/learnings/_template.md`.
- J owns `plan/TASKS.md` and `plan/DECISIONS.md`.
- If it changes a locked decision, write the learning file and ask J/human.
- J reviews and merges learnings with the task branch.

Examples:

```text
plan/learnings/T04-dinov2-resolution-limit.md
plan/learnings/T10-upper-arlington-cloud-cover.md
plan/learnings/general-procrustes-centering-bug.md
```

## Scope cuts

Cut in this order:

1. CLIP baseline.
2. Map view.
3. Confidence threshold.
4. Failure gallery.
5. Video.
6. ViT-B comparison.
7. Local tiles from 50 to 30.

Never cut:

1. Held-out evaluation.
2. Majority baseline.
3. ResNet-18 baseline.
4. Random rotation.
5. One failure case.
6. Working demo.
7. Write-up and credits.
8. Link check.

## Do-not-do list

- Do not train a multimodal model.
- Do not add audio.
- Do not add multilingual.
- Do not add a third modality.
- Do not use private data.
- Do not claim parcel-level canopy measurement.
- Do not tune on test.
- Do not compare methods on different splits.
- Do not hide weak results.
- Do not spend more than 30 minutes on a data download.
- Do not start a new idea after Phase 1.
