# J / A / D Agent Work Plan

This file defines what each primary agent does, what they own, and how subagents are used.

## 1. Primary agents

| Agent | Role | Owns | Tasks |
|---|---|---|---|
| **J** | Integrator + data/alignment lead | `configs/`, `src/data.py`, `src/captions.py`, `src/encoders.py`, `src/alignment.py`, `src/retrieval.py`, notebooks 01–03, integration | T01–T07 |
| **A** | Baselines + local validation lead | `src/baselines.py`, `src/evaluate.py`, `src/local_tiles.py`, notebooks 04–05, `data/local_tiles/`, `artifacts/metrics/` | T08–T13 |
| **D** | Demo + documentation lead | `app.py`, `assets/`, implementation `README.md`, write-up, video | T14–T17 |

J is the only agent who merges to `main` and pushes `main`.

## 2. Phase plan

### Phase 0 — Setup

**J**

- T01 repo scaffold and config.
- Start T02 EuroSAT loader.
- Write `results/T01.md`.

**A**

- Start T10 Upper Arlington tile collection.
- Prepare the ResNet-18 script against the expected split format.
- Write `results/T10.md` when collection is done.

**D**

- Build mock Gradio UI with placeholder tiles and scores.
- Write the demo script.
- Start `results/T14.md` with status `IN PROGRESS`; do not mark T14 done until real data is wired.

**Exit:** J has scaffold and EuroSAT loading; A has tiles underway; D has a clickable mock UI.

### Phase 1 — Data foundation

**J**

- T02 EuroSAT splits.
- T03 captions.
- Write `results/T02.md`, `results/T03.md`.

**A**

- T10 finish tile collection.
- T11 label and QA.
- T08 prepare ResNet-18.
- Write `results/T10.md`, `results/T11.md`.

**D**

- Refine mock UI.
- Prepare example queries and screenshots.
- Keep `results/T14.md` current.

**Exit:** splits and captions exist; local tiles collected; mock UI ready.

### Phase 2 — Embeddings and alignment

**J**

- T04 DINOv2 embeddings.
- T05 MiniLM caption embeddings.
- T06 Procrustes fit.
- Write `results/T04.md`, `results/T05.md`, `results/T06.md`.

**A**

- T08 ResNet-18.
- T09 majority baseline.
- Write `results/T08.md`, `results/T09.md`.

**D**

- Prepare assets.
- Update mock UI.
- Do not block J.

**Exit:** Procrustes beats random on validation; baselines run; UI ready to wire.

### Phase 3 — Retrieval and local validation

**J**

- T07 retrieval evaluation.
- Save metrics and confusion matrix.
- Write `results/T07.md`.

**A**

- T12 local OOD evaluation.
- T13 failure taxonomy and figures.
- Write `results/T12.md`, `results/T13.md`.

**D**

- T14 wire real artifacts into Gradio.
- Add baseline comparison and failure example.
- Write `results/T14.md`.

**Exit:** retrieval metrics; local N=50 metrics; failure example; real demo.

### Phase 4 — Integration and demo

**J**

- Merge J/A/D branches.
- Run final evaluation on `main`.
- Update `plan/TASKS.md`.
- Check leakage and splits.

**A**

- Finish metrics and figures.
- Review claims against runs.

**D**

- Polish demo.
- Start T15 write-up.
- Prepare screenshots.

**Exit:** `main` has working code; final metrics stable; demo end-to-end; write-up draft.

### Phase 5 — Write-up and video

**J**

- Review results and claims.
- Final integration fixes.
- Prepare submission links.

**A**

- Review metrics and failure examples.
- Confirm local sample size and labels.

**D**

- T15 write-up.
- T16 video.
- Write `results/T15.md`, `results/T16.md`.

**Exit:** write-up complete; video recorded; links work.

### Phase 6 — Submission

**J**

- T17 checklist.
- Submit.

**A / D**

- Stand by for questions.

**Exit:** submitted.

## 3. Subagents

J, A, and D can each create as many subagents as they want.

### Naming

```text
J1, J2, J3, ...
A1, A2, A3, ...
D1, D2, D3, ...
```

### Rules

- **No fixed cap.** Create as many as they want, provided each has a bounded task and file scope.
- Prefer one level of delegation. If a subagent needs more help, it reports to its parent, and the parent creates another subagent.
- Each subagent gets one bounded task and one file scope.
- Subagent branch examples:
  - `j/sub-j1-t02`
  - `j/sub-j2-t04`
  - `a/sub-a1-t08`
  - `a/sub-a2-t10`
  - `d/sub-d1-t14`
- Subagents cannot merge to `main`.
- Parent writes the `results/Txx.md` report.
- No two subagents edit the same file.
- If a subagent is blocked for more than 30 minutes, the parent stops it and reassigns the work.
- The parent is accountable for every subagent it creates.

### Suggested assignments

**J**

- `J1` — EuroSAT loader and splits (T02).
- `J2` — DINOv2 and MiniLM embedding extraction (T04/T05).

J keeps T06/T07 and integration.

**A**

- `A1` — ResNet-18 training loop (T08).
- `A2` — Upper Arlington tile collection (T10).

A keeps T12/T13 and metrics.

**D**

- `D1` — Gradio UI components (T14).
- `D2` — visuals and demo script (T15/T16).

D keeps T15/T16.

These are starting suggestions. Create more subagents whenever they improve speed.

## 4. Git workflow

No message board. No board branch. Plain Git only.

### Feature branch

```sh
git fetch --all
git switch main
git pull --rebase origin main
git switch -c j/t01-scaffold
```

### Commit

```sh
git add configs/ src/
git commit -m "T01: add repo scaffold and config"
git push -u origin j/t01-scaffold
```

### J merges

```sh
git fetch --all
git switch main
git pull --rebase origin main
git merge --no-ff origin/j/t01-scaffold
git push origin main
```

No `gh`. No GitHub CLI.

## 5. Handoff

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

This is not a message board. It is a per-task report with one owner.

## 6. Phase sync

J reviews and merges at these points:

- After Phase 0: scaffold, tiles started, mock UI.
- After Phase 1: splits, captions, tiles collected.
- After Phase 2: embeddings, Procrustes, baselines.
- After Phase 3: retrieval, local eval, demo.
- After Phase 4: integration, final eval, write-up draft.
- After Phase 5: write-up and video.
- Phase 6: submit.

No clock times. Move when the phase exit is met.

## 7. Continuous planning

Plans change when the work teaches you something.

### Update a plan file when

- A task reveals a new constraint.
- A dataset or model behaves differently than expected.
- A metric changes the interpretation.
- A failure becomes important enough to document.
- A new artifact or script becomes part of the workflow.

### Create a learning file when

The finding is important but does not fit an existing section.

```text
plan/learnings/Txx-topic.md
plan/learnings/general-topic.md
```

Examples:

```text
plan/learnings/T04-dinov2-resolution-limit.md
plan/learnings/T10-upper-arlington-cloud-cover.md
plan/learnings/general-procrustes-centering-bug.md
```

Use `plan/learnings/_template.md`.

### Rules

- Any agent may create a learning file.
- Do not edit another agent's learning file.
- J owns `plan/TASKS.md` and `plan/DECISIONS.md`.
- Agents may update plan sections local to their task; use a learning file for cross-cutting findings.
- If it changes a locked decision, write the learning file and ask J/human.
- J reviews and merges learnings with the task branch.
- Learning files are a durable knowledge base, not a chat board.

## 8. Conflict rules

- One owner per module.
- J only writes `plan/TASKS.md`.
- Each task owner writes only their own `results/Txx.md`.
- No direct pushes to `main` except J.
- No two subagents on the same file.
- Pull before work.
- Small commits.
- If blocked, commit `Txx: BLOCKED ...` on the task branch and push.
