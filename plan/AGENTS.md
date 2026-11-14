# AGENTS.md — Working Rules for J, A, D, and Subagents

This file is the contract for the Platonic Earth implementation. Read it before writing code.

## 0. Read order

1. **Your runbook:** `agents/J.md`, `agents/A.md`, or `agents/D.md`
2. `DECISIONS.md`
3. `TASKS.md`
4. `INTERFACES.md`
5. `03-platonic-earth-spec.md`
6. `05-data-plan.md`
7. `06-alignment-and-model-plan.md`
8. `07-evaluation-plan.md`
9. `14-upper-arlington-data-collection.md`

If a task conflicts with `DECISIONS.md`, stop and ask the human. Do not silently change the project.
If your runbook and this file disagree, follow the runbook and tell J.

## 1. Team and ownership

| Agent | Role | Owns | Tasks |
|---|---|---|---|
| **J** | Integrator + data/alignment lead | `configs/`, `src/data.py`, `src/captions.py`, `src/encoders.py`, `src/alignment.py`, `src/retrieval.py`, notebooks 01–03, integration | T01–T07, T17, T18 |
| **A** | Baselines + local validation lead | `src/baselines.py`, `src/evaluate.py`, `src/local_tiles.py`, `src/failures.py`, `src/local_collect.py`, notebooks 04–05, `data/local_tiles/`, `artifacts/metrics/` | T08–T13 |
| **D** | Demo + documentation lead | `app.py`, `assets/`, implementation `README.md`, write-up, video | T14–T16 (T16 optional) |

**J is the only agent who merges to `main` and pushes `main`.**

## 2. Subagents

J, A, and D can each create subagents to improve speed.

### Names

```text
J1, J2, J3, ...
A1, A2, A3, ...
D1, D2, D3, ...
```

### Rules

- **No fixed cap.** Create as many as they want, provided each has a bounded task and file scope.
- Prefer one level of delegation. If a subagent needs more help, it reports to its parent, and the parent creates another subagent.
- Each subagent gets one bounded task and one file scope.
- Subagent branch: `j/sub-j1-t02`, `a/sub-a1-t08`, `d/sub-d1-t14`.
- Subagents cannot merge to `main`.
- The parent writes the `results/Txx.md` report.
- No two subagents edit the same file.
- If a subagent is blocked for more than 30 minutes, the parent stops it and reassigns the work.
- The parent is accountable for every subagent it creates.

## 3. Non-negotiable project decisions

- **Track:** AI + Climate only.
- **Project:** Platonic Earth only.
- **Test site:** Upper Arlington, Ohio.
- **Method:** frozen DINOv2 + frozen MiniLM + closed-form Procrustes.
- **No encoder training.** No multimodal training. No fine-tuning.
- **No audio.** No multilingual branch. No three-modality branch.
- **Data:** EuroSAT for fitting; Upper Arlington Sentinel-2 tiles for local test.
- **Baselines:** majority class and ResNet-18 are required. Random rotation is required for the alignment claim.
- **Evaluation:** held-out test split only. Never tune on test.
- **Claims:** report honest numbers, including weak results. No overclaiming.
- **User:** Upper Arlington sustainability/urban-forestry staff. Not “general users.”

## 4. Allowed tools and data

Allowed:

- Public datasets: EuroSAT, local Sentinel-2 tiles from Copernicus Browser.
- Pretrained models: DINOv2, MiniLM, ResNet-18, optional CLIP.
- Libraries: PyTorch, torchvision, Hugging Face Transformers, Sentence Transformers, scikit-learn, SciPy, NumPy, pandas, Gradio.
- Google Colab or a local GPU.
- AI coding assistants, with citation.

Not allowed:

- Private or personal data.
- Real patient records.
- Scraped private social media.
- Non-public drone imagery.
- Pretrained models with unclear licenses, unless cited and allowed.
- Claiming a pretrained model as your own.

## 5. Repository rules

Implementation lives in `/home/john/veritas`.

```text
veritas/
  configs/
  data/
  artifacts/
  results/
  src/
  notebooks/
  app.py
  README.md
  requirements.txt
```

Rules:

- Do not commit large data or embedding files. Use `.gitignore`.
- Save embeddings and rotations to `artifacts/`; commit only small reports and figures.
- Use `configs/default.yaml` for paths and model names.
- Keep notebooks runnable top-to-bottom.
- Do not hardcode absolute paths.
- Do not edit another agent’s file without coordinating.
- Do not rewrite specs in the plan folder unless the human asks.

## 6. Git workflow

Plain Git only. No GitHub CLI. No message board. No board branch.

### Task branch

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

A and D never push to `main`.

## 7. Task workflow

1. Read `TASKS.md`.
2. Confirm your assigned task.
3. Read its inputs, outputs, and acceptance criteria.
4. Create a task branch.
5. Implement only that task.
6. Run the acceptance check.
7. Save artifacts to the paths in `INTERFACES.md`.
8. Write `results/Txx.md`.
9. Commit and push the branch.
10. J merges at the phase boundary.

Do not start a downstream task until the upstream artifact exists and passes acceptance.

## 8. Handoff format

Every task writes one report:

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
- top1=0.96
- random=0.18
How to run: notebooks/03_alignment.ipynb
Known issues: ViT-S only
Next: T07
```

This is not a message board. It is a per-task report with one owner.

## 9. Definition of done

A task is done only when:

- The code runs from a clean state.
- The output artifact exists at the agreed path.
- The acceptance check passes.
- `results/Txx.md` exists.
- The result is logged in `11-experiment-log.md`.
- A teammate can reproduce it.
- J has merged the branch or it is ready to merge.

## 10. Experiment discipline

- Use the split in `data/splits/`.
- Never fit on test.
- Never tune on test.
- Save the exact config for every run.
- Log every run in `11-experiment-log.md`.
- Include model names, versions, seed, and split sizes.
- If a result looks too good, check for leakage.

## 11. Claim discipline

Every claim in the write-up or demo must trace to a logged run.

Allowed:

- “Text→image search top-1 was X over 50 caption queries on the held-out test split.”
- “Image→class accuracy was X over 4,050 test images; ResNet-18 reached Y.”
- “The random-rotation baseline was Y.”
- “ResNet-18 reached Z on fixed-class classification.”
- “Local tiles showed a domain-shift drop.”

Not allowed:

- Comparing a 50-query search number directly to per-image ResNet accuracy.
- Calling the rotation zero-shot.
- “The AI understands land use.”
- “Accurate land-use determination.”
- “Replaces GIS.”
- “Beat the baseline” without a number.
- “Works well” without evidence.

## 12. Data discipline

- EuroSAT only from torchvision or the official/HF mirror.
- Local tiles only from Copernicus Browser / Sentinel-2.
- Clip local tiles to the Upper Arlington city boundary.
- Record coordinates, date, source, label, and notes for every local tile.
- Use the label rubric in `14-upper-arlington-data-collection.md`.
- Do not label a tile you cannot identify; mark `Other` and note why.
- Do not collect personal data.

## 13. Demo discipline

- Precompute embeddings and rotation before the demo.
- The demo must load without internet.
- Include one success and one failure.
- Show the baseline comparison.
- Include “prototype, not a land-use determination.”
- Keep the demo under two minutes.
- Record a fallback screen capture.

## 14. Stop conditions

Stop and ask the human if:

- EuroSAT does not load after a reasonable attempt.
- Procrustes retrieval is at chance after checking centering and normalization.
- A task requires a private dataset.
- A result would require changing the track or project.
- The deadline is close and the write-up is not started.
- An agent wants to add audio, multilingual, or a third modality.

## 15. Escalation defaults

If no human is available:

- Keep the current track and project.
- Cut stretch goals before cutting evaluation.
- Submit a complete, honest, narrow project.
- Prefer a working classifier demo over a broken alignment demo.
- Log the decision in `11-experiment-log.md`.

## 16. Conflict rules

- One owner per module.
- If you need a function in another owner's module, add a stub with `TODO(owner)` or write a learning file. Do not implement it yourself.
- J only writes `plan/TASKS.md`.
- Each task owner writes only their own `results/Txx.md`.
- No direct pushes to `main` except J.
- No two subagents on the same file.
- Pull before work.
- Small commits.
- If blocked, commit `Txx: BLOCKED ...` on the task branch and push.

## 17. Continuous planning

Plans are living documents. Update them when new information changes future work.

### When to update

Update a plan file when:

- A task reveals a constraint that changes later tasks.
- A dataset turns out to be unusable or different from expected.
- A model behaves differently than planned.
- A metric changes the interpretation of the project.
- A failure mode becomes important enough to document.
- A new artifact or script becomes part of the workflow.

### What to update

- `plan/TASKS.md` — J updates task ownership, status, dependencies, acceptance after a finding.
- `plan/DECISIONS.md` — only J changes locked decisions.
- `plan/08-48-hour-runbook.md` — any agent may update if the change is local to their task.
- `plan/15-agent-work-plan.md` — J updates agent/subagent assignments; agents may add their own subagent notes.
- `plan/05-data-plan.md` — A updates local-tile details; J updates EuroSAT details.
- `plan/06-alignment-and-model-plan.md` — J updates model/alignment details.
- `plan/07-evaluation-plan.md` — A updates metrics/evaluation details.
- `plan/14-upper-arlington-data-collection.md` — A updates local tile details.
- Cross-cutting or uncertain findings — create a learning file instead of editing shared plans.

### Create a learning file when

Create a new file under `plan/learnings/` when the finding is important but does not fit cleanly into an existing plan section.

Naming:

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

Use the template in `plan/learnings/_template.md`.

### Rules

- Any agent may create a learning file.
- Do not edit another agent's learning file.
- Keep the finding short: what happened, evidence, impact, next action.
- If the finding changes a locked decision, do not change the decision yourself. Write the learning file and ask J/human.
- J reviews and merges learnings with the task branch.
- Learning files are a durable knowledge base, not a chat board.
