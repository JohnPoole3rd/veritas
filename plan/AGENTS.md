# AGENTS.md — Working Rules for Implementation Agents

This file is the contract for any agent or human working on Platonic Earth.

## 0. Read order

Before writing code, read:

1. `DECISIONS.md`
2. `TASKS.md`
3. `INTERFACES.md`
4. `03-platonic-earth-spec.md`
5. `05-data-plan.md`
6. `06-alignment-and-model-plan.md`
7. `07-evaluation-plan.md`

If a task conflicts with `DECISIONS.md`, stop and ask the human. Do not silently change the project.

## 1. Non-negotiable project decisions

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

## 2. Allowed tools and data

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

## 3. Repository rules

The future implementation lives in `/home/john/veritas`.

Planned layout:

```text
veritas/
  configs/
    default.yaml
  data/
    raw/eurosat/
    raw/local_tiles/
    splits/
  artifacts/
    embeddings/
    alignment/
    metrics/
    figures/
  src/
    data.py
    captions.py
    encoders.py
    alignment.py
    retrieval.py
    evaluate.py
    baselines.py
    local_tiles.py
  notebooks/
    01_data.ipynb
    02_embeddings.ipynb
    03_alignment.ipynb
    04_evaluation.ipynb
    05_local_tiles.ipynb
  app.py
  README.md
  requirements.txt
```

Rules:

- Do not commit large data or embedding files. Use `.gitignore`.
- Save embeddings and rotations to `artifacts/`; commit only small reports and figures.
- Use `configs/default.yaml` for paths and model names.
- Keep notebooks runnable top-to-bottom.
- Do not hardcode absolute paths; use config values.
- Do not edit another agent’s file without coordinating.
- Do not rewrite specs in the plan folder unless the human asks.

## 4. Task workflow

1. Open `TASKS.md`.
2. Pick a `TODO` task with satisfied dependencies.
3. Change status to `IN PROGRESS` and add your name/handle.
4. Read the task’s inputs, outputs, and acceptance criteria.
5. Implement only that task.
6. Run the acceptance check.
7. Save artifacts to the paths in `INTERFACES.md`.
8. Log results in `11-experiment-log.md`.
9. Change status to `DONE` or `BLOCKED`.
10. Write a one-paragraph handoff note in `TASKS.md`.

Do not start a downstream task until the upstream artifact exists and passes its acceptance check.

## 5. Definition of done

A task is done only when:

- The code runs from a clean state.
- The output artifact exists at the agreed path.
- The acceptance check passes.
- The result is logged.
- The task is marked `DONE`.
- A teammate can reproduce it from the README or notebook.

“It works on my machine” is not done.

## 6. Experiment discipline

- Use the split in `data/splits/`.
- Never fit on test.
- Never tune on test.
- Save the exact config for every run.
- Log every run in `11-experiment-log.md`.
- Include model names, versions, seed, and split sizes.
- If a result looks too good, check for leakage before reporting it.

## 7. Claim discipline

Every claim in the write-up or demo must trace to a logged run.

Allowed:

- “Top-1 retrieval was X on the held-out test split.”
- “The random-rotation baseline was Y.”
- “ResNet-18 reached Z on fixed-class classification.”
- “Local tiles showed a domain-shift drop.”

Not allowed:

- “The AI understands land use.”
- “Accurate land-use determination.”
- “Replaces GIS.”
- “Beat the baseline” without a number.
- “Works well” without evidence.

## 8. Data discipline

- EuroSAT only from torchvision or the official/HF mirror.
- Local tiles only from Copernicus Browser / Sentinel-2.
- Clip local tiles to the Upper Arlington city boundary.
- Record coordinates, date, source, label, and notes for every local tile.
- Use the label rubric in `14-upper-arlington-data-collection.md`.
- Do not label a tile you cannot identify; mark `other` and note why.
- Do not collect personal data.

## 9. Demo discipline

- Precompute embeddings and rotation before the demo.
- The demo must load without internet.
- Include one success and one failure.
- Show the baseline comparison.
- Include “prototype, not a land-use determination.”
- Keep the demo under two minutes.
- Record a fallback screen capture.

## 10. Stop conditions

Stop and ask the human if:

- EuroSAT does not load within 30 minutes.
- Procrustes retrieval is at chance after checking centering and normalization.
- A task requires a private dataset.
- A result would require changing the track or project.
- The deadline is within 6 hours and the write-up is not started.
- An agent wants to add audio, multilingual, or a third modality.

## 11. Escalation defaults

If no human is available:

- Keep the current track and project.
- Cut stretch goals before cutting evaluation.
- Submit a complete, honest, narrow project.
- Prefer a working classifier demo over a broken alignment demo.
- Log the decision in `11-experiment-log.md`.

## 12. Handoff format

When handing off, write:

```text
Task: T__
Status: DONE / BLOCKED
What works:
Artifacts:
How to run:
Known issues:
Next task:
```

Put it in `TASKS.md` under the task. Keep it short.
