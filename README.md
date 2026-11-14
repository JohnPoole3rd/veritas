# Platonic Earth

**Track:** AI + Climate
**Test site:** Upper Arlington, Ohio
**Status:** Planning complete. Implementation not started.
**Team:** J (integrator + data/alignment), A (baselines + local validation), D (demo + docs)
**Deadline:** Sunday, November 15, 2026, 11:59 PM ET

> Align frozen satellite-image embeddings (DINOv2) with frozen text embeddings (MiniLM) using a closed-form Procrustes rotation, then search Upper Arlington Sentinel-2 tiles in plain language. Compare against majority class, ResNet-18, and a random rotation.

## Start here

- [Plan index](plan/README.md)
- [Agent rules](plan/AGENTS.md)
- [Locked decisions](plan/DECISIONS.md)
- [Task board](plan/TASKS.md)
- [Interfaces](plan/INTERFACES.md)
- [Phase runbook](plan/08-48-hour-runbook.md)
- [Upper Arlington data guide](plan/14-upper-arlington-data-collection.md)
- [Agent work plan](plan/15-agent-work-plan.md)
- [Repo blueprint](plan/16-repo-blueprint.md)
- [Learnings / durable notes](plan/learnings/README.md)

## Working method

Plain Git only. One `main`. Normal task branches. No message board. No board branch. J merges. A and D push task branches. Handoffs use per-task `results/Txx.md` reports. Important findings go to `plan/learnings/`; update plans when beneficial.

## What is in this repo right now

- `configs/` — default configuration.
- `src/` — data, alignment, retrieval, evaluation, and baseline code.
- `scripts/` — runnable data/embedding/alignment/evaluation pipeline.
- `notebooks/` — J's T01–T03 notebooks.
- `tests/` — core alignment, retrieval, and caption tests.
- `plan/` — complete hackathon planning package.
- `plan/learnings/` — durable findings and plan updates.
- `results/` — per-task reports.
- `projectBrief.pdf` — original Veritas AI project brief.
- `LICENSE` — repository license.

## Implementation quick start

```sh
python -m pip install -r requirements.txt
python scripts/prepare_data.py
python scripts/extract_embeddings.py
python scripts/fit_alignment.py
python scripts/evaluate_retrieval.py
```

For a quick smoke test, add `--limit 64` to the embedding/alignment/evaluation scripts.

## Current J results

| Method | Top-1 | Top-5 | N |
|---|---:|---:|---:|
| Procrustes | 0.94 | 0.96 | 512 |
| Random rotation | 0.10 | 0.32 | 512 |
| Unaligned cosine | 0.16 | 0.36 | 512 |

These are from a stratified 512-tile subset because full DINOv2 extraction was too slow on CPU. The pipeline supports the full split; rerun without `--limit` on a GPU and report the larger N.

Local Nix environment used by J:

```sh
nix-shell -p python3Packages.torch python3Packages.torchvision \
  python3Packages.transformers python3Packages.sentence-transformers \
  python3Packages.numpy python3Packages.scipy python3Packages.scikit-learn \
  python3Packages.pyyaml python3Packages.pillow python3Packages.tqdm \
  --run 'python scripts/prepare_data.py'
```

## First implementation tasks

1. `T01` — J: repo scaffold and config. **DONE**
2. `T02` — J/J1: EuroSAT loader and splits. **DONE**
3. `T03` — J: caption templates. **DONE**
4. `T04` — J/J2: DINOv2 embeddings. **DONE** (512-tile stratified subset)
5. `T05` — J/J2: MiniLM caption embeddings. **DONE**
6. `T06` — J: Procrustes alignment. **DONE** (val top-1 0.94)
7. `T07` — J: retrieval evaluation. **DONE** (test top-1 0.94, random 0.10)
8. `T10` — A/A2: Upper Arlington tile collection.
9. `T14` — D/D1: mock Gradio shell.

See `plan/TASKS.md` for the full task board.
