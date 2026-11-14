# Platonic Earth

**Track:** AI + Climate
**Test site:** Upper Arlington, Ohio
**Status:** Implementation in progress. J T01–T07 and D T14 are done; A T08–T13 are pending.
**Team:** J (integrator + data/alignment), A (baselines + local validation), D (demo + docs)
**Deadline:** Sunday, November 15, 2026, 11:59 PM ET

> Align frozen satellite-image embeddings (DINOv2) with frozen text embeddings (MiniLM) using a closed-form Procrustes rotation, then search Upper Arlington Sentinel-2 tiles in plain language. Compare against majority class, ResNet-18, and a random rotation.

## Start here

- [Plan index](plan/README.md)
- [J runbook](plan/agents/J.md)
- [A runbook](plan/agents/A.md)
- [D runbook](plan/agents/D.md)
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
pytest -q
```

For a quick smoke test, add `--limit 64` to the embedding/alignment/evaluation scripts.

## A quick start

```sh
python scripts/evaluate_majority.py
python scripts/train_resnet.py --epochs 2 --batch-size 32   # CPU smoke
python scripts/extract_local_embeddings.py                 # after T10/T11 manifest
python scripts/evaluate_local.py
```

Exact A instructions: `plan/agents/A.md`.

## D quick start

```sh
python app.py --mock     # UI rehearsal
python app.py            # real mode after J artifacts
```

Exact D instructions: `plan/agents/D.md`.

## Current results

### Text→image search — 50 caption queries, full test

| Method | Top-1 | Top-5 |
|---|---:|---:|
| Procrustes | 0.96 | 1.00 |
| Random rotation | 0.18 | 0.34 |
| Unaligned cosine | 0.10 | 0.34 |

### Image→class — 4,050 test images

| Method | Accuracy | Top-5 | Macro-F1 |
|---|---:|---:|---:|
| ResNet-18 | 0.9817 | — | 0.9812 |
| Procrustes | 0.8121 | 0.9901 | 0.7948 |
| Random rotation | 0.1590 | 0.6588 | 0.1559 |
| Unaligned cosine | 0.0827 | 0.5326 | 0.0720 |
| Majority | 0.1111 | — | 0.0200 |

These are A's full-split reruns of J's scripts. J's CPU smoke run used a 512-tile subset and is retained in `results/T04.md`–`T07.md`. Do not compare the 50-query search number with per-image ResNet accuracy.

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
4. `T04` — J/J2: DINOv2 embeddings. **DONE** (J CPU subset; A full-split rerun)
5. `T05` — J/J2: MiniLM caption embeddings. **DONE**
6. `T06` — J: Procrustes alignment. **DONE** (full text search 0.96/1.00; image→class 0.8121)
7. `T07` — J: retrieval evaluation. **DONE** (search and image→class tables)
8. `T08` — A/A1: ResNet-18 baseline. **DONE** (0.9817)
9. `T09` — A: majority baseline. **DONE** (0.1111)
10. `T10` — A/A2: Upper Arlington tile collection. **DONE** (50 tiles)
11. `T11` — A: local labeling + QA. **DONE**
12. `T12` — A: local OOD evaluation. **DONE** (0.64)
13. `T13` — A: failure taxonomy + figures. **DONE**
14. `T14` — D/D1: Gradio demo. **DONE** (real backend verified)

## Run the demo

```sh
python app.py
```

Expected with current artifacts:

```text
[demo] REAL mode — gallery N=512 of 4050, transforms from src.alignment
[demo] NOTE — using 512-tile stratified subset
```

Use `python app.py --mock` only for UI rehearsal. Pre-warm MiniLM and DINOv2 before recording; see `plan/09-demo-writeup-submission.md`.

See `plan/TASKS.md` for the full task board.
