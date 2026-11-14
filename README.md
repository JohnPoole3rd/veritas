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

## Working method

Plain Git only. One `main`. Normal task branches. No message board. No board branch. J merges. A and D push task branches. Handoffs use per-task `results/Txx.md` reports.

## What is in this repo right now

- `plan/` — complete hackathon planning package.
- `projectBrief.pdf` — original Veritas AI project brief.
- `LICENSE` — repository license.

No implementation code, data, or artifacts exist yet.

## First implementation tasks

1. `T01` — J: repo scaffold and config.
2. `T02` — J/J1: EuroSAT loader and splits.
3. `T03` — J: caption templates.
4. `T10` — A/A2: Upper Arlington tile collection.
5. `T14` — D/D1: mock Gradio shell.

See `plan/TASKS.md` for the full task board.
