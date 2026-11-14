# Platonic Earth — J / A / D Plan

**Status:** Implementation in progress.
**Location:** `/home/john/veritas/plan`
**Track:** AI + Climate
**Project:** Platonic Earth
**Test site:** Upper Arlington, Ohio, USA
**Deadline:** Sunday, November 15, 2026, 11:59 PM ET

## Team

| Agent | Role | Tasks |
|---|---|---|
| **J** | Integrator + data/alignment lead | T01–T07, T17, T18 |
| **A** | Baselines + local validation lead | T08–T13 |
| **D** | Demo + documentation lead | T14–T16 (T16 optional) |

**J is the only agent who merges to `main` and pushes `main`.**

J, A, and D can each create as many subagents as they want to improve speed:

```text
J1, J2, J3, ...
A1, A2, A3, ...
D1, D2, D3, ...
```

Subagent rules are in [`AGENTS.md`](AGENTS.md) and [`15-agent-work-plan.md`](15-agent-work-plan.md).

**If you are J, A, or D, read your specific runbook first:**

- [`agents/J.md`](agents/J.md)
- [`agents/A.md`](agents/A.md)
- [`agents/D.md`](agents/D.md)

When something important is learned, create a durable note under [`learnings/`](learnings/README.md) and update the relevant plan if beneficial.

## Working method

- **Plain Git only.**
- One `main` branch.
- Normal task branches:
  - `j/t01-scaffold`
  - `a/t08-resnet18`
  - `d/t14-gradio`
- No message board.
- No board branch.
- No orphan branch.
- No GitHub CLI.
- Handoff through per-task reports in `results/Txx.md` and commit messages.
- J merges at phase boundaries.
- Update plans when useful; create `plan/learnings/Txx-topic.md` for important findings.

## The project

> Align frozen satellite-image embeddings (DINOv2) with frozen text embeddings (MiniLM) using a closed-form Procrustes rotation, then let Upper Arlington sustainability/urban-forestry staff search Sentinel-2 tiles in plain language. Compare against majority class, ResNet-18, and a random rotation.

## Why Upper Arlington

- City center: **40.02056° N, 83.06639° W**
- About **9.83 sq mi**
- Bordered west by the **Scioto River**, north/east by Columbus, south by Marble Cliff and Grandview Heights
- **34.2% tree canopy**
- About **33% possible additional canopy**
- **21 parks**, about **186 acres**
- **Tree City USA** since 1990
- 2026 **Urban Forestry Plan** and Master Plan/Parks & Recreation Plan updates
- Sustainability agenda includes zero waste, zero pollution, and zero emissions by 2050

## Canonical settings

| Setting | Value |
|---|---|
| Track | AI + Climate |
| Project | Platonic Earth |
| Test site | Upper Arlington, Ohio |
| Image encoder | DINOv2 ViT-S/14, frozen |
| Text encoder | `sentence-transformers/all-MiniLM-L6-v2`, frozen |
| Alignment | Orthogonal Procrustes (384→384) |
| Training data | EuroSAT RGB |
| Local data | 50 Sentinel-2 tiles of Upper Arlington + river corridor (40/50 fully inside; 10 water 25–37% inside) |
| Local categories | residential, commercial/industrial, road/highway, park/forest/herbaceous, water/river |
| Baselines | majority class, ResNet-18, random rotation, unaligned cosine |
| Demo | Gradio |
| Primary metric | text→image top-1 retrieval |
| Local metric | out-of-distribution accuracy on 50 tiles |
| Submission | code link + one-page write-up |

## Read this first

1. [`agents/J.md`](agents/J.md), [`agents/A.md`](agents/A.md), or [`agents/D.md`](agents/D.md) — your exact runbook.
2. [`AGENTS.md`](AGENTS.md) — J/A/D roles, subagent rules, Git rules.
3. [`DECISIONS.md`](DECISIONS.md) — locked decisions.
4. [`TASKS.md`](TASKS.md) — task board and acceptance criteria.
5. [`INTERFACES.md`](INTERFACES.md) — data, artifacts, function contracts.
6. [`03-platonic-earth-spec.md`](03-platonic-earth-spec.md) — full product spec.
7. [`05-data-plan.md`](05-data-plan.md) — EuroSAT and Upper Arlington tiles.
8. [`06-alignment-and-model-plan.md`](06-alignment-and-model-plan.md) — PRH and Procrustes.
9. [`07-evaluation-plan.md`](07-evaluation-plan.md) — baselines and metrics.
10. [`08-48-hour-runbook.md`](08-48-hour-runbook.md) — phase-based runbook.
11. [`09-demo-writeup-submission.md`](09-demo-writeup-submission.md) — demo and submission.
12. [`10-risks-fallbacks.md`](10-risks-fallbacks.md) — risk register.
13. [`11-experiment-log.md`](11-experiment-log.md) — results log.
14. [`12-glossary-references.md`](12-glossary-references.md) — terms and citations.
15. [`13-open-questions.md`](13-open-questions.md) — remaining unknowns.
16. [`14-upper-arlington-data-collection.md`](14-upper-arlington-data-collection.md) — local tile guide.
17. [`15-agent-work-plan.md`](15-agent-work-plan.md) — J/A/D phase plan.
18. [`16-repo-blueprint.md`](16-repo-blueprint.md) — repo layout.
19. [`learnings/README.md`](learnings/README.md) — durable findings and plan updates.

## Phase overview

| Phase | J | A | D |
|---|---|---|---|
| 0 — Setup | T01 scaffold | Start T10 local tiles | Mock Gradio shell |
| 1 — Data | T02 EuroSAT, T03 captions | T10 tiles, prepare T08 | Refine mock UI |
| 2 — Embeddings/alignment | T04 DINOv2, T05 MiniLM, T06 Procrustes | T08 ResNet, T09 majority | Prepare assets |
| 3 — Retrieval/local | T07 retrieval | T11 QA, T12 local eval, T13 failures | T14 real demo |
| 4 — Integration | Merge branches, final eval | Finish metrics | Polish demo, T15 |
| 5 — Docs | Review claims | Review metrics | T15 write-up, T16 video |
| 6 — Submit | T17 checklist | Stand by | Stand by |

No clock times. Start when the human gives the signal.

## Current status

- [x] Track chosen: AI + Climate.
- [x] Project chosen: Platonic Earth.
- [x] Test site chosen: Upper Arlington, Ohio.
- [x] J/A/D roles assigned.
- [x] Task ownership assigned.
- [x] Subagent rules defined.
- [x] Continuous planning and learnings defined.
- [x] Plain Git workflow defined.
- [x] Implementation started.
- [x] J T01–T07 complete; full-split metrics adopted from A.
- [x] A T08–T13 complete and merged.
- [x] D T14 real backend verified against J/A artifacts.
- [ ] D T15/T16 write-up and optional video.
- [ ] J T17 submission.
- [ ] J T18 final review.

## Next actions

1. D writes T15 using A's full-split and local results.
2. D records T16 video with the real demo, or marks it not recorded.
3. J runs T17 submission checklist and T18 final review.
4. Read your runbook before starting: `plan/agents/J.md`, `A.md`, `D.md`.
