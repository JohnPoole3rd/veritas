# Platonic Earth — Agent-Ready Plan

**Status:** Planning complete. Implementation not started.
**Location:** `/home/john/veritas/plan`
**Track:** AI + Climate
**Project:** Platonic Earth
**Test site:** Upper Arlington, Ohio, USA
**Deadline:** Sunday, November 15, 2026, 11:59 PM ET

## The project

> Align frozen satellite-image embeddings (DINOv2) with frozen text embeddings (MiniLM) using a closed-form Procrustes rotation, then let Upper Arlington sustainability/urban-forestry staff search Sentinel-2 tiles in plain language. Compare against majority class, ResNet-18, and a random rotation.

## Why Upper Arlington

Upper Arlington is a 9.8-square-mile inner-ring suburb of Columbus, Ohio:

- City center: **40.02056° N, 83.06639° W**
- Bordered west by the **Scioto River**, north/east by Columbus, south by Marble Cliff and Grandview Heights.
- **34.2% tree canopy**, with about **33% of land** identified as possible additional tree canopy.
- **21 parks**, about **186 acres**.
- **Tree City USA** since 1990.
- The city is preparing an **Urban Forestry Plan** and updating its Master Plan and Parks & Recreation Plan in 2026.
- Sustainability agenda includes zero waste, zero pollution, and zero emissions by 2050.

That gives the project a concrete user, a real decision, and a local test set.

## Canonical settings

| Setting | Value |
|---|---|
| Track | AI + Climate |
| Project | Platonic Earth |
| Test site | Upper Arlington, Ohio |
| Image encoder | DINOv2 ViT-S/14, frozen (ViT-B/14 optional) |
| Text encoder | `sentence-transformers/all-MiniLM-L6-v2`, frozen |
| Alignment | Orthogonal Procrustes (square 384→384) |
| Training data | EuroSAT RGB |
| Local data | 50 Sentinel-2 tiles clipped to Upper Arlington |
| Local categories | residential, commercial/industrial, road/highway, park/forest/herbaceous, water/river |
| Baselines | majority class, ResNet-18, random rotation, unaligned cosine |
| Demo | Gradio |
| Primary metric | text→image top-1 retrieval |
| Local metric | out-of-distribution accuracy on 50 tiles |
| Submission | code link + one-page write-up |

## Read this first

Any agent starting work should read, in order:

1. [`AGENTS.md`](AGENTS.md) — working rules and handoff protocol.
2. [`DECISIONS.md`](DECISIONS.md) — locked decisions and revisit conditions.
3. [`TASKS.md`](TASKS.md) — task board and acceptance criteria.
4. [`INTERFACES.md`](INTERFACES.md) — file formats, function signatures, artifacts.
5. [`03-platonic-earth-spec.md`](03-platonic-earth-spec.md) — full product spec.
6. [`05-data-plan.md`](05-data-plan.md) — EuroSAT and Upper Arlington tiles.
7. [`06-alignment-and-model-plan.md`](06-alignment-and-model-plan.md) — PRH and Procrustes.
8. [`07-evaluation-plan.md`](07-evaluation-plan.md) — baselines and metrics.
9. [`08-48-hour-runbook.md`](08-48-hour-runbook.md) — execution timeline.
10. [`09-demo-writeup-submission.md`](09-demo-writeup-submission.md) — demo and submission.
11. [`10-risks-fallbacks.md`](10-risks-fallbacks.md) — risk register.
12. [`11-experiment-log.md`](11-experiment-log.md) — results log.
13. [`12-glossary-references.md`](12-glossary-references.md) — terms and citations.
14. [`13-open-questions.md`](13-open-questions.md) — remaining unknowns.
15. [`14-upper-arlington-data-collection.md`](14-upper-arlington-data-collection.md) — local tile guide.
16. [`15-agent-work-plan.md`](15-agent-work-plan.md) — phase-by-phase work plan.
17. [`16-repo-blueprint.md`](16-repo-blueprint.md) — concrete repo structure.

## File map

| File | Purpose |
|---|---|
| `README.md` | Entry point and canonical settings |
| `AGENTS.md` | Rules for implementation agents |
| `DECISIONS.md` | Locked decisions and decision log |
| `TASKS.md` | Task board with owners, dependencies, acceptance |
| `INTERFACES.md` | Data schemas, artifact paths, function signatures |
| `00-executive-summary.md` | One-page pitch |
| `01-brief-constraints.md` | Hackathon rules and compliance |
| `02-project-decision.md` | Why this project, alternatives archived |
| `03-platonic-earth-spec.md` | Full product and research spec |
| `04-picture-dictionary-spec.md` | Archived alternative — not selected |
| `05-data-plan.md` | EuroSAT, local tiles, splits, licenses |
| `06-alignment-and-model-plan.md` | PRH, Procrustes math, model plan |
| `07-evaluation-plan.md` | Baselines, metrics, experiment matrix |
| `08-48-hour-runbook.md` | Hour-by-hour execution |
| `09-demo-writeup-submission.md` | Demo, video, write-up, checklist |
| `10-risks-fallbacks.md` | Risks, kill criteria, fallbacks |
| `11-experiment-log.md` | Tables to fill during the build |
| `12-glossary-references.md` | Terms and citations |
| `13-open-questions.md` | Remaining unknowns and defaults |
| `14-upper-arlington-data-collection.md` | Local tile collection guide |
| `15-agent-work-plan.md` | Phase plan and parallelization |
| `16-repo-blueprint.md` | Repo layout and module responsibilities |

## Current status

- [x] Track chosen: AI + Climate.
- [x] Project chosen: Platonic Earth.
- [x] Test site chosen: Upper Arlington, Ohio.
- [x] User and decision defined.
- [x] Data sources identified.
- [x] Model and baseline plan defined.
- [x] Evaluation plan defined.
- [x] 48-hour runbook defined.
- [x] Agent task board created.
- [ ] Implementation started.
- [ ] Local tiles collected.
- [ ] Experiments run.
- [ ] Demo built.
- [ ] Write-up submitted.

## Next actions

1. Assign owners in `TASKS.md`.
2. Start T01 (repo scaffold) and T02 (EuroSAT loader).
3. Run the T+2 data gate.
4. Log every run in `11-experiment-log.md`.
