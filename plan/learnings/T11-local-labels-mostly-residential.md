# Learning: Upper Arlington local tiles are 62 % Residential by honest labelling

**Date:** 2026-11-14
**Author:** A
**Task:** T11
**Status:** confirmed

## What we learned

Labelled by dominant land cover (the plan/14 rubric), 31/50 local tiles are Residential, even though
they were sampled 10 per local category. Upper Arlington has no highways, little commercial land and
little closed forest that is visible at 10 m. A constant "Residential" guess scores 0.62 locally.

## Evidence

- `data/local_tiles/manifest.csv`, `data/local_tiles/labels_qa.csv`, `results/T11.md`.
- Blind QA (subagent A2) agreed on 6/10. Every disagreement was Residential vs Forest (canopy) or
  Residential vs River (boundary water tiles), and the reviewer never chose a non-Residential label
  that the primary labeller had not.

## Impact

- Evaluation: T12 must report the **local** majority baseline (0.62), per-local-category accuracy
  (N=10 each), and accuracy on the 37 non-ambiguous tiles. Overall local accuracy alone is misleading.
- Write-up (D): "Upper Arlington is overwhelmingly tree-covered residential land; the planned
  highway/commercial/forest classes barely exist at 10 m." This is a good "where it fails / new
  setting" story (brief §3, "Your own question").
- Demo: a residential tile with heavy canopy is a natural failure/ambiguity example for Forest vs Residential.

## Plan changes

- [x] `results/T11.md` documents the deviation from "10 tiles per category".
- [ ] J: T11 acceptance in `plan/TASKS.md` ("10 tiles per category") is met per stratum, not per label; J to confirm.
- [ ] Human: spot-check the 13 `ambiguous` tiles in `labels_qa.csv` (AI reviewer ≠ human second person).
