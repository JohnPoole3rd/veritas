# Learning: "94% top-1" means 47 of 50 caption searches, not 94% of tiles

**Date:** 2026-11-14
**Author:** D
**Task:** T14 (affects T07, T15, T16)
**Status:** confirmed

## What we learned

`src/evaluate.evaluate_retrieval` scores **text→image**. Each of the 50 captions is one query, and a hit means
the best-matching test tile has the caption's class. So T07's top-1 = 0.94 means 47 of 50 caption queries,
measured over a 512-tile gallery. The demo's upload feature goes the other way (**image→text**: tile → nearest
class-caption centroid), and that direction is right about **77%** of the time on the same 512 tiles.
Also, `permanent crop` returns a PermanentCrop tile first (2nd and 3rd are AnnualCrop), so the planned
"permanent crop → AnnualCrop" failure claim is false as stated.

## Evidence

- I reproduced J's pipeline locally with `--limit 512` and got identical metrics (0.94 / 0.96; random 0.10 / 0.32;
  unaligned 0.16 / 0.36).
- The demo's fresh DINOv2 encoding of a test tile matches `dinov2_test.npy` (cosine 1.0), so the demo is faithful.
- Image→text, using precomputed test embeddings vs. the 10 caption centroids: 0.768 (393/512). This is NOT a logged run.
- Live queries: `permanent crop` → PermanentCrop 0.47, AnnualCrop 0.45, AnnualCrop 0.44.
  `climate change` → SeaLake 0.16, AnnualCrop 0.15, Forest 0.15.

## Impact

- Evaluation: the "Test N = 512" next to 94% invites misreading. Write-up/video should say "50 caption queries".
  Majority (N=4,050, classification) and retrieval (N=512, search) are not the same metric or the same N.
- Evaluation: image→text accuracy is a planned secondary metric (plan/03 §10). It should be logged by J/A before
  anyone quotes it.
- Demo: the Results tab now runs failure probes live instead of hardcoding a failure.
- Write-up: use "abstract query" (`climate change`) as the clean failure; describe crop confusion as partial.

## Plan changes

- [ ] Updated `plan/TASKS.md` (J: consider adding image→text accuracy to T07 outputs)
- [ ] Updated `plan/DECISIONS.md` (J only)
- [ ] Updated runbook
- [ ] Updated data/model/eval plan
- [x] Demo script updated (`assets/demo_script.md`); no locked decision changes

## Follow-up

J logs the image→text accuracy as a run in `11-experiment-log.md`, or tells D to keep it out of the demo/write-up.
