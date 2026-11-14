# Learning: "96% top-1" means 48 of 50 caption searches, not 96% of tiles

**Date:** 2026-11-14
**Author:** D (updated by J after the full split)
**Task:** T14 (affects T07, T15, T16)
**Status:** confirmed

## What we learned

`src/evaluate.evaluate_retrieval` scores **text→image**. Each of the 50 captions is one query, and a hit means the best-matching test tile has the caption's class. The full-split top-1 is **0.96 = 48 of 50 caption queries**; N is 50 queries, not 4,050 images.

The comparable **image→class** metric is **0.8121** over 4,050 test images (`caption_classification_test.csv`). ResNet-18 is 0.9817. Do not compare 0.96 (50 queries) with 0.9817 (4,050 images).

A live `permanent crop` query returns PermanentCrop first; the crop failure is image→class confusion (PermanentCrop→AnnualCrop 45/375), not a text-search failure. The clean demo failure is `ua_049` (River→Highway) and the abstract query `climate change`.

## Evidence

- `artifacts/metrics/retrieval.csv`: Procrustes 0.96 / 1.00 over 50 queries, 4,050 test tiles.
- `artifacts/metrics/caption_classification_test.csv`: Procrustes image→class 0.8121 / macro-F1 0.7948, N=4,050.
- `artifacts/metrics/procrustes_test_top_confusions.csv`: River→Highway 122; PermanentCrop→AnnualCrop 45.
- Live demo probes: `permanent crop` → PermanentCrop first; `climate change` → low scores around 0.05.

## Impact

- Write-up/video: say “50 caption queries” for the 0.96 number.
- Main classification table: use image→class Procrustes 0.8121 next to ResNet 0.9817.
- Demo: use ua_049/abstention as the failure case; crop confusion is secondary.
- Do not quote the old 512-subset 0.94/0.96 or the unlogged 77% image→text number.

## Plan changes

- [x] Updated `plan/TASKS.md` and `plan/07-evaluation-plan.md` for the two metric families.
- [x] Updated `plan/09-demo-writeup-submission.md` failure example.
- [x] Updated `assets/demo_script.md`.
- [x] Updated `results/T06.md`, `T07.md`, `T14.md`.

## Follow-up

D uses the full-split tables in `results/T07.md` for T15/T16.
