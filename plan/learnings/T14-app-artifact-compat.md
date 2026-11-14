# Learning: Demo artifact contract must match J's actual files

**Date:** 2026-11-14
**Author:** J
**Task:** T14 integration
**Status:** confirmed

## What we learned

D's `app.py` initially assumed `procrustes.npz` had a `config` key and that `dinov2_test.npy` had exactly the same row count as `test.json`. J's actual artifacts use `metadata` and a 512-tile subset, so RealBackend would fail or fall back to MOCK.

## Evidence

- `src/alignment.save_procrustes` writes `W`, `mu_x`, `mu_y`, `metadata`.
- J's `dinov2_test.npy` has 512 rows; `data/splits/test.json` has 4,050 items.
- After the fix, `load_backend()` reports `REAL mode — Upper Arlington N=50; EuroSAT test N=512 of 4050`.

## Impact

- `app.py` now reads `metadata` and accepts `config` as a legacy alias.
- It slices `test.json` to the embedding count for stratified subsets.
- It reads `majority.csv` and `test_n` columns.
- `results/T14.md` now records the integration check.

## Plan changes

- [x] Updated `plan/INTERFACES.md` with exact artifact keys and CSV columns.
- [x] Updated `plan/09-demo-writeup-submission.md` with runtime encoder requirements.
- [x] Updated `plan/agents/D.md` with exact commands.

## Follow-up

If J reruns full extraction, `app.py` automatically uses all 4,050 test items and the subset note disappears.
