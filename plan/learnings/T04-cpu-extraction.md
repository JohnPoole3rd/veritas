# Learning: Full DINOv2 extraction is too slow on CPU

**Date:** 2026-11-14
**Author:** J
**Task:** T04
**Status:** confirmed

## What we learned

Full 27k EuroSAT extraction with DINOv2 ViT-S/14 on CPU was stopped after several minutes because it would not finish inside the phase gate. A 512-per-split stratified subset completes quickly and gives a usable smoke result.

## Evidence

- Full extraction process ran for ~6 minutes and had not finished the 18,900-tile train split.
- 512-per-split extraction completed in under a minute.
- 512 subset: Procrustes test top-1 = 0.94, random rotation = 0.10, unaligned = 0.16.

## Impact

- T04/T06/T07 are reported with N=512, not the full split.
- The code supports the full split; rerun without `--limit` on a GPU.
- The write-up must state N and the CPU limitation honestly.

## Plan changes

- [x] Updated `results/T04.md`, `T06.md`, `T07.md`.
- [x] Updated `plan/11-experiment-log.md`.
- [ ] Full-GPU run is a stretch goal.

## Follow-up

If a GPU becomes available, run:

```sh
python scripts/extract_embeddings.py
python scripts/fit_alignment.py
python scripts/evaluate_retrieval.py
```
