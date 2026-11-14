# Learning: EuroSAT debug subsets must be shuffled

**Date:** 2026-11-14
**Author:** J
**Task:** T02
**Status:** confirmed

## What we learned

EuroSAT's underlying samples are grouped by class. Taking the first `N` items from a split produces a single-class debug subset, which makes retrieval look artificially low or high.

## Evidence

- A 32-item `--limit` run produced top-1 = 0.10 because the subset contained only `AnnualCrop`.
- After shuffling split items with a seeded RNG, a 64-item subset gave Procrustes top-1 = 0.72 and random-rotation top-1 = 0.14.

## Impact

- Debug subsets are now mixed-class.
- The full split is unchanged in size, only item order is shuffled.

## Plan changes

- [x] Updated `src/data.py` to shuffle items after splitting.
- [ ] No decision change needed.

## Follow-up

Use `--limit` only for smoke tests; report full-split numbers in the write-up.
