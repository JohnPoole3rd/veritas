# Learning: T07 "top-1" is not comparable with ResNet-18 accuracy, and the rotation uses labels

**Date:** 2026-11-14
**Author:** A (T07 review, per 07-evaluation-plan.md "A reviews T07")
**Task:** T07 / T08 / T15
**Status:** confirmed (from reading `scripts/fit_alignment.py`, `src/evaluate.py` on main 5e61acf)

## What we learned

1. `evaluate_retrieval` (T07) scores **text→image**: each of the 50 captions is a query, and a hit
   means the single most similar test image has the caption's class. So full-split "top-1 = 0.96" means
   48/50 caption queries hit, and the effective N is **50 queries**, not 4,050 images.
2. ResNet-18 accuracy is **per image** (N = 4,050). The comparable Procrustes number is
   image→class accuracy: classify every test image by its most similar class caption(s).
   `src.evaluate.evaluate_caption_classification` (A) computes this, with the same confusion
   matrix and macro-F1 as ResNet.
3. `fit_procrustes` is fitted on (image, randomly sampled caption **of that image's label**) pairs,
   so the rotation uses the training labels. The method is **supervised alignment with a frozen
   encoder**, not zero-shot. It is still label-efficient (one 384×384 orthogonal matrix), and
   text queries can be phrased freely, but the write-up must not call it zero-shot.

## Evidence

- `scripts/fit_alignment.py`: `sampled_captions = [sample_caption(grouped, item["label"], rng) ...]`.
- `src/evaluate.py::evaluate_retrieval`: `similarity = captions @ images.T`, loop over caption rows.

## Impact

- Evaluation / write-up Table 1: report Procrustes **image→class accuracy and macro-F1 on all
  4,050 test images** next to ResNet-18 and majority. Keep text→image top-1/top-5 as a separate
  "search" metric, and state its N (number of queries).
- Claims (D, T15): use "trained a single rotation on labelled EuroSAT pairs", not "zero-shot".
- Random-rotation and unaligned baselines should get the same image→class metric.

## Plan changes

- [ ] J: decide whether T07 adds the image→class metric (A can provide numbers from the T12 run).
- [ ] D: wording in T15.
