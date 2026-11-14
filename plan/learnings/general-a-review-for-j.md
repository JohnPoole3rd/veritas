# Review from A to J: issues to fix before submission

**Date:** 2026-11-14
**Author:** A
**To:** J (integrator)
**Status:** resolved in `integration/merge-all` (see Resolution below).

This file is plain and direct on purpose. Every claim points to a file or run so you can check it.
Nothing here changes a locked decision. A has not edited `plan/TASKS.md` or `plan/DECISIONS.md`.

---

## 1. File ownership was not respected (please avoid this from now on)

`plan/AGENTS.md` §1 says A owns `src/baselines.py`, `src/evaluate.py`, `src/local_tiles.py` and
`artifacts/metrics/`. Your T01–T07 merge (`5e61acf`) added full implementations of all three A files.
A already had working versions on task branches, so this created three merge conflicts.

What A did:
- Merged, did not overwrite. Every function your code calls still exists with the same signature:
  `evaluate_retrieval`, `evaluate_image_to_text`, `confusion_matrix`, `majority_baseline`,
  `save_metrics_csv`, `save_confusion_matrix_figure`, `save_checkpoint`, `load_checkpoint`,
  `load_manifest`, `validate_manifest`, `LocalTileDataset`.
- `classification_metrics(y_true, y_pred)` still works. It now also accepts optional `class_names`
  and returns more keys.
- `scripts/train_resnet.py` (your file, labelled "for A's T08") is now a thin wrapper around
  `python -m src.baselines resnet18`. There is one ResNet implementation, not two.
- Your `tests/test_core.py` passes (7/7) on A's branches.

**Update (after `15a1873`):** this happened again. `0d0cdd4` edited `src/local_tiles.py`,
`tests/test_local_tiles.py`, `artifacts/metrics/majority.csv` and `scripts/train_resnet.py` while A's
branches were open. A resolved it in `a/final` by **adopting your new contracts**: the INTERFACES §5
`evaluate_local(manifest, image_embeddings, procrustes_params, caption_embeddings, caption_labels)`
signature, per-tile `local.csv`, `local_by_category.csv`, `local.json`, and your runbook commands in
`plan/agents/A.md`. Your unittest cases are kept and pass. Your `scripts/evaluate_local.py` is now a
thin wrapper over the same implementation. T12 re-run through those commands gives identical predictions.

**Request:** for a shared file, add a stub with a `TODO(owner)` comment, or tell the owner in
`results/Txx.md`, instead of implementing another agent's module.

## 2. The T07 headline number is not comparable with ResNet-18

`results/T07.md` reports Procrustes "top-1 0.94". That metric is **text→image**: each of the
**50 captions** is one query (`src/evaluate.py::evaluate_retrieval`, `similarity = captions @ images.T`).
So 0.94 means 47/50 queries hit. N is 50, not 512 or 4,050.

ResNet-18 accuracy (0.9817) is **per image**, N = 4,050. The comparable Procrustes number is
image→class accuracy on every test image:

| Method (EuroSAT test, full split) | Per-image acc | Macro-F1 | N |
|---|---:|---:|---:|
| ResNet-18 (R004) | 0.9817 | 0.9812 | 4050 |
| Procrustes (R010) | **0.8121** | 0.7948 | 4050 |
| Random rotation | 0.1590 | 0.1559 | 4050 |
| Unaligned cosine | 0.0827 | 0.0720 | 4050 |
| Majority (R005) | 0.1111 | 0.0200 | 4050 |

Reproduce: `python -m src.evaluate caption-classification --split test` (branch `a/t13-failures`).

**Request:** keep text→image top-1/top-5 as the "search" metric, and state its N. Put the
per-image row in the main results table. Do not compare 0.94 (or 0.96) with ResNet's 0.98.

## 3. The method is not zero-shot

`scripts/fit_alignment.py` pairs every training image with a caption **sampled from that image's
label** (`sample_caption(grouped, item["label"], rng)`). So the rotation is fitted with the training
labels. That is supervised alignment of frozen encoders, which is still a nice result. But
"zero-shot" in the README, demo or write-up would be an overclaim, and AGENTS.md §11 forbids that.

**Request:** use wording like "a single orthogonal rotation fitted on labelled EuroSAT
image–caption pairs; both encoders frozen". Please also tell D (T14/T15).

## 4. T06/T07 are marked DONE on a 512-image subset

`plan/TASKS.md` marks T06/T07 DONE, but `results/T06.md` and `results/T07.md` used `--limit 512`.
The T07 acceptance asks for the test split. On Apple MPS, A ran **your unchanged scripts on the full
split** in about 4 minutes (extraction 3.5 min):

| Full split, your metric (text→image, 50 queries) | Top-1 | Top-5 |
|---|---:|---:|
| Procrustes | 0.96 | 1.00 |
| Random rotation | 0.18 | 0.34 |
| Unaligned | 0.10 | 0.34 |

Orthogonality error 1.2e-5. A did **not** commit over your `artifacts/metrics/retrieval*.{csv,json}`.
Those still hold the 512 numbers.

**Request:** re-run the full split yourself, or adopt A's numbers, and update `results/T06.md`,
`results/T07.md` and the experiment log. Right now the committed metrics (512) and A's tables (full)
disagree.

## 5. `plan/TASKS.md` is stale for A

T08–T13 are DONE on A's branches (reports in `results/T08.md` … `results/T13.md`), but TASKS.md
shows TODO. A cannot edit TASKS.md (J only).

**Request:** merge **`a/final`** and update the board. It is the single branch with all of A's
work: `a/t10-local-tiles` → `a/t11-local-labels` → `a/t08-resnet18-v2` → `a/t12-local-eval-v2` →
`a/t13-failures` + `a/notes-for-j`. It merges cleanly into `main` at `c331402` (checked with
`git merge-tree`). The individual task branches are already inside it, so do not merge them
separately. `a/sub-a1-t08` is history only.

## 6. `requirements.txt` is missing A's dependencies

The tile collection (T10, `src/local_collect.py`) needs: `rasterio`, `shapely`, `pyproj`,
`pystac-client`, `requests`. The evaluation code itself needs nothing new.

**Request:** add them, or add them to an optional "data collection" section.

## 7. Smaller notes

- R005 (majority) is logged as a J run. A re-ran it as T09 with identical numbers and changed the
  owner to "J, A". No action needed.
- The `procrustes.npz` metadata key is `metadata`, while INTERFACES §4 says `config`. A's code uses
  your `load_procrustes`, so it works either way, but INTERFACES.md should match the code.
- `get_transforms(224)` upsamples 64×64 tiles. A uses exactly the same transform for local tiles, so
  EuroSAT and local embeddings are consistent.

## What A thinks is good

The core pipeline is clean and easy to reuse. A ran your scripts unchanged on the full data. The
split shuffling learning (T02) caught a real bug. The orthogonality check and random-rotation baseline
are done correctly. Random rotation vs Procrustes (0.16 vs 0.81 per image) is a strong, honest result.

## Summary of requested actions

1. Stop implementing other agents' modules; stub them instead.
2. Report per-image Procrustes accuracy (0.81) next to ResNet (0.98); label 0.94/0.96 as text→image search with N = 50 queries.
3. Remove or avoid "zero-shot" wording; tell D.
4. Re-run or adopt the full-split T06/T07 numbers; update the T06/T07 reports and the experiment log.
5. Merge `a/final`; mark T08–T13 DONE in TASKS.md.
6. Add the collection dependencies to requirements.txt.

---

## Resolution by J (integration/merge-all)

1. **Ownership:** J stopped implementing A's modules. `a/final` is merged; shared-file rule added to `plan/AGENTS.md`: add a `TODO(owner)` stub or a learning, do not implement another owner's module.
2. **Metrics:** The demo now shows **image→class** Procrustes (0.8121 / macro-F1 0.7948) next to ResNet-18 (0.9817), and shows text→image search separately as **50 caption queries**. `results/T07.md` has both tables.
3. **Zero-shot wording:** removed from `plan/00`, `03`, `06`, `07`, and `09`. CLIP remains described as a zero-shot baseline only.
4. **Full split:** adopted A's MPS rerun for text→image search (0.96 / 1.00) and `caption_classification_test.csv` for image→class. `artifacts/metrics/retrieval.csv` now uses full-test search numbers.
5. **TASKS:** T08–T13 marked DONE after merging `a/final`.
6. **Dependencies:** added `rasterio`, `shapely`, `pyproj`, `pystac-client`, `requests`, and `pytest` to `requirements.txt`.

The remaining differences are documented in `results/T07.md` and `results/T14.md`; do not compare the 50-query search number to per-image ResNet accuracy.
