# Code review: combined notes (J, A, D)

**Branch:** `review/code-audit` (one branch for everyone's review notes and fixes)
**Base:** `main` @ 742ed89
**Started:** 2026-11-14 by A

How to use this file:
- One row per finding. Use the next free ID (`CR-xx`) and put your agent letter in "By".
- Severity: **bug** (wrong result or crash), **inconsistency** (code/docs/contracts disagree),
  **simplification** (dead or duplicate code).
- Fix it in the same branch when you can, then put the commit hash in "Fix". If you can't fix it,
  write `open` and say who should.
- Pull before you edit (`git pull --rebase origin review/code-audit`) and keep commits small.

## Findings

Reviewers for this pass: A, plus three independent read-only subagent reviews (J pipeline, A code, app.py). A verified every item and made the fixes. Rows say “A” because A made the change; ownership of each file is unchanged.

| ID | By | Severity | File | Finding | Fix |
|---|---|---|---|---|---|
| CR-01 | A | inconsistency | artifacts/metrics/retrieval_test.json, retrieval_val.{csv,json} | Still the 512-subset run (test top-1 0.94, val 0.94/0.98, N=512). `retrieval.csv` and every other metric are full split (test 0.96/1.00, val 0.98/1.00, N=4050). Mixed N in one folder. | Fixed: full pipeline re-run; val 0.98/1.00, test 0.96/1.00, N=4050 |
| CR-02 | A | simplification | artifacts/metrics/retrieval_test.json vs retrieval_test_full.json | Two files for the same full-split test numbers (one stale). | Fixed: J's naming adopted in the merge with ea8e575: `retrieval_test_full.json` (full) or `retrieval_test_<N>.json` (--limit), written by `evaluate_retrieval.py`; stale `retrieval_test.json` removed |
| CR-03 | A | inconsistency | artifacts/metrics/majority.json vs majority.csv | Two majority implementations break the 5-way class tie differently: `majority.json` says SeaLake (`Counter.most_common`), `majority.csv` says AnnualCrop (`class_names` order). Same accuracy, contradictory "majority class". | Fixed: see CR-07 |
| CR-04 | A | simplification | requirements.txt | `scipy` and `tqdm` are required but never imported anywhere in src/, scripts/, app.py or tests/. | Fixed: removed |
| CR-05 | A | inconsistency | scripts/evaluate_local.py | `--config` is parsed and used for validation, but `run_local_eval()` ignores it and always uses the default config/paths. | Fixed: `run_local_eval(config_path)` |
| CR-06 | A | bug | src/data.py `make_splits` | Val and test shares swapped: `test_size=val_share/(val+test)` but the held-out half is named `test_idx`. No effect at 0.15/0.15 (re-generated splits are byte-identical), wrong for any other config. | `test_size=test_share/(val+test)` |
| CR-07 | A | bug | scripts/evaluate_majority.py | Crashed (`save_json` on a numpy confusion matrix) and used a second majority implementation with a different tie-break (SeaLake vs AnnualCrop). | Script is a thin wrapper over `src.baselines.run_majority`, the only implementation, which writes majority.csv and majority.json (AnnualCrop, 0.1111). `evaluate.majority_baseline` removed. |
| CR-08 | A | bug | src/local_tiles.py `local_metrics` | “Local majority” (always Residential, 0.62) was the mode of the **test** labels: an oracle, presented as a baseline. | Renamed `oracle_majority_local` and labelled as an oracle everywhere; added the fitted baseline `majority_train` (EuroSAT train majority AnnualCrop) = **0/50**. Docs, app and figures updated. |
| CR-09 | A | bug | src/failures.py `local_confusion` | Crosstab rows and columns had different label sets, so the visual diagonal was not “correct”. | Reindex both axes to the same sorted label set. |
| CR-10 | A | bug | src/failures.py `taxonomy` | Counts were read from the top-8 confusion pairs only; any other pair silently became 0. | `run_caption_classification` now saves the full confusion matrix (`procrustes_{split}_confusion.csv`); failures reads it instead of recomputing. |
| CR-11 | A | bug | src/failures.py | Crashed with `AttributeError` when no ResNet checkpoint existed (`run_local_eval` silently drops those columns). | ResNet parts are optional; `run_local_eval` prints a warning. |
| CR-12 | A | bug | src/local_tiles.py, app.py | Local embedding cache: `run_local_eval` checked nothing, the app checked tile_ids only, the app wrote a second sidecar (`.ids.json`), and the meta file could go stale. A model change would silently reuse old vectors. | One cache API (`local_embeddings` / `save_local_embeddings`): reuse only if tile_ids (in order), model name and input size match; used by the app, `extract_local_embeddings.py` and `run_local_eval`. |
| CR-13 | A | bug | app.py `_load_local` | Any local-gallery error (one missing PNG, offline without DINOv2) bubbled up and switched the whole demo to MOCK. | Local gallery failures disable only that gallery, with a warning. |
| CR-14 | A | bug | app.py `tile_image` | A missing image in REAL mode showed a tile stamped “MOCK”. | Neutral grey “image missing” tile; one module-level `tile_image` for both backends. |
| CR-15 | A | bug | scripts/extract_embeddings.py | Device fallback was cuda→cpu, ignoring MPS (unlike `get_device` used everywhere else). | `get_device()`. |
| CR-16 | A | bug | scripts/evaluate_retrieval.py, src/evaluate.py | “Unaligned” baseline crashes for ViT-B (768-d images vs 384-d text), which INTERFACES allows. | Skipped when dims differ. |
| CR-17 | A | bug | scripts/fit_alignment.py, evaluate_retrieval.py, src/evaluate.py | No check that embeddings match their split beyond (sometimes) a count; a re-run of prepare_data would silently misalign everything. | `dinov2_meta.json` records tile_ids per split; `src.encoders.check_split_embeddings` checks count and order before every fit/eval, printing both shapes. |
| CR-18 | A | inconsistency | scripts/evaluate_retrieval.py, src/evaluate.py | Random-rotation baseline: hard-coded seed 42, one shuffle. 50-query top-1 varies 0.10–0.18 by seed. | Config seed; mean of 5 seeds (42–46) with std in notes. Text search 0.152 ± 0.043 (was 0.18); per-image 0.1551 (was 0.159). Docs updated. |
| CR-19 | A | inconsistency | fit_alignment.py, evaluate_retrieval.py, src/evaluate.py | Three copies of training-pair (caption) sampling that had to stay in lockstep. | One definition: `src.captions.sample_caption_indices` + `src.alignment.training_pairs`; random rotation via `random_rotation_params`. evaluate_retrieval no longer loads MiniLM or re-encodes 18,900 strings. |
| CR-20 | A | inconsistency | scripts/evaluate_retrieval.py | Caption labels came from captions.jsonl while vectors came from the npy; order never checked. | Labels from `minilm_caption_index.json` with a length check. |
| CR-21 | A | inconsistency | configs/default.yaml | Keys never read (`alignment.normalize/method`, `local_tiles_root`, `local.*`, `demo.confidence_threshold`); `alignment.center` and `*.embedding_dim` ignored; `data.download` dead from the CLI. | Unused keys removed; `center` wired into fit and baselines; embedding dims asserted; `prepare_data` honours `data.download` unless `--no-download`; `demo.top_k` drives the app. |
| CR-22 | A | inconsistency | scripts/fit_alignment.py | INTERFACES §9 says stop if W is not orthogonal; script only printed. `--device` parsed but unused. | Refuses to save if error > 1e-3; `--device` passed to MiniLM. |
| CR-23 | A | inconsistency | scripts/evaluate_retrieval.py | Subset note hard-coded `< 4050`. | Compares with the split size. |
| CR-24 | A | inconsistency | plan/INTERFACES.md | §3/§5 drifted from code (caption ids, split paths, `load_dinov2` “and processor”, signatures, classify_tile, evaluate_classifier keys, local cache, config schema). | Rewritten to match the code. |
| CR-25 | A | inconsistency | src/baselines.py vs src/data.py, src/config.py, src/encoders.py | Duplicate config loader (+ DEFAULT_CONFIG), `SplitDataset`, `load_split`, `repo_path`, `pick_device`, `seed_everything`, ImageNet constants. | baselines uses `src.config`, `src.data.SplitDataset`, `get_device`, `set_seed`, shared constants. Test fixture switched to J's real split layout. |
| CR-26 | A | inconsistency | app.py | Re-implemented caption loading (no length check), class scoring and top-k; duplicated EUROSAT_CLASSES and TOP_K. | Uses `load_caption_artifacts`, `src.retrieval.class_scores` / `retrieve_top_k_with_scores` (the same rule as evaluation), `EUROSAT_CLASSES`, `config.demo.top_k`. Uploads still match logged T12 predictions on 50/50. |
| CR-27 | A | inconsistency | app.py, assets/demo_script.md | Hard-coded numbers (31/50, 62%, “by 1 tile”, 50 captions, 50 tiles, 32%); demo script heading “Where it fails” didn't exist; clicking an example didn't search; the “low scores” claim was contradicted by real queries (“open water” scored below “climate change”); OOD warning said “Upper Arlington” for every upload. | Numbers formatted from metric files; heading renamed; examples run the search; “open water” example dropped (no lake tiles locally); probe/demo text reworded; generic OOD warning. |
| CR-28 | A | inconsistency | app.py | `--root` was half honoured (config and local tiles still used the code tree); app never set offline mode though the script says Wi-Fi off; subset note said “stratified” for a first-N subset. | `--root` removed; `HF_HUB_OFFLINE=1` by default unless `--prefetch`; “first N of M” note shown in the caption. |
| CR-29 | A | inconsistency | src/local_tiles.py | `Other` tiles: local.json used all rows, the tables used scorable rows. Blank `path` crashed with TypeError. `ambiguous` parsing fragile. local_predictions.csv duplicated four columns. `--config` ignored. Docstring claimed a plain 64→224 upscale. | All metrics on scorable tiles with `n_other_excluded`; blank path → ManifestError; robust bool parse; no duplicate columns; `run_local_eval(config_path)`; docstring states Resize(256)+CenterCrop(224). |
| CR-30 | A | inconsistency | tests/test_local_tiles.py | cwd-dependent path; failed when run outside the repo. | Uses `ROOT / DEFAULT_MANIFEST`; new tests for `Other` exclusion and blank path. |
| CR-31 | A | bug | src/baselines.py | `epochs=0` → KeyError; `predict_resnet18` lru_cache keyed by path served a stale model after retraining; unknown string labels gave a confusing `int()` error. | Explicit error; cache keyed by (path, mtime); clear `label not in class_names` error. |
| CR-32 | A | simplification | src/retrieval.py, src/evaluate.py, src/encoders.py, src/data.py, src/config.py | Dead or duplicate: `classify_tile`, `class_centroids`, `evaluate_image_to_text`, `confusion_matrix`, `save_confusion_matrix_figure`, `save/load_checkpoint`, `save/load_metadata`, `split_labels`, `return_id`, unused train transform, identical batch branches, runtime PYTHONHASHSEED, unused imports. | Removed (`src.retrieval` now holds the one scoring rule `class_scores`). |
| CR-33 | A | simplification | src/local_collect.py | Recomputed the score grid per tile; silenced every DeprecationWarning process-wide. | Score grids once per sub-type; filter only shapely's `ops.transform` notice. |
| CR-34 | A | simplification | scripts/run_j_pipeline.sh, README | Pipeline script stopped before the per-image, local and failure steps the README reports. | Runs the full verified pipeline (ResNet optional via `SKIP_RESNET=1`). |
| CR-35 | A | inconsistency | app.py | No tests for the demo. | tests/test_app.py: mock UI builds, results tables from committed metrics, logged examples, missing-tile fallback. |
| CR-36 | A | note | src/data.py `get_transforms` | Resize(256)+CenterCrop(224) crops ~12.5 % of each 64×64 tile's border. | **Kept by design** (applies identically to EuroSAT and local tiles; changing it would invalidate all logged embeddings). Documented; candidate ablation. |
| CR-37 | A | note | data | Local L2A TCI tiles are darker and much less blue than EuroSAT (mean RGB ≈ [71, 82, 65] vs [90, 98, 105]). | Not a code bug; a likely contributor to the OOD drop. Mention in the write-up (D). |
| CR-38 | A | inconsistency | merge with main ea8e575 (J, “Cohere code…”) | J changed the same files on main in parallel. | Kept J's improvements on top of the review fixes: `--limit` overwrite guards (now one helper, `src.config.refuse_subset_overwrite`, also applied to `fit_alignment.py`), `retrieval_test_full.json` / `retrieval_test_<N>.json` naming, legacy `config` key in `load_procrustes`, table-1 wording, `metrics_table` alias, data-derived canopy examples (J's correction of ua_030/032–034). Dropped J text that still quoted 512-tile galleries, random 15.9 % or an unlabelled “local majority”. |
| CR-39 | A | bug | src/encoders.py `check_split_embeddings` (added in this review) | With `--limit`, compared the full recorded tile_id list with the truncated split, so every smoke run failed. | Compare the recorded prefix. Verified: `--limit 64` refused without `--force`; with `--force` writes `retrieval_test_64.json`; full rerun restores `retrieval.csv` identically. |

## Verification log

| When | By | Command | Result |
|---|---|---|---|
| start | A | every documented `python ...` command `--help` (scripts/*, app.py, src.baselines/local_tiles/evaluate) | all flags in README/runbooks/results exist |
| start | A | in-memory recompute of EuroSAT val/test metrics vs committed CSVs | test matches (0.96/1.00; image→class 0.8121); val files stale (CR-01) |
| start | A | fresh `git clone` (no gitignored artifacts): unittest core+local, test_baselines, `app.load_backend()` + `build_ui` | 13+4 tests pass; app falls back to MOCK with a clear missing-artifact message; results tables still render from committed CSVs |
| after fixes | A | `scripts/prepare_data.py --no-download` | splits + captions byte-identical to committed (CR-06 has no effect at 15/15) |
| after fixes | A | full pipeline: extract → fit → retrieval → majority → caption-classification val/test → local extract → local eval → failures | all steps pass; orthogonality 1.2e-5; numbers below |
| after fixes | A | metrics | ResNet-18 0.9817 (unchanged checkpoint); Procrustes per-image test 0.8121 / val 0.8114; text→image test 0.96/1.00, val 0.98/1.00; random rotation 0.152±0.043 (search), 0.1551 (per-image); unaligned 0.10 / 0.0827; majority 0.1111 (AnnualCrop); local Procrustes 0.64, ResNet 0.64, EuroSAT-majority 0.00, oracle 0.62 |
| after fixes | A | `HF_HUB_OFFLINE=1` REAL app: uploads of all 50 local tiles vs logged T12 predictions | 50/50 identical; 3 tables populated; REAL + MOCK UIs build |
| after fixes | A | `unittest tests.test_core tests.test_local_tiles tests.test_app` + `tests/test_baselines.py` | 18 + 4 pass (also from outside the repo for local tiles) |

| after merge | A | tests + REAL app after merging main ea8e575 | 18 + 4 tests pass; uploads 50/50 match logged T12; guards verified (CR-39) |

## Open (owner decides)

- J: confirm the random-rotation change (mean of 5 seeds) is how T07 should be reported.
- D: write-up wording for CR-08 (oracle vs real majority) and CR-37 (brightness shift).
- Human: spot-check the 13 `ambiguous` local tiles (from T11).
