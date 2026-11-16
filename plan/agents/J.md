# J — Integrator and Data/Alignment Lead

Read `plan/AGENTS.md`, `plan/DECISIONS.md`, and `plan/INTERFACES.md` first.

## Owns

- `configs/`
- `src/config.py`
- `src/data.py`
- `src/captions.py`
- `src/encoders.py`
- `src/alignment.py`
- `src/retrieval.py`
- `scripts/prepare_data.py`
- `scripts/extract_embeddings.py`
- `scripts/fit_alignment.py`
- `scripts/evaluate_retrieval.py`
- Notebooks 01–03
- Integration and final merge

## Tasks

| Task | Command / action | Report |
|---|---|---|
| T01 scaffold | `python -m compileall -q src scripts`; `pytest -q` | `results/T01.md` |
| T02 EuroSAT | `python scripts/prepare_data.py` | `results/T02.md` |
| T03 captions | same as T02 | `results/T03.md` |
| T04 DINOv2 | CPU smoke: `python scripts/extract_embeddings.py --limit 512`; GPU: no `--limit` | `results/T04.md` |
| T05 MiniLM | `python scripts/fit_alignment.py --limit 512` | `results/T05.md` |
| T06 Procrustes | same as T05 | `results/T06.md` |
| T07 retrieval | `python scripts/evaluate_retrieval.py --limit 512` | `results/T07.md` |
| Majority | `python scripts/evaluate_majority.py` | A owns `results/T09.md` |

**Do not run `--limit` commands on `main` after adopting A's full-split metrics.** They overwrite `procrustes.npz` and `retrieval.csv`. Use a scratch worktree or restore the committed files afterward.

## Current status

- T01–T07 are done. A's full-split rerun provides the headline numbers:
  - text→image search: **0.96 top-1 / 1.00 top-5** over 50 caption queries;
  - image→class: **0.8121 accuracy / 0.7948 macro-F1** over 4,050 test images.
- J's CPU run used the first 512 test tiles and is retained as a smoke test.
- A's `a/final`, D's `d/t14-gradio`, and D's `d/review-for-j` are merged.
- D's `app.py` RealBackend is verified with Upper Arlington N=50 and EuroSAT N=512; A's full-split metrics are in the Results tables.

## Integration

```sh
git fetch --all
git switch main
git pull --rebase origin main
git merge --no-ff <agent-branch> -m "Merge <task>"
git push origin main
```

Only J merges or pushes `main`.

## App integration check

```sh
python -c 'import sys, types; sys.modules["gradio"] = types.ModuleType("gradio"); from app import load_backend; b=load_backend(); print(b.name, len(b.items), b.full_test_n, b.subset_note)'
```

Expected with current artifacts:

```text
[demo] REAL mode — Upper Arlington N=50; EuroSAT test N=512 of 4050; transforms from src.alignment
backend REAL 512 4050 first 512 of 4050 test tiles (--limit run)
```

## Do not

- Do not implement A's or D's modules. Add a `TODO(owner)` stub or a learning file instead.
- Do not edit A's local tile manifest or data.
- Do not edit D's write-up or video script.
- Do not change `app.py` unless fixing an integration bug; record the fix in `results/T14.md`.
- Do not compare the 50-query text search number with per-image ResNet accuracy.
- Do not call the method zero-shot; the rotation is fitted on labelled pairs.
- Do not hide a failed merge or test.

## Current blocker

Implementation is clear. J still owns T17/T18 after D finishes T15/T16: verify links, metrics against logged runs, and credits.
