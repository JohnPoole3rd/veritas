# D — Demo and Documentation Lead

Read `plan/AGENTS.md`, `plan/DECISIONS.md`, and `plan/INTERFACES.md` first.

## Owns

- `app.py`
- `assets/`
- `README.md` implementation section
- one-page write-up
- demo video
- `results/T14.md`, `T15.md`, `T16.md`

## Current state

- `app.py` has MockBackend and RealBackend.
- RealBackend verified with J's artifacts + A's metrics: Upper Arlington gallery N=50, EuroSAT test N=512 of 4,050, upload works. On the EuroSAT gallery, `dense forest` returns Forest; on the default Upper Arlington gallery, use local queries like `commercial area` or `river corridor`.
- The Results tab shows image→class Procrustes (81.2%) next to ResNet-18 (98.2%), and text search separately (50 queries, 96% top-1).
- Current limitation: the live EuroSAT gallery uses the committed 512-tile embeddings. Full-split 4,050 metrics are benchmarks from A's separate run.

## Exact commands

### Mock mode

```sh
python app.py --mock
```

Use this only for UI rehearsal. Do not screenshot mock data for the write-up.

### Real mode

```sh
python app.py
```

Expected:

```text
[demo] REAL mode — Upper Arlington N=50; EuroSAT test N=512 of 4050; transforms from src.alignment
[demo] NOTE — using 512-tile stratified subset
```

If it says `MOCK mode`, the missing artifact list tells you what is absent.

### Offline model cache

The demo needs MiniLM for new text queries and DINOv2 for uploads. Pre-warm both on the demo machine:

```sh
python app.py --prefetch
HF_HUB_OFFLINE=1 python app.py
```

## Artifact contract the app reads

- `artifacts/alignment/procrustes.npz` — `W`, `mu_x`, `mu_y`, `metadata` JSON.
- `artifacts/embeddings/dinov2_test.npy` — may be a subset; app slices `test.json` to match.
- `artifacts/embeddings/minilm_captions.npy`
- `artifacts/embeddings/minilm_caption_index.json`
- `data/splits/test.json`
- `artifacts/metrics/majority.csv`
- `artifacts/metrics/resnet18.csv` (A)
- `artifacts/metrics/caption_classification_test.csv` — image→class rows
- `artifacts/metrics/retrieval.csv` — text search rows (50 queries)
- `artifacts/metrics/local.json` / `local.csv` (A, optional for the demo)

The table shows image→class Procrustes separately from the 50-query text search row. Do not compare them.

## Write-up

Write the one-page write-up to `WRITEUP.md` at the repo root and link it from `results/T15.md`. Use the brief's template in `plan/09-demo-writeup-submission.md`.

Use the full-split tables in `results/T07.md`:
- text→image search = 50 caption queries over 4,050 test tiles: Procrustes 0.96/1.00, random 0.18/0.34, unaligned 0.10/0.34;
- image→class = 4,050 images: Procrustes 0.8121/0.7948, ResNet-18 0.9817/0.9812, majority 0.1111/0.0200.

Required sections: problem, user, input/output, data, model, results (two tables), failure (ua_049: Procrustes Highway, ResNet HerbaceousVegetation; plus `climate change`), what we added, next steps, sources/credits.

Do not quote the old 512-subset 0.94/0.96 values except as J's CPU smoke-test caveat. Do not compare the 50-query search number directly with per-image ResNet accuracy. Do not call the method zero-shot.

## Rules

- Do not edit J's artifacts or `src/alignment.py`.
- Do not edit A's local data or metrics.
- If `app.py` fails, copy the exact error into `results/T14.md`.
- If a fix belongs in J's code, write a learning file and ask J.
- If you change `app.py`, re-run both `--mock` and real mode.
- Do not claim that model scores are probabilities. They are cosine similarity.
