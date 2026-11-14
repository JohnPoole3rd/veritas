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
- J verified RealBackend with real artifacts: gallery N=512, `dense forest` returns Forest, upload works.
- Current limitations: failure-gallery images are placeholders; local Upper Arlington tiles are not in the gallery.

## Exact commands

### Mock mode

```sh
python app.py --mock
```

Use this while A's local metrics are missing. Do not screenshot mock data for the write-up.

### Real mode

```sh
python app.py
```

Expected:

```text
[demo] REAL mode — gallery N=512, transforms from src.alignment
[demo] NOTE — using 512-tile stratified subset
```

If it says `MOCK mode`, the missing artifact list tells you what is absent.

### Offline model cache

The demo needs MiniLM for new text queries and DINOv2 for uploads. Pre-warm both on the demo machine:

```sh
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
python -c "from transformers import AutoModel, AutoImageProcessor; AutoModel.from_pretrained('facebook/dinov2-small'); AutoImageProcessor.from_pretrained('facebook/dinov2-small')"
```

Then run with Wi-Fi off and confirm Real mode still loads.

## Artifact contract the app reads

- `artifacts/alignment/procrustes.npz` — `W`, `mu_x`, `mu_y`, `metadata` JSON.
- `artifacts/embeddings/dinov2_test.npy` — may be a subset; app slices `test.json` to match.
- `artifacts/embeddings/minilm_captions.npy`
- `artifacts/embeddings/minilm_caption_index.json`
- `data/splits/test.json`
- `artifacts/metrics/majority.csv`
- `artifacts/metrics/resnet18.csv` (A)
- `artifacts/metrics/retrieval.csv`
- `artifacts/metrics/local.json` / `local.csv` (A, optional for the demo)

## Write-up

Use only numbers from:

- `results/T07.md`
- `results/T08.md`
- `results/T09.md`
- `results/T12.md`
- `results/T13.md`

State N=512 for J's retrieval numbers and the CPU limitation. Do not invent full-split numbers.

## Rules

- Do not edit J's artifacts or `src/alignment.py`.
- Do not edit A's local data or metrics.
- If `app.py` fails, copy the exact error into `results/T14.md`.
- If a fix belongs in J's code, write a learning file and ask J.
- If you change `app.py`, re-run both `--mock` and real mode.
- Do not claim that model scores are probabilities. They are cosine similarity.
