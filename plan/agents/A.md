# A — Baselines and Local Validation Lead

Read `plan/AGENTS.md`, `plan/DECISIONS.md`, and `plan/INTERFACES.md` first.

## Owns

- `src/baselines.py`
- `src/evaluate.py` metrics helpers
- `src/local_tiles.py`
- `scripts/train_resnet.py`
- `scripts/evaluate_majority.py`
- `scripts/extract_local_embeddings.py`
- `scripts/evaluate_local.py`
- `data/local_tiles/manifest.csv`
- `data/raw/local_tiles/`
- `artifacts/metrics/resnet18.csv`, `majority.csv`, `local.csv`
- Notebooks 04–05

## Tasks and exact commands

### T09 — Majority baseline

J already ran this while scaffolding. Verify the metrics and write `results/T09.md` under your name.

```sh
python scripts/evaluate_majority.py
```

Current output: accuracy `0.1111`, macro-F1 `0.0200`, majority class `SeaLake`, N=4050.

Outputs:

- `artifacts/metrics/majority.json`
- `artifacts/metrics/majority.csv`
- `results/T09.md`

### T08 — ResNet-18 baseline

GPU:

```sh
python scripts/train_resnet.py --epochs 10
```

CPU smoke (faster, but report the lower epoch count):

```sh
python scripts/train_resnet.py --epochs 2 --batch-size 32
```

Outputs:

- `artifacts/metrics/resnet18.json`
- `artifacts/metrics/resnet18.csv`
- `artifacts/checkpoints/resnet18.pt`
- `results/T08.md`

The app reads `resnet18.csv` columns:

```csv
method,accuracy,macro_f1,test_n,notes
resnet18,0.96,0.95,4050,epochs=10
```

### T10–T11 — Upper Arlington tiles

Follow `plan/14-upper-arlington-data-collection.md`.

Required paths:

- Images: `data/raw/local_tiles/ua_001.png` … `ua_050.png`
- Manifest: `data/local_tiles/manifest.csv`
- Manifest `path` column: **repo-relative**, e.g. `data/raw/local_tiles/ua_001.png`

Manifest columns must be exactly:

```csv
tile_id,path,label,local_category,lat,lon,date,source,notes
ua_001,data/raw/local_tiles/ua_001.png,Residential,residential,40.02,-83.06,2026-06-01,Copernicus,Old Arlington
```

Validation:

```sh
python -c 'from src.local_tiles import load_manifest, validate_manifest; from src.config import load_config, resolve_path; c=load_config(); m=load_manifest(resolve_path(c["data"]["local_manifest"], c)); print(validate_manifest(m))'
```

If you have fewer than 50 tiles, use `--allow-partial` and report N honestly.

### T12 — Local OOD evaluation

```sh
python scripts/extract_local_embeddings.py
python scripts/evaluate_local.py
```

If partial:

```sh
python scripts/extract_local_embeddings.py --allow-partial
python scripts/evaluate_local.py --allow-partial
```

Outputs:

- `artifacts/embeddings/dinov2_local.npy`
- `artifacts/embeddings/dinov2_local_meta.json`
- `artifacts/metrics/local.csv`
- `artifacts/metrics/local_by_category.csv`
- `artifacts/metrics/local.json`
- `results/T12.md`

`local.csv` columns:

```csv
tile_id,true_label,local_category,predicted_label,score,correct
```

### T13 — Failure taxonomy

Write `results/T13.md` with:

- At least 3 failure types.
- One concrete tile/query per type.
- Likely cause.
- One failure for D's demo.

Suggested types:

1. AnnualCrop vs PermanentCrop.
2. Forest vs HerbaceousVegetation.
3. Industrial vs Residential.
4. Upper Arlington domain shift.
5. Cloud/shadow/season.

## Rules

- Do not edit J's `src/alignment.py`, `src/encoders.py`, `src/data.py`, or `scripts/fit_alignment.py`.
- Do not edit D's `app.py`.
- Do not change `procrustes.npz`.
- Do not claim full EuroSAT results if you trained on a subset.
- If a script fails, copy the exact error into `results/Txx.md` and `plan/learnings/`.
- If you need a change in J's module, write a learning file and ask J.

## Current blocker

J's Procrustes artifacts are ready (N=512). A can start T08/T09 immediately. Local scripts are ready in `scripts/`.
