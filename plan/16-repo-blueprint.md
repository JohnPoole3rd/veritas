# Repository Blueprint

This is the concrete implementation blueprint for `/home/john/veritas`. J/A/D implementation is merged through T14; T15–T18 remain.

## 1. Directory tree

```text
veritas/
  configs/
    default.yaml
  data/
    raw/
      eurosat/
      local_tiles/
    splits/
      train.json
      val.json
      test.json
    captions.jsonl
    local_tiles/
      manifest.csv
  artifacts/
    embeddings/
      dinov2_train.npy
      dinov2_val.npy
      dinov2_test.npy
      dinov2_meta.json
      dinov2_local.npy
      dinov2_local_meta.json
      minilm_captions.npy
      minilm_caption_index.json
    alignment/
      procrustes.npz
    checkpoints/
      resnet18.pt
    metrics/
      retrieval.csv
      retrieval_test_full.json
      caption_classification_test.csv
      caption_classification_val.csv
      resnet18.csv
      majority.csv
      local.csv
      local_by_category.csv
      local.json
      local_summary.csv
      failure_taxonomy.csv
      procrustes_test_per_class.csv
      procrustes_test_top_confusions.csv
    figures/
      failure_gallery.png
      local_confusion.png
      procrustes_test_confusion_matrix.png
      procrustes_val_confusion_matrix.png
      resnet18_confusion_matrix.png
  results/
    T01.md
    T02.md
    ...
    T18.md
  WRITEUP.md
  plan/
    agents/
      README.md
      J.md
      A.md
      D.md
    learnings/
      README.md
      _template.md
      T04-cpu-extraction.md
  scripts/
    prepare_data.py
    extract_embeddings.py
    fit_alignment.py
    evaluate_retrieval.py
    evaluate_majority.py
    train_resnet.py
    extract_local_embeddings.py
    evaluate_local.py
    run_j_pipeline.sh
  src/
    __init__.py
    config.py
    data.py
    captions.py
    encoders.py
    alignment.py
    retrieval.py
    evaluate.py
    baselines.py
    local_tiles.py
    failures.py
    local_collect.py
  notebooks/
    01_data.ipynb
    02_embeddings.ipynb
    03_alignment.ipynb
    # 04/05 not used; A implemented T08-T13 as scripts
  tests/
    test_core.py
    test_baselines.py
    test_local_tiles.py
  app.py
  README.md
  requirements.txt
  .gitignore
```

## 2. Module responsibilities

### `src/config.py`

- Load `configs/default.yaml`.
- Resolve paths relative to repo root.
- Validate required keys.
- Set seed.

### `src/data.py`

- Download/load EuroSAT.
- Create stratified splits.
- Save/load split JSON.
- Return PyTorch Dataset/DataLoader.
- Deduplicate.

### `src/captions.py`

- Build caption templates for all 10 classes.
- Save/load `captions.jsonl`.
- Sample a caption for a label.
- Optionally build class centroid captions.

### `src/encoders.py`

- Load DINOv2 (frozen).
- Extract CLS embeddings in batches.
- Load MiniLM (frozen).
- Encode texts.
- Save/load embeddings.
- Handle device (CPU/GPU).

### `src/alignment.py`

- Center embeddings.
- Fit Procrustes via SVD.
- Save/load `W`, `mu_x`, `mu_y`.
- Transform images/texts.
- L2-normalize.
- Validate orthogonality.

### `src/retrieval.py`

- Top-k cosine retrieval.
- Image→text classification.
- Class centroid search.
- Optional confidence threshold.

### `src/evaluate.py`

- Top-1/top-5 retrieval.
- Macro-F1.
- Per-class metrics.
- Confusion matrix.
- Save CSV/figures.

### `src/baselines.py`

- Majority baseline.
- ResNet-18 training/eval.
- Optional CLIP zero-shot.

### `src/local_tiles.py`

- Load/validate Upper Arlington manifest.
- Load local images.
- Evaluate Procrustes and ResNet.
- Save local CSV.

### `src/local_collect.py`

- Collect Upper Arlington Sentinel-2 tiles via STAC.
- Build the manifest and collection log.
- Used for T10, not the demo.

### `src/failures.py`

- Build failure taxonomy, per-class confusions, and failure gallery figures.
- Used for T13.

### `app.py`

- Gradio UI.
- Load cached artifacts.
- Text search.
- Image upload.
- Baseline comparison.
- Failure example.
- Prototype disclaimer.

## 3. Notebook sequence

### `01_data.ipynb`

- Load EuroSAT.
- Print class counts.
- Visualize sample tiles.
- Create splits.
- Save split JSON.
- Build captions.
- Save captions.

**Checkpoint:** class counts, split sizes, no duplicates.

### `02_embeddings.ipynb`

- Load DINOv2.
- Extract train/val/test embeddings.
- Load MiniLM.
- Encode captions.
- Save `.npy` and metadata.
- Print shapes/norms.

**Checkpoint:** `(N, 384)`, no NaNs.

### `03_alignment.ipynb`

- Load embeddings.
- Center.
- Fit Procrustes.
- Save `procrustes.npz`.
- Evaluate validation top-1.
- Compare random/unaligned.
- Visualize 5 retrievals.

**Checkpoint:** Procrustes beats random.

### `04_evaluation.ipynb` — not used

A implemented T08–T13 as scripts (`python -m src.baselines`, `python -m src.evaluate`, `python -m src.failures`). Do not create this notebook unless you need a Colab walkthrough.

### `05_local_tiles.ipynb` — not used

Use `scripts/extract_local_embeddings.py` and `scripts/evaluate_local.py` instead.

**Checkpoint:** 10 per sampling category, N=50.

## 4. Config

`configs/default.yaml` must include:

```yaml
seed: 42
data:
  eurosat_root: data/raw/eurosat
  local_tiles_root: data/raw/local_tiles
  splits_dir: data/splits
  captions_path: data/captions.jsonl
  local_manifest: data/local_tiles/manifest.csv
split:
  train: 0.70
  val: 0.15
  test: 0.15
image:
  model_name: facebook/dinov2-small
  input_size: 224
  batch_size: 32
  embedding_dim: 384
text:
  model_name: sentence-transformers/all-MiniLM-L6-v2
  embedding_dim: 384
alignment:
  center: true
  normalize: true
  method: orthogonal_procrustes
baseline:
  resnet18_epochs: 10
  resnet18_batch_size: 64
  resnet18_lr: 0.001
local:
  city: Upper Arlington, Ohio
  tile_count: 50
demo:
  top_k: 3
  confidence_threshold: null
```

## 5. Requirements

`requirements.txt`:

```text
torch
torchvision
transformers
sentence-transformers
scikit-learn
scipy
numpy
pandas
matplotlib
gradio
pyyaml
tqdm
Pillow
pytest

# optional tile collection
rasterio
shapely
pyproj
pystac-client
requests
```

Pin versions after the first successful run.

## 6. `.gitignore`

```text
data/raw/
artifacts/embeddings/
artifacts/alignment/
artifacts/checkpoints/
*.npy
*.npz
*.pt
*.pth
*.log
__pycache__/
.ipynb_checkpoints/
.env
.venv/
```

The 50 local tiles are force-added under `data/raw/local_tiles/`; new tiles will be ignored unless force-added. Commit small CSVs, figures, notebooks, and code. Do not commit large binaries.

## 7. README requirements

The implementation README must include:

- Project one-liner.
- Track and site.
- Setup instructions.
- How to run notebooks in order.
- How to run the demo.
- Where artifacts go.
- Where task reports go (`results/Txx.md`).
- Where learnings go (`plan/learnings/`).
- Where the write-up goes (`WRITEUP.md`) and the optional video link (`results/T16.md`).
- Known limitations.
- Credits.

## 8. Test plan

### Unit checks

- [ ] Split JSON has no duplicates.
- [ ] Caption every class has ≥5.
- [ ] Embedding shapes match.
- [ ] No NaNs.
- [ ] `W.T @ W ≈ I`.
- [ ] Transform is deterministic.
- [ ] Top-k returns k unique indices.
- [ ] Local manifest has 50 rows.
- [ ] Local categories have 10 each.
- [ ] `tests/test_local_tiles.py` passes.
- [ ] `pytest -q` passes the full suite.
- [ ] Every completed task has `results/Txx.md`.
- [ ] Important findings have a file under `plan/learnings/`.

### Integration checks

- [ ] Notebook 01 runs top-to-bottom.
- [ ] Notebook 02 runs after 01.
- [ ] Notebook 03 runs after 02.
- [ ] Notebook 04 runs after 03.
- [ ] Notebook 05 runs after 04.
- [ ] `app.py` loads cached artifacts.
- [ ] `python app.py --mock` builds.
- [ ] Real backend prints `[demo] REAL mode` with J's artifacts.
- [ ] Real backend slices `test.json` when `dinov2_test.npy` is a subset.
- [ ] `results_tables()` (and the legacy `metrics_table()` wrapper) reads majority, ResNet-18, image→class, and text-search metrics.
- [ ] Demo works offline after the models are cached.

### Evaluation checks

- [ ] Test split untouched.
- [ ] Majority baseline reported (AnnualCrop, N=4050).
- [ ] ResNet-18 reported (0.9817/0.9812).
- [ ] Procrustes image→class reported (0.8121/0.7948, N=4050).
- [ ] Procrustes text search reported separately (0.96/1.00, 50 queries).
- [ ] Random rotation/unaligned reported for image→class and search.
- [ ] Local N=50 and 37-unambiguous reported.
- [ ] One failure explained (ua_049 + abstract query).
- [ ] No zero-shot claim.
- [ ] Live gallery N=50/512 vs full-split metrics clearly distinguished.

## 9. Run order

```text
01_data → 02_embeddings → 03_alignment → app.py
```

`04_evaluation.ipynb` and `05_local_tiles.ipynb` are not used; A implemented T08–T13 as scripts (`scripts/train_resnet.py`, `evaluate_local.py`, etc.).

Do not skip a notebook; each saves artifacts the next one needs.

## 10. Commit conventions

- `feat: add EuroSAT loader`
- `feat: extract DINOv2 embeddings`
- `feat: fit Procrustes rotation`
- `eval: add ResNet-18 baseline`
- `data: add Upper Arlington manifest`
- `demo: add Gradio app`
- `docs: add results and limitations`

Keep commits small and scoped.

## 11. Definition of done

The repo is done when:

- All notebooks run top-to-bottom.
- All artifacts exist.
- All baselines are reported.
- Local tiles are evaluated.
- Demo runs offline.
- Write-up links to the repo.
- README explains how to reproduce.

## 12. No-code rule for the plan folder

The `/home/john/veritas/plan` folder contains planning only. Implementation goes in the repo root (`/home/john/veritas`), not inside `plan/`. Do not put notebooks or data in the plan folder.
