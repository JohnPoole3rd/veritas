# Repository Blueprint

This is the concrete implementation blueprint for `/home/john/veritas`. No code exists yet; this is the target structure.

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
      minilm_captions.npy
      minilm_caption_index.json
    alignment/
      procrustes.npz
    metrics/
      retrieval.csv
      resnet18.csv
      majority.csv
      local.csv
    figures/
      confusion_matrix.png
      retrieval_examples.png
      failure_gallery.png
  results/
    T01.md
    T02.md
    ...
    T17.md
  plan/
    learnings/
      README.md
      _template.md
      T04-dinov2-resolution-limit.md
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
  notebooks/
    01_data.ipynb
    02_embeddings.ipynb
    03_alignment.ipynb
    04_evaluation.ipynb
    05_local_tiles.ipynb
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

### `04_evaluation.ipynb`

- Load `W`.
- Evaluate test top-1/top-5/per-class.
- Train/evaluate ResNet-18.
- Majority baseline.
- Confusion matrix.
- Save metrics/figures.

**Checkpoint:** all baselines reported.

### `05_local_tiles.ipynb`

- Load manifest.
- Validate 50 tiles.
- Evaluate Procrustes/ResNet.
- Save local CSV.
- Select success/failure examples.

**Checkpoint:** 10 per category, N=50.

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
```

Pin versions after the first successful run.

## 6. `.gitignore`

```text
data/
artifacts/embeddings/
artifacts/alignment/
*.npy
*.npz
*.pt
*.pth
__pycache__/
.ipynb_checkpoints/
.env
```

Commit small CSVs, figures, notebooks, and code. Do not commit large binaries.

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
- [ ] Every completed task has `results/Txx.md`.
- [ ] Important findings have a file under `plan/learnings/`.

### Integration checks

- [ ] Notebook 01 runs top-to-bottom.
- [ ] Notebook 02 runs after 01.
- [ ] Notebook 03 runs after 02.
- [ ] Notebook 04 runs after 03.
- [ ] Notebook 05 runs after 04.
- [ ] `app.py` loads cached artifacts.
- [ ] Demo works offline.

### Evaluation checks

- [ ] Test split untouched.
- [ ] Majority baseline reported.
- [ ] ResNet-18 reported.
- [ ] Random rotation reported.
- [ ] Procrustes reported.
- [ ] Local N=50 reported.
- [ ] One failure explained.

## 9. Run order

```text
01_data → 02_embeddings → 03_alignment → 04_evaluation → 05_local_tiles → app.py
```

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
