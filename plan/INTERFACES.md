# INTERFACES.md — Data, Artifacts, and Function Contracts

This file defines the contracts between agents. Do not change a shape or path without updating this file and notifying the human.

## 1. Repository layout

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

## 2. Config schema

`configs/default.yaml`:

```yaml
seed: 42

data:
  eurosat_root: data/raw/eurosat
  splits_dir: data/splits
  captions_path: data/captions.jsonl
  local_manifest: data/local_tiles/manifest.csv
  download: true            # prepare_data.py; --no-download overrides

split:
  train: 0.70
  val: 0.15
  test: 0.15

image:
  model_name: facebook/dinov2-small
  input_size: 224
  batch_size: 32
  embedding_dim: 384        # asserted by extract_embeddings.py

text:
  model_name: sentence-transformers/all-MiniLM-L6-v2
  embedding_dim: 384        # asserted by fit_alignment.py

alignment:
  center: true              # used by the fit and by the random-rotation baselines

baseline:
  resnet18_epochs: 10
  resnet18_batch_size: 64
  resnet18_lr: 0.001

demo:
  top_k: 3                  # app.py
```

Every key above is read by code; unused keys were removed in the code review (plan/learnings/general-code-review.md).

## 3. Data schemas

### Split record

`data/splits/train.json`:

```json
{
  "seed": 42,
  "class_names": ["AnnualCrop", "Forest", "..."],
  "items": [
    {"tile_id": "eurosat_000001", "path": "eurosat/2750/AnnualCrop/AnnualCrop_1.jpg", "label": "AnnualCrop"}
  ]
}
```

### Caption record

`data/captions.jsonl`:

```json
{"caption_id": "annualcrop_01", "label": "AnnualCrop", "text": "a satellite image of annual cropland", "template_id": 1}
```

### Local tile record

`data/local_tiles/manifest.csv`:

```csv
tile_id,path,label,local_category,lat,lon,date,source,notes
ua_001,data/raw/local_tiles/ua_001.png,Residential,residential,40.02,-83.06,2026-06-01,Copernicus,old Arlington
```

Required columns:

- `tile_id` — unique, `ua_001` … `ua_050`.
- `path` — relative path.
- `label` — EuroSAT label or `Other`.
- `local_category` — one of the five local categories.
- `lat`, `lon` — decimal degrees.
- `date` — imagery date.
- `source` — `Copernicus`.
- `notes` — short description.

## 4. Artifact contracts

### DINOv2 embeddings

- File: `artifacts/embeddings/dinov2_{split}.npy`
- Shape: `(N, 384)` for ViT-S; `(N, 768)` for ViT-B.
- Order: exactly the order in the split JSON.
- dtype: `float32`.
- No NaNs.

`dinov2_meta.json`:

```json
{
  "model_name": "facebook/dinov2-small",
  "input_size": 224,
  "embedding_dim": 384,
  "split_sizes": {"train": 18900, "val": 4050, "test": 4050},
  "limit": null,
  "tile_ids": {"train": ["eurosat_011587", "..."], "val": ["..."], "test": ["..."]}
}
```

`tile_ids` (in row order) are checked against the split JSON by `src.encoders.check_split_embeddings`
before any fit or evaluation; a mismatch stops with both shapes printed.

### MiniLM caption embeddings

- File: `artifacts/embeddings/minilm_captions.npy`
- Shape: `(num_captions, 384)`.
- Order: matches `minilm_caption_index.json`.

`minilm_caption_index.json`:

```json
{"0": {"caption_id": "annualcrop_01", "label": "AnnualCrop"}, "1": {"caption_id": "annualcrop_02", "label": "AnnualCrop"}}
```

### Procrustes artifact

`artifacts/alignment/procrustes.npz`:

- `W` — `(384, 384)` or `(768, 384)`.
- `mu_x` — image mean, shape `(384,)` or `(768,)`.
- `mu_y` — text mean, shape `(384,)`.
- `metadata` — JSON string with method, center, shuffle_y, scale, image_model, text_model, eurosat_root,
  limit, train_n, seed. `fit_alignment.py` refuses to save if `W.T @ W` deviates from I by more than 1e-3.

Validation:

- `W.T @ W ≈ I`.
- `mu_x` and `mu_y` finite.
- Reloading reproduces the reported top-1.

### Local tile embeddings

`artifacts/embeddings/dinov2_local.npy` + `dinov2_local_meta.json` (`tile_ids`, `model_name`, `input_size`,
`embedding_dim`, `n_tiles`, `manifest`). Written by `src.local_tiles.save_local_embeddings`
(scripts/extract_local_embeddings.py, app.py, run_local_eval); reused only if tile_ids, model and input size match.

### Metrics

`artifacts/metrics/retrieval.csv`:

```csv
method,split,top1,top5,macro_f1,n_images,test_n,n_captions,notes
procrustes,test,0.96,1.0,,4050,4050,50,
random_rotation,test,<mean>,<mean>,,4050,4050,50,mean of 5 seeds 42-46; std ...
unaligned,test,0.10,0.34,,4050,4050,50,
```

Text→image: each of the 50 captions is a query (N = 50 queries). `unaligned` is omitted when the image
and text dims differ (e.g. ViT-B). The per-image comparison with ResNet-18 is
`caption_classification_{val,test}.csv` (`python -m src.evaluate caption-classification`).

`artifacts/metrics/majority.csv`:

```csv
method,accuracy,macro_f1,test_n,most_common_class,notes
majority,0.1111,0.0200,4050,AnnualCrop,train count 2100/18900; seed 42
```

`artifacts/metrics/resnet18.csv`:

```csv
method,accuracy,macro_f1,test_n,epochs,notes
resnet18,0.9817,0.9812,4050,10,best_epoch=10
```

`artifacts/metrics/caption_classification_test.csv`:

```csv
method,split,accuracy,top5,macro_f1,n_images,n_captions,notes
procrustes,test,0.8121,0.9901,0.7948,4050,50,"image->class, agg=max"
random_rotation,test,0.1590,0.6588,0.1559,4050,50,"image->class, agg=max"
unaligned,test,0.0827,0.5326,0.0720,4050,50,"image->class, agg=max"
```

`artifacts/metrics/local.csv`:

```csv
tile_id,true_label,local_category,predicted_label,score,correct
ua_001,Residential,residential,Residential,0.74,True
```

`artifacts/metrics/local.json`:

```json
{
  "n": 50,
  "accuracy": 0.50,
  "per_category": [{"local_category": "residential", "n": 10, "accuracy": 0.6}],
  "manifest": "data/local_tiles/manifest.csv"
}
```

### Demo artifact contract

`app.py` RealBackend requires:

- `artifacts/alignment/procrustes.npz`
- `artifacts/embeddings/dinov2_test.npy`
- `artifacts/embeddings/minilm_captions.npy`
- `artifacts/embeddings/minilm_caption_index.json`
- `data/splits/test.json`

Optional metrics for the Results tab:

- `artifacts/metrics/majority.csv` (`majority.json` has the same numbers; both from `src.baselines.run_majority`)
- `artifacts/metrics/resnet18.csv`
- `artifacts/metrics/caption_classification_test.csv` — image→class, comparable with ResNet
- `artifacts/metrics/retrieval.csv` — text→image search, 50 caption queries

The demo shows image→class Procrustes next to ResNet, and text search as a separate 50-query row. Do not compare the two Ns.

The demo loads MiniLM and DINOv2 at runtime for new queries/uploads. Pre-warm both models and run once with Wi-Fi off before recording.

## 5. Function contracts

These are planning signatures. Implementations may differ internally, but inputs/outputs must match.

### `src/data.py`

```python
def load_eurosat(root: str, download: bool = True):
    """Return torchvision's EuroSAT dataset."""

def make_splits(config: dict, download: bool | None = None) -> dict:
    """Create and save stratified train/val/test splits (shares from config["split"])."""

class SplitDataset:  # (items, root, class_names, transform=None)
    """Torch dataset over split items; paths are relative to data.eurosat_root."""

def load_split(path: str) -> dict:
    """Load a split JSON."""
```

### `src/captions.py`

```python
def build_captions(class_names: list[str]) -> list[dict]:
    """Return caption records for all classes."""

def sample_caption(grouped: dict[str, list[str]], label: str, rng) -> str:
    """Return one caption text for a label."""

def sample_caption_indices(items: list[dict], captions: list[dict], seed: int) -> list[int]:
    """The Procrustes training pairs: one caption index per split item (single definition)."""
```

### `src/encoders.py`

```python
def load_dinov2(model_name: str, device: str | None = None):
    """Return a frozen DINOv2 model (inputs come from src.data.get_transforms)."""

def extract_image_embeddings(model, dataset, batch_size: int = 32, device: str | None = None) -> np.ndarray:
    """Return (N, D) CLS embeddings in dataset order."""

def load_minilm(model_name: str, device: str | None = None):
    """Return a frozen sentence-transformer."""

def check_split_embeddings(embeddings, items, split, meta=None) -> None:
    """Raise if embeddings and split items differ in count or (via dinov2_meta tile_ids) order."""

def encode_texts(model, texts: list[str]) -> np.ndarray:
    """Return (N, 384) text embeddings."""
```

### `src/alignment.py`

```python
def center_embeddings(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return centered X, centered Y, mu_x, mu_y."""

def fit_procrustes(X: np.ndarray, Y: np.ndarray, *, center=True, shuffle_y=False, seed=42) -> dict:
    """Return {'W': W, 'mu_x': mu_x, 'mu_y': mu_y, ...}."""

def training_pairs(config, caption_embeddings, captions, limit=None) -> tuple[np.ndarray, np.ndarray]:
    """(train image embeddings, caption embedding of each image's label): the fit pairs."""

def random_rotation_params(X, Y, seeds, center=True) -> list[dict]:
    """Null baseline: Procrustes on shuffled pairs, one fit per seed."""

def transform_images(X: np.ndarray, params: dict) -> np.ndarray:
    """Center, rotate, and L2-normalize image embeddings."""

def transform_texts(Y: np.ndarray, params: dict) -> np.ndarray:
    """Center and L2-normalize text embeddings."""
```

### `src/retrieval.py`

```python
def retrieve_top_k(query_embedding: np.ndarray, gallery_embeddings: np.ndarray, k: int = 3) -> list[int]:
    """Return indices of top-k most similar items."""

def retrieve_top_k_with_scores(query_embedding, gallery_embeddings, k: int = 3) -> tuple[list[int], list[float]]:
    """Top-k indices and cosine scores (k is clipped to [0, len(gallery)])."""

def class_scores(image_embeddings, caption_embeddings, caption_labels, class_names, agg="max") -> np.ndarray:
    """(N, C) image→class scores: the one classification rule (evaluation, local tiles, demo)."""
```

### `src/evaluate.py`

```python
def evaluate_retrieval(image_embeddings: np.ndarray, caption_embeddings: np.ndarray, image_labels: list[str], caption_labels: list[str], k_values: tuple[int, ...] = (1, 5)) -> dict:
    """Return top1, top5, per_class, and counts."""

def evaluate_classifier(model, loader, device=None, class_names=None) -> dict:
    """Return accuracy, macro_f1, per_class, confusion_matrix, n, probs, y_true, y_pred."""

def evaluate_caption_classification(image_emb, caption_emb, labels, caption_labels, class_names=None, agg="max") -> dict:
    """Per-image image→class metrics (comparable with ResNet-18)."""
```

### `src/local_tiles.py`

```python
def load_local_manifest(path: str) -> pd.DataFrame:
    """Load and validate the local tile manifest."""

def evaluate_local(manifest: pd.DataFrame, image_embeddings: np.ndarray, procrustes_params: dict, caption_embeddings: np.ndarray, caption_labels: list[str]) -> pd.DataFrame:
    """Return predictions for each local tile."""
```

## 6. Task reports

Every task writes one report:

```text
results/T01.md
results/T02.md
...
results/T17.md
```

Required format:

```markdown
# T06 — Procrustes

Status: DONE
Owner: J
Branch: j/t06-procrustes
Artifacts:
- artifacts/alignment/procrustes.npz
Metrics:
- top1=0.42
- random=0.10
How to run: notebooks/03_alignment.ipynb
Known issues: ViT-S only
Next: T07
```

Rules:

- One owner per report.
- No shared message board.
- Commit the report with the task branch.
- J reads reports when merging.
- Do not edit another task's report.

### Learning files

Important findings go to:

```text
plan/learnings/Txx-topic.md
plan/learnings/general-topic.md
```

Use `plan/learnings/_template.md`. Do not edit another agent's learning file. J reviews and merges learnings.

## 7. Naming conventions

- Task branches: `j/t01-scaffold`, `a/t08-resnet18`, `d/t14-gradio`.
- Subagent branches: `j/sub-j1-t02`, `a/sub-a1-t08`, `d/sub-d1-t14`.
- Tile IDs: `eurosat_000001`, `ua_001`.
- Caption IDs: `{class_lower}_{template:02d}`.
- Run IDs: `R001`, `R002`, …
- Figures: `confusion_matrix.png`, `retrieval_examples.png`, `failure_gallery.png`.
- No spaces in filenames.

## 8. Acceptance tests

Before marking a task done:

- [ ] Shapes match this file.
- [ ] No NaNs or infinities.
- [ ] Artifact path matches.
- [ ] Reloading the artifact reproduces the metric.
- [ ] Config is saved with the run.
- [ ] Result is logged in `11-experiment-log.md`.
- [ ] No test leakage.

## 9. Error handling

- If a file is missing, fail with a clear message naming the path.
- If shapes mismatch, print both shapes.
- If `W.T @ W` is not close to identity, stop.
- If retrieval is at chance, check centering and normalization before changing models.
- If the local manifest has missing coordinates, reject the row.

## 10. Versioning

- Pin model names exactly.
- Record library versions in `requirements.txt`.
- Record the seed in every run.
- Do not silently upgrade a model.
