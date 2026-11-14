# Alignment and Model Plan

## 1. Hypothesis

The **Platonic Representation Hypothesis (PRH)** proposes that independently trained models converge toward a shared representation of reality. This project tests a practical version:

> A single closed-form rotation can align frozen DINOv2 satellite-image embeddings to frozen MiniLM text embeddings well enough for zero-shot land-cover retrieval.

If true, text search works without multimodal training. If false, the negative result is still informative.

## 2. Default models

| Role | Model | Dim | Frozen |
|---|---|---:|---|
| Image | `facebook/dinov2-small` (ViT-S/14) | 384 | yes |
| Text | `sentence-transformers/all-MiniLM-L6-v2` | 384 | yes |
| Baseline | torchvision `resnet18` | 10 classes | trained |
| Optional | `openai/clip-vit-base-patch32` | — | frozen |

**Why ViT-S/14:** fast enough for Colab, 384-dim, square Procrustes with MiniLM, and strong enough for a first result. Switch to ViT-B/14 (768-dim) only if time and retrieval quality justify it.

## 3. Procrustes math

### Square case (384→384)

Given centered matrices \(X \in \mathbb{R}^{n \times 384}\) (images) and \(Y \in \mathbb{R}^{n \times 384}\) (text):

\[
M = X^T Y
\]

\[
M = U S V^T
\]

\[
W = U V^T
\]

Transform:

\[
\hat{Y} = XW
\]

`scipy.linalg.orthogonal_procrustes(A, B)` requires `A.shape == B.shape`. It returns `R` and a scale. The scale is \(\sum \text{svdvals}(A^T B)\), not a multiplier. Do not multiply embeddings by it.

### Rectangular case (768→384)

If using ViT-B/14:

\[
M = X^T Y, \quad M = U S V^T, \quad W = U V^T
\]

with \(W \in \mathbb{R}^{768 \times 384}\) and \(W^T W = I\). This is a semi-orthogonal projection plus rotation. It cannot increase separability.

### Centering

Procrustes assumes centered data:

1. Compute training image mean \(\mu_x\).
2. Compute training text mean \(\mu_y\).
3. Center \(X_c = X - \mu_x\), \(Y_c = Y - \mu_y\).
4. Fit \(W\) on centered data.
5. At inference, subtract the **training** means.

Do not center a query with its own mean. Do not skip centering. This is the most common bug.

### Normalization

After centering and rotation, L2-normalize before cosine similarity:

\[
\hat{z} = \frac{z}{\|z\|_2}
\]

Do not normalize before fitting unless you have a specific reason.

## 4. Fitting procedure

```text
1. Load EuroSAT train split.
2. For each tile:
   a. Load image, resize to 224, normalize.
   b. DINOv2 CLS embedding.
   c. Sample a caption from the tile's class.
   d. MiniLM embedding.
3. Stack X and Y.
4. Center with training means.
5. Fit W.
6. Save W, mu_x, mu_y, config.
```

**Caption pairing:**

- MVP: one random caption per tile.
- Stable version: average the 5 captions per class to form a class centroid.
- Optional: multiple captions per tile.

Start with random caption per tile. If retrieval is unstable, switch to class centroids.

## 5. Inference

### Text → image

```text
1. Embed query with MiniLM.
2. Subtract training text mean.
3. L2-normalize.
4. Cosine similarity against rotated test image embeddings.
5. Return top-k.
```

### Image → text

```text
1. Embed uploaded tile with DINOv2.
2. Subtract training image mean.
3. Rotate with W.
4. L2-normalize.
5. Compare with class caption embeddings.
6. Return nearest labels.
```

### Upper Arlington local tiles

Use the same mean and \(W\) fitted on EuroSAT. Do **not** refit on local tiles. The local result is an out-of-distribution test.

## 6. Baselines and ablations

| ID | Method | Purpose |
|---|---|---|
| B0 | Majority class | Required simple baseline |
| B1 | ResNet-18 | Required supervised baseline |
| B2 | Random rotation | Null alignment baseline |
| B3 | Unaligned cosine | Tests whether rotation helps |
| B4 | Procrustes (ours) | Core method |
| B5 | CLIP zero-shot | Optional external baseline |
| B6 | Procrustes + class centroids | Stability ablation |
| B7 | ViT-B vs. ViT-S | Model-size ablation |

**Key comparison:** B4 vs. B2. If Procrustes does not beat random rotation, the alignment is not working.

## 7. Code architecture

See `16-repo-blueprint.md` for the full layout. Minimum modules:

- `src/data.py` — EuroSAT loader, splits.
- `src/captions.py` — templates, sampling.
- `src/encoders.py` — DINOv2, MiniLM.
- `src/alignment.py` — centering, Procrustes, transform.
- `src/retrieval.py` — top-k, classification.
- `src/evaluate.py` — metrics, confusion matrix.
- `src/baselines.py` — majority, ResNet-18.
- `src/local_tiles.py` — Upper Arlington manifest and eval.
- `app.py` — Gradio.

## 8. Compute

- Colab free GPU is enough for 27k 64×64 tiles with ViT-S.
- Cache embeddings to Drive; do not recompute after runtime restart.
- Batch size 32–64 for ViT-S.
- CPU fallback: use a 5k subset.
- Time: a few minutes on GPU; 30–90 minutes on CPU.

## 9. Common bugs

| Bug | Symptom | Fix |
|---|---|---|
| No centering | Retrieval near chance | Subtract training means |
| Query centered with its own mean | Inconsistent scores | Use training mean |
| Shape mismatch | Procrustes error | Use rectangular SVD |
| No L2 normalization | Scores outside [-1, 1] | Normalize |
| Test leakage | Suspiciously high accuracy | Split before fitting |
| Caption/class mismatch | Confused retrieval | Check caption index |
| `model.train()` | Dropout noise | `model.eval()` |
| No `torch.no_grad()` | Memory blow-up | Disable gradients |
| Upscaling artifacts | Blurry tiles | Report limitation |
| Duplicate tiles | Leakage | Deduplicate |

## 10. Reproducibility

- Seed splits and caption sampling.
- Save model names and library versions.
- Save split indices.
- Save `W`, `mu_x`, `mu_y`.
- Save the config with each run.
- Log every run in `11-experiment-log.md`.

## 11. Optional improvements

Only after the MVP works:

- Class centroids.
- Whitening.
- Multi-caption averaging.
- Temperature scaling.
- Abstention threshold.
- Patch-token mean instead of CLS.
- ViT-B/14.
- Remote-sensing-specific text encoder.
- CLIP baseline.

Do not add these before the core evaluation is complete.

## 12. Why no encoder training

The research question is specifically whether alignment exists **without** joint training. Fine-tuning would make the result less interesting and would violate the project scope. Keep both encoders frozen.
