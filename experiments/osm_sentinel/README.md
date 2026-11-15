# Experiment 6 — OSM-captioned Sentinel-2 training

**Owner:** J / D experiment branch · **Branch:** `d/exp-osm-sentinel` · **Status:** implementation + smoke test

**Question:** does training the rotation on real, globally sampled Sentinel-2 chips captioned from
OpenStreetMap beat EuroSAT template captions, especially outside Upper Arlington?

The method is unchanged: frozen image encoder + frozen MiniLM + **one closed-form orthogonal
Procrustes rotation** fitted on labelled image–caption pairs. This is not zero-shot; the rotation
uses the OSM captions as labels.

## Pipeline

```text
regions.csv
  -> Sentinel-2 L2A scene per region (STAC / Element84 Earth Search)
  -> local OSM PBF per region (pyrosm)
  -> random 1.28 km chips on the 10 m grid
  -> OSM caption per chip
  -> DINOv3 / DINOv2 embeddings + MiniLM caption embeddings
  -> Procrustes fit on train regions
  -> held-out regions: caption->image recall, image->label, label search, Upper Arlington transfer
```

### Visible OSM tags only

Kept:

- `landuse`: residential, commercial, retail, industrial, farmland, orchard, vineyard, forest,
  meadow, quarry, landfill, railway, construction, port, aquaculture.
- `natural`: water, wood, scrub, heath, grassland, wetland, beach, sand, bare rock, glacier, coastline.
- `leisure`: park, golf course, pitch, nature reserve, marina.
- `building`: houses, apartments, shops, warehouses, schools, hospitals, farms, etc.
- `highway`, `waterway`, `railway`, `man_made`, selected `amenity` (parking, school, hospital,
  place of worship), `aeroway`.

Excluded: names, brands, operators, addresses, phone/email/website, opening hours, religion,
description/note, and any other non-visible metadata. The caption is a short sentence such as:

> A 10 m Sentinel-2 satellite image of residential area with houses, major road, and park.

The full mapping is in `osm_captions.py`; it is covered by `tests/test_osm_captions.py`.

## Run

Install the project requirements (including `pyrosm`):

```sh
pip install -r requirements.txt
```

On NixOS, `pyrosm` is not in nixpkgs. A normal Python 3.12 venv works (PyPI provides a
manylinux wheel):

```sh
nix-shell -p python312 --run 'python -m venv .venv'
.venv/bin/pip install -r requirements.txt
```

`build_dataset.py` always uses a cropped local OSM PBF via `pyrosm`. The first run for a
region downloads the smallest covering Geofabrik/BBBike/Movisda extract; later runs reuse the
cropped PBF and GeoParquet cache.

Smoke test one region and one chip:

```sh
python experiments/osm_sentinel/build_dataset.py \
  --region us_columbus --chips-per-region 1 --max-chips 1
```

Scale to the 28-region default list (about 224 chips at 8/region):

```sh
python experiments/osm_sentinel/build_dataset.py \
  --chips-per-region 8
```

Larger runs are just more regions or more chips per region:

```sh
python experiments/osm_sentinel/build_dataset.py \
  --chips-per-region 40 --max-chips 5000
```

Data is written to `data/raw/osm_sentinel/` (gitignored) and cached by region in `_cache/`.
Existing chips are skipped on rerun; use `--overwrite` to rebuild.

### DINOv3

The default backbone is the same DINOv3 ViT-S/16 used by the other experiments:

```sh
export DINOV3_REPO=/path/to/facebookresearch/dinov3
export DINOV3_WEIGHTS=/path/to/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
python experiments/osm_sentinel/train.py --backbone dinov3-small
```

If those variables are unset, `backbones.py` tries the gated Hugging Face model.

### DINOv3 SAT-493M (satellite ViT-L)

The satellite checkpoint is **gated** by Meta and Hugging Face. Accept the gate at
`https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m`, then:

```sh
hf auth login
hf download facebook/dinov3-vitl16-pretrain-sat493m \
  --local-dir weights/dinov3-vitl16-sat493m
python experiments/osm_sentinel/train.py --backbone dinov3-sat-large --dtype bf16
```

Or use Meta's signed URL (from the DINOv3 download form) and the exact filename
`dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth` with the official repo:

```sh
export DINOV3_REPO=/path/to/facebookresearch/dinov3
export DINOV3_SAT_WEIGHTS=/path/to/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
python experiments/osm_sentinel/train.py --backbone dinov3-sat-large --dtype bf16
```

SAT-493M returns a **1024-d** CLS token; the Procrustes fit maps it to the 384-d MiniLM space.
`train.py` automatically uses the official SAT mean/std for this backbone. The weights are not
committed; they are 1.2 GB and subject to the DINOv3 License.

For a quick non-DINOv3 smoke test:

```sh
python experiments/osm_sentinel/train.py --backbone dinov2-small
```

The OSM experiment reports:

- held-out **caption→image recall@1/@10** (the real free-text retrieval metric);
- held-out **image→coarse OSM label** accuracy / macro-F1;
- one-canonical-caption-per-class **label search** top-1;
- optional **Upper Arlington** transfer (`--eval-local`, on by default);
- optional **EuroSAT test** (`--eval-eurosat`, slower);
- a random-rotation control, so the rotation's contribution is visible.

Outputs:

- `experiments/osm_sentinel/results_<backbone>.json`
- `artifacts/alignment/osm_sentinel_<backbone>.npz` (gitignored)
- `artifacts/embeddings/osm_sentinel/*.npy` (gitignored cache)

## Hardware acceleration

The main `src/encoders.py` path now auto-selects **CUDA/ROCm → Apple MPS → Intel XPU → CPU**.
Use `--device` (or `PLATONIC_DEVICE`) when you want to override. DINOv3/DINOv2 extraction uses:

- `torch.inference_mode()`;
- a worker-backed `DataLoader` with pinned host memory on CUDA/XPU;
- optional autocast (`--dtype auto` uses bf16 where supported, fp32 otherwise);
- cached embeddings, so reruns skip encoding.

We deliberately **do not** ship a Vulkan or cross-vendor runtime layer. Stock PyTorch has no usable
Vulkan backend for these ViT models (no official wheels, incomplete operator coverage, and no
training path), so it would be a false promise. Native PyTorch already covers the practical
accelerators; ONNX Runtime/DirectML/OpenVINO remain deployment-specific follow-ups, not the default.

`scripts/extract_embeddings.py` and `scripts/extract_local_embeddings.py` accept `--dtype` and
`--num-workers`, and `src.get_device` covers all of the above.

## Scale on an RTX 5060 Ti

These are planning estimates, not measured on this machine. The pipeline is frozen-encoder embedding
extraction plus a tiny Procrustes SVD, so the GPU is not the binding constraint. ViT-S/16 at 224 px,
bf16, batch 512, `inference_mode`: roughly **900–1,500 images/s**; ViT-L/16 SAT-493M: roughly
**95–200 images/s**. The 8 GB and 16 GB 5060 Ti have the same compute; 16 GB just allows larger
batches.

| Chips | ViT-S/16 (384-d) | ViT-L/16 SAT-493M (1024-d) | JPEG + embeddings |
|---:|---:|---:|---:|
| 10k | ~10–20 s | ~1–2 min | ~0.3 GB |
| 50k | ~0.7–1.5 min | ~5–9 min | ~1.5 GB |
| 100k | ~1.5–3 min | ~10–18 min | ~3 GB |
| 500k | ~7–14 min | ~50–90 min | ~15 GB |
| 1M | ~15–30 min | ~2–3 h | ~30 GB |

The real cost is **Sentinel-2 COG acquisition and OSM PBF preparation**, not GPU time. A 1M-chip run
needs roughly tens of GB of chips, a few GB of embeddings, and potentially several hours to a day of
data prep. Procrustes itself is a 384×384 or 1024×384 cross-covariance GEMM plus a small SVD:
seconds even at 1M rows, provided embeddings are float32 and not a full N×d SVD.

Practical recommendation: prototype at 20k–50k chips, scale to 250k for a real result, and go to
1M only if the OSM coverage/diversity actually improves held-out metrics. A 5060 Ti 8 GB is enough;
16 GB is convenience/headroom.

## Data scale and leakage

- The default `regions.csv` has 28 regions across 16 countries and six continents.
- Splits are by **region**, not random chip, so nearby tiles cannot leak between train and test.
- `train.py` refuses to run if a split is empty.
- Changing `regions.csv` or adding chips invalidates cached embeddings only if tile IDs change; use
  `--force` when in doubt.
- The local-PBF path may download roughly 9 GB of Geofabrik subregions for the default region list
  (not full country extracts). `pyrosm` caches the cropped PBFs under `data/raw/osm_sentinel/_cache/`.

## Limits

- OSM is volunteer data. Missing or stale features produce weak captions; this is a data-quality
  limit, not a model bug. The coarse label is derived from the dominant visible phrase, so it is
  noisier than EuroSAT's one-label-per-tile benchmark.
- OSM PBF extracts come from Geofabrik/BBBike/Movisda via `pyrosm.get_data_by_bbox`; record the
  snapshot date. The first run per region downloads a cropped PBF; later chip lookups are in-memory
  and have no API rate limit. OSM data is © OpenStreetMap contributors, ODbL 1.0.
- Sentinel-2 is 10 m; a 1.28 km chip is 128 px before the model resize. Fine-grained land use
  (e.g. individual buildings) is not resolvable at this scale.
- DINOv3 is licensed by Meta under the DINOv3 License, not an open-source license. Weights are not
  committed. Cite Siméoni et al., “DINOv3”, arXiv:2508.10104 (2025).
- Sentinel-2: Copernicus Sentinel data, processed by ESA; accessed through the public
  Element84 Earth Search STAC API. OSM: © OpenStreetMap contributors, ODbL.

## Files

| File | Purpose |
|---|---|
| `regions.csv` | Global region list with country, biome, split, centre, radius |
| `build_dataset.py` | STAC scene search, COG chip reading, pyrosm lookup, manifest/captions |
| `osm_captions.py` | Visible-tag mapping, geometry clipping, caption builder |
| `osm_source.py` | pyrosm local PBF backend with per-region GeoParquet cache |
| `backbones.py` | DINOv3 local/HF loader and DINOv2 fallback |
| `train.py` | Embedding cache, Procrustes fit, held-out and transfer metrics |
| `README.md` | This document |
