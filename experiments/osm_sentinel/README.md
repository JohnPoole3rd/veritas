# Experiment 6 — OSM-captioned Sentinel-2 training

**Owner:** J / D experiment branch · **Branch:** `d/exp-osm-sentinel` · **Status:** implementation + smoke test

**Question:** does training the rotation on real, globally sampled Sentinel-2 chips captioned from
OpenStreetMap beat EuroSAT template captions, especially outside Upper Arlington?

The method is unchanged: frozen image encoder + frozen MiniLM + **one closed-form orthogonal
Procrustes rotation** fitted on labelled image–caption pairs. This is not zero-shot; the rotation
uses the OSM captions as labels.

**Apple Silicon run instructions:** see [`RUNBOOK.md`](RUNBOOK.md). It covers setup, pyrosm caches,
smoke runs, resume behavior, MPS training, and what to report.

## Pipeline

```text
regions.csv
  -> Sentinel-2 L2A scene per region (STAC / Element84 Earth Search)
  -> local OSM PBF per region (pyrosm)
  -> random 1.28 km chips on the 10 m grid
  -> OSM caption per chip
  -> DINOv2 embeddings + MiniLM caption embeddings
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

> Residential area with houses, major road, and park.

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

Scale to the 199-region default list (about 1,592 chips at 8/region):

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
Existing chips are skipped on rerun; `manifest.csv` and `captions.jsonl` are appended per chip and
repaired on restart. Use `--overwrite` to rebuild from scratch.

### DINOv2

The experiment uses the same public DINOv2-small encoder as the main project:

```sh
python experiments/osm_sentinel/train.py
```

DINOv2 is Apache-2.0 and downloads automatically through `transformers`; no gate, token, or
manual checkpoint path is needed. Use `--device` and `--dtype` for the hardware path.

The OSM experiment reports:

- held-out **caption→image recall@1/@10** (the real free-text retrieval metric);
- held-out **image→coarse OSM label** accuracy / macro-F1;
- one-canonical-caption-per-class **label search** top-1;
- optional **Upper Arlington** transfer (`--eval-local`, on by default);
- optional **EuroSAT test** (`--eval-eurosat`, slower);
- a random-rotation control, so the rotation's contribution is visible.

Outputs:

- `experiments/osm_sentinel/results_dinov2-small.json`
- `artifacts/alignment/osm_sentinel_dinov2-small.npz` (gitignored)
- `artifacts/embeddings/osm_sentinel/*.npy` (gitignored cache)

## Hardware acceleration

The main `src/encoders.py` path now auto-selects **CUDA/ROCm → Apple MPS → Intel XPU → CPU**.
Use `--device` (or `PLATONIC_DEVICE`) when you want to override. DINOv2 extraction uses:

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
extraction plus a tiny Procrustes SVD, so the GPU is not the binding constraint. DINOv2 ViT-S/14 at
224 px, bf16, batch 512, `inference_mode`: roughly **900–1,500 images/s**. The 8 GB and 16 GB 5060 Ti
have the same compute; 16 GB just allows larger batches.

| Chips | DINOv2 ViT-S/14 (384-d) | JPEG + embeddings |
|---:|---:|---:|
| 10k | ~10–20 s | ~0.3 GB |
| 50k | ~0.7–1.5 min | ~1.5 GB |
| 100k | ~1.5–3 min | ~3 GB |
| 500k | ~7–14 min | ~15 GB |
| 1M | ~15–30 min | ~30 GB |

The real cost is **Sentinel-2 COG acquisition and OSM PBF preparation**, not GPU time. A 1M-chip run
needs roughly tens of GB of chips, a few GB of embeddings, and potentially several hours to a day of
data prep. Procrustes itself is a 384×384 cross-covariance GEMM plus a small SVD: seconds even at
1M rows, provided embeddings are float32 and not a full N×d SVD.

Practical recommendation: prototype at 20k–50k chips, scale to 250k for a real result, and go to
1M only if the OSM coverage/diversity actually improves held-out metrics. A 5060 Ti 8 GB is enough;
16 GB is convenience/headroom.

## Data scale and leakage

- The default `regions.csv` has 199 regions across 93 countries and six continents. At 1.28 km per
  chip that is roughly 77,000 **non-overlapping** chips. A 1M-chip run is technically possible by
  drawing many random crops per region, but those crops overlap heavily and do not add geographic
  diversity. For a real 1M run, expand `regions.csv` further or use a larger sampling radius per
  region.
- Splits are by **region**, not random chip, so nearby tiles cannot leak between train and test.
- `build_dataset.py` appends `manifest.csv` and `captions.jsonl` after each chip. On restart it skips
  completed tile IDs and repairs captions from the manifest.
- `fit_procrustes` accumulates the cross-covariance in chunks, so 1M rows do not require full
  centered copies of both embedding matrices.
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
- Sentinel-2: Copernicus Sentinel data, processed by ESA; accessed through the public
  Element84 Earth Search STAC API. OSM: © OpenStreetMap contributors, ODbL.

## Files

| File | Purpose |
|---|---|
| `regions.csv` | Global region list with country, biome, split, centre, radius |
| `build_dataset.py` | STAC scene search, COG chip reading, pyrosm lookup, manifest/captions |
| `osm_captions.py` | Visible-tag mapping, geometry clipping, caption builder |
| `osm_source.py` | pyrosm local PBF backend with per-region GeoParquet cache |
| `train.py` | Embedding cache, Procrustes fit, held-out and transfer metrics |
| `README.md` | This document |
