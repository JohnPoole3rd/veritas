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
  -> local OSM PBF per region (pyrosm; Overpass fallback for smoke tests)
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

Install the existing project requirements plus the optional data packages:

```sh
pip install -r requirements.txt
# recommended for more than a few chips:
pip install pyrosm
```

`build_dataset.py` defaults to `--osm-source auto`: it uses a cropped local OSM PBF via
`pyrosm` when available, and falls back to Overpass for a one-chip smoke test. Use
`--osm-source overpass` or `--osm-source pyrosm` to force one path.

Smoke test one region and one chip:

```sh
python experiments/osm_sentinel/build_dataset.py \
  --region us_columbus --chips-per-region 1 --max-chips 1
```

Scale to the 28-region default list (about 224 chips at 8/region):

```sh
python experiments/osm_sentinel/build_dataset.py \
  --chips-per-region 8 --sleep 1.0
```

Larger runs are just more regions or more chips per region:

```sh
python experiments/osm_sentinel/build_dataset.py \
  --chips-per-region 40 --max-chips 5000 --sleep 1.0
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

If those variables are unset, `backbones.py` tries the gated Hugging Face model. The satellite
ViT-L is available as `--backbone dinov3-sat-large` when your account has access. For a quick
non-DINOv3 smoke test:

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
- Overpass is rate-limited and is only a fallback. For real runs, use `pyrosm` with a local PBF:
  one download per region, then in-memory chip lookups. The public Overpass endpoint is not
  intended for thousands of per-chip requests. Keep `--sleep` at 1 second or more when using it,
  respect the mirror's usage policy, and cache extracts.
- OSM PBF extracts come from Geofabrik/BBBike/Movisda via `pyrosm.get_data_by_bbox`; record the
  snapshot date. OSM data is © OpenStreetMap contributors, ODbL 1.0.
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
| `build_dataset.py` | STAC scene search, COG chip reading, Overpass fetch, manifest/captions |
| `osm_captions.py` | Visible-tag mapping, geometry clipping, caption builder |
| `osm_source.py` | pyrosm local PBF backend with per-region GeoParquet cache |
| `backbones.py` | DINOv3 local/HF loader and DINOv2 fallback |
| `train.py` | Embedding cache, Procrustes fit, held-out and transfer metrics |
| `README.md` | This document |
