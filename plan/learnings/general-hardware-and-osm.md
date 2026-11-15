# Learning: hardware selection and OSM source choice

**Date:** 2026-11-15
**Author:** J
**Task:** hardware acceleration + OSM-Sentinel experiment
**Status:** confirmed

## Hardware

- Native PyTorch covers the practical backends: CUDA (including ROCm), Apple MPS, Intel XPU, CPU.
- `src.encoders.get_device()` autoselects in that order. `--device` or `PLATONIC_DEVICE` overrides.
- Vulkan is **not** a usable Python/PyTorch path for these ViT models: no official wheels, incomplete operator
  coverage, and no training support. We deliberately did not add a fake cross-vendor layer.
- The useful optimizations are ordinary: `inference_mode`, worker-backed DataLoader, pinned memory
  on CUDA/XPU, optional bf16 autocast, and embedding caches. `torch.compile` stays optional because
  backend coverage is uneven.

## OSM data

- Per-chip public Overpass requests do not scale and get 502s under load.
- The robust path is one local OSM PBF per region via `pyrosm.get_data_by_bbox`, then an in-memory
  GeoPandas spatial index for each chip. It preserves raw OSM tags and avoids API rate limits.
- `pyrosm` is a required dependency; there is no Overpass runtime path.
- Only visible tags are captioned. Names, brands, addresses, opening hours, contact details, and
  religion are excluded.
- Linear features are converted to a metric area (`metric length × nominal width`) before caption
  subject selection. Weighting lines by raw degrees made rivers outrank every polygon and made
  east-west lines latitude-dependent; that is fixed in `osm_captions._weight`.
- `build_dataset.py` appends a manifest row and caption per completed chip, flushes both, and on
  restart skips completed tile IDs. This is required for the 1M-chip run.
- The default sampler is a local metric grid (`--sample-mode grid`) with stable cell IDs; the legacy
  random sampler is available with `--sample-mode random` but overlaps.
- WorldCover 2021 is the auxiliary natural-land-cover source. It is read lazily per region only when
  a chip lacks an OSM land-cover polygon, at ~31 m for 640 m caption units (9x less memory than
  native 10 m). It fills forest/grass/crop/built/water gaps and is recorded in the manifest; it is
  not a replacement for OSM buildings or roads. Ocean-only WorldCover tiles are missing and are
  skipped per URL.
- `fit_procrustes` accumulates the cross-covariance in chunks; 1M rows do not allocate full centered
  copies of both embedding matrices.
- The default list has 399 regions across 125 countries, roughly 5.8M non-overlapping 640 m chips.
  A uniform 3,500-cell cap reaches the 1M budget in the 398th region (last partial); an 8,000 cap
  stops around 286 regions and loses most later city/test regions. The legacy random sampler is not
  the 1M path.

## DINOv2

- DINOv2-small is public (Apache-2.0) and downloads automatically through `transformers`; no gate,
  token, or manual checkpoint path is needed.
- It is the backbone for the OSM-Sentinel experiment.

## Sources

- PyTorch Vulkan status: https://docs.pytorch.org/docs/stable/notes/vulkan.html
- PyTorch device backends: https://pytorch.org/docs/stable/notes/cuda.html and https://pytorch.org/docs/stable/notes/mps.html
- pyrosm: https://pyrosm.readthedocs.io/en/stable/
- Geofabrik extracts: https://download.geofabrik.de/
- OSM copyright / ODbL: https://www.openstreetmap.org/copyright

