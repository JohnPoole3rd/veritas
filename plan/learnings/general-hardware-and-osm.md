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
- Linear features are converted to an area (`length × nominal width`) before caption subject
  selection. Weighting lines by raw degrees made rivers outrank every polygon; that is fixed in
  `osm_captions._weight`.
- `build_dataset.py` appends a manifest row and caption per completed chip, flushes both, and on
  restart skips completed tile IDs. This is required for the 1M-chip run.
- `fit_procrustes` accumulates the cross-covariance in chunks; 1M rows do not allocate full centered
  copies of both embedding matrices.
- The default list has 399 regions across 125 countries, roughly 1.46M non-overlapping 1.28 km chips
  worth of area. The 1M run is intentionally overlapping random crops from 399 regions, not 1M
  independent locations.

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

