# Learning: hardware selection and OSM source choice

**Date:** 2026-11-15
**Author:** J
**Task:** hardware acceleration + OSM-Sentinel experiment
**Status:** confirmed

## Hardware

- Native PyTorch covers the practical backends: CUDA (including ROCm), Apple MPS, Intel XPU, CPU.
- `src.encoders.get_device()` autoselects in that order. `--device` or `PLATONIC_DEVICE` overrides.
- Vulkan is **not** a usable Python/PyTorch path for DINOv3: no official wheels, incomplete operator
  coverage, and no training support. We deliberately did not add a fake cross-vendor layer.
- The useful optimizations are ordinary: `inference_mode`, worker-backed DataLoader, pinned memory
  on CUDA/XPU, optional bf16 autocast, and embedding caches. `torch.compile` stays optional because
  backend coverage is uneven.

## OSM data

- Per-chip public Overpass requests do not scale and get 502s under load.
- The robust path is one local OSM PBF per region via `pyrosm.get_data_by_bbox`, then an in-memory
  GeoPandas spatial index for each chip. It preserves raw OSM tags, avoids rate limits, and is
  reproducible from a snapshot.
- Overpass remains a one-chip smoke-test fallback (`--osm-source overpass`).
- Only visible tags are captioned. Names, brands, addresses, opening hours, contact details, and
  religion are excluded.
- `pyrosm` is optional; `build_dataset.py --osm-source auto` falls back when it is not installed.

## Sources

- PyTorch Vulkan status: https://docs.pytorch.org/docs/stable/notes/vulkan.html
- PyTorch device backends: https://pytorch.org/docs/stable/notes/cuda.html and https://pytorch.org/docs/stable/notes/mps.html
- pyrosm: https://pyrosm.readthedocs.io/en/stable/
- Geofabrik extracts: https://download.geofabrik.de/
- OSM copyright / ODbL: https://www.openstreetmap.org/copyright

