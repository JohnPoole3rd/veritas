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
  GeoPandas spatial index for each chip. It preserves raw OSM tags and avoids API rate limits.
- `pyrosm` is a required dependency; there is no Overpass runtime path.
- Only visible tags are captioned. Names, brands, addresses, opening hours, contact details, and
  religion are excluded.

## DINOv3 satellite weights

- SAT-493M is the ViT-L/16 model: `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth`, 1024-d CLS.
- Official URL pattern: `https://dl.fbaipublicfiles.com/dinov3/dinov3_vitl16/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth`.
  The unprefixed URL is **403**; access requires Meta's download form or the gated HF repo.
- HF: `facebook/dinov3-vitl16-pretrain-sat493m` (manual gate; `model.safetensors` 1.21 GB).
- The SAT transform uses `mean=(0.430,0.411,0.296)`, `std=(0.213,0.156,0.143)`; `train.py` applies
  it automatically. The ViT-S LVD checkpoint keeps the project's ImageNet transform.

## Sources

- PyTorch Vulkan status: https://docs.pytorch.org/docs/stable/notes/vulkan.html
- PyTorch device backends: https://pytorch.org/docs/stable/notes/cuda.html and https://pytorch.org/docs/stable/notes/mps.html
- pyrosm: https://pyrosm.readthedocs.io/en/stable/
- Geofabrik extracts: https://download.geofabrik.de/
- OSM copyright / ODbL: https://www.openstreetmap.org/copyright

