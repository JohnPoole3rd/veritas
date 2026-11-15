# OSM-Sentinel Runbook — Apple Silicon

**Branch:** `d/exp-osm-sentinel`  
**Owner for this run:** A or D  
**Backbone:** DINOv2-small, public and auto-downloaded  
**Method:** frozen DINOv2 + frozen MiniLM + one Procrustes rotation fitted on OSM-captioned Sentinel-2 chips

This is an experiment branch. Do not merge the generated data, PBFs, embeddings, or `.venv` into Git.

## 1. Setup

Use Python 3.12 if available.

```sh
git fetch origin
git switch d/exp-osm-sentinel
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verify the important pieces:

```sh
python - <<'PY'
import torch, pyrosm, rasterio
print("torch", torch.__version__)
print("mps", torch.backends.mps.is_available())
print("pyrosm", pyrosm.__version__)
print("rasterio", rasterio.__version__)
PY
```

If MPS is false, stop and report the exact output.

## 2. Tiny smoke run first

Use one region and one chip.

```sh
python experiments/osm_sentinel/build_dataset.py \
  --region us_columbus \
  --chips-per-region 1 \
  --max-chips 1
```

Expected behavior:

- finds a Sentinel-2 L2A scene,
- downloads/crops an OSM PBF with `pyrosm`,
- writes one image under `data/raw/osm_sentinel/images/`,
- appends one row to `manifest.csv`,
- appends one plain caption to `captions.jsonl`.

Inspect:

```sh
head -n 2 data/raw/osm_sentinel/captions.jsonl
head -n 2 data/raw/osm_sentinel/manifest.csv
```

The caption should look like:

```text
Residential area with commercial area, golf course, dense buildings.
```

No “A 10 m Sentinel-2 satellite image of …” prefix.

## 3. Small multi-region run

Run a few diverse regions before committing to a large download:

```sh
python experiments/osm_sentinel/build_dataset.py \
  --region us_columbus \
  --region us_iowa \
  --region br_amazon \
  --region nl_rotterdam \
  --region ke_nairobi \
  --chips-per-region 20 \
  --max-chips 100
```

`--max-chips` is the **total** chip count, not the number of new chips.

Check:

```sh
python - <<'PY'
import csv, collections, json
rows = list(csv.DictReader(open("data/raw/osm_sentinel/manifest.csv")))
print("chips", len(rows))
print("splits", collections.Counter(r["split"] for r in rows))
print("labels", collections.Counter(r["label"] for r in rows))
print("countries", collections.Counter(r["country"] for r in rows))
PY
```

## 4. Resume and interruption behavior

`manifest.csv` and `captions.jsonl` are appended after every completed chip.

- Rerunning the same command skips completed tile IDs.
- If the run crashes after the manifest append but before the caption append,
  `captions.jsonl` is repaired from `manifest.csv` on the next start.
- `dataset_stats.json` is rewritten after each region.
- `--overwrite` resets `manifest.csv` and `captions.jsonl`; old images are left on disk.

Do not delete `manifest.csv` during a run unless you intend to rebuild that run.

## 5. Larger data runs

The default region list has **199 regions across 93 countries**:

- 139 train regions
- 30 validation regions
- 30 test regions

Start with one continent or one biome rather than all 199 at once:

```sh
# Example: Europe and North Africa subset
python experiments/osm_sentinel/build_dataset.py \
  --region gb_london \
  --region fr_bordeaux \
  --region es_andalusia \
  --region it_po_valley \
  --region de_bavaria \
  --region pl_farmland \
  --region ma_atlas \
  --region eg_nile \
  --chips-per-region 100 \
  --max-chips 800
```

Then expand. The PBF/GeoParquet caches live in:

```text
data/raw/osm_sentinel/_cache/pbf/
data/raw/osm_sentinel/_cache/osm/
```

Expect a large cache across many countries. Run in batches and keep the cache.

### 1M note

The 199-region list has roughly **77,000 non-overlapping 1.28 km chips**. A 1M run is technically
possible with overlapping random crops, but the extra rows do not add true geographic diversity.

For 1M, either:

1. add many more regions to `regions.csv`, or
2. use a larger `--radius-km` so each region covers more land, or
3. accept that the 1M set is heavily oversampled.

Do not treat an oversampled 1M run as equivalent to 1M independent locations.

## 6. Training

Once there is at least one train, one validation, and one test region:

```sh
python experiments/osm_sentinel/train.py \
  --device mps \
  --dtype fp32 \
  --batch-size 32 \
  --num-workers 4
```

On Apple Silicon:

- use `--dtype fp32` or `--dtype auto` (auto resolves to fp32 on MPS),
- avoid fp16 for these ViT attention paths,
- if MPS DataLoader workers cause trouble, set `--num-workers 0`.

Optional slower evaluation on EuroSAT:

```sh
python experiments/osm_sentinel/train.py \
  --device mps --dtype fp32 --eval-eurosat
```

Training outputs:

```text
experiments/osm_sentinel/results_dinov2-small.json
artifacts/alignment/osm_sentinel_dinov2-small.npz
artifacts/embeddings/osm_sentinel/
```

The result JSON reports:

- held-out caption → image recall@1/@10,
- held-out image → coarse OSM label accuracy / macro-F1,
- one-caption-per-class label search,
- Upper Arlington transfer,
- a random-rotation control.

The rotation is fitted on OSM image–caption pairs. This is **not zero-shot**.

## 7. What to report

Create a short report, for example:

```text
results/T19-osm-sentinel.md
```

Include:

- commit hash,
- machine/device,
- number of chips and regions,
- train/val/test counts,
- countries and label distribution,
- exact commands,
- caption examples,
- held-out recall/accuracy,
- random-rotation control,
- Upper Arlington transfer,
- failures and caveats.

Do not commit:

```text
data/raw/osm_sentinel/
.venv/
*.npy
*.npz
```

## 8. Quick checklist

- [ ] `torch.backends.mps.is_available()` is true
- [ ] pyrosm and rasterio import
- [ ] one-chip smoke run writes image + manifest + caption
- [ ] 100-chip multi-region run has all three splits represented
- [ ] captions are plain visible-feature sentences
- [ ] training run produces `results_dinov2-small.json`
- [ ] random-rotation control is present
- [ ] report includes exact commands and caveats
