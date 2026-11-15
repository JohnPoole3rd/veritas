# OSM-Sentinel Runbook — Apple Silicon

**Branch:** `j/exp-osm-sentinel`  
**Owner for this run:** A or D  
**Backbone:** DINOv2-small, public and auto-downloaded  
**Method:** frozen DINOv2 + frozen MiniLM + one Procrustes rotation fitted on OSM-captioned Sentinel-2 chips

This is an experiment branch. Do not merge the generated data, PBFs, embeddings, or `.venv` into Git.

## Run budget

**Run target: 1,000,000 chips.** The region list has 399 regions across 125 countries. Its combined
area contains roughly 5.8M non-overlapping 640 m grid cells. The default sampler is a
non-overlapping grid, so the 1M run is **1,000,000 non-overlapping grid chips from 399 regions**.
Some cells are skipped for water/cloud/OSM reasons; report the actual `n`.

The first 199 anchors are hand-picked biome/location regions; the extra 200 are Natural Earth 1:50m
populated places with a 60 km radius.

The old random sampler remains available with `--sample-mode random`, but it produces overlapping
chips and is not the 1M path.

## A: full 1M rerun

This is the authoritative A run. Start from a clean 1M output directory. Do **not** reuse, copy,
symlink, resume, or recaption anything from `data/raw/osm_sentinel` (the old 77k build).

```sh
cd /home/john/veritas
git switch j/exp-osm-sentinel
git pull --ff-only origin j/exp-osm-sentinel
```

Use Python 3.12+ (3.14.5 already works on A's machine):

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python - <<'PY'
import torch, pyrosm, rasterio
print("mps", torch.backends.mps.is_available())
print("pyrosm", pyrosm.__version__)
print("rasterio", rasterio.__version__)
PY
```

Remove only the fresh 1M directory if it exists. Leave the 77k build alone:

```sh
rm -rf data/raw/osm_sentinel_1m_640
```

Run the full grid + WorldCover build:

```sh
mkdir -p data/raw/osm_sentinel_1m_640
python experiments/osm_sentinel/build_dataset.py \
  --out-dir data/raw/osm_sentinel_1m_640 \
  --chip-m 640 --chip-px 64 \
  --sample-mode grid \
  --chips-per-region 8000 \
  --max-chips 1000000 \
  2>&1 | tee data/raw/osm_sentinel_1m_640/_run.log
```

Defaults are intentional:

- `grid`: non-overlapping cells, stable IDs, resume-safe
- WorldCover on
- 399 regions / 125 countries
- fresh PBF and GeoParquet caches under `data/raw/osm_sentinel_1m_640/_cache/`

Do not add `--no-worldcover`. Do not point this at the old 77k directory.

The build is resumable: if it stops, rerun the same command. It will continue from the new
`data/raw/osm_sentinel_1m_640/manifest.csv`.

While it runs:

```sh
tail -f data/raw/osm_sentinel_1m_640/manifest.csv
grep "REGION FAILED" data/raw/osm_sentinel_1m_640/_run.log 2>/dev/null || true
cat data/raw/osm_sentinel_1m_640/dataset_stats.json
```

When it finishes, record the actual `n`, `failed_regions`, and `skipped_chips`. If `n` is below
1,000,000, report it; do not fill the gap from the 77k build.

Train on the new 1M data:

```sh
python experiments/osm_sentinel/train.py \
  --data data/raw/osm_sentinel_1m_640 \
  --device mps \
  --dtype fp32 \
  --batch-size 32 \
  --num-workers 4 \
  --eval-eurosat
```

Run the EuroSAT-vs-OSM comparison on the same 1M data:

```sh
python experiments/osm_sentinel/compare_eurosat.py \
  --data data/raw/osm_sentinel_1m_640 \
  --tag full_1m \
  --dtype fp32 \
  --batch-size 64 \
  --num-workers 0
```

Write the results to `results/T20-osm-1m.md`. The old `results/T19-osm-sentinel.md` remains the
77k historical report; do not merge its data into the 1M run.

### Performance check

The optimized build reads each Sentinel scene window **once per region**, then crops chips from
memory. On the Malta smoke region, this changed the measured build from ~4.3 to **~26.6 chips/s**.
A 1M-chip run is still data-bound, but it should be hours rather than days.

Each region prints a line like:

```text
[region] timing: 29 chips, 1.1s, 26.6 chips/s,
append=0.01s, caption=1.03s, image_save=0.01s, osm_load=0.43s,
raster_read=0.03s, scene_open=1.39s, scene_read=0.95s,
scene_search=2.62s, worldcover_load=3.70s
```

Use that to find the next bottleneck:

- `scene_read` / `worldcover_load` high → region I/O
- `caption` high → OSM geometry/caption cost
- `raster_read` high → Sentinel cropping or memory
- `scene_search` high → STAC

Do not run the old 128 px build again for this comparison.

## 1. Setup

Use Python 3.12+ if available. A ran this successfully on Python 3.14.5; 3.12 is not required.

```sh
git fetch origin
git switch j/exp-osm-sentinel
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
  --region nl_rotterdam \
  --region br_amazon \
  --region gb_london \
  --region eg_nile \
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

The default region list has **399 regions across 125 countries**:

- 279 train regions
- 60 validation regions
- 60 test regions

Start with one continent or one biome rather than all 399 at once:

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

### Target run: 1,000,000 chips

Use the 1M budget. Use a fresh output directory so you do not resume A's existing 77k build:

```sh
python experiments/osm_sentinel/build_dataset.py \
  --out-dir data/raw/osm_sentinel_1m_640 \
  --chip-m 640 --chip-px 64 \
  --chips-per-region 8000 \
  --max-chips 1000000
```

With the non-overlapping grid, the 399 regions contain about 5.8M candidate cells. `--chips-per-region 8000`
allows every region to contribute its full grid, and `--max-chips 1000000` stops at 1M. The build is resumable, so you can run it in batches and rerun the same command to
continue.

Do **not** point this at `data/raw/osm_sentinel`: that is A's existing 77k build made with the old
captioner. Keep it for comparison; the 1M run uses `data/raw/osm_sentinel_1m_640`.

If a region has too few OSM features, the actual count may be slightly below 1,000,000; report the
actual `n` from `dataset_stats.json`, along with `failed_regions` and `skipped_chips`.

Budget disk for roughly 25–40 GB of JPEG chips, ~1.5 GB of DINOv2 embeddings, and a large OSM PBF
cache. Run in batches and keep the cache.

Some grid cells will be skipped because the Sentinel-2 window is invalid or OSM has no visible
features. WorldCover can still label water and natural land cover; report the actual `n` from
`dataset_stats.json`; the target is 1M but the achieved count may be lower.

The random sampler is opt-in (`--sample-mode random`) and produces overlapping chips. Do not mix
random and grid results in the same manifest.

### WorldCover

WorldCover is on by default and is used as a fallback subject when OSM has no land-cover polygon:

- forest, shrub, grassland, cropland, built-up, bare, water, wetland, mangrove, snow, moss
- the manifest records `worldcover_class` and `worldcover_fraction`
- disable it with `--no-worldcover`

WorldCover is 2021, 10 m, and broad. It does not replace OSM for buildings, roads, or detailed land
use.

### Re-caption an older build

If a build was made before the line-weight fix, do not re-download images. Re-caption from the cached
OSM GeoParquet:

```sh
python experiments/osm_sentinel/recaption.py \
  --src data/raw/osm_sentinel \
  --dst data/raw/osm_sentinel_recaption
```

Then train with `--data data/raw/osm_sentinel_recaption`. This is the migration path for A's current
build.

## 6. Training

Once there is at least one train, one validation, and one test region:

```sh
python experiments/osm_sentinel/train.py \
  --data data/raw/osm_sentinel_1m_640 \
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
  --data data/raw/osm_sentinel_1m_640 \
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

Metric caveat: `caption_to_image recall@1` asks for the exact chip among thousands that often share
an identical caption; it will be low by construction. Use `image_to_label` accuracy and the
random-rotation control as the meaningful numbers. Report `caption_to_image` only as a secondary
metric.

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
- [ ] full run stops at `--max-chips 1000000`
- [ ] `dataset_stats.json` reports the actual `n` and it is at most 1,000,000
- [ ] `failed_regions` and `skipped_chips` are recorded
- [ ] captions are plain visible-feature sentences
- [ ] training run produces `results_dinov2-small.json`
- [ ] random-rotation control is present
- [ ] caption-to-image recall is reported as secondary, not headline
- [ ] report includes exact commands and caveats
