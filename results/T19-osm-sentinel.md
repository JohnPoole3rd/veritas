# T19 — OSM-Sentinel experiment (77k chips): run report from A

Status: **IN PROGRESS.** The dataset build is running. A **preliminary EuroSAT-vs-OSM comparison** on the first
10,118 chips is below; the final runs start automatically when the build finishes.
Owner of this run: A (instructions: `experiments/osm_sentinel/RUNBOOK.md`)
For: J (owner of the experiment), cc D
Branch: `j/exp-osm-sentinel`; A's commits: `ae69074` (crash fixes), `a3f7fee` (report + first `recaption.py`), this comparison.
J's response: `c68f8f1` fixed the line weighting (issue 2) and the runbook (issue 4); J's `recaption.py` replaced A's.
Machine: Apple M1 Pro, 16 GB, MPS available; Python 3.14.5 (3.12 not installed); torch 2.15.0, pyrosm 0.29.0, rasterio 1.5.2

## Done so far

| Runbook step | Result |
|---|---|
| 1 Setup | `pip install -r requirements.txt` works on Python 3.14; MPS = True; 32/32 branch tests pass |
| 2 One-chip smoke run | OK: image + manifest row + plain caption (`Residential area with commercial area, grassland, dense buildings.`), no "Sentinel-2 image of" prefix |
| 3 100-chip, 5-region run | 90 chips in 98 s. **All 5 example regions are `train`**, so the checklist item "all three splits represented" cannot pass with the runbook's own command (issue 4). `br_amazon` gave 9/20 (sparse OSM) |
| 5 Full run | `build_dataset.py --chips-per-region 390 --max-chips 77000` started 14:17 ET, resumable, ~5 chips/s including downloads. **ETA ≈ 19:00–19:30 ET** for ~77k chips (exact `n` will come from `dataset_stats.json`) |
| 6 Training / testing | Preliminary comparison done on a 10k-chip snapshot (next section). Full `train.py` + comparison queued to run when the build exits |

## Does OSM training beat the original EuroSAT training? (preliminary)

**Short answer so far: no.** On a 10,118-chip snapshot, the OSM-fitted rotation is much worse than the
EuroSAT rotation on EuroSAT and on Upper Arlington. On its own held-out OSM regions it is only
marginally better with the as-built captions, and worse with the fixed captions. Fitting on
**EuroSAT + OSM together** slightly improves EuroSAT but costs Upper Arlington accuracy.

Script: `experiments/osm_sentinel/compare_eurosat.py`. It uses the same frozen DINOv2-small, MiniLM
and Procrustes for every fit; only the fitting pairs change. Scoring is shared with the main
pipeline (`src.retrieval.class_scores`, max over captions). Every cell has a random-rotation control
(mean of 5 seeds). Results: `experiments/osm_sentinel/compare_eurosat_prelim10k_{asbuilt,areafix}.json`.

Snapshot: OSM train 7,388 / val 1,950 / test 780 chips from 26 regions. EuroSAT fit: 18,900 pairs.
Accuracy shown; random-rotation control in brackets.

| Fit (captions) | EuroSAT test (4,050) | EuroSAT text search top-1 (50 queries) | Upper Arlington (50) | UA unambiguous (37) | OSM held-out test (780) |
|---|---|---|---|---|---|
| **EuroSAT** (main pipeline) | **0.812** (0.155) | 0.960 (0.152) | **0.640** (0.108) | **0.811** (0.124) | 0.219 / 0.328 † |
| OSM, as-built captions | 0.310 (0.098) | 0.360 (0.104) | 0.460 (0.172) | 0.568 (0.195) | 0.291 (0.065) |
| OSM, re-captioned (fixed weighting) | 0.250 (0.095) | 0.300 (0.100) | 0.380 (0.188) | 0.460 (0.205) | 0.221 (0.071) |
| EuroSAT + OSM (as-built) | **0.828** (0.083) | **0.980** (0.064) | 0.560 (0.052) | 0.703 (0.059) | 0.255 (0.080) |
| EuroSAT + OSM (re-captioned) | 0.825 (0.082) | **0.980** (0.064) | 0.580 (0.064) | 0.730 (0.086) | 0.294 (0.094) |

† EuroSAT fit scored against the as-built / re-captioned OSM test labels respectively.
The preliminary "re-captioned" rows used A's first `recaption.py`; its weighting is the same idea as J's
`c68f8f1` (J adds more line widths and normalises to the chip). The final run uses J's version.
Upper Arlington references: always "Residential" = 0.62 (oracle); EuroSAT majority = 0.00.

Reading:
- **Upper Arlington, the out-of-distribution test that matters for our user: OSM training does not help.**
  The OSM fit (0.38–0.46) is below the EuroSAT fit (0.64) and below the always-Residential oracle (0.62).
  Adding OSM to EuroSAT also lowers it (0.56–0.58).
- **EuroSAT: combined fitting is a small gain** (+1.3–1.6 points accuracy, text search 48/50 → 49/50).
  That is within what more pairs alone could explain; it needs the full run to confirm.
- **OSM held-out regions: every fit is weak** (≤ 0.33 accuracy, macro-F1 ≤ 0.11). The OSM labels are
  hard to predict from 10 m RGB (issues 2–3). Here the "test set" is only 2 regions (Tortuguero
  rainforest and the Havana coast), so treat those numbers as anecdotal until the full run has 30.
- Why OSM-only fitting is weak, likely causes:
  - captions describe *mapped* features, not visible cover (issue 3);
  - many chips share near-identical captions (low text diversity for Procrustes);
  - label noise from the line-weight bug (issue 2);
  - 1.28 km chips at 128 px vs EuroSAT's 640 m at 64 px, so the image statistics differ.

Caveats: 10% of the planned data, 2 test regions, single run. **The final numbers on the full ~77k build
will replace this table.**

## Issues found

**1. Bug, fixed (`ae69074`): one invalid OSM polygon crashed the whole build.**
The first full run died after 42 chips with `GEOSException: TopologyException: side location
conflict at -83.0434 40.0002` in `osm_captions._weight` (us_columbus). OSM routinely contains invalid
geometry, so a 77k run would keep dying. Fix:
- `_weight` repairs the geometry with `make_valid`, or ignores it if that fails;
- per-chip read/caption errors and region-level STAC/Geofabrik/pyrosm errors are logged and skipped
  instead of aborting (rerunning the same command retries skipped regions).

Caption logic is unchanged and the branch tests pass.

**2. Bug, now fixed by J in `c68f8f1` (the running 77k build still uses the old captioner; re-caption with J's
`recaption.py`): water lines dominated captions and labels.**
`_weight` returns **area** for polygons but **length** for lines, both in degrees. A river line
across a 1.28 km chip weighs ~1.06e-2, which is **about 80× the area of the whole chip** (1.32e-4).
So any mapped river or stream becomes the caption subject and the `water` label, even in a dense
city. Evidence:
- in the 100-chip run, 30/90 chips were labelled `water`;
- every Amazon chip was captioned just `River.`;
- Rotterdam, Iowa and Nairobi neighbourhoods became `River with …`.

I did **not** change this in the running build (changing the captioner mid-run would mix two
caption versions). Instead `experiments/osm_sentinel/recaption.py` re-captions an existing build
from the cached OSM GeoParquet, with no image re-download. Its only change is that a line's weight
becomes length × a nominal width (river 30 m, stream 4 m, road 15 m, default 5 m). Examples:

| Region | As built | Area-fixed |
|---|---|---|
| nl_rotterdam | River with residential area, grassland, dense buildings. (water) | Residential area with grassland, water, dense buildings. (residential) |
| us_iowa | River with park, water, dense buildings. (water) | Park with water, residential area, dense buildings. (green) |
| ke_nairobi | River with commercial area, residential area, dense buildings. (water) | Commercial area with residential area, park, dense buildings. (commercial) |

**Plan:** train and test on both the as-built captions (exactly per the runbook) and the area-fixed
captions, and report both. J decides which goes into `osm_captions.py`.

**3. Data caveat: OSM leaves most natural land cover unmapped.**
Even after the fix, Amazon chips stay `water` ("Stream with river."), because the surrounding forest
has no OSM polygon. Captions describe *what is mapped*, not what is visible. Labels will be biased
toward well-mapped cities and Europe/US, and against forest, desert and cropland in the Global South.

**4. Runbook inaccuracies.**
- *"~77,000 non-overlapping chips"* is not true. The 199 regions' total area fits ~76,700 chips of
  1.28 km only with perfect packing. 101 regions are asked for 390 chips when fewer fit, and chips
  are placed at random, so the 77k chips overlap substantially. Splits are by region, so this does
  **not** leak between train/val/test, but report the result as "77k overlapping chips from 199
  regions", not 77k independent locations.
- The step-3 multi-region example uses only `train` regions (see the table above).
- "Use Python 3.12": it works on 3.14 too; worth saying so.

**5. Metric caveat in `train.py` (not a bug).**
`caption_to_image recall@1` asks for the *exact* chip out of thousands that often share an
identical caption ("Residential area with … dense buildings."), so it will be low by construction.
`image_to_label` accuracy and the random-rotation control are the meaningful numbers.

## Work left

0. Queued, runs automatically after the 77k build exits (old captioner, kept as the comparison baseline per J's runbook):
   - `train.py --eval-eurosat` on `data/raw/osm_sentinel` (as built);
   - `compare_eurosat.py --tag full_asbuilt`;
   - J's `recaption.py --src data/raw/osm_sentinel --dst data/raw/osm_sentinel_recaption`;
   - `compare_eurosat.py --tag full_recaption` and `train.py --data data/raw/osm_sentinel_recaption --eval-eurosat`.

   A updates this report with the final tables. The new 1M-chip run (`data/raw/osm_sentinel_1m`) is a separate, later job.

1. Let the build finish (about 4 more hours); record the actual `n` from `dataset_stats.json`,
   split/country/label counts and any `REGION FAILED` lines.
2. `python experiments/osm_sentinel/train.py --device mps --dtype fp32 --batch-size 32 --num-workers 4 --eval-eurosat`
   on the as-built data; then `recaption.py` + the same command with `--data data/raw/osm_sentinel_areafix`.
3. Fill in this report: held-out recall/accuracy, random-rotation control, Upper Arlington, EuroSAT, caption examples.
4. J: choose the caption weighting; update RUNBOOK §5/§3 wording (issue 4).

## Suggestions (expand / tidy)

**Expand**
- Grid sampling with no-overlap instead of random points, and per-region quotas from area (`π r² / chip²`).
- Fill OSM gaps with ESA WorldCover (10 m, global) for land-cover labels. Keep OSM for built features
  and captions, so forests and cropland are labelled even where OSM is empty.
- Use the val split for something: pick the caption weighting / class prompts on val, report test once.
- Rebalance by label (cap `water`/`residential`) or report macro-F1 as the headline.
- Optional: multiple scenes per region (seasons) as augmentation; NIR band to settle water vs road.

**Tidy**
- Move the shared builder fixes (`make_valid`, skip-and-log) into `osm_source`/`osm_captions` with tests.
- `train.py` re-implements `class_scores` (already `src.retrieval.class_scores` on main) and the
  EuroSAT/local evaluation (`src.evaluate`, `src.local_tiles`). Reuse them so the numbers are
  comparable with the main results by construction.
- The branch is up to date with `main` and merges cleanly, but it changes **shared** code
  (`src/encoders.py`, `src/alignment.py`, `src/local_tiles.py`, the extraction scripts), not just
  `experiments/`. Review those as core-pipeline changes before merging, or split them into their own PR.
- Keep `data/raw/osm_sentinel/` (~1.5 GB cache already, growing) out of Git, as the runbook says.
- Write `dataset_stats.json` with `failed_regions` and `skipped_chips` counts.
