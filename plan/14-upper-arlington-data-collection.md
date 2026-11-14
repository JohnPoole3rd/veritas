# Upper Arlington Data Collection Guide

**Owner:** A
**Tasks:** T10, T11.

This is the step-by-step guide for the 50 local Sentinel-2 tiles. Read `05-data-plan.md` first for context.

## 0. Goal

Collect 50 cloud-free Sentinel-2 tiles clipped to the Upper Arlington city boundary:

- 10 residential
- 10 commercial/industrial
- 10 road/highway
- 10 park/forest/herbaceous
- 10 water/river

These tiles are an **out-of-distribution test set**. They are never used to fit Procrustes or train ResNet-18.

## 1. City boundary

Do not use a simple rectangle. Upper Arlington is bordered by:

- **West:** Scioto River
- **North/east:** Columbus
- **South:** Marble Cliff and Grandview Heights

Use one of:

- OpenStreetMap boundary polygon for Upper Arlington.
- U.S. Census TIGER/Line boundary.

Approximate center: `40.02056, -83.06639`.

If a tile includes Grandview Heights, Marble Cliff, or Columbus, move it or clip it. Label the dominant land cover only after clipping.

## 2. Imagery settings

**Source:** Copernicus Browser.

**Mission:** Sentinel-2 L2A.

**Resolution:** 10 m.

**Layer:** true color (RGB).

**Date:** recent, low cloud, late spring/summer preferred.

**Tile size:** 64×64 pixels = **640 m × 640 m** at 10 m.

**Format:** PNG.

**Processing:**

- Download a larger crop.
- Resize/crop to 64×64.
- Keep the original crop if possible for QA.
- Record the exact imagery date.

## 3. Sampling design

- 10 tiles per category.
- Spread across the city; do not cluster all 10 in one neighborhood.
- Avoid overlapping tiles where possible.
- If a category is scarce, use the best available example and note it.
- If a tile is ambiguous, label `Other` and replace it if possible.
- Do not use tiles with heavy cloud, shadow, or snow.
- Do not use tiles from different seasons within the same category if avoidable.

## 4. Category guidance

### 4.1 Residential (ua_001–ua_010)

**Goal:** tile dominated by houses, streets, and yards.

**Search anchors:**

- Old Arlington / South of Lane historic district.
- North Arlington.
- River Ridge.
- Brandon Heights.
- Neighborhoods around Northam Park.
- Streets off Tremont, Reed, Kenny, and Riverside.

**Label:** `Residential`.

**Avoid:** tiles dominated by a park, a commercial strip, or the river.

### 4.2 Commercial/industrial (ua_011–ua_020)

**Goal:** tile dominated by commercial buildings, parking lots, or warehouses.

**Search anchors:**

- Lane Avenue corridor (Kingsdale, Tremont Center, Lane Avenue shopping area).
- Kenny Road / Henderson Road commercial nodes.
- Upper Arlington Municipal Services Center area.
- Mixed-use and parking-heavy corridors.

**Label:** `Industrial` (EuroSAT proxy).

**Note:** Upper Arlington has little heavy industry. This category is a **commercial/industrial proxy**. Say so in the write-up.

**Avoid:** pure residential blocks and pure road tiles.

### 4.3 Road/highway (ua_021–ua_030)

**Goal:** tile dominated by a major road corridor.

**Search anchors:**

- Lane Avenue.
- Henderson Road.
- Kenny Road.
- Riverside Drive.
- Fishinger Road.
- Northwest Boulevard.
- Tremont Road.

**Label:** `Highway`.

**Avoid:** a tile where the road is a thin line through houses; center on the corridor so pavement dominates.

### 4.4 Park/forest/herbaceous (ua_031–ua_040)

**Goal:** 5 tiles dominated by trees, 5 by grass/low vegetation.

**Search anchors:**

- Northam Park.
- Thompson Park.
- Fancyburg Park.
- Reed Road Park.
- Miller Park.
- Sunny 95 Park.
- Amelita Mirolo Barn area.
- Scioto Country Club / golf course edges (check boundary).

**Labels:**

- `Forest` for tree-dominated tiles (ua_031–ua_035).
- `HerbaceousVegetation` for grass/low-vegetation tiles (ua_036–ua_040).

**Avoid:** a park with a large parking lot or pool dominating the tile.

### 4.5 Water/river (ua_041–ua_050)

**Goal:** 5 river tiles, 5 lake/pond tiles.

**Search anchors:**

- Scioto River western edge.
- Riverside Drive riverfront.
- Any lake/pond inside a park.
- Griggs Reservoir only if within/immediately adjacent to the boundary; label carefully.

**Labels:**

- `River` for flowing river channel (ua_041–ua_045).
- `SeaLake` for a pond/lake (ua_046–ua_050).

**Avoid:** tiles dominated by trees along the riverbank; center on the water.

## 5. Copernicus workflow

1. Go to `https://browser.dataspace.copernicus.eu`.
2. Create a free account.
3. Search `Upper Arlington, Ohio`.
4. Enable the city boundary overlay if available; otherwise keep a boundary map open.
5. Select Sentinel-2 L2A.
6. Choose a recent date with low cloud.
7. Set true color.
8. Zoom so 64 pixels ≈ 640 m. Use the scale bar.
9. Export/download a crop.
10. Crop/resize to 64×64 PNG.
11. Save as `ua_001.png` … `ua_050.png`.
12. Record `tile_id`, `path`, `label`, `local_category`, `lat`, `lon`, `date`, `source`, `notes` in the manifest.

## 6. Labeling rubric

For each tile:

1. Identify the dominant land cover.
2. Map to the closest EuroSAT class.
3. Assign the local category.
4. If mixed, assign the dominant class and note the mix.
5. If unclear, label `Other` and explain.
6. Have a second person review 10 tiles.
7. Record disagreements; do not silently overwrite.

**Priority:**

- Water > built-up > vegetation > bare soil.
- Mostly road → `Highway`.
- Mostly trees → `Forest`.
- Mostly grass → `HerbaceousVegetation` or `Pasture`.
- Mostly houses → `Residential`.
- Mostly commercial buildings/pavement → `Industrial`.
- Mostly river → `River`.
- Mostly lake/pond → `SeaLake`.

## 7. Manifest schema

`data/local_tiles/manifest.csv`:

```csv
tile_id,path,label,local_category,lat,lon,date,source,notes
ua_001,data/raw/local_tiles/ua_001.png,Residential,residential,40.02,-83.06,2026-06-01,Copernicus,Old Arlington
```

Required columns:

- `tile_id`
- `path`
- `label`
- `local_category`
- `lat`
- `lon`
- `date`
- `source`
- `notes`

## 8. Run the local pipeline

After the manifest exists:

```sh
python scripts/extract_local_embeddings.py
python scripts/evaluate_local.py
```

If you have fewer than 50 tiles:

```sh
python scripts/extract_local_embeddings.py --allow-partial
python scripts/evaluate_local.py --allow-partial
```

The scripts use the **repo root** as the image root, so the manifest `path` column must be repo-relative:

```csv
ua_001,data/raw/local_tiles/ua_001.png,Residential,residential,40.02,-83.06,2026-06-01,Copernicus,Old Arlington
```

Outputs:

- `artifacts/embeddings/dinov2_local.npy`
- `artifacts/embeddings/dinov2_local_meta.json`
- `artifacts/metrics/local.csv`
- `artifacts/metrics/local_by_category.csv`
- `artifacts/metrics/local.json`

## 9. QA

- [ ] 50 tiles total.
- [ ] 10 per category.
- [ ] All within the city boundary.
- [ ] Cloud-free or minimal cloud.
- [ ] Coordinates recorded.
- [ ] Date recorded.
- [ ] Source recorded.
- [ ] Label matches rubric.
- [ ] 10 tiles reviewed by a second person.
- [ ] Disagreements documented.
- [ ] No personal data.
- [ ] Tiles never used for fitting.

## 10. Common pitfalls

| Pitfall | Fix |
|---|---|
| Tile includes Grandview/Columbus | Clip to boundary; move tile |
| Tile is mostly trees along river | Center on water; label water only if water dominates |
| Residential tile includes a park | Move or relabel |
| Commercial tile is mostly parking | That is okay; note it |
| Road tile is a thin line | Zoom/center on the corridor |
| Cloud/shadow | Choose another date |
| Snow | Avoid winter imagery |
| Mixed tile | Label dominant class; note mix; use `Other` if unclear |
| Same location used twice | Keep a map of tile centers; avoid overlap |
| Local tile used in fitting | Never fit on local; keep separate |

## 11. 50-tile tracking table

Fill this in as you collect. Use the manifest as the source of truth.

| Tile ID | Category | Location / anchor | Lat | Lon | Date | Label | QA | Notes |
|---|---|---|---|---|---|---|---|---|
| ua_001 | residential | | | | | | | |
| ua_002 | residential | | | | | | | |
| ua_003 | residential | | | | | | | |
| ua_004 | residential | | | | | | | |
| ua_005 | residential | | | | | | | |
| ua_006 | residential | | | | | | | |
| ua_007 | residential | | | | | | | |
| ua_008 | residential | | | | | | | |
| ua_009 | residential | | | | | | | |
| ua_010 | residential | | | | | | | |
| ua_011 | commercial/industrial | | | | | | | |
| ua_012 | commercial/industrial | | | | | | | |
| ua_013 | commercial/industrial | | | | | | | |
| ua_014 | commercial/industrial | | | | | | | |
| ua_015 | commercial/industrial | | | | | | | |
| ua_016 | commercial/industrial | | | | | | | |
| ua_017 | commercial/industrial | | | | | | | |
| ua_018 | commercial/industrial | | | | | | | |
| ua_019 | commercial/industrial | | | | | | | |
| ua_020 | commercial/industrial | | | | | | | |
| ua_021 | road/highway | | | | | | | |
| ua_022 | road/highway | | | | | | | |
| ua_023 | road/highway | | | | | | | |
| ua_024 | road/highway | | | | | | | |
| ua_025 | road/highway | | | | | | | |
| ua_026 | road/highway | | | | | | | |
| ua_027 | road/highway | | | | | | | |
| ua_028 | road/highway | | | | | | | |
| ua_029 | road/highway | | | | | | | |
| ua_030 | road/highway | | | | | | | |
| ua_031 | park/forest | | | | | | | |
| ua_032 | park/forest | | | | | | | |
| ua_033 | park/forest | | | | | | | |
| ua_034 | park/forest | | | | | | | |
| ua_035 | park/forest | | | | | | | |
| ua_036 | park/herbaceous | | | | | | | |
| ua_037 | park/herbaceous | | | | | | | |
| ua_038 | park/herbaceous | | | | | | | |
| ua_039 | park/herbaceous | | | | | | | |
| ua_040 | park/herbaceous | | | | | | | |
| ua_041 | river | | | | | | | |
| ua_042 | river | | | | | | | |
| ua_043 | river | | | | | | | |
| ua_044 | river | | | | | | | |
| ua_045 | river | | | | | | | |
| ua_046 | lake/pond | | | | | | | |
| ua_047 | lake/pond | | | | | | | |
| ua_048 | lake/pond | | | | | | | |
| ua_049 | lake/pond | | | | | | | |
| ua_050 | lake/pond | | | | | | | |

## 12. Acceptance

The local dataset is ready when:

- 50 tiles exist.
- 10 per category.
- All are within the boundary.
- All have coordinates, date, and source.
- Labels are reviewed.
- Manifest validates.
- Local tiles are excluded from fitting.
