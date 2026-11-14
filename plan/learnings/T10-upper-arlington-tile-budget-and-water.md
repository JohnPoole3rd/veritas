# T10 — Upper Arlington tile budget, water outside the boundary, and data source

**Date:** 2026-11-14
**Author:** A
**Task:** T10
**Status:** confirmed

## What we learned

The water lies just outside the OSM city boundary, and the city holds only about 61 non-overlapping
640 m tiles. So the 50 local tiles cannot all be fully inside the city, disjoint, and category-pure.

## Details

1. **Water is outside the city.** The OSM city boundary (relation 181876, 9.76 sq mi) follows the
   *east* bank of the Scioto River / Griggs Reservoir. Inside the boundary, ESA WorldCover finds only
   ~3 ha of open water (golf-course ponds). No 640 m tile fully inside Upper Arlington is
   water-dominated.
2. **The city only fits ~61 non-overlapping 640 m tiles.** 50 category-pure, fully disjoint tiles is
   not possible; strict non-overlap left residential/forest slots unfilled or filled with poor tiles.
3. **Category purity is limited at 10 m.** Arterials (Lane, Tremont, Riverside, Fishinger, Kenny)
   cover 5–11 % of a 640 m tile; residential tiles have ~70–85 % WorldCover tree cover (the city's
   34 % canopy), so "forest" tiles and treed residential tiles look alike.
4. **Source.** Tiles come from Sentinel-2 L2A (Copernicus) scene `S2C_17TLE_20250801_0_L2A`, read
   from the public Copernicus Sentinel-2 COG mirror on AWS (Element84 Earth Search) instead of the
   Copernicus Browser GUI, so collection is scripted and reproducible (`python -m src.local_collect`).
   Same product, same 10 m TCI true-colour band.

## Evidence

- `data/local_tiles/collection_log.csv` — per-tile `inside`, `overlap`, WorldCover and OSM fractions.
- Water tiles ua_041–ua_050: 25–37 % of the tile inside the city, 10–40 % water.
- Road tiles ua_021–ua_030: OSM major-road pavement 5–11 %.

## Impact

- Water tiles straddle the western boundary ("immediately adjacent", allowed by plan/14 §4.5).
  The write-up must say the water itself is Columbus-owned Scioto/Griggs Reservoir.
- Tiles may overlap earlier picks by ≤ 25 % (recorded per tile). Local N = 50 is not 50 independent
  samples; treat as a case study (already the plan's framing).
- Highway labels follow EuroSAT's convention (road is the salient, centred feature, not the majority
  of pixels). Expect Highway/Residential confusion; this is a real failure type for T13.
- The planned 5 River / 5 SeaLake split is replaced by visual labelling in T11.

## Plan changes

- [x] Updated `plan/14-upper-arlington-data-collection.md` (§1, §3, §4.5 notes; A-owned).
- [ ] J: decide whether `plan/TASKS.md` T10/T11 acceptance ("clipped to city boundary", "10 per EuroSAT sub-type") should be relaxed to match.

## Next action

- T11: label visually by rubric; record mixes; use `Other` where no class dominates.
- T12/T13: report per-category results with these caveats; Highway and Forest are expected weak spots.
- D (T15): use the wording "tiles in and immediately adjacent to Upper Arlington".

## Credits (brief §5 requires every dataset/tool with links)

- Sentinel-2 L2A: "Contains modified Copernicus Sentinel data [2025]", accessed via Element84 Earth Search / AWS Open Data `sentinel-cogs` (https://registry.opendata.aws/sentinel-2-l2a-cogs/).
- ESA WorldCover 10 m 2021 v200, CC BY 4.0 (https://esa-worldcover.org) — used only to choose tile locations.
- OpenStreetMap contributors, ODbL (https://www.openstreetmap.org/copyright) — city boundary (relation 181876) and land-use/roads used only to choose tile locations.

## Idea for J (brief §4.3.3 "make it your own": "look for real change over time")

`src/local_collect.py --scene <id>` can re-cut the *same 50 tile windows* from an earlier leaf-on scene
(e.g. summer 2019) so the demo can show per-tile prediction change for the canopy question. Not started;
needs J's go-ahead because it adds scope.
