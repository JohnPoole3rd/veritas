# Data Plan — EuroSAT + Upper Arlington

**Owners:** J for EuroSAT, splits, captions; A for Upper Arlington local tiles.

## Part A — EuroSAT

### A.1 What it is

| Property | Value |
|---|---|
| Tiles | ~27,000 |
| Tile size | 64×64 pixels |
| Source | Sentinel-2 |
| Classes | 10 |
| Bands | 13 spectral; RGB version available |
| Task | scene classification |
| Access | `torchvision.datasets.EuroSAT` |
| License | Verify on the official dataset page; cite Helber et al. |

**Classes:**

1. AnnualCrop
2. Forest
3. HerbaceousVegetation
4. Highway
5. Industrial
6. Pasture
7. PermanentCrop
8. Residential
9. River
10. SeaLake

**Why it fits:**

- Small enough for Colab.
- Public and widely used.
- Clean labels.
- Easy to pair with text captions.
- The brief’s suggested Climate project already uses it, so judges understand it.

**Download:**

```python
from torchvision.datasets import EuroSAT
ds = EuroSAT(root="data/raw/eurosat", download=True)
```

If the download fails, use the Hugging Face mirror or TensorFlow Datasets. Do not spend more than 30 minutes; the brief says switch.

### A.2 Splits

| Split | Share | Use |
|---|---:|---|
| Train | 70% | Fit Procrustes; train ResNet-18 |
| Validation | 15% | Choose captions, model size, threshold |
| Test | 15% | Final numbers only |
| Local | separate | Upper Arlington OOD case study |

Rules:

- Stratify by class.
- Seed the split.
- Deduplicate before splitting.
- Never fit or tune on test.
- Never use local tiles for fitting.

Expected approximate sizes: 18,900 train / 4,050 val / 4,050 test.

### A.3 Captions

Use 5–10 templates per class. Store in `data/captions.jsonl`.

Examples:

**AnnualCrop**

- “a satellite image of annual cropland”
- “farmland with seasonal crops”
- “agricultural fields with bare soil between rows”
- “a patchwork of crop fields”
- “cropland seen from above”

**Forest**

- “a satellite image of dense forest”
- “tree canopy seen from above”
- “a wooded area with closed canopy”
- “forest land”
- “dark green tree cover”

**HerbaceousVegetation**

- “a satellite image of grassland”
- “low vegetation and shrubs”
- “herbaceous vegetation”
- “open grassy land”
- “sparse natural vegetation”

**Highway**

- “a satellite image of a highway”
- “a major road with vehicles”
- “a multi-lane highway”
- “transport corridor”
- “road network”

**Industrial**

- “a satellite image of an industrial site”
- “factories and warehouses”
- “large commercial buildings”
- “industrial buildings and pavement”
- “manufacturing area”

**Pasture**

- “a satellite image of pasture”
- “grazing land”
- “open grass for livestock”
- “meadow”
- “managed grassland”

**PermanentCrop**

- “a satellite image of permanent crops”
- “orchards and vineyards”
- “tree crops in rows”
- “long-lived agricultural plants”
- “fruit tree plantation”

**Residential**

- “a satellite image of a residential area”
- “houses and streets”
- “a neighborhood”
- “suburban housing”
- “dense residential buildings”

**River**

- “a satellite image of a river”
- “a winding waterway”
- “a river through land”
- “flowing water”
- “river channel”

**SeaLake**

- “a satellite image of a lake”
- “open water”
- “a sea or lake”
- “large body of water”
- “blue water surface”

Caption rules:

- Keep captions class-consistent.
- Do not claim tile-specific details.
- Start with 5 per class; add more only if retrieval is weak.
- Optional: LLM-generated paraphrases, manually checked.

## Part B — Upper Arlington local data

**Owner:** A (T10, T11).

### B1. Why Upper Arlington

- City center: **40.02056° N, 83.06639° W**.
- Area: **9.83 sq mi**.
- Bordered west by the **Scioto River**, north/east by Columbus, south by Marble Cliff and Grandview Heights.
- **34.2% tree canopy**; about **33% possible additional canopy**.
- **21 parks**, ~**186 acres**.
- **Tree City USA** since 1990.
- 2026 **Urban Forestry Plan** and Master Plan/Parks & Recreation Plan updates.
- Sustainability agenda: zero waste, zero pollution, zero emissions by 2050.

### B2. Boundary

Do **not** use a simple rectangle. Use the Upper Arlington city boundary polygon from:

- OpenStreetMap, or
- U.S. Census TIGER/Line boundary.

Clip Sentinel-2 tiles to that polygon. This avoids accidentally including Grandview Heights, Marble Cliff, or Columbus.

Approximate center for search: `40.02056, -83.06639`.

### B3. Imagery source

**Copernicus Browser:** `https://browser.dataspace.copernicus.eu`

Steps:

1. Create a free account.
2. Search “Upper Arlington, Ohio.”
3. Select **Sentinel-2 L2A**.
4. Choose a recent, mostly cloud-free date.
5. Set the layer to true color (RGB).
6. Download tiles at 10 m resolution.
7. Crop/resize to 64×64 pixels for model input.
8. Record the date, coordinates, and source.

Recommended dates:

- Late spring / summer for vegetation contrast.
- Early fall for crop/grass contrast.
- Avoid snow and heavy cloud.
- If possible, use the same season as EuroSAT imagery.

### B4. Local tile plan — 50 tiles

| Local category | Count | EuroSAT mapping | Notes |
|---|---:|---|---|
| Residential | 10 | Residential | Old Arlington, North Arlington, River Ridge |
| Commercial/industrial | 10 | Industrial (proxy) | Lane Ave, Kenny Rd, Henderson Rd, Tremont Center |
| Road/highway | 10 | Highway | Lane Ave, Henderson Rd, Kenny Rd, Riverside Dr, Fishinger Rd |
| Park/forest/herbaceous | 10 | Forest (5) + HerbaceousVegetation (5) | Northam, Thompson, Fancyburg, Reed Road, Miller, Sunny 95 |
| Water/river | 10 | River (5) + SeaLake (5) | Scioto River corridor |

If a tile does not fit, label `Other` and note why. Replace it if possible so each category has 10 useful examples.

### B5. Suggested locations

Use these as search anchors, not as exact coordinates:

**Residential**

- Old Arlington / South of Lane historic district.
- North Arlington neighborhood.
- River Ridge.
- Brandon Heights.
- Neighborhoods around Northam Park.
- Streets off Tremont, Reed, Kenny, and Riverside.

**Commercial/industrial**

- Lane Avenue corridor (Kingsdale, Tremont Center, Lane Avenue shopping area).
- Kenny Road / Henderson Road commercial nodes.
- Upper Arlington Municipal Services Center area (3600 Tremont Rd).
- Mixed-use and parking-heavy corridors.

**Road/highway**

- Lane Avenue.
- Henderson Road.
- Kenny Road.
- Riverside Drive.
- Fishinger Road.
- Northwest Boulevard.
- Tremont Road.

**Park/forest/herbaceous**

- Northam Park.
- Thompson Park.
- Fancyburg Park.
- Reed Road Park.
- Miller Park.
- Sunny 95 Park.
- Amelita Mirolo Barn area.
- Scioto Country Club / golf course edges (check boundary).

**Water/river**

- Scioto River western edge.
- Riverside Drive riverfront.
- Any lake/pond inside a park (if present).
- Griggs Reservoir area only if it falls within or immediately adjacent to the city boundary; label carefully.

### B6. Local labeling rubric

For each tile:

1. Identify the dominant land cover.
2. Map to the closest EuroSAT class.
3. Assign the local category.
4. If mixed, assign the dominant class and note the mix.
5. If unclear, label `Other` and explain.
6. Have a second person review 10 tiles.
7. Record disagreements; do not silently overwrite.

**Label priority:**

- Water > built-up > vegetation > bare soil.
- If a tile is mostly a road, label Highway even if buildings are visible.
- If a tile is mostly trees, label Forest.
- If a tile is mostly grass, label HerbaceousVegetation or Pasture.
- If a tile is mostly houses, label Residential.
- If a tile is mostly commercial buildings/pavement, label Industrial.
- If a tile is mostly river, label River.
- If a tile is mostly lake/pond, label SeaLake.

### B7. File naming

```text
ua_001.png
ua_002.png
...
ua_050.png
```

Manifest:

```csv
tile_id,path,label,local_category,lat,lon,date,source,notes
ua_001,data/raw/local_tiles/ua_001.png,Residential,residential,40.02,-83.06,2026-06-01,Copernicus,Old Arlington
```

The `path` column is **repo-relative**. `scripts/extract_local_embeddings.py` uses the repo root as the image root.

### B8. Local QA checklist

- [ ] 50 tiles total.
- [ ] 10 per local category.
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

## Part C — Licensing and citations

| Resource | Use | License / terms | Citation |
|---|---|---|---|
| EuroSAT | training tiles | verify dataset page | Helber et al., 2019 |
| Sentinel-2 / Copernicus | local tiles | Copernicus terms | ESA |
| DINOv2 | image encoder | check model card | Oquab et al., 2023 |
| all-MiniLM-L6-v2 | text encoder | check model card | Reimers & Gurevych |
| ResNet-18 | baseline | torchvision | He et al., 2016 |
| Gradio | demo | open source | Gradio |
| Google Colab | compute | Colab terms | Google |

## Part D — Fallbacks

If EuroSAT fails:

- Hugging Face EuroSAT mirror.
- TensorFlow Datasets EuroSAT.
- BigEarthNet.
- UC Merced Land Use.
- RESISC45.

If local tiles fail:

- Use 20 tiles instead of 50; report N.
- Use a nearby Franklin County area, clearly labeled.
- Use only EuroSAT test as the evaluation and treat local as a qualitative example.
- Do not fabricate local tiles.

## Part E — Storage

| Item | Estimate |
|---|---|
| EuroSAT RGB | ~90 MB |
| DINOv2 ViT-S | ~90 MB |
| MiniLM | ~90 MB |
| Embeddings (27k × 384 float32) | ~40 MB |
| Local tiles | < 10 MB |
| Figures | < 20 MB |
