# Platonic Earth

**Track:** AI + Climate  **Team:** J, A, D *(full names, emails and schools go on the submission form)*
**Try it:** https://johnpoole3rd.github.io/veritas/  **Code:** https://github.com/JohnPoole3rd/veritas  **Video:** *(YouTube link)*

> Prototype, not a land-use determination. Land cover is a model estimate; greenness and surface temperature are satellite measurements.

## The problem
Upper Arlington, Ohio's urban-forestry and sustainability staff decide each season which streets and lots to inspect first for new trees. Hot, bare places gain the most from shade. Satellites photograph the whole city every few days, but nobody can look through all those photos. Normal land-cover classifiers only answer a fixed list of classes. You can't ask them in your own words, e.g. “parking lots” or “tall trees near a river”.

## What we built
A website, with a public copy that runs entirely in the browser.
- **Search:** type words and get the best-matching 640 m Sentinel-2 squares, each with a similarity score (not a probability).
- **Heat Lab:** pick an area and get satellite greenness (Sentinel-2 NDVI), ground surface temperature (Landsat), how much cooler greener squares are, a “plant first” list of hot squares with few plants, and CSV/GeoJSON exports.

## What we added
We started from the suggested land-use mapping project (EuroSAT + ResNet-18), which is now our baseline, and added:
- **our own user and place:** Upper Arlington urban forestry;
- **our own data:** 50 Sentinel-2 tiles of the city that we labelled, plus live Landsat heat;
- **our own question:** can **one rotation** join an image AI and a text AI that were trained separately (the *Platonic Representation Hypothesis*)? That rotation is what makes search in words possible.

## How it works
- **Data:** EuroSAT, Sentinel-2 photos in 10 land types, split 18,900 / 4,050 / 4,050 (train / validation / test, seed 42), plus the 50 Upper Arlington tiles (Sentinel-2 L2A, 1 Aug 2025).
- **Models:** frozen DINOv2 ViT-S/14 (image → 384 numbers) and frozen all-MiniLM-L6-v2 (text → 384 numbers). Neither is retrained.
- **The rotation:** one orthogonal Procrustes rotation, a closed-form step with no training loop, fit on training images and captions to line the two up.
- **Baselines:** a random rotation, no rotation, always guessing the most common class, and a ResNet-18 fine-tuned on the labels.
- **Testing:** on held-out test data only.

## Results
<!-- auto:writeup_results -->
**1 · Search in words** (text → photos; 50 caption queries, each over all 4,050 held-out EuroSAT test photos; top-1 = the first photo has the right land type)

| Method | Top-1 | Top-5 | Verdict |
|---|---:|---:|---|
| **Procrustes rotation (ours)** | **96%** | **100%** | ahead |
| Random rotation (control, mean of 5 seeds) | 15.2% | 28% | |
| No rotation (control) | 10% | 34% | |

**2 · Label each photo** (image → one of 10 land types; N = 4,050 held-out EuroSAT test photos)

| Method | Accuracy | Macro-F1 | Verdict |
|---|---:|---:|---|
| ResNet-18, fine-tuned on the labels (strong baseline) | 98.2% | 98.1% | |
| **Procrustes rotation (ours, no training)** | **81.2%** | 79.5% | **behind ResNet**, far ahead of the controls |
| Random rotation (control) | 15.5% | 13.9% | |
| Always the most common class (“AnnualCrop”) | 11.1% | 2.0% | |

**3 · Our own town** (N = 50 Sentinel-2 tiles of Upper Arlington, Ohio, labelled by the team)

| Method | All tiles | Clear-cut tiles | Verdict |
|---|---:|---:|---|
| **Procrustes rotation (ours)** | **64%** | 81.1% | **tie** |
| ResNet-18 | 64% | 81.1% | |
| Always “Residential” (uses the local answers) | 62% | 78.4% | |

**4 · Heat, measured** (Upper Arlington; Sentinel-2 greenness vs Landsat ground temperature; 164 land squares of 640 m)

- Cooling: **−1.69 °C per +0.1 NDVI** (straight-line fit, r = −0.64).
- Least-green third vs greenest third: **42.4 °C vs 38.3 °C** (54 squares each). Phoenix, Arizona: −1.13 °C per +0.1 NDVI (r = −0.61).

**Where it fails:** rivers. 0 of 10 Upper Arlington river tiles get the right label; at 10 m per pixel a thin river looks like a road (River → Highway is also the top EuroSAT mix-up: 122 river photos, 32%).
<!-- /auto:writeup_results -->

**What this means:**
- The rotation really does connect the two models: search in words works, and a random rotation does not.
- A trained ResNet labels better, but it cannot take a question in words.
- On our own town the result is a **tie** with always answering “Residential”, so we treat it as a case study, not proof.
- The scores can't tell right answers from wrong ones, so the tool can't yet say “I don't know”. Abstract queries such as “climate change” still return tiles.

## What we'd do next
1. Use sharper aerial photos (about 0.3 m) and smaller squares, to find open grass and bare lots block by block. That is the real planting question.
2. Add an “I don't know” answer and a water check for rivers.
3. Get more local labels, balanced across land types and checked by city staff.
4. Try newer image models and a slightly more flexible map than a pure rotation, labelled clearly as a different test.

## Sources and credits
- **Data:** EuroSAT (Helber et al., 2019); Copernicus Sentinel-2 (ESA) via Earth Search; Landsat 8/9 Collection 2 (USGS/NASA); OpenStreetMap contributors; ESA WorldCover 2021; Esri World Imagery basemap.
- **Models:** DINOv2 (Oquab et al., 2024); all-MiniLM-L6-v2 (sentence-transformers); ResNet-18 (He et al., 2016).
- **Idea:** Platonic Representation Hypothesis (Huh, Cheung, Wang & Isola, 2024), arXiv:2405.07987.
- **Software:** PyTorch, Hugging Face Transformers, Sentence-Transformers, scikit-learn, SciPy, FastAPI, MapLibre, transformers.js, Remotion.
- **AI tools:** Claude (Anthropic) via Claude Code helped write code, the website, the video and these documents. The team chose the project, labelled the local tiles and checked every number against the logged results in `artifacts/metrics/`.
