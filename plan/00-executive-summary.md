# Executive Summary

## Project

**Name:** Platonic Earth
**Track:** AI + Climate
**Test site:** Upper Arlington, Ohio, USA
**Level:** Intermediate / Advanced
**Team:** 1–3 high school students
**Deadline:** Sunday, November 15, 2026, 11:59 PM ET

## One-liner

> Type a land-cover description and find matching Sentinel-2 tiles in Upper Arlington — using frozen vision and text encoders aligned by a closed-form rotation, with no multimodal training.

## The problem

Upper Arlington is a built-out inner-ring suburb with a mature urban forest. The city’s tree canopy is about **34.2%** of land area, and a 2011–2021 canopy assessment found that about **33% of land** could support additional tree canopy. The city is preparing an **Urban Forestry Plan** and updating its Master Plan and Parks & Recreation Plan in 2026.

A sustainability or urban-forestry staff member who wants to find candidate areas for tree planting, canopy expansion, or land-cover inspection cannot easily search satellite imagery in plain language. They either browse tiles manually or rely on a fixed supervised classifier that answers only pre-defined classes.

## The user

**Primary user:** a City of Upper Arlington sustainability coordinator or urban-forestry staff member.

**Moment they reach for the tool:**

> “I need to find candidate areas in Upper Arlington for tree-canopy expansion or land-cover inspection, but I cannot browse every Sentinel-2 tile.”

**Decision supported:** which neighborhoods, parks, or corridors to inspect first.

**Important limitation:** the tool is a land-cover triage aid, not a canopy measurement system. It can find candidate residential, park, commercial, road, and water tiles. It cannot measure tree canopy per parcel.

## What goes in, what comes out

**Input:**

- A short natural-language query: `dense forest`, `residential neighborhood`, `park or grassland`, `commercial area`, `river corridor`.
- Or an uploaded Sentinel-2 tile from Upper Arlington.

**Output:**

- Top-3 matching tiles with EuroSAT labels and confidence scores.
- For an uploaded tile: nearest label and confidence.
- Coordinates so the user can inspect the location.

## Core AI

1. Frozen **DINOv2 ViT-S/14** encodes satellite tiles.
2. Frozen **all-MiniLM-L6-v2** encodes class descriptions.
3. A closed-form **orthogonal Procrustes** rotation maps image embeddings into text space.
4. Retrieval uses cosine similarity between the rotated image embedding and the query text embedding.

This is a direct test of the **Platonic Representation Hypothesis**:

> Independently trained image and text models share enough geometry that one closed-form rotation can support zero-shot land-cover retrieval.

## Data

**Training / validation / test:** EuroSAT.

- 27,000 labeled 64×64 Sentinel-2 tiles.
- 10 land-use classes.
- RGB version loads with `torchvision.datasets.EuroSAT`.
- Text side: 5–10 template captions per class.

**Local test:** 50 Sentinel-2 tiles clipped to the Upper Arlington city boundary.

- 10 residential.
- 10 commercial/industrial.
- 10 road/highway.
- 10 park/forest/herbaceous.
- 10 water/river.

Collected from Copernicus Browser, hand-labeled, never used for fitting.

## Baselines

Required by the brief:

- **Majority class** — always predict the most common EuroSAT class.
- **ResNet-18** — supervised classifier trained on EuroSAT.

For the alignment claim:

- **Random rotation** — fit Procrustes on shuffled pairs.
- **Unaligned cosine** — raw DINOv2 vs. MiniLM similarity.
- **CLIP zero-shot** — optional external baseline.

## Metrics

- Text→image top-1 and top-5 retrieval on the held-out EuroSAT test split.
- Image→text per-class accuracy and macro-F1.
- Local Upper Arlington out-of-distribution accuracy, with N = 50 reported honestly.
- Failure taxonomy: annual vs. permanent crop, forest vs. herbaceous, industrial vs. residential, clouds, season, domain shift, abstract queries.

## Demo

A Gradio app:

1. Type `dense forest` → top-3 Upper Arlington or EuroSAT tiles.
2. Upload a local tile → nearest label and confidence.
3. Show the baseline comparison: majority, ResNet-18, Procrustes.
4. Show one failure and explain it.

## Why Upper Arlington works as the test site

- Specific city with a clear boundary.
- Public Sentinel-2 coverage.
- Real climate/urban-forestry decision.
- Local tree-canopy data and policy context.
- Small enough to collect 50 tiles in a few hours.
- The team can tie the demo to a real place and a real user.

## Why this is creative

The suggested Climate project is supervised land-use classification. This project asks a different question:

> Can a closed-form rotation between frozen satellite-image and text embeddings replace joint multimodal training for land-cover search?

The addition is the **text search capability**, the **PRH test**, and the **Upper Arlington local evaluation**.

## Success criteria

**Minimum viable submission:**

- EuroSAT loads.
- DINOv2 and MiniLM embeddings extracted.
- Procrustes fitted.
- Top-1 retrieval measured against random rotation.
- ResNet-18 and majority baselines reported.
- Gradio demo works.
- One-page write-up and code link submitted.

**Expected:**

- Retrieval beats random rotation.
- ResNet-18 is stronger at fixed-class classification.
- Local tiles show a domain-shift drop.
- Honest failure analysis.

**Stretch:**

- CLIP baseline.
- Local tile map.
- Confidence/abstention.
- Per-class failure gallery.
- Feedback from a real Upper Arlington sustainability/forestry contact.

## Main risks

- DINOv2 may not separate 64×64 upscaled EuroSAT tiles well.
- Template captions may be too simple.
- Procrustes may underperform ResNet-18.
- Local tiles may not map cleanly to EuroSAT classes.
- Time: 48 hours is short.

See `10-risks-fallbacks.md`.
