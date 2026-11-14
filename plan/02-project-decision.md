# Project Decision Record

## Selected project

**Platonic Earth — AI + Climate — Upper Arlington, Ohio.**

## Why this project

1. **Track fit:** directly addresses a community climate decision (tree-canopy expansion / land-cover inspection).
2. **Specific user:** Upper Arlington sustainability / urban-forestry staff.
3. **Data readiness:** EuroSAT loads easily; local Sentinel-2 tiles are public.
4. **Baseline clarity:** majority class and ResNet-18 are obvious and required.
5. **48-hour feasibility:** frozen models, closed-form rotation, small dataset, Gradio demo.
6. **PRH centrality:** the rotation is the search mechanism, not a side feature.
7. **Local relevance:** Upper Arlington has a real tree-canopy and Urban Forestry Plan context.
8. **Clear failure story:** annual vs. permanent crop; local domain shift.

## Why Upper Arlington

| Criterion | Why it fits |
|---|---|
| Specific place | 9.83-square-mile city, clear boundary |
| Public imagery | Sentinel-2 coverage via Copernicus |
| Climate relevance | 34.2% tree canopy; 33% possible additional canopy; 2026 Urban Forestry Plan |
| Local decision | Where to inspect for canopy expansion / land-cover change |
| Data size | 50 tiles is feasible in a few hours |
| User | City sustainability / urban-forestry staff |
| Story | “Help our city prioritize where to look” |

## Decision criteria

| Criterion | Weight | Platonic Earth score |
|---|---|---|
| Track fit | High | 5/5 |
| Data readiness | High | 5/5 |
| Baseline clarity | High | 5/5 |
| 48-hour feasibility | High | 4/5 |
| Demo strength | Medium | 4/5 |
| PRH centrality | Medium | 5/5 |
| Impact | Medium | 4/5 |
| Novelty | Medium | 4/5 |

## Alternatives considered and archived

### Platonic Picture Dictionary — AI + Languages

**Why not selected:** strong impact, but requires translation, multilingual text encoding, TTS, and user testing. Higher data risk, less certain technical result, and a weaker fit for the team’s current data access.

Archived spec: `04-picture-dictionary-spec.md`.

### Platonic Synesthesia — three modalities

**Why not selected:** does not fit one track, no single user, too many modalities for 48 hours, no natural baseline.

### Platonic Bioacoustics

**Why not selected:** indirect climate fit; audio-text alignment risk; no local Upper Arlington angle.

### Platonic Protein

**Why not selected:** domain complexity; less accessible to judges; data processing too heavy.

### Platonic Interoperability

**Why not selected:** practical but not visually compelling; weaker track fit.

## Locked scope

- One track: AI + Climate.
- One site: Upper Arlington, Ohio.
- One method: frozen DINOv2 + MiniLM + Procrustes.
- One local test: 50 Sentinel-2 tiles, five categories.
- One demo: Gradio text search + image upload.
- One required baseline: majority class.
- One strong baseline: ResNet-18.
- One alignment baseline: random rotation.

## Out of scope

- Audio.
- Multilingual.
- Three modalities.
- Object detection/segmentation.
- Change detection over time.
- Real-time satellite ingestion.
- Parcel-level canopy measurement.
- Legal/regulatory land-use decisions.
- Fine-tuning encoders.

## Revisit conditions

Change the project only if:

- EuroSAT cannot load in Phase 1.
- The local tiles cannot be collected in Phase 1.
- Procrustes is at chance after debugging.
- The team is Beginner and cannot run PyTorch.

Otherwise, finish the selected project.
