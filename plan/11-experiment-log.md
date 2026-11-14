# Experiment Log

Fill this in during the build. Do not reconstruct from memory.

Owner values: `J`, `A`, `D`, `J1`, `J2`, … , `A1`, `A2`, … , `D1`, `D2`, …

## 1. Run log

| Run ID | Date/time | Owner | Image encoder | Text encoder | Alignment | Train N | Val N | Test N | Top-1 | Top-5 | Macro-F1 | Notes |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| R001 | 2026-11-14 | J | DINOv2 ViT-S/14 | MiniLM-L6 | Procrustes | 64 | 64 | 64 | 0.72 | 0.98 | | smoke subset; val retrieval |
| R002 | 2026-11-14 | J | DINOv2 ViT-S/14 | MiniLM-L6 | Random rotation | 64 | 64 | 64 | 0.14 | 0.36 | | smoke subset; test retrieval |
| R003 | 2026-11-14 | J | DINOv2 ViT-S/14 | MiniLM-L6 | Unaligned cosine | 64 | 64 | 64 | 0.26 | 0.48 | | smoke subset; test retrieval |
| R004 | 2026-11-14 | A | ResNet-18 (ImageNet init) | — | — (supervised) | 18900 | 4050 | 4050 | 0.9817 | | 0.9812 | 10 ep, 64px, MPS, best ep 10 by val; results/T08.md |
| R005 | 2026-11-14 | J, A | — | — | Majority | 18900 | 4050 | 4050 | 0.1111 | | 0.0200 | J scaffold run reproduced by A (T09); AnnualCrop, 5-way tie |
| R006 | 2026-11-14 | J | DINOv2 ViT-S/14 | MiniLM-L6 | Procrustes | 512 | 512 | 512 | 0.94 | 0.98 | | 512 subset; val retrieval |
| R007 | 2026-11-14 | J | DINOv2 ViT-S/14 | MiniLM-L6 | Random rotation | 512 | 512 | 512 | 0.10 | 0.32 | | 512 subset; test null baseline |
| R008 | 2026-11-14 | J | DINOv2 ViT-S/14 | MiniLM-L6 | Unaligned cosine | 512 | 512 | 512 | 0.16 | 0.36 | | 512 subset; test baseline |
| R009 | 2026-11-14 | J | DINOv2 ViT-S/14 | MiniLM-L6 | Procrustes | 512 | 512 | 512 | 0.94 | 0.96 | | 512 subset; test retrieval |
| R010 | 2026-11-14 | A | DINOv2 ViT-S/14 | MiniLM-L6 | Procrustes (image→class) | 18900 | 4050 | 4050 | 0.8121 | 0.9901 | 0.7948 | full split, J's scripts re-run on MPS; per-image (comparable with R004); random rot 0.159, unaligned 0.083; text→image top-1 0.96 (50 queries) |
| R011 | 2026-11-14 | A | DINOv2 ViT-S/14 | MiniLM-L6 | Procrustes | 18900 | | 50 local | 0.64 | | | Upper Arlington OOD; 0.81 on 37 non-ambiguous; local majority 0.62 |
| R012 | 2026-11-14 | A | ResNet-18 | — | — | 18900 | | 50 local | 0.64 | | | Upper Arlington OOD; 0.81 on 37 non-ambiguous |

## 2. Data inventory

| Dataset | Version | Split | Count | Source | License | Notes |
|---|---|---|---:|---|---|---|
| EuroSAT | | train | | torchvision | | |
| EuroSAT | | val | | torchvision | | |
| EuroSAT | | test | | torchvision | | |
| Upper Arlington local | S2C_17TLE_20250801_0_L2A (2025-08-01) | local | 50 | Copernicus Sentinel-2 L2A via AWS Earth Search | Copernicus open data | 10 per stratum; T10 |
| Upper Arlington local | manifest.csv | local | 10 | blind second labeller (AI subagent A2) | | QA reviewed; 6/10 agree; T11 |

## 3. Upper Arlington local inventory

| Local category | Planned N | Collected N | Labeled N | QA N | Notes |
|---|---:|---:|---:|---:|---|
| Residential | 10 | 10 | 10 | 2 | stratum; all labelled Residential |
| Commercial/industrial | 10 | 10 | 10 | 2 | 3 Industrial, 7 Residential |
| Road/highway | 10 | 10 | 10 | 2 | 1 Highway, 9 Residential (arterials 5–11 % of pixels) |
| Park/forest/herbaceous | 10 | 10 | 10 | 2 | 1 Forest, 4 HerbaceousVegetation, 5 Residential |
| Water/river | 10 | 10 | 10 | 2 | 10 River; straddle W boundary; no SeaLake |
| **Total** | **50** | **50** | **50** | **10** | blind QA 6/10 agree; see results/T11.md |

## 4. Failure log

| ID | Input | True label | Predicted label | Score | Failure type | Likely cause | Fix idea |
|---|---|---|---|---|---:|---|---|---|
| F001 | | | | | annual vs. permanent | lifecycle not visible | temporal data |
| F002 | | | | | forest vs. herbaceous | color similarity | multi-band |
| F003 | | | | | industrial vs. residential | built-up similarity | more context |
| F004 | | | | | cloud/snow | not in training | local fine-tune |
| F005 | | | | | abstract query | not a land-cover class | query rewriting |
| F006 | | | | | Upper Arlington domain shift | different distribution | local labels |

## 5. Decision log

| Date/time | Decision | Reason | Alternatives | Reversible? |
|---|---|---|---|---|
| | Track = Climate | Best fit | Languages | until Phase 1 |
| | Project = Platonic Earth | Feasible, PRH central | Picture Dictionary | until Phase 1 |
| | Site = Upper Arlington | Local, public imagery | Columbus, Franklin County | yes |
| | Dataset = EuroSAT | Small, public | BigEarthNet, UC Merced | yes |
| | Image encoder = ViT-S/14 | Fast, 384-dim | ViT-B/14 | yes |
| | Text encoder = MiniLM | Simple, 384-dim | MPNet, LaBSE | yes |
| | Alignment = Procrustes | Closed form | CCA, ridge | no |
| | Baseline = ResNet-18 | Required, strong | logistic regression | yes |
| | Demo = Gradio | Fast | Streamlit | yes |

## 6. Hyperparameter log

| Run ID | DINOv2 | Input size | Batch | Text model | Caption mode | Centering | Normalization | W shape |
|---|---|---|---:|---|---|---|---|---|
| R001 | ViT-S/14 | 224 | 32 | MiniLM-L6 | random | train mean | L2 | 384×384 |
| R007 | ViT-S/14 | 224 | 32 | MiniLM-L6 | class centroid | train mean | L2 | 384×384 |
| R008 | ViT-S/14 | 224 | 32 | MiniLM-L6 | class centroid | train mean | L2 | 384×384 |

## 7. Results

### Main results

| Method | Top-1 | Top-5 | Macro-F1 | Test N | Notes |
|---|---:|---:|---:|---:|---|
| Majority | 0.1111 | | 0.0200 | 4050 | R005; constant AnnualCrop |
| ResNet-18 | 0.9817 | | 0.9812 | 4050 | R004; per-image accuracy |
| Random rotation | 0.1590 | | 0.1559 | 4050 | R010; image→class |
| Unaligned cosine | 0.0827 | | 0.0720 | 4050 | R010; image→class |
| Procrustes | 0.8121 | 0.9901 | 0.7948 | 4050 | R010; image→class (text→image top-1 0.96 over 50 queries) |
| CLIP | | | | | optional |

### Per-class results

| Class | Procrustes top-1 | ResNet top-1 | Most confused with |
|---|---:|---:|---|
| AnnualCrop | | | |
| Forest | | | |
| HerbaceousVegetation | | | |
| Highway | | | |
| Industrial | | | |
| Pasture | | | |
| PermanentCrop | | | |
| Residential | | | |
| River | | | |
| SeaLake | | | |

### Upper Arlington local results

| Local category | N | Procrustes acc. | ResNet-18 acc. | Notes |
|---|---:|---:|---:|---|
| Residential | 10 | 1.0 | 1.0 | all labelled Residential |
| Commercial/industrial | 10 | 0.9 | 1.0 | 3 Industrial + 7 Residential labels |
| Road/highway | 10 | 0.9 | 0.8 | 9 Residential labels |
| Park/forest/herbaceous | 10 | 0.4 | 0.3 | golf turf → Pasture (P) / Forest p≈1 (R) |
| Water/river | 10 | 0.0 | 0.1 | P: 6/10 → Highway |
| **Total** | **50** | **0.64** | **0.64** | local majority (Residential) 0.62; R011/R012 |

### Local tile detail

| Tile ID | True label | Local category | Procrustes | ResNet-18 | Notes |
|---|---|---|---|---|---|
| ua_001 | | | | | |
| ua_002 | | | | | |
| ... | | | | | |

## 8. Claim log

Every claim in the write-up must trace to a run.

| Claim | Evidence | Run ID | Confidence |
|---|---|---|---|
| “Procrustes beats random rotation” | table | R007 vs R002 | |
| “ResNet is stronger at fixed classes” | table | R004 vs R007 | |
| “Upper Arlington shows domain shift” | table | R008 | |
| “Annual vs. permanent crop is the main confusion” | confusion matrix | R007 | |

## 9. Phase log

| Phase | Planned outcome | Actual | Status | Notes |
|---|---|---|---|---|
| Phase 0 — Setup | T01 scaffold; A starts T10; D mock UI | J T01 done; A/D in progress | J done | J |
| Phase 1 — Data | T02 splits; T03 captions; T10/T11 tiles | J T02/T03 done; A tiles in progress | J done | J |
| Phase 2 — Alignment | T04/T05 embeddings; T06 Procrustes; T08/T09 baselines | J T04-T06 done on 512 subset; A baselines pending | J done | J |
| Phase 3 — Retrieval | T07 retrieval; T12 local eval; T13 failures; T14 demo | J T07 done; D T14 real backend verified; A local pending | partial | J/D |
| Phase 4 — Integration | Merge branches; final eval; T15 draft | Main synced; app verified with N=512 | partial | J |
| Phase 5 — Docs | T15 write-up; T16 video | Pending A local results | pending | D |
| Phase 6 — Submit | T17 checklist; submit | | | |

## 10. Post-mortem

- What worked?
- What failed?
- What would you do differently?
- What is the one result to highlight?
- What is the one limitation to tell a judge?
