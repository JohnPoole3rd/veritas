# Experiment Log

Fill this in during the build. Do not reconstruct from memory.

## 1. Run log

| Run ID | Date/time | Owner | Image encoder | Text encoder | Alignment | Train N | Val N | Test N | Top-1 | Top-5 | Macro-F1 | Notes |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| R001 | | | DINOv2 ViT-S/14 | MiniLM-L6 | Procrustes | | | | | | | first sanity run |
| R002 | | | | | Random rotation | | | | | | | null baseline |
| R003 | | | | | Unaligned cosine | | | | | | | no rotation |
| R004 | | | | | ResNet-18 | | | | | | | supervised |
| R005 | | | | | Majority | | | | | | | required baseline |
| R006 | | | | | CLIP | | | | | | | optional |
| R007 | | | | | Procrustes | | | | | | | full test |
| R008 | | | | | Procrustes | | | | | | | Upper Arlington local |

## 2. Data inventory

| Dataset | Version | Split | Count | Source | License | Notes |
|---|---|---|---:|---|---|---|
| EuroSAT | | train | | torchvision | | |
| EuroSAT | | val | | torchvision | | |
| EuroSAT | | test | | torchvision | | |
| Upper Arlington local | | local | 50 | Copernicus | | 10 per category |
| Upper Arlington local | | local | | Copernicus | | QA reviewed |

## 3. Upper Arlington local inventory

| Local category | Planned N | Collected N | Labeled N | QA N | Notes |
|---|---:|---:|---:|---:|---|
| Residential | 10 | | | | |
| Commercial/industrial | 10 | | | | |
| Road/highway | 10 | | | | |
| Park/forest/herbaceous | 10 | | | | |
| Water/river | 10 | | | | |
| **Total** | **50** | | | | |

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
| | Track = Climate | Best fit | Languages | until T+6 |
| | Project = Platonic Earth | Feasible, PRH central | Picture Dictionary | until T+6 |
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
| Majority | | | | | |
| ResNet-18 | | | | | |
| Random rotation | | | | | |
| Unaligned cosine | | | | | |
| Procrustes | | | | | |
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
| Residential | 10 | | | |
| Commercial/industrial | 10 | | | |
| Road/highway | 10 | | | |
| Park/forest/herbaceous | 10 | | | |
| Water/river | 10 | | | |
| **Total** | **50** | | | |

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

## 9. Time log

| Checkpoint | Planned | Actual | Status | Notes |
|---|---|---|---|---|
| Data loaded | T+2 | | | |
| Embeddings | T+6 | | | |
| First alignment | T+10 | | | |
| Full run | T+16 | | | |
| Local tiles | T+24 | | | |
| Demo skeleton | T+30 | | | |
| Failure case | T+36 | | | |
| Write-up | T+42 | | | |
| Video | T+45 | | | |
| Submitted | T+47 | | | |

## 10. Post-mortem

- What worked?
- What failed?
- What would you do differently?
- What is the one result to highlight?
- What is the one limitation to tell a judge?
