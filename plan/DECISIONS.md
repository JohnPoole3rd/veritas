# DECISIONS.md — Locked Decisions

Status: **Accepted**. Revisit only with a human decision.

| ID | Decision | Value | Rationale |
|---|---|---|---|
| D001 | Track | **AI + Climate** | Best fit for local land-use search and Upper Arlington. |
| D002 | Project | **Platonic Earth** | Keeps PRH + Procrustes central; feasible in 48h. |
| D003 | Test site | **Upper Arlington, Ohio** | Local, specific, data available via Sentinel-2. |
| D004 | User | Upper Arlington sustainability / urban-forestry staff | Real decision: where to inspect for canopy/land-cover change. |
| D005 | Primary decision | Prioritize areas for tree-canopy expansion / land-cover inspection | Ties to 2026 Urban Forestry Plan and 34.2% canopy. |
| D006 | Training data | **EuroSAT RGB** | 27k tiles, 10 classes, easy loader. |
| D007 | Local data | **50 Sentinel-2 tiles** clipped to Upper Arlington | Local OOD test and “make it your own.” |
| D008 | Local categories | Residential, commercial/industrial, road/highway, park/forest/herbaceous, water/river | 10 tiles each; maps to EuroSAT classes. |
| D009 | Image encoder | **DINOv2 ViT-S/14**, frozen | Fast, 384-dim, square Procrustes with MiniLM. |
| D010 | Text encoder | **all-MiniLM-L6-v2**, frozen | 384-dim, simple, reliable. |
| D011 | Alignment | **Orthogonal Procrustes** | Closed form, interpretable, PRH test. |
| D012 | Required baselines | Majority class, ResNet-18 | Brief requires a simple baseline; ResNet is the strong supervised comparison. |
| D013 | Alignment baseline | Random rotation | Tests whether the rotation is meaningful. |
| D014 | Optional baseline | CLIP zero-shot | Strong external comparison if time. |
| D015 | Primary metric | Text→image top-1 retrieval | Direct test of the alignment. |
| D016 | Secondary metrics | Top-5, macro-F1, per-class, local OOD accuracy | Honest evaluation. |
| D017 | Demo | **Gradio** | Fast, recommended by the brief. |
| D018 | Audio | **Out of scope** | Not needed for Climate track; saves time. |
| D019 | Multilingual | **Out of scope** | Not needed for Climate track. |
| D020 | Three modalities | **Out of scope** | Too ambitious for 48h. |
| D021 | Encoder training | **Not allowed** | The research question is alignment without joint training. |
| D022 | Test split | 15% held out | Never fit or tune on it. |
| D023 | Local split | Separate OOD case study | Never used for fitting. |
| D024 | Submission | Code link + one-page write-up; video optional | Brief requirements. |
| D025 | Deadline buffer | Submit a complete version before the final hour | Avoid last-minute link failures. |
| D026 | Integrator | **J** | J merges to `main` and runs final evaluation. |
| D027 | Baselines/local lead | **A** | A owns ResNet-18, majority, Upper Arlington tiles, local eval. |
| D028 | Demo/docs lead | **D** | D owns Gradio, write-up, video. |
| D029 | Subagents | **J1/J2, A1/A2, D1/D2** | Max 2 active per primary; one level deep; one file scope. |
| D030 | Message board | **Not used** | Handoffs use per-task `results/Txx.md` and commit messages. |
| D031 | Git model | **Plain Git feature branches; J merges** | No GitHub CLI, no board branch, no orphan branch. |

## Canonical values

```yaml
track: AI + Climate
project: Platonic Earth
test_site: Upper Arlington, Ohio, USA
team:
  integrator: J
  baselines_local: A
  demo_docs: D
  subagents: [J1, J2, A1, A2, D1, D2]
city_center:
  lat: 40.02056
  lon: -83.06639
image_encoder: facebook/dinov2-small
image_dim: 384
text_encoder: sentence-transformers/all-MiniLM-L6-v2
text_dim: 384
alignment: orthogonal_procrustes
train_data: EuroSAT
local_tiles: 50
local_categories:
  - residential
  - commercial_industrial
  - road_highway
  - park_forest_herbaceous
  - water_river
baselines:
  - majority
  - resnet18
  - random_rotation
  - unaligned_cosine
primary_metric: top1_retrieval
local_metric: ood_accuracy
demo: gradio
```

## Revisit conditions

Revisit a decision only if:

- **D006/D007:** EuroSAT or local tiles cannot be loaded in Phase 1.
- **D009:** DINOv2 ViT-S is too weak; try ViT-B.
- **D011:** Procrustes is at chance after centering/normalization checks.
- **D012:** ResNet-18 cannot train in time; use logistic regression or majority only.
- **D017:** Gradio is blocked; use Streamlit.
- **D024:** Video is optional; cut it before cutting the write-up.
- **D029:** Subagents collide; reduce to one active subagent per primary.

## Archived alternatives

- **Platonic Picture Dictionary (AI + Languages):** not selected. See `04-picture-dictionary-spec.md` for the archived spec.
- **Platonic Synesthesia (three modalities):** not selected. Too broad for one track and 48 hours.
- **Platonic Bioacoustics:** not selected. Track fit is indirect.
- **Platonic Protein:** not selected. Too much domain complexity for 48 hours.

## Open decisions

See `13-open-questions.md`. Only remaining unknowns:

- Team size and experience.
- Exact local tile collection date.
- Whether to include CLIP baseline.
- Whether to record a video.
