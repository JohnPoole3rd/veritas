# Open Questions

Track, project, and test site are locked. These are the remaining unknowns.

## 1. Resolved decisions

| Question | Decision |
|---|---|
| Track | AI + Climate |
| Project | Platonic Earth |
| Test site | Upper Arlington, Ohio |
| User | Upper Arlington sustainability / urban-forestry staff |
| Primary decision | Where to inspect for tree-canopy expansion / land-cover change |
| Training data | EuroSAT RGB |
| Local data | 50 Sentinel-2 tiles, 10 per category |
| Image encoder | DINOv2 ViT-S/14 |
| Text encoder | MiniLM-L6-v2 |
| Alignment | Orthogonal Procrustes |
| Baselines | Majority, ResNet-18, random rotation, unaligned cosine |
| Demo | Gradio |
| Audio | Out of scope |
| Multilingual | Out of scope |
| Deadline | Sunday, Nov 15, 11:59 PM ET |

## 2. Remaining unknowns

| Question | Options | Default | Decision |
|---|---|---|---|
| Team size? | 1 / 2 / 3 | 3 | |
| Experience level? | Beginner / Intermediate / Advanced | Intermediate | |
| Who owns data? | name | Agent A | |
| Who owns alignment? | name | Agent A | |
| Who owns baselines/local? | name | Agent B | |
| Who owns demo/write-up? | name | Agent C | |
| Exact local imagery date? | date | most recent cloud-free | |
| Include CLIP baseline? | yes / no | optional | |
| Record video? | yes / no | recommended | |
| Contact a real Upper Arlington user? | yes / no | optional | |
| Use 50 or 30 local tiles? | 50 / 30 | 50 | |

## 3. Team roles

| Role | Responsibilities | Default owner |
|---|---|---|
| Data lead | EuroSAT, splits, captions, local tiles | Agent A |
| Alignment lead | DINOv2, MiniLM, Procrustes, retrieval | Agent A |
| Evaluation lead | ResNet-18, majority, metrics, failure analysis | Agent B |
| Demo lead | Gradio, visuals, video | Agent C |
| Writer | Write-up, credits, submission | Human + Agent C |

## 4. Local tile collection decisions

| Question | Default |
|---|---|
| Boundary source | OpenStreetMap or Census TIGER/Line |
| Imagery source | Copernicus Browser, Sentinel-2 L2A |
| Season | Late spring / summer |
| Cloud cover | As low as possible |
| Tile size | 64×64 at 10 m (640 m × 640 m) |
| Overlap | Avoid where possible |
| Labels per category | 10 |
| `Other` allowed? | Yes, with note; replace if possible |
| QA | Second person reviews 10 tiles |

## 5. Submission decisions

| Question | Default |
|---|---|
| Code platform | GitHub or Colab |
| Write-up platform | Google Doc or GitHub README |
| Video platform | YouTube unlisted |
| Submit first version | T+45 |
| Resubmit | Only for link fixes or final metrics |

## 6. Defaults if no answers

Proceed with:

- 3-person roles as assigned in `TASKS.md`.
- 50 local tiles.
- Most recent cloud-free summer imagery.
- CLIP optional; skip if behind.
- Video recommended but optional.
- No real user contact required; design for the role.
- Submit at T+45.

## 7. Decisions that cannot wait

1. Team roles — by T+2.
2. EuroSAT data loaded — by T+2.
3. Local boundary source — by T+12.
4. Local tile date — by T+12.
5. Whether Procrustes works — by T+10; if not, pivot.
6. Whether to record video — by T+42.

## 8. Questions for the human

- How many people are on the team, and what is their experience level?
- Do you want to assign the three agent roles, or should the plan default?
- Do you have a preferred date/season for the Upper Arlington tiles?
- Do you want to include the CLIP baseline?
- Do you want to try to contact a real Upper Arlington sustainability/forestry staff member for feedback?
