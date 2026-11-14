# Open Questions

Track, project, test site, and roles are locked. These are the remaining unknowns.

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
| Message board | Not used |
| Git workflow | Plain Git feature branches; J merges to `main` |
| Integrator | J |
| Data/alignment lead | J |
| Baselines/local lead | A |
| Demo/docs lead | D |
| Subagents | J1/J2, A1/A2, D1/D2 |
| Deadline | Sunday, Nov 15, 11:59 PM ET |

## 2. Remaining unknowns

| Question | Options | Default | Decision |
|---|---|---|---|
| Exact local imagery date? | date | most recent cloud-free | |
| Include CLIP baseline? | yes / no | optional | |
| Record video? | yes / no | recommended | |
| Contact a real Upper Arlington user? | yes / no | optional | |
| Use 50 or 30 local tiles? | 50 / 30 | 50 | |
| Start signal? | human tells agents | wait | |

## 3. Roles

| Role | Responsibilities | Default owner |
|---|---|---|
| Integrator | Merge branches, final eval, submission | J |
| Data/alignment | EuroSAT, splits, captions, DINOv2, MiniLM, Procrustes, retrieval | J |
| Baselines | ResNet-18, majority, metrics | A |
| Local data | Upper Arlington tiles, labels, QA | A |
| Demo | Gradio, visuals, video | D |
| Writer | Write-up, credits, video | D |
| Submission | Final checklist and form | J |

## 4. Subagents

| Primary | Subagents | Suggested work |
|---|---|---|
| J | J1, J2, J3, … | J1 EuroSAT loader; J2 embedding extraction |
| A | A1, A2, A3, … | A1 ResNet-18; A2 local tile collection |
| D | D1, D2, D3, … | D1 Gradio UI; D2 visuals/video |

Rules:

- No fixed cap: create as many as they want.
- Prefer one level of delegation. If a subagent needs more help, it reports to its parent, and the parent creates another subagent.
- One bounded task and file scope per subagent.
- Subagent cannot merge to `main`.
- Parent writes `results/Txx.md`.
- Parent is accountable for every subagent.

## 5. Local tile collection decisions

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

## 6. Submission decisions

| Question | Default |
|---|---|
| Code platform | GitHub or Colab |
| Write-up platform | Google Doc or GitHub README |
| Video platform | YouTube unlisted |
| Submit buffer | Submit before the final hour |
| Resubmit | Only for link fixes or final metrics |

## 7. Defaults if no answers

Proceed with:

- J/A/D roles as assigned.
- 50 local tiles.
- Most recent cloud-free summer imagery.
- CLIP optional; skip if behind.
- Video recommended but optional.
- No real user contact required; design for the role.
- Submit before the final hour.

## 8. Continuous planning

- Update the relevant plan file when a finding changes future work and is local to your task.
- Create `plan/learnings/Txx-topic.md` for important findings.
- Use `plan/learnings/_template.md`.
- Any agent may create a learning file; do not edit another agent's.
- J owns `plan/TASKS.md` and `plan/DECISIONS.md`.
- If it changes a locked decision, write the learning and ask J/human.
- J reviews and merges learnings with the task branch.

## 9. Questions for the human

- When should everyone start?
- Do you want to include the CLIP baseline?
- Do you want to try to contact a real Upper Arlington sustainability/forestry staff member?
- Do you want a video, or code + write-up only?
