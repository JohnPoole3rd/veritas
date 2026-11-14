# Risks and Fallbacks

## 1. Risk register

| ID | Risk | Likelihood | Impact | Mitigation | Owner |
|---|---|---|---|---|---|
| R1 | EuroSAT download fails | Low | High | HF mirror/TFDS; switch after 30 min | Agent A |
| R2 | DINOv2 poor on 64×64 tiles | Medium | High | Try ViT-B; patch mean; report limit | Agent A |
| R3 | Procrustes near chance | Medium | High | Check centering/normalization; random baseline; pivot to classifier demo | Agent A |
| R4 | Captions too simple | Medium | Medium | Add paraphrases; class centroids | Agent A |
| R5 | ResNet-18 beats Procrustes | High | Medium | Expected; frame as different tasks; report honestly | Agent B |
| R6 | Upper Arlington tiles hard to label | High | Medium | Use `Other`; replace; 5 categories only | Agent C |
| R7 | Local tiles outside city boundary | Medium | Medium | Use OSM/Census polygon; clip | Agent C |
| R8 | Clouds/snow in local tiles | Medium | Medium | Pick cloud-free date; note if unavoidable | Agent C |
| R9 | Colab disconnects | Medium | High | Cache embeddings to Drive; 5k subset | All |
| R10 | Time runs out | High | High | Scope cuts; submit at T+45 | Human |
| R11 | Demo breaks on video | Medium | High | Precompute; incognito test; screen capture fallback | Agent C |
| R12 | Results look too good | Low | High | Check leakage/deduplication | Agent B |
| R13 | Project judged as generic classifier | Medium | High | Emphasize text search, PRH, local tiles | Writer |
| R14 | Missing citations | Medium | Medium | Maintain credits table | Writer |
| R15 | Canopy overclaim | Medium | High | Say “land-cover triage,” not canopy measurement | Writer |
| R16 | Teammate unavailable | Medium | Medium | Cross-train; reduce scope | Human |

## 2. Kill criteria

Stop and pivot if:

- Data has not loaded by T+2.
- Embeddings are not extracted by T+6.
- No working demo by T+30.
- Write-up not started by T+40.
- Team cannot explain the rotation.

Pivot options:

1. **Classifier-only demo:** ResNet-18 land-use classifier + Upper Arlington tiles; Procrustes as ablation.
2. **Coarse text search:** only forest / water / residential / commercial.
3. **Simpler suggested project:** if the team is Beginner.

## 3. Fallback branches

### Fallback A — Procrustes works

- Full demo: text search + image upload.
- Main result: Procrustes beats random.
- Failure: annual vs. permanent crop.
- Stretch: CLIP baseline.

### Fallback B — Procrustes is weak

- Demo: coarse categories only.
- Main result: “PRH holds for coarse categories, not fine-grained ones.”
- Compare with ResNet-18.
- Use as the research finding.

### Fallback C — DINOv2 fails

- Use ResNet-18 as the main demo.
- Procrustes as an ablation with a smaller model.
- Frame: “we tested whether PRH can replace supervised classification; it cannot at this resolution.”

### Fallback D — Time is nearly gone

- Submit:
  - EuroSAT classifier,
  - local tile test,
  - one failure,
  - write-up.
- Drop Procrustes if it is not working.
- Do not submit a broken demo.

### Fallback E — Local tiles fail

- Use 20–30 tiles; report N.
- Use nearby Franklin County tiles, clearly labeled.
- Treat local as qualitative.
- Do not fabricate.

## 4. What to say if results are weak

Avoid:

- “The model is 42% accurate, but it is still good.”
- “It works well” without numbers.
- “The AI understands land use.”

Say:

- “Our top-1 retrieval was X on held-out tiles.”
- “The random-rotation baseline was Y, so the alignment [did/did not] add value.”
- “ResNet-18 reached Z because it is supervised; our method supports text queries it cannot.”
- “This is a prototype and not a land-use determination.”

## 5. Scope-creep rules

Do not add:

- audio,
- multilingual,
- a third modality,
- a custom neural network,
- a large language model API unless it replaces a failed component,
- real GIS layers,
- user accounts,
- a database,
- a mobile app,
- parcel-level canopy measurement.

Add only after MVP:

- CLIP baseline,
- map view,
- confidence threshold,
- failure gallery.

## 6. Data leakage checklist

- [ ] Split before fitting.
- [ ] Deduplicate tiles.
- [ ] No test tiles in rotation fit.
- [ ] No test tiles in caption selection.
- [ ] No local tiles in training.
- [ ] No tuning on test.
- [ ] Report split sizes.
- [ ] Check no tile in train and test.

## 7. Demo failure checklist

- [ ] Embeddings saved.
- [ ] \(W\) saved.
- [ ] Caption embeddings saved.
- [ ] App loads without internet.
- [ ] Example queries work.
- [ ] Upload works.
- [ ] Failure example works.
- [ ] Screenshot fallback.
- [ ] Video recorded before deadline.

## 8. Team contingency

If a teammate drops:

- Reduce to one modality pair.
- Use 5k subset.
- Skip CLIP.
- Skip map.
- Keep code, write-up, one failure.

If the team disagrees:

- Default to Platonic Earth.
- Reversible until T+6.
- After T+6, do not switch.

## 9. Final rule

A complete, honest, narrow project beats an incomplete, ambitious one.
