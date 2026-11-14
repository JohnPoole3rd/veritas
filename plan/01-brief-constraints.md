# Brief Constraints and Compliance

This file distills the Veritas AI International High School Hackathon 2026 brief for the selected project.

## Selected project

- **Track:** AI + Climate
- **Project:** Platonic Earth
- **Test site:** Upper Arlington, Ohio
- **Deadline:** Sunday, November 15, 2026, 11:59 PM ET

## Dates and format

- 48-hour virtual hackathon.
- Teams of 1–3 high school students.
- One project per team, one track per project.
- Resubmission allowed; judges see the latest version.
- No kickoff call or check-in.

## Climate track requirements

The brief’s AI + Climate challenge:

> Help a community, city, or energy system measure, predict, or reduce its climate impact.

What the project needs:

- Focus on one location or region.
- For forecasts, train on earlier dates and test on later ones.
- Compare with a simple baseline.
- Explain what decision the prediction supports and who makes it.

**How Platonic Earth satisfies this:**

| Requirement | How the project satisfies it |
|---|---|
| One location | Upper Arlington, Ohio |
| Measure/act on climate | Land-cover triage for tree-canopy expansion and inspection |
| Specific decision | Which areas to inspect first |
| Specific user | City sustainability / urban-forestry staff |
| Simple baseline | Majority class and ResNet-18 |
| Honest evaluation | Held-out EuroSAT test + 50 local tiles |
| Failure case | Annual vs. permanent crop; local domain shift |

The project is not a forecast, so the time-split rule does not apply. It is a retrieval/classification tool.

## Project level

**Intermediate / Advanced.**

The project uses:

- PyTorch / torchvision
- Hugging Face Transformers
- Sentence Transformers
- SciPy
- scikit-learn
- Gradio
- Google Colab or a GPU

If the team is Beginner, simplify to a supervised ResNet-18 land-use classifier and use Procrustes as an ablation.

## Rules

1. **Teams:** 1–3 high school students; everyone listed must contribute.
2. **Build during the weekend:** no extending a pre-built project. Learning tools and browsing datasets beforehand is allowed.
3. **AI must be central:** the Procrustes alignment and retrieval are the core.
4. **Allowed:** public datasets, pretrained models, AI APIs, open-source libraries, tutorial code, coding assistants. Cite everything.
5. **Not allowed:** someone else’s project; personal data without permission; private records.
6. **One project, one track.**
7. **Disclaimers:** health/finance only; not applicable here, but include “prototype, not a land-use determination.”
8. **Honest results:** report weak results too.

## What every project must show

1. **Who is it for?**
   - Upper Arlington sustainability / urban-forestry staff.

2. **What goes in, what comes out?**
   - Text query or uploaded tile → top-3 matching tiles with labels and scores.

3. **Does it beat a simple baseline?**
   - Compare Procrustes retrieval with random rotation, majority class, and ResNet-18.

4. **Where does it fail?**
   - Annual vs. permanent crop; local domain shift; abstract queries; clouds/season.

## “Make it your own”

The brief requires at least one addition:

- **Own user:** Upper Arlington sustainability / urban-forestry staff.
- **Own data:** 50 local Sentinel-2 tiles clipped to the city boundary.
- **Own question:** Does PRH support text-to-satellite land-cover search without multimodal training?

All three are present.

## Suggested schedule

| By | Where the team should be |
|---|---|
| Saturday, first 2 hours | Track and idea picked; data loading |
| Saturday evening | Baseline and first model work |
| Sunday midday | One measured improvement; demo runs end-to-end |
| Sunday, 2 hours before deadline | Write-up done, video uploaded, submitted |

## Submission requirements

| Item | Required? | Details |
|---|---|---|
| Code | Yes | Colab notebook or GitHub repo, viewable by link |
| Write-up | Yes | About one page, brief template |
| Demo video | Optional but recommended | ≤2 minutes, unlisted |
| Team details | Yes | Track, names, emails, schools |
| Credits | Yes | Datasets, models, tutorials, AI tools |

## Write-up template

- Project name
- Track
- Team
- The problem
- What we built
- What we added
- How it works
- Results (baseline vs. model, failure)
- What we’d do next
- Sources

## Demo video structure

| Part | Length | Content |
|---|---|---|
| Problem | ~15 seconds | User and why it matters |
| Demo | ~60 seconds | Real input → output |
| How it works | ~30 seconds | Data, model, baseline |
| Next steps | ~15 seconds | One limitation and next step |

## Judging criteria

| Criterion | What judges look for | How Platonic Earth shows it |
|---|---|---|
| Technical quality | Works, sound, fairly tested | Baselines, held-out test, failure analysis |
| Impact | Real problem for real people | Upper Arlington sustainability/forestry user |
| Creativity | Fresh approach | PRH + closed-form rotation + local tiles |
| Clarity | Understandable in two minutes | Text query → tiles demo |

## Compliance checklist

- [ ] Track is AI + Climate.
- [ ] One specific user is named.
- [ ] Input and output are shown with a real example.
- [ ] Majority baseline reported.
- [ ] ResNet-18 baseline reported.
- [ ] Random rotation baseline reported.
- [ ] Results are on held-out test data.
- [ ] At least one failure case shown and explained.
- [ ] Local Upper Arlington tiles used.
- [ ] AI is central.
- [ ] All datasets and models cited.
- [ ] Code link opens without signing in.
- [ ] Write-up link opens without signing in.
- [ ] No personal data collected.
- [ ] No prior project extended.
- [ ] Results are honest, including weak ones.
- [ ] “Prototype, not a land-use determination” included.
