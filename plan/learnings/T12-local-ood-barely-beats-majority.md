# Learning: On Upper Arlington, both models barely beat "always Residential"

**Date:** 2026-11-14
**Author:** A
**Task:** T12
**Status:** confirmed (N=50 case study)

## What we learned

Out of distribution, Procrustes (0.64) and ResNet-18 (0.64) are 1 tile better than always answering
"Residential" (0.62). That constant is an *oracle*: it is the most common local label, read from the
test labels (code review CR, 2026-11-14). The real majority baseline, the EuroSAT train majority
(AnnualCrop), scores 0/50. On EuroSAT test they score 0.81 and 0.98 per image. The specific
failures are systematic, not random:

- Water: the Scioto/Griggs channel → **Highway** (Procrustes 6/10). A thin linear feature at 10 m.
- Golf turf → ResNet **Forest, p ≈ 0.9999**. Procrustes says Pasture.
- Confidence is uninformative: Procrustes top score 0.377 when right vs 0.371 when wrong; ResNet p 0.82 when wrong.

## Evidence

`artifacts/metrics/local*.csv`, `results/T12.md` (R011, R012).

## Impact

- Write-up (D, T15): the honest headline is "works on EuroSAT, mostly does not transfer to a leafy
  US suburb at 10 m", plus the brief §6 note that judges prefer an honest modest result.
- Demo (D, T14): the river→Highway tile (ua_049) is the best failure example; ua_011 is the success.
- Do not add an "abstain below threshold" feature based on these scores; it would not work.
- Next step worth naming: higher-resolution imagery (NAIP 1 m) or local labelled tiles.

## Plan changes

- [x] `plan/11-experiment-log.md` R010–R012 and local results table.
- [ ] J/D: use per-image Procrustes 0.81 (not 0.96) when comparing with ResNet 0.98.
