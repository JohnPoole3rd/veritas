# Code review: combined notes (J, A, D)

**Branch:** `review/code-audit` (one branch for everyone's review notes and fixes)
**Base:** `main` @ 742ed89
**Started:** 2026-11-14 by A

How to use this file:
- One row per finding. Use the next free ID (`CR-xx`) and put your agent letter in "By".
- Severity: **bug** (wrong result or crash), **inconsistency** (code/docs/contracts disagree),
  **simplification** (dead or duplicate code).
- Fix it in the same branch when you can, then put the commit hash in "Fix". If you can't fix it,
  write `open` and say who should.
- Pull before you edit (`git pull --rebase origin review/code-audit`) and keep commits small.

## Findings

| ID | By | Severity | File | Finding | Fix |
|---|---|---|---|---|---|
| CR-01 | A | inconsistency | artifacts/metrics/retrieval_test.json, retrieval_val.{csv,json} | Still the 512-subset run (test top-1 0.94, val 0.94/0.98, N=512). `retrieval.csv` and every other metric are full split (test 0.96/1.00, val 0.98/1.00, N=4050). Mixed N in one folder. | pending (re-run J's unchanged `fit_alignment.py` + `evaluate_retrieval.py` on the full split) |
| CR-02 | A | simplification | artifacts/metrics/retrieval_test_full.json | Hand-written duplicate of the full-split test numbers; becomes redundant once CR-01 regenerates `retrieval_test.json`. | pending |
| CR-03 | A | inconsistency | artifacts/metrics/majority.json vs majority.csv | Two majority implementations break the 5-way class tie differently: `majority.json` says SeaLake (`Counter.most_common`), `majority.csv` says AnnualCrop (`class_names` order). Same accuracy, contradictory "majority class". | pending (one implementation) |
| CR-04 | A | simplification | requirements.txt | `scipy` and `tqdm` are required but never imported anywhere in src/, scripts/, app.py or tests/. | pending |
| CR-05 | A | inconsistency | scripts/evaluate_local.py | `--config` is parsed and used for validation, but `run_local_eval()` ignores it and always uses the default config/paths. | pending |

## Verification log

| When | By | Command | Result |
|---|---|---|---|
| start | A | every documented `python ...` command `--help` (scripts/*, app.py, src.baselines/local_tiles/evaluate) | all flags in README/runbooks/results exist |
| start | A | in-memory recompute of EuroSAT val/test metrics vs committed CSVs | test matches (0.96/1.00; image→class 0.8121); val files stale (CR-01) |
| start | A | fresh `git clone` (no gitignored artifacts): unittest core+local, test_baselines, `app.load_backend()` + `build_ui` | 13+4 tests pass; app falls back to MOCK with a clear missing-artifact message; results tables still render from committed CSVs |
