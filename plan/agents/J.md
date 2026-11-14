# J — Integrator and Data/Alignment Lead

Read `plan/AGENTS.md`, `plan/DECISIONS.md`, and `plan/INTERFACES.md` first.

## Owns

- `configs/`
- `src/config.py`
- `src/data.py`
- `src/captions.py`
- `src/encoders.py`
- `src/alignment.py`
- `src/retrieval.py`
- `scripts/prepare_data.py`
- `scripts/extract_embeddings.py`
- `scripts/fit_alignment.py`
- `scripts/evaluate_retrieval.py`
- Notebooks 01–03
- Integration and final merge

## Tasks

| Task | Command / action | Report |
|---|---|---|
| T01 scaffold | `python -m compileall -q src scripts`; `python -m unittest discover -s tests -v` | `results/T01.md` |
| T02 EuroSAT | `python scripts/prepare_data.py` | `results/T02.md` |
| T03 captions | same as T02 | `results/T03.md` |
| T04 DINOv2 | CPU smoke: `python scripts/extract_embeddings.py --limit 512`; GPU: no `--limit` | `results/T04.md` |
| T05 MiniLM | `python scripts/fit_alignment.py --limit 512` | `results/T05.md` |
| T06 Procrustes | same as T05 | `results/T06.md` |
| T07 retrieval | `python scripts/evaluate_retrieval.py --limit 512` | `results/T07.md` |
| Majority | `python scripts/evaluate_majority.py` | A owns `results/T09.md` |

## Current status

- T01–T07 are done on a **512-tile stratified subset**.
- Full 27k extraction was stopped on CPU because it was too slow. Use a GPU for the full run.
- D's `app.py` RealBackend is verified against J's artifacts with N=512.

## Integration

```sh
git fetch --all
git switch main
git pull --rebase origin main
git merge --no-ff <agent-branch> -m "Merge <task>"
git push origin main
```

Only J merges or pushes `main`.

## App integration check

```sh
python -c 'import sys, types; sys.modules["gradio"] = types.ModuleType("gradio"); from app import load_backend; b=load_backend(); print(b.name, len(b.items), b.subset_note)'
```

Expected with current artifacts:

```text
[demo] REAL mode — gallery N=512
backend REAL 512 512-tile stratified subset
```

## Do not

- Do not edit A's local tile manifest or data.
- Do not edit D's write-up or video script.
- Do not change `app.py` unless fixing an integration bug; record the fix in `results/T14.md`.
- Do not claim full-split results when N=512.
- Do not hide a failed merge or test.

## Current blocker

A's T08/T09/T10–T13 are still pending. D's T14 real backend is verified; failure gallery and local examples wait on A.
