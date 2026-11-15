# T20 — OSM-Sentinel 1M run: status and decision needed from J

> **Superseded:** this is the old 128 px / 1.28 km run. The authoritative scale-matched run is
> `results/T21-osm-1m-640.md` (64 px / 640 m, fresh `data/raw/osm_sentinel_1m_640`).

Status: **RUNNING. Awaiting J's call** (A is not changing anything until J decides).
From: A, 2026-11-15 15:25 ET. For: J (cc D)
Code: `j/exp-osm-sentinel` @ `b7e2e8e` (your authoritative 1M runbook), run from a separate worktree.
Machine: Apple M1 Pro, 16 GB, MPS. Deadline: **today 23:59 ET** (8.5 h from this note).

## What is running (exactly per RUNBOOK "A: full 1M rerun")

```sh
python experiments/osm_sentinel/build_dataset.py \
  --out-dir data/raw/osm_sentinel_1m --sample-mode grid \
  --chips-per-region 8000 --max-chips 1000000 2>&1 | tee data/raw/osm_sentinel_1m/_run.log
```

- Fresh output dir and caches; nothing is reused from the 77k build. Grid mode, WorldCover on.
- All 37 branch tests pass. MPS = True; pyrosm 0.29.0; rasterio 1.5.2; Python 3.14.5.
- Started 15:12 ET. At 15:25: **2,582 chips, 14 regions, no failures.** Captions look right,
  e.g. `Residential area with …` for us_arizona.
- The old 77k build (old captioner) is still finishing alongside it: 22,224 chips, done ≈ 18:15.
  Its queued final runs follow automatically (train, compare, your `recaption.py`) for T19.

## The problem: 1M cannot finish by the deadline on this machine

- Measured rate is **~2.4–3.3 chips/s** while sharing the network with the 77k build. Maybe ~5/s
  after 18:15.
- 1M chips ⇒ **~2.5–4 days**, plus ~2 h to embed 1M images on MPS.
- By 23:59 the build will reach roughly **100k–150k chips**. The first 199 hand-picked regions
  (~77k grid cells) come first; the 200 × 60 km city regions come after.

## Already scheduled (user-approved): interim test before the deadline

At **19:30 ET** a script freezes a snapshot of the 1M manifest (complete rows; expect ~60k chips,
all three splits) into `data/raw/osm_sentinel_1m_interim`, then runs:

```sh
python experiments/osm_sentinel/train.py --data data/raw/osm_sentinel_1m_interim \
  --device mps --dtype fp32 --batch-size 32 --num-workers 4 --eval-eurosat
python experiments/osm_sentinel/compare_eurosat.py --data data/raw/osm_sentinel_1m_interim \
  --tag 1m_interim --dtype fp32 --batch-size 64 --num-workers 0
```

It runs with its own artifact dirs, so it cannot clobber the 77k results. Expected results by
~20:30–21:00 ET, leaving ~3 h to write this report and for you/D to use the numbers.

## Decision needed from J (pick one; A will execute and finish by 23:59)

| Option | What A does | Done by deadline? |
|---|---|---|
| **A. Interim only (current plan)** | Keep the 1M build running untouched; report the 19:30 snapshot numbers as "interim, n≈60k of 1M" in this file by ~21:00. The full 1M continues after the deadline (post-submission). | Yes (interim) |
| B. Later snapshot | Same as A but snapshot at ~21:30 (≈90–110k chips); results ~22:45. Tighter, little slack if anything fails. | Yes, barely |
| C. Stop at a smaller cap | Stop the build at whatever exists ~19:30 and treat that as the run (no 1M claim). | Yes |
| D. Pause the 77k leftovers | Kill the 77k build's remaining work (T19 keeps its 10k preliminary) to give the 1M build all bandwidth (~+50% rate). Combine with A or B. | Yes |
| E. Parallel shards | Not recommended tonight: it changes your single-command design, needs a manifest merge, and is RAM-risky on 16 GB with the 60 km city regions; still ~12–20 h. | No |

**A recommends A** (optionally with D at 18:00 if you want a bigger snapshot): it gives real 1M-pipeline
numbers before the deadline without changing your runbook. Nothing will be submitted as "1M" unless
n actually reaches 1M.

## Also for J

- `run_final` for the 77k build uses your `recaption.py` (output `data/raw/osm_sentinel_recaption`)
  and writes the T19 final tables. T19 stays the historical 77k report, as your runbook says.
- When the full 1M build eventually finishes (days), it still needs `train.py` + `compare_eurosat.py`
  on `data/raw/osm_sentinel_1m` and this report's final tables. That part is not scheduled.
