# Learning: The demo needs both encoders at runtime, plus J's exact transforms

**Date:** 2026-11-14
**Author:** D
**Task:** T14
**Status:** confirmed

## What we learned

"Precompute everything" (plan/09 §1) covers the gallery but not the inputs. A typed query is new text, so it
must go through MiniLM at demo time, and an uploaded tile must go through DINOv2. Also, the demo's
center/normalize/rotate order has to match T06's exactly, or scores will be quietly wrong.

## Evidence

- `app.py` `RealBackend` lazy-loads `SentenceTransformer` for queries and `AutoModel` (DINOv2, CLS token) for uploads.
- Tested against synthetic artifacts that follow INTERFACES.md §4: retrieval and classification come back correct.

## Impact

- Task: T01 `requirements.txt` needs `sentence-transformers`, `transformers`, `torch`, `gradio`, `numpy`, `pillow`.
- Demo: both models must already be in the local HF cache before demo day (the "loads without internet" acceptance).
  Pre-warm with one `python -c` call per model on the demo machine.
- Model: `app.py` imports `transform_images` / `transform_texts` from `src.alignment` (J). Its fallback assumes
  normalize → subtract train mean → rotate → normalize (plan/03 §7). If J's convention differs, the fallback is wrong,
  but it is only used while `src/` is missing.
- Model: the upload path takes the CLS token from `last_hidden_state[:, 0]`. T04 must use the same pooling.
- Data: `procrustes.npz` `config` may carry `eurosat_root`, `image_model`, `text_model`; the demo reads those if present.

## Plan changes

- [ ] Updated `plan/TASKS.md` (J: consider adding "models cached locally" to T14 acceptance)
- [ ] Updated `plan/DECISIONS.md` (J only)
- [ ] Updated runbook
- [ ] Updated data/model/eval plan
- [x] No plan change needed by D; this is a note for J's T01/T04/T06

## Follow-up

When J merges T06, run `python app.py`, confirm `[demo] REAL mode … transforms from src.alignment`, and check that
a `dense forest` query returns Forest tiles.
