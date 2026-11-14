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
- Data: `procrustes.npz` `metadata` carries `eurosat_root`, `image_model`, `text_model`, `limit`; `app.py` reads it.

## Plan changes

- [x] Updated `plan/TASKS.md` (T14 acceptance now includes cached models and real-mode verification)
- [x] Updated `plan/DECISIONS.md` (no decision change; artifact contract clarified in INTERFACES)
- [x] Updated `plan/09-demo-writeup-submission.md` with runtime encoder and offline-cache instructions
- [x] Updated `plan/INTERFACES.md` with exact artifact keys and CSV columns
- [x] Updated `plan/agents/D.md` with exact commands

## Follow-up

J verified `python app.py` in real mode with N=512; `dense forest` returns Forest tiles and upload classification works.
D still needs to replace the failure-gallery placeholders after A's T12/T13.
