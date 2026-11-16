# Demo video (Remotion)

A 2-minute video with burned-in captions, built from real screen recordings of the public site. Every number comes from `src/numbers.json`, which `scripts/make_numbers.py` writes from the logged metrics.

```sh
# from the repo root, with the public copy served at http://localhost:8765/veritas/
.venv/bin/python video/scripts/make_numbers.py     # numbers, timeline, captions.srt, VOICEOVER.md, images
.venv/bin/python video/scripts/record.py           # real clips → video/public/clips/
cd video && npm i && npx remotion studio           # preview
npx remotion render PlatonicEarth ~/Movies/PlatonicEarth_demo.mp4
```

Or run all of it with `scripts/d_refresh.sh --video`. Upload `captions.srt` to YouTube as the English subtitles. `VOICEOVER.md` is a script to read aloud if you add a voice.
