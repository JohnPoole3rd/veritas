#!/usr/bin/env bash
# Rebuild everything D publishes from the current logged metrics (owner: D).
#   scripts/d_refresh.sh            # site data, public copy, README/WRITEUP numbers, tests
#   scripts/d_refresh.sh --video    # ...plus re-record the site and re-render the demo video into ~/Movies
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY -m web.bake_story
$PY -m web.build_static
$PY scripts/sync_docs.py
$PY -m pytest -q tests
if [[ "${1:-}" == "--video" ]]; then
  PAGES=$(mktemp -d); ln -s "$PWD/docs" "$PAGES/veritas"
  (cd "$PAGES" && python3 -m http.server 8765 >/dev/null 2>&1 & echo $! > "$PAGES/pid")
  sleep 1
  $PY video/scripts/make_numbers.py
  [[ -f video/public/audio/music.wav ]] || $PY video/scripts/make_music.py 125
  $PY video/scripts/record.py --url http://localhost:8765/veritas/
  kill "$(cat "$PAGES/pid")" || true
  (cd video && npx remotion render PlatonicEarth "$HOME/Movies/PlatonicEarth_demo.mp4" --codec=h264 --crf=18)
  $PY scripts/readme_media.py
fi
echo "[refresh] done. Check: git diff --stat; then QA the site (desktop, mobile, dark)."
