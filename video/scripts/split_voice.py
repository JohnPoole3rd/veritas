"""Split one voice-over take into the video's scenes (owner: D).

    .venv/bin/python video/scripts/split_voice.py ~/Desktop/voice.m4a [--gap 1.2]

Finds the silent gaps (>= --gap seconds) between the parts of VOICEOVER.md, expects one part per scene,
trims and normalises each part, and writes video/public/audio/voice_<scene>.wav plus voice.json (lengths).
Then run make_numbers.py and render. If the count of parts is wrong it prints the parts it found.
"""
import argparse, json, subprocess, sys, wave
from pathlib import Path
import numpy as np
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[2]
AUD = ROOT / "video" / "public" / "audio"
SR = 44100
ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("--gap", type=float, default=1.2)
ap.add_argument("--thresh", type=float, default=-40, help="silence level, dBFS")
a = ap.parse_args()
keys = [sc["key"] for sc in json.loads((ROOT / "video" / "src" / "numbers.json").read_text())["timeline"]]
raw = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-i", a.src, "-ac", "1", "-ar", str(SR),
                      "-af", "highpass=f=80,afftdn=nf=-25", "-f", "s16le", "-"], capture_output=True, check=True).stdout
x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
win = int(0.05 * SR)
rms = np.sqrt(np.convolve(x ** 2, np.ones(win) / win, "same") + 1e-12)
loud = 20 * np.log10(rms) > a.thresh
# speech regions, merged across gaps shorter than --gap
idx = np.flatnonzero(loud)
if not idx.size: sys.exit("No speech found. Is the file silent?")
parts, s0, prev = [], idx[0], idx[0]
for i in idx[1:]:
    if i - prev > a.gap * SR:
        parts.append((s0, prev)); s0 = i
    prev = i
parts.append((s0, prev))
parts = [(s, e) for s, e in parts if e - s > 0.4 * SR]  # drop clicks / breaths
print(f"[voice] found {len(parts)} parts for {len(keys)} scenes: " + ", ".join(f"{(e - s) / SR:.1f}s" for s, e in parts))
if len(parts) != len(keys):
    sys.exit("Part count doesn't match. Try --gap 0.9 (more parts) or --gap 1.6 (fewer), or re-record with clearer pauses.")
AUD.mkdir(parents=True, exist_ok=True)
lens = {}
for k, (s, e) in zip(keys, parts):
    s, e = max(0, s - int(0.12 * SR)), min(len(x), e + int(0.25 * SR))
    seg = x[s:e].copy()
    seg *= 10 ** (-3 / 20) / max(1e-6, np.abs(seg).max())     # peak -3 dBFS
    fade = int(0.03 * SR); seg[:fade] *= np.linspace(0, 1, fade); seg[-fade:] *= np.linspace(1, 0, fade)
    with wave.open(str(AUD / f"voice_{k}.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((seg * 32767).astype("<i2").tobytes())
    lens[k] = round(len(seg) / SR, 2)
(AUD / "voice.json").write_text(json.dumps(lens, indent=1))
print("[voice] wrote", AUD / "voice.json", lens)
