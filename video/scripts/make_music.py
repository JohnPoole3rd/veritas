"""Quiet ambient background music for the demo video, synthesised here (no licence needed; owner: D).

    .venv/bin/python video/scripts/make_music.py [seconds]

Soft pad chords (Cmaj9 – Am9 – Fmaj7 – G6sus), slow swell, gentle high shimmer. Written to video/public/audio/music.wav.
"""
import sys, wave
from pathlib import Path
import numpy as np

SR = 44100
secs = float(sys.argv[1]) if len(sys.argv) > 1 else 125
OUT = Path(__file__).resolve().parents[1] / "public" / "audio" / "music.wav"
hz = lambda m: 440 * 2 ** ((m - 69) / 12)
CHORDS = [[48, 55, 64, 71, 74], [45, 52, 60, 67, 71], [41, 48, 57, 64, 69], [43, 50, 59, 62, 69]]
BAR = 8.0
n = int(secs * SR)
t = np.arange(n) / SR
out = np.zeros((n, 2))
rng = np.random.default_rng(7)
for k in range(int(np.ceil(secs / BAR)) + 1):
    t0 = k * BAR - 1.5  # overlap for smooth crossfades
    i0, i1 = max(0, int(t0 * SR)), min(n, int((t0 + BAR + 3) * SR))
    if i0 >= i1:
        continue
    tt = t[i0:i1] - t0
    env = np.clip(tt / 2.5, 0, 1) * np.clip((BAR + 3 - tt) / 2.5, 0, 1)
    env = env ** 1.5
    for j, m in enumerate(CHORDS[k % 4]):
        f = hz(m)
        for det, pan in ((-0.12, 0.3), (0.12, 0.7)):
            ph = rng.uniform(0, 2 * np.pi)
            w = np.sin(2 * np.pi * (f + det) * tt + ph) + 0.18 * np.sin(2 * np.pi * 2 * (f + det) * tt + ph)
            g = (0.5 if j == 0 else 0.32) * env
            out[i0:i1, 0] += w * g * (1 - pan)
            out[i0:i1, 1] += w * g * pan
# shimmer: slow high notes from the chord, every 2 s
for b in np.arange(0.5, secs, 2.0):
    k = int(b // BAR) % 4
    f = hz(CHORDS[k][rng.integers(2, 5)] + 12)
    i0, i1 = int(b * SR), min(n, int((b + 3) * SR))
    tt = t[i0:i1] - b
    w = np.sin(2 * np.pi * f * tt) * np.exp(-tt * 1.4) * np.clip(tt / 0.05, 0, 1) * 0.12
    pan = rng.uniform(0.25, 0.75)
    out[i0:i1, 0] += w * (1 - pan); out[i0:i1, 1] += w * pan
# soften: one-pole low-pass, overall swell in/out, normalise to -14 dBFS peak
a = np.exp(-2 * np.pi * 2200 / SR)
for c in range(2):
    y = out[:, c]; z = np.empty_like(y); s = 0.0
    for i in range(0, n, 1):
        s = a * s + (1 - a) * y[i]; z[i] = s
    out[:, c] = z
out *= (np.clip(t / 3, 0, 1) * np.clip((secs - t) / 4, 0, 1))[:, None]
out *= 10 ** (-14 / 20) / np.abs(out).max()
OUT.parent.mkdir(parents=True, exist_ok=True)
with wave.open(str(OUT), "wb") as f:
    f.setnchannels(2); f.setsampwidth(2); f.setframerate(SR)
    f.writeframes((out * 32767).astype("<i2").tobytes())
print(f"[music] {OUT} {secs:.0f} s")
