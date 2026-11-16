"""Cut a voice-over recording with retakes into per-caption pieces (owner: D).

    .venv/bin/python video/scripts/cut_voice.py ~/Downloads/'FINAL SCRIPT .m4a'

TAKES lists the clean take of each scene (seconds in the source; find them by transcribing the file).
Each take is sped up by TEMPO, then split at its quietest point near each caption boundary so every
caption line gets its own piece: video/public/audio/voice_<scene>_<i>.wav, and voice.json lists the
piece lengths. Then run make_numbers.py (each piece starts at its caption's beat) and render.
"""
import json, re, subprocess, wave, sys
from pathlib import Path
import numpy as np, imageio_ffmpeg
ROOT = Path(__file__).resolve().parents[2]; AUD = ROOT/"video/public/audio"; SR = 44100
src = sys.argv[1]
# caption line texts per scene, to know where to split each take
caps = {sc["key"]: [c["text"] for c in sc["captions"]] for sc in json.loads((ROOT/"video/src/numbers.json").read_text())["timeline"]}
# best clean take per scene (seconds in the source recording); lists are spliced with a short pause
TAKES = {
 "hook":    [(4.30, 9.85)],
 "problem": [(12.55, 19.42)],
 "idea":    [(26.90, 31.28)],
 "search":  [(33.25, 39.12)],
 "heat":    [(40.70, 54.52)],
 "how":     [(57.28, 70.14)],
 "test":    [(80.20, 89.58)],
 "baseline":[(99.00, 104.78)],
 "limits":  [(117.50, 127.00)],
 "next":    [(134.50, 138.22)],
 "end":     [(139.02, 141.72)],
}
raw = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-i", src, "-ac", "1", "-ar", str(SR),
      "-af", "highpass=f=80,afftdn=nf=-25", "-f", "s16le", "-"], capture_output=True, check=True).stdout
x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
TEMPO = 1.08
# first spoken words of each caption line after the first (as whisper writes them, lower case, no punctuation)
ANCHORS = {"problem": ["satellites|satellite"], "search": ["type", "the best"], "heat": ["greener|griner|greater", "plant", "download|down"],
           "how": ["we", "fit|fitted"], "test": ["about", "for parking|for"], "limits": ["and in"]}
_asr = []
def asr():
    if not _asr:
        from transformers import pipeline
        _asr.append(pipeline("automatic-speech-recognition", model="openai/whisper-base.en", device="cpu"))
    return _asr[0]
lens = {}
for k, rs in TAKES.items():
    pieces = []
    for a, b in rs:
        seg = x[int(a*SR):int(b*SR)].copy(); f = int(0.03*SR)
        seg[:f] *= np.linspace(0,1,f); seg[-f:] *= np.linspace(1,0,f)
        pieces += [seg, np.zeros(int(0.3*SR), np.float32)]
    seg = np.concatenate(pieces[:-1])
    # shorten pauses inside a take: any quiet run over 0.4 s becomes 0.3 s
    w = int(0.03*SR); db = 10*np.log10(np.convolve(seg**2, np.ones(w)/w, "same") + 1e-12)
    quiet = db < max(-50, np.percentile(db, 95) - 35)
    keep = np.ones(len(seg), bool); i = 0
    while i < len(seg):
        if quiet[i]:
            j = i
            while j < len(seg) and quiet[j]: j += 1
            if j - i > 0.4*SR: keep[i + int(0.15*SR): j - int(0.15*SR)] = False
            i = j
        else: i += 1
    seg = seg[keep]
    r = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
        "-af", f"atempo={TEMPO}", "-f", "f32le", "-"], input=seg.astype("<f4").tobytes(), capture_output=True, check=True).stdout
    seg = np.frombuffer(r, "<f4").copy()
    seg *= 10 ** (-3/20) / max(1e-6, np.abs(seg).max())
    # split points: just before the first spoken words of each later caption line (whisper word timestamps),
    # at the quietest moment between that word and the one before it
    cuts = [0]
    if ANCHORS.get(k):
        x16 = np.interp(np.arange(0, len(seg), SR/16000), np.arange(len(seg)), seg).astype(np.float32)
        words = [(c["timestamp"][0], c["timestamp"][1] or c["timestamp"][0], re.sub(r"[^a-z0-9]", "", c["text"].lower()))
                 for c in asr()({"raw": x16, "sampling_rate": 16000}, return_timestamps="word")["chunks"]]
        j = 0
        for anchor in ANCHORS[k]:
            alts = [a.split() for a in anchor.split("|")]
            while not any([w[2] for w in words[j:j+len(aw)]] == aw for aw in alts):
                j += 1
                if j >= len(words): sys.exit(f"[voice] {k}: can't find '{anchor}' in: " + " ".join(w[2] for w in words))
            lo, hi = int(words[j-1][1]*SR) if j else 0, int(words[j][0]*SR)
            win = int(0.06*SR); e = np.convolve(seg**2, np.ones(win), "same")
            cuts.append(lo + int(np.argmin(e[lo:hi])) if hi - lo > win else max(0, hi - int(0.04*SR)))
            j += 1
    cuts.append(len(seg))
    for f in AUD.glob(f"voice_{k}*.wav"): f.unlink()
    lens[k] = []
    for i, (a, b) in enumerate(zip(cuts, cuts[1:])):
        piece = seg[a:b].copy(); f = int(0.02*SR); piece[:f] *= np.linspace(0,1,f); piece[-f:] *= np.linspace(1,0,f)
        with wave.open(str(AUD/f"voice_{k}_{i}.wav"), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((piece*32767).astype("<i2").tobytes())
        lens[k].append(round(len(piece)/SR, 2))
(AUD/"voice.json").write_text(json.dumps(lens, indent=1))
print(json.dumps(lens), "total", round(sum(map(sum, lens.values())), 1))
