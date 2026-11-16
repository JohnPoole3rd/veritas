"""Write video/src/numbers.json from the static site's API files (docs/api), which come straight from
artifacts/metrics/ and the saved study areas. The video imports this file; no number is typed by hand.

    .venv/bin/python video/scripts/make_numbers.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
API = ROOT / "docs" / "api"
meta = json.loads((API / "meta.json").read_text())
st = json.loads((API / "static.json").read_text())
M = meta["metrics"]
reg = {r["key"]: r for r in st["regions"]}
ua = json.loads((API / "region" / reg["upper-arlington"]["id"] / "climate.json").read_text())["stats"]
ph = json.loads((API / "region" / reg["phoenix"]["id"] / "climate.json").read_text())["stats"] if "phoenix" in reg else None
river = json.loads((API / "search" / "local" / "river-corridor.json").read_text())["hits"]
probe = meta.get("probe") or {}
cat = {r["local_category"]: r for r in M["local_by_category"]}
pc = lambda v, d=0: round(v * 100, d)
n = {
    "search": {"ours": pc(M["retrieval"]["procrustes"]["top1"]), "random": pc(M["retrieval"]["random_rotation"]["top1"], 1),
               "none": pc(M["retrieval"]["unaligned"]["top1"]), "queries": M["retrieval"]["procrustes"]["n_captions"],
               "tiles": M["retrieval"]["procrustes"]["n_images"]},
    "label": {"ours": pc(M["caption_classification"]["procrustes"]["accuracy"], 1), "resnet": pc(M["resnet18"]["accuracy"], 1),
              "majority": pc(M["majority"]["accuracy"], 1), "random": pc(M["caption_classification"]["random_rotation"]["accuracy"], 1),
              "n": M["caption_classification"]["procrustes"]["n_images"]},
    "local": {"ours": pc(M["local"]["procrustes"]["accuracy"]), "resnet": pc(M["local"]["resnet18"]["accuracy"]),
              "oracle": pc(M["local"]["oracle_majority_local"]["accuracy"]), "n": M["local"]["procrustes"]["n"],
              "river_right": round(cat["water_river"]["accuracy"] * cat["water_river"]["n"]), "river_n": cat["water_river"]["n"]},
    "heat": {"slope": ua["regression"]["slope_per_0p1_ndvi"], "r": ua["regression"]["r"], "n": ua["regression"]["n"],
             "tercile": ua["tercile"]["hot_minus_green_c"], "hot": ua["tercile"]["least_green_mean"],
             "green": ua["tercile"]["greenest_mean"], "lst_max": ua["lst_max"],
             "phoenix_slope": ph["regression"]["slope_per_0p1_ndvi"] if ph else None},
    "river_demo": {"right": sum(h["label"] == "River" for h in river), "k": len(river)},
    "eurosat_n": next((g["n"] for g in meta["galleries"] if g["key"] == "eurosat"), None),
    "local_n": next((g["n"] for g in meta["galleries"] if g["key"] == "local"), None),
    "probe": {"query": probe.get("query"), "top": (probe.get("hits") or [{}])[0].get("score")},
    "disclaimer": "Prototype, not a land-use determination.",
}
out = ROOT / "video" / "src" / "numbers.json"
out.write_text(json.dumps(n, indent=1))
print(json.dumps(n))

# ---------------------------------------------------------------- assets the video shows (real images)
import shutil
from datetime import datetime
PUB = ROOT / "video" / "public"
for sub in ("img", "fonts", "tiles"):
    (PUB / sub).mkdir(parents=True, exist_ok=True)
for f in ("ua-lst.png", "ua-ndvi.png", "ua-truecolor.jpg"):
    shutil.copy(ROOT / "docs" / "static" / "img" / f, PUB / "img" / f)
for f in (ROOT / "docs" / "static" / "fonts").glob("*.woff2"):
    shutil.copy(f, PUB / "fonts" / f.name)
wrong = [t for t in meta["local_tiles"] if t["correct"] is False]
river_wrong = [t for t in wrong if t["label"] == "River"]
for t in meta["local_tiles"]:
    shutil.copy(API / "tiles" / "local" / f"{t['tile_id']}.png", PUB / "tiles" / f"{t['tile_id']}.png")
for h in river:
    pass
story = json.loads((ROOT / "web" / "static" / "data" / "story.json").read_text())
d = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00")).strftime("%-d %b %Y")
n["dates"] = {"s2": d(story["ua"]["scene"]["datetime"]), "lst": d(story["ua"]["lst_scene"]["datetime"])}
n["river_tiles"] = [h["tile_id"] for h in river]
n["river_wrong"] = [{"id": t["tile_id"], "pred": t["pred"]} for t in river_wrong]
n["site_url"] = "johnpoole3rd.github.io/veritas"
n["repo_url"] = "github.com/JohnPoole3rd/veritas"

# ---------------------------------------------------------------- timeline + captions (seconds within each scene)
f1 = lambda v: f"{v:.1f}".rstrip("0").rstrip(".")
sgn = lambda v: ("−" if v < 0 else "+") + f"{abs(v):.2f}"
H, S, L, Lo = n["heat"], n["search"], n["label"], n["local"]
r0 = lambda v: f"{v:.0f}"
SCENES = [
    ("hook", 5, [(0, 5, f"In Upper Arlington, Ohio, the least green part of town was {f1(H['tercile'])} degrees hotter than the greenest part.")]),
    ("problem", 8, [(0, 4, "So where should the city plant new trees?"),
                    (4, 8, "Satellites take a new picture of every street every few days. That's too much to look through.")]),
    ("idea", 6, [(0, 6, "So we built a way to search satellite photos in plain English, without training a new AI.")]),
    ("search", 19, [(0, 3.5, "Type “river corridor”."),
                    (3.5, 9, f"All {n['river_demo']['k']} top results really are the river. We checked them against our own labels." if n['river_demo']['right'] == n['river_demo']['k']
                     else f"{n['river_demo']['right']} of the top {n['river_demo']['k']} really are the river. We checked them against our own labels."),
                    (9, 19, "Any words work, like “tall trees near a river”. The number is a match score, not a percent.")]),
    ("heat", 23, [(0, 6, "The Heat Lab adds two satellite measurements: how green, and how hot."),
                  (6, 12, f"Greener squares are cooler, by about {abs(H['slope']):.1f} degrees for every step in green."),
                  (12, 17, "“Plant first” marks hot squares with few trees."),
                  (17, 23, "Search “parking lot”, see the hottest ones, and download the list.")]),
    ("how", 19, [(0, 6, "How does it work? One AI turns photos into numbers. Another turns words into numbers."),
                 (6, 12, "We line the two up with one rotation. No training."),
                 (12, 19, f"On new photos, the right land type comes first {r0(S['ours'])}% of the time. A random rotation gets {r0(S['random'])}%.")]),
    ("baseline", 8, [(0, 8, f"A trained ResNet labels photos better: {r0(L['resnet'])}% to our {r0(L['ours'])}%. But it can't search by words.")]),
    ("limits", 15, [(0, 7, f"In our own town, it's a tie: {r0(Lo['ours'])}% right, versus {r0(Lo['oracle'])}% for always guessing “Residential”."),
                    (7, 15, f"Rivers: {'zero' if Lo['river_right'] == 0 else Lo['river_right']} out of {'ten' if Lo['river_n'] == 10 else Lo['river_n']} right. From space, a thin river looks like a road.")]),
    ("next", 7, [(0, 7, "Next: sharper photos, and an AI that can say “I don't know”.")]),
    ("end", 5, [(0, 5, "Platonic Earth. Try it at the link below.")]),
]
# If a voice-over has been cut (video/scripts/cut_voice.py), voice.json holds one piece per caption line.
# Each piece starts at its caption's beat (0.3 s in), or right after the previous piece if that runs long;
# the caption shows while its piece plays, and the scene ends 0.8 s after its last piece.
VOICE = ROOT / "video" / "public" / "audio" / "voice.json"
voice = json.loads(VOICE.read_text()) if VOICE.exists() else {}
t = 0
tl = []
for key, dur, caps in SCENES:
    v = voice.get(key)
    if isinstance(v, list) and len(v) == len(caps):
        at, prev_end = [], 0.0
        for (a, b, x), vd in zip(caps, v):
            st = round(max(a + 0.3, prev_end + 0.25), 2); at.append(st); prev_end = st + vd
        new = round(prev_end + 0.8, 2) if key != "end" else max(dur, round(prev_end + 0.8, 2))
        bounds = [max(0, s - 0.2) for s in at] + [new]
        tl.append({"key": key, "start": round(t, 2), "dur": new,
                   "voices": [{"src": f"audio/voice_{key}_{i}.wav", "at": s} for i, s in enumerate(at)],
                   "captions": [{"from": round(bounds[i], 2), "to": round(bounds[i + 1], 2), "text": x} for i, (_, _, x) in enumerate(caps)]})
    else:
        tl.append({"key": key, "start": round(t, 2), "dur": dur, "voices": [],
                   "captions": [{"from": a, "to": b, "text": x} for a, b, x in caps]})
    t += tl[-1]["dur"]
t = round(t, 2)
n["timeline"] = tl
n["total_s"] = t
out.write_text(json.dumps(n, indent=1, ensure_ascii=False))
ts = lambda s: f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{int(s % 60):02d},{int(round(s % 1 * 1000)):03d}"
srt, k = [], 1
for sc in tl:
    for c in sc["captions"]:
        srt.append(f"{k}\n{ts(sc['start'] + c['from'])} --> {ts(sc['start'] + c['to'])}\n{c['text']}\n"); k += 1
(ROOT / "video" / "captions.srt").write_text("\n".join(srt))
(ROOT / "video" / "VOICEOVER.md").write_text(
    "# Voice-over script — Platonic Earth demo\n\n"
    "**How to record:** Voice Memos or QuickTime, quiet room, phone about a hand's width from your mouth. "
    "Read slowly — slower than feels normal. Don't read the part numbers.\n\n"
    "- **Messed up? Just stop, wait 2 seconds, and start that sentence again.** No need to say anything or restart "
    "the whole thing — D keeps the last clean try of each sentence.\n"
    "- Wait about 2 seconds between parts.\n"
    "- One file for everything is fine. Send it to D when done.\n\n"
    "Numbers: say “four point one”, “one point seven”, “ninety-six percent”.\n\n"
    + "\n".join(f"**{i}. {sc['key'].capitalize()}** *(about {sc['dur']:g} s)*\n\n> " + " ".join(c["text"] for c in sc["captions"]) + "\n\n*[pause 2 s]*\n"
                for i, sc in enumerate(tl, 1))
    + f"\nVideo length now: {t:g} s. Generated by video/scripts/make_numbers.py from the logged metrics.\n")
print("total seconds:", t)
