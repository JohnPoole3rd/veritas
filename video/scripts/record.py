"""Record real screen clips of the website for the demo video (owner: D).

    .venv/bin/python video/scripts/record.py [--url http://localhost:8765/veritas/]

Serve the static copy first (python -m http.server inside a folder that holds docs/ as veritas/),
or point --url at the live server. Writes MP4s to video/public/clips/. Nothing here types a number.
"""
import argparse, asyncio, json, shutil, subprocess
from pathlib import Path
from playwright.async_api import async_playwright
import imageio_ffmpeg

OUT = Path(__file__).resolve().parents[1] / "public" / "clips"
W, H = 1600, 900


async def smooth_scroll_to(pg, sel, offset=0, steps=60, ms=16):
    y0 = await pg.evaluate("scrollY")
    y1 = await pg.evaluate("([s,o])=>document.querySelector(s).getBoundingClientRect().top+scrollY+o", [sel, offset])
    for k in range(1, steps + 1):
        t = k / steps; e = 3 * t * t - 2 * t * t * t
        await pg.evaluate(f"scrollTo(0,{y0 + (y1 - y0) * e})"); await pg.wait_for_timeout(ms)


async def slow_scroll(pg, dy, seconds):
    n = int(seconds * 30)
    for _ in range(n):
        await pg.mouse.wheel(0, dy / n); await pg.wait_for_timeout(33)


async def clip_hero(pg):
    await pg.wait_for_timeout(2500)
    await slow_scroll(pg, 2600, 9)
    await pg.wait_for_timeout(800)


MARKS, _T0 = {}, [0.0]
def mark(clip, name):
    """seconds into the finished clip at which this action happens (the clip is trimmed 0.3 s after page load)"""
    MARKS.setdefault(clip, {})[name] = round(asyncio.get_event_loop().time() - _T0[0] - 0.3, 2)


async def lab_in_view(pg):
    await pg.evaluate("scrollTo(0, document.querySelector('#lab-app').getBoundingClientRect().top + scrollY - 54)")
    await pg.evaluate("(()=>{const y=scrollY; addEventListener('scroll',()=>{ if(Math.abs(scrollY-y)>2) scrollTo(0,y); });})()")  # keep the app framed
    await pg.wait_for_timeout(5000)  # map tiles and the saved Upper Arlington pass


async def clip_search(pg):
    # Search anywhere: radius -> circle on the map -> typed query -> matches light up
    await lab_in_view(pg)
    await pg.locator('#layer-switch [data-layer="rgb"]').click(); await pg.wait_for_timeout(2500)
    mark("search", "radius"); await pg.locator("#radius-btn").click(); await pg.wait_for_timeout(1900)
    mark("search", "circle"); await pg.mouse.click(560, 470); await pg.wait_for_timeout(2200)
    mark("search", "type"); await pg.locator('[data-tab="ask"]').click(); await pg.wait_for_timeout(300)
    a = pg.locator("#ask-q"); await a.click(); await a.type("parking lots", delay=60); await a.press("Enter")
    await pg.locator("text=Best matches for").first.wait_for(timeout=20000)
    mark("search", "results"); await pg.wait_for_timeout(7000)


async def clip_lab(pg):
    await lab_in_view(pg)
    mark("lab", "ndvi"); await pg.locator('#layer-switch [data-layer="ndvi"]').click(); await pg.wait_for_timeout(2300)
    mark("lab", "lst"); await pg.locator('#layer-switch [data-layer="lst"]').click(); await pg.wait_for_timeout(3500)
    mark("lab", "heat"); await pg.locator('[data-tab="heat"]').click(); await pg.wait_for_timeout(6500)
    mark("lab", "priority"); await pg.locator('#layer-switch [data-layer="priority"]').click(); await pg.wait_for_timeout(5000)
    mark("lab", "export"); await pg.locator('[data-tab="export"]').click(); await pg.wait_for_timeout(5000)


async def clip_results(pg):
    await smooth_scroll_to(pg, "#evidence", -20, steps=2); await pg.wait_for_timeout(3500)
    await slow_scroll(pg, 700, 5); await pg.wait_for_timeout(1500)


async def clip_limits(pg):
    await smooth_scroll_to(pg, "#limits", -20, steps=2); await pg.wait_for_timeout(3000)
    await smooth_scroll_to(pg, "#wrong-grid", -200, steps=60); await pg.wait_for_timeout(4000)


async def clip_heat(pg):
    await smooth_scroll_to(pg, "#heat", 0, steps=2); await pg.wait_for_timeout(1500)
    await slow_scroll(pg, 4200, 14); await pg.wait_for_timeout(1500)


CLIPS = {"hero": clip_hero, "search": clip_search, "heat": clip_heat, "lab": clip_lab, "results": clip_results, "limits": clip_limits}


async def main(url, only, dark):
    OUT.mkdir(parents=True, exist_ok=True)
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="chrome")
        for name, fn in CLIPS.items():
            if only and name not in only:
                continue
            tmp = OUT / f"_{name}"; shutil.rmtree(tmp, ignore_errors=True)
            ctx = await b.new_context(viewport={"width": W, "height": H}, record_video_dir=str(tmp),
                                      record_video_size={"width": W, "height": H}, color_scheme="dark" if dark else "light")
            await ctx.add_init_script("window.PE_QUIET = true; window.PE_NO_PRELOAD = true;")
            # warm-up page: caches the site and the in-browser word model so the take has no loading gaps
            warm = await ctx.new_page(); await warm.goto(url, wait_until="networkidle")
            await warm.evaluate("fetch('/api/search?q=warm%20up%20query&gallery=local&k=6').then(r=>r.json())")
            await warm.close()
            t0 = asyncio.get_event_loop().time()
            pg = await ctx.new_page()
            await pg.goto(url, wait_until="networkidle")
            lead = asyncio.get_event_loop().time() - t0
            _T0[0] = asyncio.get_event_loop().time()
            await fn(pg)
            await ctx.close()
            webm = Path(await pg.video.path())
            dst = OUT / f"{name}.mp4"
            subprocess.run([ff, "-y", "-loglevel", "error", "-ss", f"{lead + 0.3:.2f}", "-i", str(webm), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "16", "-r", "30", "-vf", f"scale={W}:{H}", str(dst)], check=True)
            shutil.rmtree(tmp)
            print("[record]", dst.name)
        await b.close()
    mp = OUT.parents[1] / "src" / "marks.json"
    old = json.loads(mp.read_text()) if mp.exists() else {}
    mp.write_text(json.dumps({**old, **MARKS}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8765/veritas/")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--dark", action="store_true")
    a = ap.parse_args()
    asyncio.run(main(a.url, a.only, a.dark))
