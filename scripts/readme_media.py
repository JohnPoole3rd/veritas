"""README images (owner: D): hero + Lab screenshots from the static copy, and a short GIF from the search clip.

    .venv/bin/python scripts/readme_media.py [--url http://localhost:8765/veritas/]
"""
import argparse, asyncio, subprocess
from pathlib import Path
import imageio_ffmpeg
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "readme"


async def shots(url):
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="chrome")
        pg = await b.new_page(viewport={"width": 1440, "height": 810}, device_scale_factor=1)
        await pg.add_init_script("window.PE_QUIET = true; window.PE_NO_PRELOAD = true;")
        await pg.goto(url, wait_until="networkidle"); await pg.wait_for_timeout(2500)
        await pg.screenshot(path=str(OUT / "hero.png"))
        await pg.locator("#lab-app").scroll_into_view_if_needed(); await pg.wait_for_timeout(7000)
        await pg.locator('[data-tab="heat"]').click(); await pg.wait_for_timeout(800)
        await pg.locator('#layer-switch [data-layer="lst"]').click(); await pg.wait_for_timeout(2500)
        await pg.locator("#lab-app").screenshot(path=str(OUT / "lab.png"))
        await b.close()


def gif():
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    clip = ROOT / "video" / "public" / "clips" / "search.mp4"
    if not clip.exists():
        print("[readme] no search clip; skipping GIF"); return
    vf = "fps=10,crop=1000:620:60:180,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer"
    subprocess.run([ff, "-y", "-loglevel", "error", "-ss", "1.5", "-t", "7", "-i", str(clip), "-vf", vf, str(OUT / "search.gif")], check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://localhost:8765/veritas/")
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    asyncio.run(shots(a.url)); gif()
    for f in sorted(OUT.iterdir()): print(f"[readme] {f.name} {f.stat().st_size // 1024} KB")
