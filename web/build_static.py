"""Build a public, static copy of the website for GitHub Pages (owner: D).

    .venv/bin/python -m web.build_static            # writes docs/
    .venv/bin/python -m web.build_static --out /tmp/site

How it works: the real FastAPI app runs in-process (REAL mode, offline models). Every response the page needs
for the saved study areas is captured to a file, so the numbers are byte-for-byte what the live server gives.
`web/static/js/static-api.js` (loaded only in the static copy) answers the page's /api calls from those files.

Text search still works for *any* words: the page runs the same MiniLM model in the browser (transformers.js),
subtracts the same text mean `mu_y`, and compares with the rotated image vectors exported here. Analysing a new
city needs the Python server (live satellite download), so the static Lab offers the saved areas only.

Re-run after any metrics or code change; nothing here holds a number.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "web" / "static"
REPO_URL = "https://github.com/JohnPoole3rd/veritas"

# Queries baked as files too (exact server answers, used if the in-browser model can't load).
BAKED_QUERIES = ["dense forest", "residential neighborhood", "park or grassland", "commercial area", "river corridor",
                 "highway", "lake or reservoir", "farmland", "golf course", "parking lot", "industrial buildings",
                 "climate change", "trees", "water"]


def slug(q: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", q.strip().lower()).strip("-") or "q"


def rel(text: str) -> str:
    """Absolute site paths -> relative, so the copy works under any GitHub Pages subpath."""
    return (text.replace('"/api/', '"api/').replace("`/api/", "`api/").replace("'/api/", "'api/")
            .replace('"/static/', '"static/').replace("`/static/", "`static/"))


class Out:
    def __init__(self, root: Path):
        self.root = root
        self.n = 0
        self.bytes = 0

    def write(self, path: str, data: bytes | str) -> None:
        p = self.root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        b = data.encode() if isinstance(data, str) else data
        p.write_bytes(b)
        self.n += 1
        self.bytes += len(b)

    def json(self, path: str, obj) -> None:
        self.write(path, rel(json.dumps(obj, separators=(",", ":"))))


def f16(a: np.ndarray) -> bytes:
    a = np.asarray(a, np.float32)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    return a.astype("<f2").tobytes()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT / "docs"))
    ap.add_argument("--no-eurosat", action="store_true", help="skip the 4,050 EuroSAT thumbnails (~10 MB)")
    args = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

    from fastapi.testclient import TestClient
    from PIL import Image

    from app import load_backend
    from web.server import create_app

    out_dir = Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out = Out(out_dir)

    backend = load_backend(ROOT, False)
    if backend.name != "REAL":
        raise SystemExit("REAL artifacts are missing; the static copy must not show mock data.")
    app = create_app(backend, ROOT, mount_classic=False)
    srv = app.state.server
    c = TestClient(app)

    def get(path, **kw):
        r = c.get(path, **kw)
        if r.status_code != 200:
            raise SystemExit(f"GET {path} -> {r.status_code} {r.text[:200]}")
        return r

    def post_job(path, body=None):
        r = c.post(path, json=body) if body is not None else c.post(path)
        r.raise_for_status()
        jid = r.json()["job"]
        import time
        for _ in range(600):
            j = c.get(f"/api/jobs/{jid}").json()
            if j["status"] == "done":
                return j
            if j["status"] == "error":
                raise SystemExit(f"{path}: {j.get('error')}")
            time.sleep(0.2)
        raise SystemExit(f"{path}: job timed out")

    # ---------------------------------------------------------------- site files
    for p in STATIC.rglob("*"):
        if p.is_file() and "__pycache__" not in p.parts:
            r = p.relative_to(STATIC).as_posix()
            data = p.read_bytes()
            if p.suffix in (".js", ".css"):
                data = rel(data.decode()).encode()
            out.write(f"static/{r}", data)
    html = (STATIC / "index.html").read_text()
    html = rel(html).replace('href="/static/', 'href="static/').replace('src="/static/', 'src="static/')
    html = html.replace('<a href="/classic/">Classic demo (Gradio)</a>', f'<a href="{REPO_URL}#run-it-yourself">Run it yourself</a>')
    shim = '<script src="static/js/static-api.js"></script>\n'
    html = html.replace('<script src="static/vendor/maplibre-gl.js">', shim + '<script src="static/vendor/maplibre-gl.js">', 1)
    assert "static-api.js" in html, "could not inject the static API shim"
    out.write("index.html", html)
    out.write(".nojekyll", "")

    # ---------------------------------------------------------------- meta + text model constants
    meta = get("/api/meta").json()
    meta["static"] = True
    out.json("api/meta.json", meta)
    out.write("api/mu_y.bin", np.asarray(backend.params["mu_y"], "<f4").tobytes())
    model_name = backend.config["text"]["model_name"]

    # ---------------------------------------------------------------- galleries (local + EuroSAT)
    galleries = {}
    for key in ("local", "eurosat"):
        if key == "eurosat" and args.no_eurosat:
            continue
        g = srv.gallery(key)
        items = g["items"]
        out.write(f"api/gallery/{key}.bin", f16(g["emb"]))
        rows = []
        for it in items:
            row = {"tile_id": it["tile_id"], "label": it["label"]}
            if key == "local":
                row["lat"], row["lon"] = it.get("lat"), it.get("lon")
            rows.append(row)
            png = get(f"/api/tiles/{key}/{it['tile_id']}.png").content
            if key == "eurosat":  # 4,050 thumbnails: JPEG keeps the copy small; the page uses the same path
                im = Image.open(io.BytesIO(png)).convert("RGB")
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=86)
                out.write(f"api/tiles/{key}/{it['tile_id']}.jpg", buf.getvalue())
            else:
                out.write(f"api/tiles/{key}/{it['tile_id']}.png", png)
        galleries[key] = {"n": len(items), "dim": int(np.asarray(g["emb"]).shape[1]), "items": rows,
                          "ext": "jpg" if key == "eurosat" else "png"}
        for q in BAKED_QUERIES:
            r = get("/api/search", params={"q": q, "gallery": key, "k": 6}).json()
            if key == "eurosat":
                for h in r["hits"]:
                    h["image"] = h["image"].replace(".png", ".jpg")
            out.json(f"api/search/{key}/{slug(q)}.json", r)
    for t in meta.get("local_tiles", []):
        if t.get("tile_id"):
            r = c.get(f"/api/classify/local/{t['tile_id']}")
            if r.status_code == 200:
                out.json(f"api/classify/local/{t['tile_id']}.json", r.json())

    # ---------------------------------------------------------------- saved study areas (Lab + heat chapter)
    regions = []
    for p in meta.get("presets", []):
        if not p.get("cached"):
            continue
        est = c.post("/api/region/estimate", json={"bbox": p["bbox"]}).json()
        rj = post_job("/api/region", {"bbox": p["bbox"], "when": None})
        region = rj["result"]
        rid = region["id"]
        cj = post_job(f"/api/region/{rid}/climate")
        out.json(f"api/region/{rid}.json", region)
        out.json(f"api/region/{rid}/climate.json", cj["result"])
        out.json(f"api/region/{rid}/estimate.json", est)
        out.json(f"api/region/{rid}/stages.json", {"region": rj["stages"], "climate": cj["stages"]})
        out.write(f"api/region/{rid}/image.png", get(f"/api/region/{rid}/image.png").content)
        for name in ("ndvi", "lst"):
            r = c.get(f"/api/region/{rid}/layer/{name}.png")
            if r.status_code == 200:
                out.write(f"api/region/{rid}/layer/{name}.png", r.content)
        for fmt in ("csv", "geojson"):
            out.write(f"api/region/{rid}/export.{fmt}", get(f"/api/region/{rid}/export.{fmt}").content)
        n = len(region["tiles"])
        for i in range(n):
            out.write(f"api/region/{rid}/tile/{i}.png", get(f"/api/region/{rid}/tile/{i}.png").content)
        entry = srv.region_entry(rid)
        from web import analysis
        out.write(f"api/region/{rid}/aligned.bin", f16(entry["aligned"]))
        valid = [bool(v) for v in analysis.valid_tiles(entry["summary"]["tiles"])]
        out.json(f"api/region/{rid}/valid.json", valid)
        for q in BAKED_QUERIES:
            out.json(f"api/region/{rid}/search/{slug(q)}.json", get(f"/api/region/{rid}/search", params={"q": q, "k": 8}).json())
        regions.append({"key": p["key"], "bbox": p["bbox"], "id": rid, "n": n})

    out.json("api/static.json", {"text_model": model_name, "browser_model": "Xenova/all-MiniLM-L6-v2",
                                 "galleries": galleries, "regions": regions, "baked_queries": BAKED_QUERIES,
                                 "repo": REPO_URL})
    print(f"[static] wrote {out.n} files, {out.bytes / 1e6:.1f} MB -> {out_dir}")
    print(f"[static] regions: {', '.join(r['key'] for r in regions) or 'none'}; galleries: {', '.join(galleries)}")


if __name__ == "__main__":
    main()
