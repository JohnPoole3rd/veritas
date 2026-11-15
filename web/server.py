"""Platonic Earth website: the FastAPI app behind web/static/ (owner: D, subagent D1).

Implements the HTTP API in web/API.md. Model inference goes through the same backend objects as the
Gradio demo (app.py: RealBackend / MockBackend); satellite I/O is web/geo.py; per-tile statistics are
web/analysis.py. Nothing numeric is typed in here: metrics come from artifacts/metrics/*, everything
else is computed live.

Run:
    python -m web.server                 # auto: REAL if artifacts exist, else MOCK
    python -m web.server --mock --port 8001
    python -m web.server --online        # allow Hugging Face downloads (default: offline cache only)

`app` is created lazily by `create_app(backend)` so tests can pass a MockBackend.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from web import analysis, geo

log = logging.getLogger("platonic-earth.web")

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "web" / "static"
METRICS_DIR = ROOT / "artifacts" / "metrics"

MAX_TILES = 400
SEARCH_K_MAX = 24
REGION_K_MAX = 100
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
JOBS_KEPT = 200
REGIONS_KEPT = 12
GEOCODE_URL = "https://nominatim.openstreetmap.org/search"
GEOCODE_HEADERS = {"User-Agent": "platonic-earth-hackathon/0.1"}
LONG_CACHE = "public, max-age=86400"
IMMUTABLE_CACHE = "public, max-age=31536000, immutable"

# Climate-relevant study areas, each <= 400 tiles at 640 m (checked in tests/test_web_server.py).
PRESETS = [
    {"key": "upper-arlington", "name": "Upper Arlington, Ohio", "note": "Our test site: suburban canopy and the Scioto River",
     "bbox": [-83.110, 39.985, -83.037, 40.070]},
    {"key": "columbus", "name": "Downtown Columbus, Ohio", "note": "Dense core next door to the test site",
     "bbox": [-83.030, 39.930, -82.960, 39.990]},
    {"key": "phoenix", "name": "Phoenix, Arizona", "note": "Desert city, one of the strongest urban heat islands in the US",
     "bbox": [-112.120, 33.410, -112.030, 33.490]},
    {"key": "las-vegas", "name": "Las Vegas, Nevada", "note": "Irrigated green against bare desert",
     "bbox": [-115.200, 36.110, -115.120, 36.190]},
    {"key": "houston", "name": "Houston, Texas", "note": "Humid sprawl, bayous and flood risk",
     "bbox": [-95.420, 29.720, -95.330, 29.790]},
    {"key": "chicago", "name": "Chicago Loop, Illinois", "note": "High-rise core on Lake Michigan",
     "bbox": [-87.680, 41.850, -87.600, 41.910]},
    {"key": "paris", "name": "Paris, France", "note": "Compact European city on the Seine",
     "bbox": [2.290, 48.830, 2.400, 48.885]},
    {"key": "madrid", "name": "Madrid, Spain", "note": "Hot, dry summers; parks as cool refuges",
     "bbox": [-3.740, 40.390, -3.650, 40.450]},
    {"key": "delhi", "name": "New Delhi, India", "note": "Extreme pre-monsoon heat",
     "bbox": [77.170, 28.580, 77.260, 28.660]},
]


class ApiError(Exception):
    """An error with a plain-language message for the client."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def _png_response(data: bytes, cache: str = LONG_CACHE) -> Response:
    return Response(content=data, media_type="image/png", headers={"Cache-Control": cache})


def _png_bytes(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _num(v: Any) -> Any:
    """CSV cell -> int / float / bool / str; '' -> None."""
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    if s in ("True", "False"):
        return s == "True"
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    try:
        f = float(s)
        return f if math.isfinite(f) else None
    except ValueError:
        return s


def _read_csv(name: str, metrics_dir: Path) -> list[dict] | None:
    path = metrics_dir / name
    if not path.exists():
        return None
    try:
        with path.open(newline="") as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error):
        log.exception("could not read %s", path)
        return None


def _pick(row: dict, keys: list[str]) -> dict:
    return {k: _num(row.get(k)) for k in keys}


# --------------------------------------------------------------------------- metrics
def read_metrics(metrics_dir: Path = METRICS_DIR, manifest_rows: list[dict] | None = None) -> dict:
    """The `metrics` block of /api/meta, straight from the logged files (null when a file is missing)."""
    out: dict[str, Any] = {}

    rows = _read_csv("majority.csv", metrics_dir)
    out["majority"] = _pick(rows[0], ["accuracy", "macro_f1", "test_n", "most_common_class"]) if rows else None
    rows = _read_csv("resnet18.csv", metrics_dir)
    out["resnet18"] = _pick(rows[0], ["accuracy", "macro_f1", "test_n"]) if rows else None

    rows = _read_csv("caption_classification_test.csv", metrics_dir)
    out["caption_classification"] = (
        {r["method"]: _pick(r, ["accuracy", "top5", "macro_f1", "n_images"]) for r in rows} if rows else None)
    rows = _read_csv("retrieval.csv", metrics_dir)
    out["retrieval"] = {r["method"]: _pick(r, ["top1", "top5", "n_captions", "n_images"]) for r in rows} if rows else None

    rows = _read_csv("local_summary.csv", metrics_dir)
    local = {r["method"]: _pick(r, ["accuracy", "n", "accuracy_unambiguous", "n_unambiguous"]) for r in rows} if rows else None
    try:
        lj = json.loads((metrics_dir / "local.json").read_text())
        for m in lj.get("methods", []):
            if m.get("method") == "oracle_majority_local":
                entry = {k: m.get(k) for k in ("accuracy", "n", "accuracy_unambiguous", "n_unambiguous")}
                hit = re.search(r"'([^']+)'", str(m.get("notes", "")))
                entry["label"] = hit.group(1) if hit else None
                local = dict(local or {})
                local["oracle_majority_local"] = entry
    except (OSError, ValueError):
        pass
    out["local"] = local

    rows = _read_csv("local_by_category.csv", metrics_dir)
    out["local_by_category"] = [{k: _num(v) if k != "local_category" else v for k, v in r.items()} for r in rows] if rows else None

    if manifest_rows:
        counts: dict[str, int] = {}
        for r in manifest_rows:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        out["local_label_counts"] = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
    else:
        out["local_label_counts"] = None

    pc = _read_csv("procrustes_test_per_class.csv", metrics_dir)
    rc = {r["class"]: r for r in (_read_csv("resnet18_per_class.csv", metrics_dir) or [])}
    out["per_class"] = [{"class": r["class"], "recall": _num(r.get("recall")),
                         "resnet18_recall": _num(rc.get(r["class"], {}).get("recall"))} for r in pc] if pc else None

    path = metrics_dir / "procrustes_test_confusion.csv"
    out["confusion"] = None
    if path.exists():
        try:
            with path.open(newline="") as f:
                table = list(csv.reader(f))
            out["confusion"] = {"classes": table[0][1:],
                                "rows": [r[0] for r in table[1:]],
                                "matrix": [[int(v) for v in r[1:]] for r in table[1:]]}
        except (OSError, ValueError, IndexError):
            log.exception("could not read %s", path)

    rows = _read_csv("procrustes_test_top_confusions.csv", metrics_dir)
    out["top_confusions"] = [_pick(r, ["true", "predicted", "count", "share_of_true"]) for r in rows] if rows else None
    rows = _read_csv("failure_taxonomy.csv", metrics_dir)
    out["failure_taxonomy"] = [dict(r) for r in rows] if rows else None
    return out


# --------------------------------------------------------------------------- server state
class Server:
    """Everything the routes share: backend, galleries, jobs, regions, the model lock."""

    def __init__(self, backend, root: Path = ROOT):
        self.backend = backend
        self.root = Path(root)
        self.mode = getattr(backend, "name", "MOCK")
        self.model_lock = threading.Lock()  # one model call at a time (torch / MPS are not re-entrant here)
        self.state_lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pe-job")
        self.jobs: "OrderedDict[str, dict]" = OrderedDict()
        self.regions: "OrderedDict[str, dict]" = OrderedDict()
        self.geocode_cache: dict[str, list] = {}
        self._probe: dict | None = None
        self._probe_done = False
        self._static_meta: dict | None = None
        self.manifest_rows = self._read_manifest()
        self.galleries = self._build_galleries()

    # -- galleries ----------------------------------------------------------
    def _config(self) -> dict:
        cfg = getattr(self.backend, "config", None)
        if cfg is None:
            from src.config import load_config
            cfg = load_config()
        return cfg

    def _read_manifest(self) -> list[dict]:
        try:
            path = self.root / self._config()["data"]["local_manifest"]
        except Exception:
            path = self.root / "data" / "local_tiles" / "manifest.csv"
        if not path.exists():
            return []
        with path.open(newline="") as f:
            return list(csv.DictReader(f))

    def _build_galleries(self) -> dict:
        """{key: {"label", "items": [{tile_id, label, path(abs), lat, lon}], "index": {tile_id: i}, "emb", "note"}}"""
        out: dict[str, dict] = {}
        man = {r["tile_id"]: r for r in self.manifest_rows}
        b = self.backend
        if self.mode == "REAL":
            local = getattr(b, "local", None)
            if local:
                items = []
                for it in local["items"]:
                    m = man.get(it["tile_id"], {})
                    items.append({"tile_id": it["tile_id"], "label": it["label"], "path": self.root / it["path"],
                                  "lat": _num(m.get("lat")), "lon": _num(m.get("lon"))})
                out["local"] = {"label": "Upper Arlington", "items": items, "emb": local["gallery"]}
            items = [{"tile_id": it["tile_id"], "label": it["label"], "path": Path(b.eurosat_root) / it["path"]}
                     for it in b.items]
            out["eurosat"] = {"label": "Europe (EuroSAT test)", "items": items, "emb": b.gallery,
                              "note": getattr(b, "subset_note", None)}
        else:
            if self.manifest_rows:
                out["local"] = {"label": "Upper Arlington", "emb": None, "items": [
                    {"tile_id": r["tile_id"], "label": r["label"], "path": self.root / r["path"],
                     "lat": _num(r.get("lat")), "lon": _num(r.get("lon"))} for r in self.manifest_rows]}
            try:
                from src.config import resolve_path
                cfg = self._config()
                split = json.loads((self.root / "data" / "splits" / "test.json").read_text())
                base = resolve_path(cfg["data"]["eurosat_root"], cfg)
                out["eurosat"] = {"label": "Europe (EuroSAT test)", "emb": None, "items": [
                    {"tile_id": it["tile_id"], "label": it["label"], "path": Path(base) / it["path"]}
                    for it in split["items"]]}
            except Exception:
                pass
        for g in out.values():
            g["index"] = {it["tile_id"]: i for i, it in enumerate(g["items"])}
        return out

    def gallery(self, key: str) -> dict:
        g = self.galleries.get(key)
        if g is None:
            raise ApiError(404 if key not in ("local", "eurosat") else 503,
                           f"The '{key}' gallery isn't available here." if key in ("local", "eurosat")
                           else "Unknown gallery. Use 'local' or 'eurosat'.")
        return g

    def tile_path(self, gallery: str, tile_id: str) -> Path:
        g = self.gallery(gallery)
        i = g["index"].get(tile_id)
        if i is None:
            raise ApiError(404, "No tile with that id.")
        return g["items"][i]["path"]

    # -- model calls (all under model_lock) -----------------------------------
    def gallery_scores(self, gallery: str, query: str) -> np.ndarray:
        g = self.gallery(gallery)
        n = len(g["items"])
        if self.mode == "REAL":
            from src.alignment import l2_normalize, transform_texts

            with self.model_lock:
                q = transform_texts(self.backend._encode_text(query), self.backend.params)[0]
            G = l2_normalize(np.asarray(g["emb"], np.float32))
            return (G @ l2_normalize(np.asarray(q, np.float32).reshape(1, -1))[0]).astype(np.float32)
        # MOCK: placeholder scores, nudged towards the class a keyword names so rehearsal looks sane.
        from app import _KEYWORDS

        s = analysis.text_scores(self.backend, query, np.zeros((n, 1), np.float32)).astype(np.float32)
        q = query.lower()
        favored = next((c for kw, c in _KEYWORDS.items() if kw in q), None)
        if favored:
            s = s + np.array([0.3 if it["label"] == favored else 0.0 for it in g["items"]], np.float32)
        return s

    def search(self, gallery: str, query: str, k: int) -> list[dict]:
        g = self.gallery(gallery)
        scores = self.gallery_scores(gallery, query)
        order = np.argsort(-scores, kind="stable")[:k]
        hits = []
        for i in order:
            it = g["items"][int(i)]
            h = {"tile_id": it["tile_id"], "label": it["label"], "score": round(float(scores[i]), 4),
                 "image": f"/api/tiles/{gallery}/{it['tile_id']}.png"}
            if gallery == "local":
                h["lat"], h["lon"] = it.get("lat"), it.get("lon")
            hits.append(h)
        return hits

    def classify(self, image) -> list[dict]:
        with self.model_lock:
            ranked = list(self.backend.classify_image(image.convert("RGB")))
        return [{"label": lab, "score": round(float(s), 4)} for lab, s in ranked[:3]]

    # -- meta ---------------------------------------------------------------
    def device(self) -> str | None:
        if self.mode != "REAL":
            return None
        d = getattr(self.backend, "_dinov2", None)
        if d is not None:
            return str(d[1])
        try:
            from src.encoders import get_device
            return get_device()
        except Exception:
            return None

    def probe(self) -> dict | None:
        if self._probe_done:
            return self._probe
        from app import FAILURE_PROBES, TOP_K

        p = FAILURE_PROBES[0]
        key = "local" if "local" in self.galleries else "eurosat"
        try:
            hits = self.search(key, p["query"], TOP_K)
            self._probe = {"query": p["query"], "why": p["why"], "gallery": key,
                           "hits": [{k: h[k] for k in ("tile_id", "label", "score", "image")} for h in hits]}
        except Exception:
            log.exception("failure probe could not run")
            self._probe = None
        self._probe_done = True
        return self._probe

    def local_tiles(self) -> list[dict]:
        preds = {r["tile_id"]: r for r in (_read_csv("local_predictions.csv", self.root / "artifacts" / "metrics") or [])}
        out = []
        for r in self.manifest_rows:
            p = preds.get(r["tile_id"], {})
            out.append({
                "tile_id": r["tile_id"], "label": r["label"], "local_category": r.get("local_category"),
                "lat": _num(r.get("lat")), "lon": _num(r.get("lon")), "date": r.get("date"), "notes": r.get("notes"),
                "pred": p.get("predicted_label"), "score": _num(p.get("score")), "correct": _num(p.get("correct")),
                "top3": p["top3"].split("|") if p.get("top3") else None,
                "resnet18_pred": p.get("resnet18_pred") or None, "resnet18_prob": _num(p.get("resnet18_prob")),
                "resnet18_correct": _num(p.get("resnet18_correct")),
                "image": f"/api/tiles/local/{r['tile_id']}.png",
            })
        return out

    def static_meta(self) -> dict:
        if self._static_meta is None:
            from app import DISCLAIMER, EXAMPLE_QUERIES, local_examples

            try:
                examples = [{"tile_id": e["tile_id"], "kind": e["kind"], "why": e["why"]} for e in local_examples(self.root)]
            except Exception:
                log.exception("local examples unavailable")
                examples = []
            self._static_meta = {
                "mode": self.mode,
                "disclaimer": DISCLAIMER,
                "classes": [dict(c) for c in analysis.CLASS_META],
                "galleries": [{"key": k, "label": g["label"], "n": len(g["items"])} for k, g in self.galleries.items()],
                "examples": list(EXAMPLE_QUERIES),
                "metrics": read_metrics(self.root / "artifacts" / "metrics", self.manifest_rows),
                "local_tiles": self.local_tiles(),
                "local_examples": examples,
            }
        return self._static_meta

    def meta(self) -> dict:
        m = dict(self.static_meta())
        m["device"] = self.device()
        m["probe"] = self.probe()
        m["limits"] = {"max_tiles": MAX_TILES, "tile_m": int(geo.TILE_M), "tile_px": geo.TILE_PX}
        m["presets"] = [dict(p, cached=geo.cached_region_for_bbox(p["bbox"]) is not None) for p in PRESETS]
        m["geo_available"] = geo.available()
        return m

    # -- jobs ---------------------------------------------------------------
    def new_job(self, kind: str) -> dict:
        job = {"id": uuid.uuid4().hex[:12], "kind": kind, "status": "running", "stages": [], "error": None,
               "result": None, "_t0": time.time()}
        with self.state_lock:
            self.jobs[job["id"]] = job
            while len(self.jobs) > JOBS_KEPT:
                self.jobs.popitem(last=False)
        return job

    def say_for(self, job: dict):
        def say(stage: str, detail: str) -> None:
            with self.state_lock:
                job["stages"].append({"stage": stage, "detail": detail, "t": round(time.time() - job["_t0"], 2)})
        return say

    def run_job(self, job: dict, fn, *args) -> None:
        def work():
            try:
                result = fn(job, *args)
                with self.state_lock:
                    job["result"], job["status"] = result, "done"
            except geo.GeoError as e:
                with self.state_lock:
                    job["error"], job["status"] = str(e), "error"
            except ApiError as e:
                with self.state_lock:
                    job["error"], job["status"] = e.message, "error"
            except Exception:
                log.exception("%s job %s failed", job["kind"], job["id"])
                with self.state_lock:
                    job["error"], job["status"] = "Something went wrong on the server. Try again, or pick another area.", "error"
        self.pool.submit(work)

    def job_view(self, job_id: str) -> dict:
        with self.state_lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise ApiError(404, "No job with that id (the server may have restarted).")
            return {k: (list(v) if k == "stages" else v) for k, v in job.items() if not k.startswith("_")}

    # -- regions ------------------------------------------------------------
    def _suffix(self) -> str:
        return "" if self.mode == "REAL" else f".{self.mode.lower()}"

    def _region_dir(self, rid: str) -> Path:
        return Path(geo.CACHE_DIR) / rid

    def _remember_region(self, rid: str, entry: dict) -> dict:
        with self.state_lock:
            self.regions[rid] = entry
            self.regions.move_to_end(rid)
            while len(self.regions) > REGIONS_KEPT:
                self.regions.popitem(last=False)
        return entry

    def region_entry(self, rid: str) -> dict:
        """{"summary", "aligned", "climate"} for a labelled region, from memory or disk; ApiError 404 otherwise."""
        if not geo._valid_id(rid):
            raise ApiError(404, "No region with that id.")
        with self.state_lock:
            hit = self.regions.get(rid)
        if hit is not None:
            return hit
        d, sfx = self._region_dir(rid), self._suffix()
        try:
            summary = json.loads((d / f"summary{sfx}.json").read_text())
            aligned = np.load(d / f"aligned{sfx}.npy")
            if summary.get("mode") != self.mode or len(aligned) != len(summary["tiles"]):
                raise ValueError("stale")
        except (OSError, ValueError, KeyError):
            if (d / "region.npz").exists():
                raise ApiError(404, "This area was downloaded before but hasn't been labelled since the server "
                                    "started. POST /api/region with its box again; it loads from the saved copy.")
            raise ApiError(404, "No region with that id.")
        climate = None
        try:
            climate = json.loads((d / f"climate{sfx}.json").read_text())
        except (OSError, ValueError):
            pass
        return self._remember_region(rid, {"summary": summary, "aligned": aligned, "climate": climate})

    def _save_region_files(self, rid: str, summary: dict, aligned: np.ndarray) -> None:
        d, sfx = self._region_dir(rid), self._suffix()
        try:
            d.mkdir(parents=True, exist_ok=True)
            geo._atomic_write(d / f"aligned{sfx}.npy", lambda f: np.save(f, aligned))
            geo._atomic_write(d / f"summary{sfx}.json", lambda f: f.write(json.dumps(summary).encode()))
        except OSError:
            log.warning("could not save region %s to disk", rid)

    def region_job(self, job: dict, bbox: list, when) -> dict:
        say = self.say_for(job)
        t0 = time.time()
        region = geo.fetch_region(bbox, max_tiles=MAX_TILES, when=when, progress=say)
        from_cache = any(s["stage"] == "cache" for s in job["stages"])
        rid = region.id
        try:
            entry = self.region_entry(rid)
        except ApiError:
            entry = None
        if entry is not None:
            say("cache", "Using the saved land-cover estimates for this pass")
            summary = dict(entry["summary"], elapsed_s=round(time.time() - t0, 1), from_cache=True)
            return summary

        images = [region.tile_image(i) for i in range(region.n_tiles)]
        with self.model_lock:
            aligned, scores = analysis.embed_and_score(self.backend, images, progress=say)
        labels = analysis.tile_labels(scores)
        valid = analysis.valid_tiles(region.tiles)
        classes = [lab["cls"] for lab in labels]
        tiles = [{**t, **lab, "valid": bool(ok)} for t, lab, ok in zip(region.tiles, labels, valid)]
        say("summary", f"Labelled {int(valid.sum()):,} tiles with imagery")
        scene = dict(region.scene)
        scene.pop("collection", None)
        summary = analysis._clean({
            "id": rid, "bbox": [float(v) for v in region.bbox], "mode": self.mode, "scene": scene,
            "rows": region.rows, "cols": region.cols, "n_tiles": region.n_tiles, "tile_m": int(geo.TILE_M),
            "area_km2": round(region.n_tiles * analysis.TILE_KM2, 1), "corners": region.corners,
            "image": f"/api/region/{rid}/image.png", "tiles": tiles,
            "composition": analysis.composition(classes, valid), "ipcc": analysis.ipcc_rollup(classes, valid),
            "elapsed_s": round(time.time() - t0, 1), "from_cache": from_cache,
        })
        self._save_region_files(rid, summary, aligned)
        self._remember_region(rid, {"summary": summary, "aligned": aligned, "climate": None, "region": region})
        return summary

    def _geo_region(self, rid: str) -> geo.Region:
        with self.state_lock:
            region = (self.regions.get(rid) or {}).get("region")
        if region is None:
            region = geo.load_cached(rid)
        if region is None:
            raise ApiError(404, "The pixels for this region aren't saved any more. POST /api/region with its box again.")
        return region

    def climate_job(self, job: dict, rid: str) -> dict:
        say = self.say_for(job)
        entry = self.region_entry(rid)
        if entry.get("climate") is not None:
            say("cache", "Using the saved climate layers")
            return entry["climate"]
        region = self._geo_region(rid)
        summary = entry["summary"]
        idx = geo.fetch_indices(region, progress=say)
        lst, lst_error = None, None
        try:
            lst = geo.fetch_lst(region, progress=say)
        except geo.GeoError as e:
            lst_error = str(e)
            say("lst-error", lst_error)
        d = self._region_dir(rid)
        lst_c = None if lst is None else np.asarray(lst["lst_c"], np.float32)
        if lst_c is not None and not np.isfinite(lst_c).any():
            lst, lst_c, lst_error = None, None, "The Landsat pass has no clear pixels over this area."
        tiles = summary["tiles"]
        stats = analysis.climate_stats(
            [t["cls"] for t in tiles], idx["tile_ndvi"], idx["tile_water_frac"],
            None if lst is None else lst["tile_lst_c"], [t["center"] for t in tiles], analysis.valid_tiles(tiles))
        say("layers", "Rendering the greenness and temperature layers")
        try:
            d.mkdir(parents=True, exist_ok=True)
            geo._atomic_write(d / "layer_ndvi.png", lambda f: f.write(geo.colorize(idx["ndvi"], 0.0, 0.9, "ndvi")))
        except OSError:
            log.warning("could not save NDVI layer for %s", rid)
        result: dict[str, Any] = {
            "ndvi": {"tile": [analysis._r(v, 3) for v in np.asarray(idx["tile_ndvi"]).ravel()],
                     "image": f"/api/region/{rid}/layer/ndvi.png", "vmin": 0.0, "vmax": 0.9,
                     "palette": geo.palette_stops("ndvi")},
            "water": {"tile": [analysis._r(v, 3) for v in np.asarray(idx["tile_water_frac"]).ravel()]},
            "lst": None, "lst_error": lst_error, "stats": stats,
        }
        if lst is not None:
            vals = lst_c[np.isfinite(lst_c)]
            vmin, vmax = float(math.floor(np.percentile(vals, 2))), float(math.ceil(np.percentile(vals, 98)))
            if vmax <= vmin:
                vmax = vmin + 1.0
            try:
                geo._atomic_write(d / "layer_lst.png", lambda f: f.write(geo.colorize(lst_c, vmin, vmax, "heat")))
            except OSError:
                log.warning("could not save LST layer for %s", rid)
            result["lst"] = {"tile": [analysis._r(v, 1) for v in np.asarray(lst["tile_lst_c"]).ravel()],
                             "image": f"/api/region/{rid}/layer/lst.png", "vmin": vmin, "vmax": vmax,
                             "scene": lst.get("scene"), "days_from_s2": lst.get("days_from_s2"),
                             "palette": geo.palette_stops("heat")}
        result = analysis._clean(result)
        try:
            geo._atomic_write(d / f"climate{self._suffix()}.json", lambda f: f.write(json.dumps(result).encode()))
        except OSError:
            log.warning("could not save climate result for %s", rid)
        with self.state_lock:
            entry["climate"] = result
        return result

    def layer_png(self, rid: str, name: str) -> bytes:
        entry = self.region_entry(rid)
        path = self._region_dir(rid) / f"layer_{name}.png"
        if path.exists():
            return path.read_bytes()
        climate = entry.get("climate")
        if not climate:
            raise ApiError(404, "No climate layers for this region yet. POST /api/region/{id}/climate first.")
        region = self._geo_region(rid)
        if name == "ndvi":
            data = geo.colorize(geo.fetch_indices(region)["ndvi"], 0.0, 0.9, "ndvi")
        else:
            if not climate.get("lst"):
                raise ApiError(404, climate.get("lst_error") or "No surface-temperature layer for this region.")
            data = geo.colorize(geo.fetch_lst(region)["lst_c"], climate["lst"]["vmin"], climate["lst"]["vmax"], "heat")
        return data


# --------------------------------------------------------------------------- app factory
def _bbox_from(body: Any) -> list[float]:
    if not isinstance(body, dict):
        raise ApiError(400, "Send JSON like {\"bbox\": [west, south, east, north]}.")
    bbox = body.get("bbox")
    try:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError
        out = [float(v) for v in bbox]
        if not all(math.isfinite(v) for v in out):
            raise ValueError
    except (TypeError, ValueError):
        raise ApiError(400, "bbox must be four numbers: [west, south, east, north] in degrees.") from None
    return out


async def _json_body(request: Request) -> Any:
    try:
        return await request.json()
    except Exception:
        raise ApiError(400, "Send a JSON body.") from None


def _int_param(value: str | None, name: str, default: int, lo: int, hi: int) -> int:
    if value in (None, ""):
        return default
    try:
        v = int(value)
    except ValueError:
        raise ApiError(400, f"{name} must be a whole number.") from None
    if not lo <= v <= hi:
        raise ApiError(400, f"{name} must be between {lo} and {hi}.")
    return v


def _query_text(q: str | None) -> str:
    q = (q or "").strip()
    if not q:
        raise ApiError(400, "Type a description to search for, e.g. 'river corridor'.")
    if len(q) > 200:
        raise ApiError(400, "Keep the description under 200 characters.")
    return q


def create_app(backend=None, root: Path = ROOT, mount_classic: bool = True) -> FastAPI:
    if backend is None:
        from app import load_backend
        backend = load_backend(root)
    srv = Server(backend, root)
    app = FastAPI(title="Platonic Earth", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.server = srv

    # -- errors: always {"error": message}, never a traceback --------------------------------
    @app.exception_handler(ApiError)
    async def _api_error(_req, exc: ApiError):
        return JSONResponse({"error": exc.message}, status_code=exc.status)

    @app.exception_handler(geo.GeoError)
    async def _geo_error(_req, exc: geo.GeoError):
        return JSONResponse({"error": str(exc)}, status_code=503 if exc.network else 422)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_req, exc: StarletteHTTPException):
        msg = "Not found." if exc.status_code == 404 else str(exc.detail)
        return JSONResponse({"error": msg}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_req, exc: RequestValidationError):
        return JSONResponse({"error": "The request is missing a field or has a wrong value."}, status_code=400)

    @app.exception_handler(Exception)
    async def _unexpected(req, exc: Exception):
        log.exception("unhandled error on %s", req.url.path)
        return JSONResponse({"error": "Something went wrong on the server."}, status_code=500)

    @app.middleware("http")
    async def _no_cache_for_code(request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.endswith((".html", ".js", ".css")):
            response.headers["Cache-Control"] = "no-cache"
        return response

    # -- pages -----------------------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    def index():
        page = STATIC_DIR / "index.html"
        if not page.exists():
            return JSONResponse({"error": "The site isn't built yet (web/static/index.html is missing)."}, status_code=404)
        return FileResponse(page, media_type="text/html", headers={"Cache-Control": "no-cache"})

    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # -- meta ------------------------------------------------------------------------------
    @app.get("/api/meta")
    def meta():
        return srv.meta()

    # -- search / classify ----------------------------------------------------------------
    @app.get("/api/search")
    def search(q: str | None = None, gallery: str = "local", k: str | None = None):
        from app import TOP_K

        text = _query_text(q)
        kk = _int_param(k, "k", max(TOP_K, 6), 1, SEARCH_K_MAX)
        hits = srv.search(gallery, text, kk)
        g = srv.gallery(gallery)
        note = "Labels are the human labels; scores are cosine similarity, not probability."
        if g.get("note"):
            note += f" Searching the {g['note']}."
        if srv.mode != "REAL":
            note = "MOCK mode: fake scores for rehearsal only. " + note
        return {"query": text, "gallery": gallery, "k": kk, "hits": hits, "note": note}

    @app.get("/api/tiles/{gallery}/{tile_id}.png")
    def tile_png(gallery: str, tile_id: str):
        path = srv.tile_path(gallery, tile_id)
        if not path.exists():
            raise ApiError(404, "That tile's image file is missing on this machine.")
        from PIL import Image

        try:
            with Image.open(path) as im:
                data = _png_bytes(im.convert("RGB"))
        except OSError:
            raise ApiError(404, "That tile's image file can't be read.") from None
        return _png_response(data, IMMUTABLE_CACHE)

    def _classified(image) -> dict:
        from app import DISCLAIMER
        return {"top": srv.classify(image), "disclaimer": DISCLAIMER}

    @app.post("/api/classify")
    async def classify_upload(file: UploadFile = File(...)):
        from PIL import Image
        from starlette.concurrency import run_in_threadpool

        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise ApiError(413, "That file is over 10 MB. Upload a small image tile.")
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
        except Exception:
            raise ApiError(400, "That file isn't an image we can read. Upload a PNG or JPEG.") from None
        return await run_in_threadpool(_classified, image)

    @app.get("/api/classify/local/{tile_id}")
    def classify_local(tile_id: str):
        from PIL import Image

        path = srv.tile_path("local", tile_id)
        try:
            with Image.open(path) as im:
                image = im.convert("RGB")
        except OSError:
            raise ApiError(404, "That tile's image file can't be read.") from None
        return _classified(image)

    # -- region explorer -------------------------------------------------------------------
    @app.post("/api/region/estimate")
    async def region_estimate(request: Request):
        bbox = _bbox_from(await _json_body(request))
        g = geo.grid_estimate(bbox)
        n = g["n_tiles"]
        ok = n <= MAX_TILES
        cached = ok and geo.cached_region_for_bbox(bbox) is not None
        eta = 2 if cached else int(round(8 + 0.06 * n))
        msg = (f"{n:,} tiles of 640 m ({g['width_km']} × {g['height_km']} km)."
               + (" Saved copy available." if cached else "") if ok
               else f"That area is {n:,} tiles; the limit here is {MAX_TILES:,}. Draw a smaller box.")
        return {"rows": g["rows"], "cols": g["cols"], "n_tiles": n, "width_km": g["width_km"],
                "height_km": g["height_km"], "max_tiles": MAX_TILES, "ok": ok, "eta_s": eta, "message": msg}

    @app.post("/api/region")
    async def region_start(request: Request):
        body = await _json_body(request)
        bbox = _bbox_from(body)
        when = body.get("when")
        if when is not None and not isinstance(when, str):
            raise ApiError(400, "when must be null, \"YYYY\" or \"YYYY-MM\".")
        msg = geo.available()
        if msg:
            raise ApiError(503, msg)
        est = geo.grid_estimate(bbox)  # fail fast on bad / oversized boxes
        if est["n_tiles"] > MAX_TILES:
            raise ApiError(422, f"That area is {est['n_tiles']:,} tiles; the limit here is {MAX_TILES:,}. Draw a smaller box.")
        geo._when_range(when)  # validates the date text (GeoError -> 422)
        job = srv.new_job("region")
        srv.run_job(job, srv.region_job, bbox, when)
        return {"job": job["id"]}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        return srv.job_view(job_id)

    @app.get("/api/regions/cached")
    def regions_cached():
        return geo.list_cached()

    @app.get("/api/region/{rid}")
    def region_get(rid: str):
        return srv.region_entry(rid)["summary"]

    @app.get("/api/region/{rid}/image.png")
    def region_image(rid: str):
        if not geo._valid_id(rid):
            raise ApiError(404, "No region with that id.")
        return _png_response(geo.rgb_png(srv._geo_region(rid)), IMMUTABLE_CACHE)

    @app.get("/api/region/{rid}/tile/{i}.png")
    def region_tile(rid: str, i: int):
        if not geo._valid_id(rid):
            raise ApiError(404, "No region with that id.")
        region = srv._geo_region(rid)
        if not 0 <= i < region.n_tiles:
            raise ApiError(404, f"Tile {i} is outside this region (0 to {region.n_tiles - 1}).")
        return _png_response(_png_bytes(region.tile_image(i)), IMMUTABLE_CACHE)

    def _score_payload(entry: dict, scores: np.ndarray, k: int, query: Any, exclude: int | None = None) -> dict:
        valid = analysis.valid_tiles(entry["summary"]["tiles"])
        s = np.asarray(scores, np.float64)
        vs = s[valid & np.isfinite(s)]
        return {"query": query, "scores": [analysis._r(v, 4) for v in s],
                "top": analysis.top_k(s, k, valid, exclude=exclude),
                "min": analysis._r(vs.min(), 4) if vs.size else None,
                "max": analysis._r(vs.max(), 4) if vs.size else None}

    @app.get("/api/region/{rid}/search")
    def region_search(rid: str, q: str | None = None, k: str | None = None):
        entry = srv.region_entry(rid)
        text = _query_text(q)
        kk = _int_param(k, "k", 8, 1, REGION_K_MAX)
        with srv.model_lock:
            scores = analysis.text_scores(srv.backend, text, entry["aligned"])
        return _score_payload(entry, scores, kk, text)

    @app.get("/api/region/{rid}/similar")
    def region_similar(rid: str, i: str | None = None, k: str | None = None):
        entry = srv.region_entry(rid)
        n = len(entry["aligned"])
        if i in (None, ""):
            raise ApiError(400, "Pick a tile: i is required.")
        ii = _int_param(i, "i", 0, 0, n - 1)
        kk = _int_param(k, "k", 8, 1, REGION_K_MAX)
        return _score_payload(entry, analysis.image_scores(entry["aligned"], ii), kk, ii, exclude=ii)

    @app.post("/api/region/{rid}/climate")
    def region_climate(rid: str):
        entry = srv.region_entry(rid)  # 404 before starting a job
        job = srv.new_job("climate")
        if entry.get("climate") is not None:
            srv.say_for(job)("cache", "Using the saved climate layers")
            with srv.state_lock:
                job["result"], job["status"] = entry["climate"], "done"
        else:
            srv.run_job(job, srv.climate_job, rid)
        return {"job": job["id"]}

    @app.get("/api/region/{rid}/layer/{name}.png")
    def region_layer(rid: str, name: str):
        if name not in ("ndvi", "lst"):
            raise ApiError(404, "Layers are 'ndvi' and 'lst'.")
        return _png_response(srv.layer_png(rid, name), "no-cache")

    @app.get("/api/region/{rid}/export.csv")
    def region_csv(rid: str):
        entry = srv.region_entry(rid)
        text = analysis.export_csv(entry["summary"], entry.get("climate"))
        return Response(content=text, media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="platonic-earth-{rid}.csv"'})

    @app.get("/api/region/{rid}/export.geojson")
    def region_geojson(rid: str):
        from app import DISCLAIMER

        entry = srv.region_entry(rid)
        data = analysis.export_geojson(entry["summary"], entry.get("climate"), DISCLAIMER)
        return Response(content=json.dumps(data), media_type="application/geo+json",
                        headers={"Content-Disposition": f'attachment; filename="platonic-earth-{rid}.geojson"'})

    @app.get("/api/geocode")
    def geocode(q: str | None = None):
        text = (q or "").strip()
        if not text or len(text) > 200:
            return []
        key = text.lower()
        if key in srv.geocode_cache:
            return srv.geocode_cache[key]
        try:
            import requests

            r = requests.get(GEOCODE_URL, params={"format": "jsonv2", "limit": 5, "q": text},
                             headers=GEOCODE_HEADERS, timeout=6)
            r.raise_for_status()
            out = []
            for it in r.json():
                s, n, w, e = (float(v) for v in it["boundingbox"])
                out.append({"name": it.get("display_name") or it.get("name"), "lat": float(it["lat"]),
                            "lon": float(it["lon"]), "bbox": [w, s, e, n],
                            "type": it.get("addresstype") or it.get("type")})
        except Exception:
            log.info("geocode failed for %r", text)
            return []
        srv.geocode_cache[key] = out
        return out

    # -- the original Gradio demo ---------------------------------------------------------
    if mount_classic:
        try:
            import gradio as gr
            from app import build_ui

            app = gr.mount_gradio_app(app, build_ui(backend, root), path="/classic")
        except Exception:
            log.exception("could not mount the Gradio demo at /classic; the site still works")
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Platonic Earth website")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--mock", action="store_true", help="Force mock mode even if artifacts exist")
    parser.add_argument("--online", action="store_true", help="Allow model downloads from the Hugging Face hub")
    parser.add_argument("--no-classic", action="store_true", help="Don't mount the Gradio demo at /classic")
    args = parser.parse_args()
    if not args.online:  # same as app.py: models come from the local cache (python app.py --prefetch)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    logging.basicConfig(level=logging.INFO, format="[web] %(levelname)s %(message)s")

    import uvicorn
    from app import load_backend

    backend = load_backend(ROOT, args.mock)
    application = create_app(backend, ROOT, mount_classic=not args.no_classic)
    shown = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    print(f"[web] {backend.name} mode — http://{shown}:{args.port}/", flush=True)
    uvicorn.run(application, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
