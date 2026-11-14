"""Platonic Earth — Gradio demo (T14, owner D).

Two backends behind one UI:
- MockBackend: placeholder tiles + fake scores, used while J's artifacts
  (T06/T07) and A's metrics (T08/T09/T12) do not exist. Nothing it shows is real.
- RealBackend: loads precomputed artifacts per plan/INTERFACES.md §4. Selected
  automatically once every file in REQUIRED_ARTIFACTS exists.

Run:
    python app.py              # auto: REAL if artifacts exist, else MOCK
    python app.py --mock       # force mock
    python app.py --port 7861
    python app.py --prefetch   # once, while online: cache both models for offline demo

The demo runs offline by default (HF_HUB_OFFLINE=1 unless --prefetch).

The UI only talks to the backend through `search_text` and `classify_image`;
the baseline table reads artifacts/metrics/*.csv directly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # demo must run offline

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from src.config import load_config, resolve_path  # noqa: E402
from src.local_tiles import EUROSAT_CLASSES  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent
CONFIG = load_config()
TOP_K = int(CONFIG["demo"]["top_k"])

DISCLAIMER = "Prototype, not a land-use determination. Do not use for permitting, taxation, or legal decisions."

# Clicking an example runs the search. There is no lake/pond tile in Upper Arlington, so
# "open water" is only meaningful on the EuroSAT gallery and is not offered as an example.
EXAMPLE_QUERIES = [
    "dense forest",
    "residential neighborhood",
    "park or grassland",
    "commercial area",
    "river corridor",
]

# Failure probes. The Results tab runs these live through the active backend and shows what
# actually comes back, so the demo never claims a failure the model did not make.
FAILURE_PROBES = [
    {
        "query": "climate change",
        "why": (
            "“Climate change” is not a land-cover class, so there is nothing to match, yet the tool "
            "still returns tiles. It cannot say “I don't know”, and a low score is not a reliable signal "
            "either: some real land-cover queries score just as low, and A found (T12) that scores don't "
            "separate right from wrong answers."
        ),
    },
]


# ---------------------------------------------------------------------------
# Placeholder tiles (mock only)
# ---------------------------------------------------------------------------

_PALETTE = {
    "AnnualCrop": [(186, 170, 96), (140, 160, 80)],
    "Forest": [(24, 70, 32), (40, 92, 44)],
    "HerbaceousVegetation": [(110, 140, 70), (150, 160, 90)],
    "Highway": [(120, 120, 115), (70, 110, 60)],
    "Industrial": [(170, 170, 175), (120, 120, 130)],
    "Pasture": [(120, 160, 80), (140, 175, 95)],
    "PermanentCrop": [(120, 130, 70), (160, 150, 90)],
    "Residential": [(150, 140, 130), (60, 100, 55)],
    "River": [(70, 110, 60), (40, 70, 120)],
    "SeaLake": [(25, 55, 110), (35, 70, 130)],
}


def _seed(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16)


def make_placeholder_tile(label: str, variant: int, size: int = 192) -> Image.Image:
    """Draw a deterministic, obviously-synthetic tile for a class."""
    rng = np.random.default_rng(_seed(label, str(variant)))
    base, accent = _PALETTE.get(label, [(128, 128, 128), (90, 90, 90)])
    noise = rng.normal(0, 12, (size, size, 1))
    img = np.clip(np.array(base, dtype=float) + noise, 0, 255).astype(np.uint8)
    tile = Image.fromarray(img)
    draw = ImageDraw.Draw(tile)

    if label in ("Residential", "Industrial"):
        step = 24 if label == "Residential" else 40
        for x in range(0, size, step):
            for y in range(0, size, step):
                if rng.random() < 0.7:
                    w = int(step * rng.uniform(0.4, 0.8))
                    draw.rectangle([x + 3, y + 3, x + w, y + w], fill=accent)
    elif label == "Highway":
        y = int(size * rng.uniform(0.3, 0.7))
        draw.rectangle([0, y - 8, size, y + 8], fill=(95, 95, 95))
    elif label == "River":
        pts = [(x, int(size / 2 + 30 * np.sin(x / 25 + variant))) for x in range(0, size + 1, 8)]
        draw.line(pts, fill=accent, width=18)
    elif label in ("AnnualCrop", "PermanentCrop", "Pasture"):
        stripe = 12 if label == "AnnualCrop" else 20
        for x in range(0, size, stripe * 2):
            draw.rectangle([x, 0, x + stripe, size], fill=accent)
    elif label in ("Forest", "HerbaceousVegetation"):
        for _ in range(60 if label == "Forest" else 25):
            cx, cy = rng.integers(0, size, 2)
            r = int(rng.integers(5, 14))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=accent)

    draw.rectangle([0, 0, 58, 16], fill=(0, 0, 0))
    draw.text((4, 3), "MOCK", fill=(255, 210, 0))
    return tile


def tile_image(path: Path, size: int = 192) -> Image.Image:
    """A real tile, upscaled with hard pixels; a neutral grey 'missing' tile if it can't be read."""
    try:
        return Image.open(path).convert("RGB").resize((size, size), Image.NEAREST)
    except OSError:
        tile = Image.new("RGB", (size, size), (128, 128, 128))
        ImageDraw.Draw(tile).text((8, size // 2 - 6), "image missing", fill=(255, 255, 255))
        return tile


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

_KEYWORDS = {
    "forest": "Forest", "tree": "Forest", "canopy": "Forest", "wood": "Forest",
    "residential": "Residential", "house": "Residential", "neighborhood": "Residential",
    "commercial": "Industrial", "industrial": "Industrial", "warehouse": "Industrial",
    "park": "HerbaceousVegetation", "grass": "HerbaceousVegetation", "meadow": "Pasture",
    "pasture": "Pasture", "river": "River", "stream": "River", "creek": "River",
    "water": "SeaLake", "lake": "SeaLake", "reservoir": "SeaLake",
    "highway": "Highway", "road": "Highway", "interstate": "Highway",
    "permanent": "AnnualCrop",  # deliberately reproduces the planned failure case
    "crop": "AnnualCrop", "farm": "AnnualCrop", "orchard": "PermanentCrop",
}


@dataclass
class Hit:
    image: Image.Image
    label: str
    score: float
    tile_id: str


class MockBackend:
    """Fake scores so the UI can be rehearsed. Replace in Phase 3."""

    name = "MOCK"
    tile_image = staticmethod(tile_image)

    def _class_scores(self, key: str, favored: str | None) -> list[tuple[str, float]]:
        rng = np.random.default_rng(_seed("scores", key))
        scores = {c: float(rng.uniform(0.35, 0.65)) for c in EUROSAT_CLASSES}
        if favored:
            scores[favored] = float(rng.uniform(0.75, 0.85))
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    def search_text(self, query: str, k: int = TOP_K, gallery: str = "local") -> list[Hit]:
        q = query.lower()
        favored = next((c for kw, c in _KEYWORDS.items() if kw in q), None)
        ranked = self._class_scores(q, favored)
        hits = []
        # two tiles from the top class, then the runner-up; scores strictly descending
        order = [ranked[0], (ranked[0][0], ranked[0][1] - 0.03), ranked[1]]
        prev = 1.0
        for i, (label, score) in enumerate(order[:k]):
            score = min(score, prev - 0.01)
            prev = score
            hits.append(Hit(make_placeholder_tile(label, i), label, round(score, 2), f"mock_{label[:4].lower()}_{i:02d}"))
        return hits

    def classify_image(self, image: Image.Image) -> list[tuple[str, float]]:
        arr = np.asarray(image.convert("RGB").resize((32, 32)), dtype=float)
        key = hashlib.sha256(arr.tobytes()).hexdigest()
        r, g, b = arr.reshape(-1, 3).mean(axis=0)
        favored = "SeaLake" if b > g and b > r else "Forest" if g > r * 1.15 else "Residential"
        return self._class_scores(key, favored)[:TOP_K]


class RealBackend:
    """Loads J's precomputed artifacts. No fitting at demo time.

    Everything numeric goes through src/, so the demo cannot drift from the logged runs:
    src.alignment (load_procrustes, transform_images/texts), src.encoders (load_dinov2,
    extract_image_embeddings, load_minilm, encode_texts), src.data.get_transforms,
    src.retrieval (retrieve_top_k_with_scores, class_scores) and src.local_tiles (captions, local cache).

    Two search galleries:
    - "local": A's Upper Arlington tiles (data/local_tiles/manifest.csv, T10/T11). Embeddings come from
      the validated cache shared with scripts/extract_local_embeddings.py (src.local_tiles.local_embeddings:
      tile_ids, model and input size must match) or are computed once and cached. If anything about the
      local gallery fails, only that gallery is disabled; EuroSAT search keeps working.
    - "eurosat": the EuroSAT test tiles that have embeddings (all 4,050 after a full-split run; the
      first N if J's --limit was used).
    Upload classification uses A's rule from T12 (agg=max: a class scores its best-matching caption),
    so it matches the logged per-image numbers.
    Text queries need MiniLM; uploads need DINOv2. Both load lazily from the local
    Hugging Face cache (run `python app.py --prefetch` once while online). `text_encoder` / `image_encoder`
    can be injected (callables → np.ndarray) for testing.
    """

    name = "REAL"

    def __init__(self, root: Path = REPO_ROOT, text_encoder=None, image_encoder=None):
        from src.alignment import load_procrustes, transform_images, transform_texts
        from src.local_tiles import load_caption_artifacts

        self.root = root
        self.config = CONFIG
        art = root / "artifacts"
        self.params = load_procrustes(art / "alignment" / "procrustes.npz")

        split = json.loads((root / "data" / "splits" / "test.json").read_text())
        X = np.load(art / "embeddings" / "dinov2_test.npy")
        if len(X) > len(split["items"]):
            raise ValueError(f"dinov2_test.npy has {len(X)} rows but test.json has only {len(split['items'])} items")
        self.items = split["items"][: len(X)]  # extract_embeddings.py --limit embeds the first N test items
        self.full_test_n = len(split["items"])
        self.subset_note = (f"first {len(X)} of {self.full_test_n} test tiles (--limit run)"
                            if len(X) < self.full_test_n else None)
        self.eurosat_root = resolve_path(self.config["data"]["eurosat_root"], self.config)
        self.gallery = transform_images(X, self.params)

        caps, self.caption_labels = load_caption_artifacts()  # T05 vectors + labels, length-checked
        self.captions_aligned = transform_texts(caps, self.params)
        self.class_names = list(EUROSAT_CLASSES)

        self._text_encoder = text_encoder
        self._image_encoder = image_encoder
        self._dinov2 = None
        self.local = self._load_local(root)

    # -- encoders (lazy, via J's src.encoders) -----------------------------
    def _encode_text(self, text: str) -> np.ndarray:
        if self._text_encoder is None:
            from src.encoders import encode_texts, load_minilm
            model = load_minilm(self.config["text"]["model_name"])
            self._text_encoder = lambda t: encode_texts(model, [t])
        return np.asarray(self._text_encoder(text), dtype=np.float32).reshape(1, -1)

    def _encode_image(self, image: Image.Image) -> np.ndarray:
        if self._image_encoder is None:
            from src.data import get_transforms
            from src.encoders import extract_image_embeddings
            model, device = self._dinov2_model()
            tf = get_transforms(int(self.config["image"]["input_size"]), train=False)
            self._image_encoder = lambda img: extract_image_embeddings(
                model, [tf(img.convert("RGB"))], batch_size=1, device=device
            )
        return np.asarray(self._image_encoder(image), dtype=np.float32).reshape(1, -1)

    def _dinov2_model(self):
        if self._dinov2 is None:
            from src.encoders import get_device, load_dinov2
            device = get_device()
            self._dinov2 = (load_dinov2(self.config["image"]["model_name"], device=device), device)
        return self._dinov2

    # -- Upper Arlington gallery (A's T10/T11 tiles) -----------------------
    def _load_local(self, root: Path):
        manifest_path = root / self.config["data"]["local_manifest"]
        if not manifest_path.exists():
            return None
        try:
            from src.alignment import transform_images
            from src.local_tiles import load_local_manifest, local_embeddings

            df = load_local_manifest(manifest_path)
            if self._image_encoder is not None:  # tests: injected encoder, no caching
                X = np.concatenate([self._encode_image(Image.open(root / p)) for p in df.path])
            else:
                X = local_embeddings(df, self.config, *self._dinov2_model())
            return {
                "items": [{"tile_id": r.tile_id, "path": r.path, "label": r.label, "notes": r.notes}
                          for r in df.itertuples()],
                "gallery": transform_images(X, self.params),
            }
        except Exception as exc:  # the EuroSAT gallery still works without it
            print(f"[demo] WARNING — Upper Arlington gallery disabled: {type(exc).__name__}: {exc}")
            return None

    # -- UI API -------------------------------------------------------------
    tile_image = staticmethod(tile_image)

    def search_text(self, query: str, k: int = TOP_K, gallery: str = "local") -> list[Hit]:
        from src.alignment import transform_texts
        from src.retrieval import retrieve_top_k_with_scores

        q = transform_texts(self._encode_text(query), self.params)[0]
        if gallery == "local" and self.local is not None:
            items, G, base = self.local["items"], self.local["gallery"], self.root
        else:
            items, G, base = self.items, self.gallery, self.eurosat_root
        idx, scores = retrieve_top_k_with_scores(q, G, k=k)
        return [Hit(self.tile_image(base / items[i]["path"]), items[i]["label"], round(s, 2), items[i]["tile_id"])
                for i, s in zip(idx, scores)]

    def classify_image(self, image: Image.Image) -> list[tuple[str, float]]:
        from src.alignment import transform_images
        from src.retrieval import class_scores

        x = transform_images(self._encode_image(image), self.params)
        scores = class_scores(x, self.captions_aligned, self.caption_labels, self.class_names)[0]  # agg=max (T12)
        order = np.argsort(-scores)[:TOP_K]
        return [(self.class_names[i], float(scores[i])) for i in order]


REQUIRED_ARTIFACTS = [
    "artifacts/alignment/procrustes.npz",
    "artifacts/embeddings/dinov2_test.npy",
    "artifacts/embeddings/minilm_captions.npy",
    "artifacts/embeddings/minilm_caption_index.json",
    "data/splits/test.json",
]


def load_backend(root: Path = REPO_ROOT, force_mock: bool = False):
    missing = [p for p in REQUIRED_ARTIFACTS if not (root / p).exists()]
    if force_mock or missing:
        if missing and not force_mock:
            print(f"[demo] MOCK mode — missing artifacts: {', '.join(missing)}")
        return MockBackend()
    try:
        backend = RealBackend(root)
    except Exception as exc:  # keep the demo usable even if an artifact is broken
        print(f"[demo] MOCK mode — real artifacts failed to load: {type(exc).__name__}: {exc}")
        return MockBackend()
    n_local = len(backend.local["items"]) if backend.local else 0
    print(f"[demo] REAL mode — Upper Arlington N={n_local}; EuroSAT test N={len(backend.items)} of {backend.full_test_n}")
    if backend.subset_note:
        print(f"[demo] NOTE — using {backend.subset_note}")
    return backend


# ---------------------------------------------------------------------------
# Results (read straight from logged metric files; nothing is typed in by hand)
# ---------------------------------------------------------------------------

def _read_csv(name: str, root: Path = REPO_ROOT) -> list[dict]:
    path = root / "artifacts" / "metrics" / name
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _fmt(v) -> str:
    if v in (None, ""):
        return "—"
    try:
        return f"{float(v):.1%}"
    except ValueError:
        return str(v)


def results_tables(root: Path = REPO_ROOT) -> list[dict]:
    """Three tables, each comparing like with like. Sources: T07 (J), T08/T09/T12 (A)."""
    tables = []

    # 1) EuroSAT, per-image classification: the apples-to-apples comparison with ResNet.
    maj = (_read_csv("majority.csv", root) or [{}])[0]
    res = (_read_csv("resnet18.csv", root) or [{}])[0]
    cc = {r["method"]: r for r in _read_csv("caption_classification_test.csv", root)}
    rows = [
        [f"Majority class (always “{maj.get('most_common_class', '?')}”)", _fmt(maj.get("accuracy")), _fmt(maj.get("macro_f1")), maj.get("test_n", "—")],
        ["ResNet-18 (trained on the labels)", _fmt(res.get("accuracy")), _fmt(res.get("macro_f1")), res.get("test_n", "—")],
    ]
    for key, label in [("procrustes", "Procrustes rotation (ours)"), ("random_rotation", "Random rotation (control)"),
                       ("unaligned", "No rotation (control)")]:
        r = cc.get(key, {})
        rows.append([label, _fmt(r.get("accuracy")), _fmt(r.get("macro_f1")), r.get("n_images", "—")])
    tables.append({
        "title": "1 · Label each EuroSAT test tile (same task for every row)",
        "headers": ["Method", "Accuracy", "Macro-F1", "Test N"],
        "rows": rows,
        "note": "Held-out EuroSAT test split. ResNet-18 is trained on these exact labels; ours never trains "
                "either encoder. It fits one rotation on labelled EuroSAT image–caption pairs. "
                "Sources: majority.csv, resnet18.csv, caption_classification_test.csv (T08, T09, T12).",
    })

    # 2) Text search: what only the rotation can do.
    ret = _read_csv("retrieval.csv", root)
    n_q = ret[0].get("n_captions", "?") if ret else "?"
    tables.append({
        "title": "2 · Text search (ResNet-18 can't do this)",
        "headers": ["Method", "Top-1", "Top-5", "Caption queries", "Benchmark tiles searched"],
        "rows": [[{"procrustes": "Procrustes rotation (ours)", "random_rotation": "Random rotation (control)",
                   "unaligned": "No rotation (control)"}.get(r["method"], r["method"]), _fmt(r.get("top1")), _fmt(r.get("top5")), r.get("n_captions", "—"),
                  r.get("n_images", "—")] for r in ret],
        "note": f"Each of the {n_q} class captions is a search; Top-1 = the best tile has the right class. "
                "Random rotation is the mean of several shuffles. Source: retrieval.csv (T07).",
    })

    # 3) Upper Arlington, out of distribution.
    loc = _read_csv("local_summary.csv", root)
    names = {"procrustes": "Procrustes rotation (ours)", "resnet18": "ResNet-18",
             "majority_train": "EuroSAT majority class (real baseline)",
             "oracle_majority_local": "Most common local label (oracle)"}
    by = {r["method"]: r for r in loc}
    note = "A small case study, not a benchmark. Source: local_summary.csv (T12)."
    if "oracle_majority_local" in by and "procrustes" in by:
        o, pr = by["oracle_majority_local"], by["procrustes"]
        n = int(o["n"])
        n_top = round(float(o["accuracy"]) * n)
        diff = round((float(pr["accuracy"]) - float(o["accuracy"])) * n)
        top_label = o["notes"].split("'")[1] if "'" in o["notes"] else "the most common label"
        note = (f"{n_top} of {n} local tiles are {top_label}, so always answering “{top_label}” scores "
                f"{float(o['accuracy']):.0%}. That oracle reads the test labels; it is the bar for "
                f"“better than a constant guess”. Procrustes differs from it by {diff:+d} tile(s). " + note)
    tables.append({
        "title": "3 · Upper Arlington tiles (never used for fitting)",
        "headers": ["Method", "Accuracy (all)", "N", "Accuracy (clear-cut tiles)", "N"],
        "rows": [[names.get(r["method"], r["method"]), _fmt(r.get("accuracy")), r.get("n", "—"),
                  _fmt(r.get("accuracy_unambiguous")), r.get("n_unambiguous", "—")] for r in loc],
        "note": note,
    })
    return [t for t in tables if t["rows"]]


def metrics_table(root: Path = REPO_ROOT) -> list[dict]:
    """Compatibility alias (older runbooks/integration checks); same as results_tables."""
    return results_tables(root)


def local_examples(root: Path = REPO_ROOT) -> list[dict]:
    """A's demo picks (T13): success ua_011, failure ua_049, with their logged predictions."""
    preds = {r["tile_id"]: r for r in _read_csv("local_predictions.csv", root)}
    conf = {(r["true"], r["predicted"]): r for r in _read_csv("procrustes_test_top_confusions.csv", root)}
    rh = conf.get(("River", "Highway"))
    rh_txt = f" River→Highway is also a top EuroSAT mistake ({float(rh['share_of_true']):.0%} of river tiles)." if rh else ""
    picks = [
        ("ua_011", "Success", "Big-box retail."),
        ("ua_049", "Failure", "The Scioto River channel. At 10 m per pixel a thin river looks like a road." + rh_txt),
    ]
    manifest = root / CONFIG["data"]["local_manifest"]
    paths = {}
    if manifest.exists():
        with manifest.open(newline="") as f:
            paths = {r["tile_id"]: root / r["path"] for r in csv.DictReader(f)}
    out = []
    for tid, kind, why in picks:
        r, path = preds.get(tid), paths.get(tid)
        if r and path and path.exists():
            out.append({"tile_id": tid, "kind": kind, "why": why, "path": path, "pred": r})
    return out


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

GALLERY_LOCAL, GALLERY_EUROSAT = "Upper Arlington (local tiles)", "EuroSAT test tiles (Europe)"
GALLERIES = {GALLERY_LOCAL: "local", GALLERY_EUROSAT: "eurosat"}


def build_ui(backend, root: Path = REPO_ROOT) -> gr.Blocks:
    mock_banner = (
        "> ⚠️ **MOCK MODE** — placeholder tiles and fake scores for UI rehearsal only."
        if backend.name == "MOCK" else ""
    )
    has_local = backend.name == "MOCK" or getattr(backend, "local", None) is not None
    gallery_choices = [GALLERY_LOCAL, GALLERY_EUROSAT] if has_local else [GALLERY_EUROSAT]
    if backend.name == "REAL" and backend.local:
        gallery_choices[0] = f"Upper Arlington ({len(backend.local['items'])} local tiles)"
        GALLERIES[gallery_choices[0]] = "local"
    subset = f" ({backend.subset_note})" if getattr(backend, "subset_note", None) else ""

    def on_search(query: str, where: str):
        query = (query or "").strip()
        if not query:
            return [], "Type a land-cover description, e.g. *dense forest*."
        try:
            hits = backend.search_text(query, gallery=GALLERIES.get(where, "eurosat"))
        except Exception as e:  # e.g. MiniLM not cached and no internet
            return [], f"Search failed: `{type(e).__name__}: {e}`"
        gallery = [(h.image, f"{h.label} · {h.score:.2f} · {h.tile_id}") for h in hits]
        caption = (
            f"Top {len(hits)} tiles for **“{query}”** in *{where}*{subset if GALLERIES.get(where) == 'eurosat' else ''}. "
            "Labels are the human labels; scores are cosine similarity, not probability."
        )
        return gallery, caption

    def on_upload(image):
        if image is None:
            return "Upload a tile to classify it."
        try:
            ranked = backend.classify_image(image)
        except Exception as e:  # e.g. DINOv2 not cached and no internet
            return f"Classification failed: `{type(e).__name__}: {e}`"
        (top, s), rest = ranked[0], ranked[1:]
        nxt = ", ".join(f"{l} ({v:.2f})" for l, v in rest)
        return (
            f"### Nearest label: {top} ({s:.2f})\n"
            f"Next: {nxt}\n\n"
            "_Imagery from outside EuroSAT (e.g. Upper Arlington) is out of distribution; treat this as a hint._\n\n"
            f"_{DISCLAIMER}_"
        )

    tables = results_tables(root)
    examples = local_examples(root)

    with gr.Blocks(title="Platonic Earth — Upper Arlington") as demo:
        gr.Markdown(
            "# Platonic Earth — Upper Arlington\n"
            "Search Sentinel-2 land cover in plain language, for sustainability and "
            "urban-forestry staff looking for places to inspect.\n\n"
            f"**{DISCLAIMER}**\n\n{mock_banner}"
        )

        with gr.Tab("Text search"):
            with gr.Row():
                query = gr.Textbox(label="Describe the land cover", placeholder="dense forest", scale=4)
                btn = gr.Button("Search", variant="primary", scale=1)
            where = gr.Radio(gallery_choices, value=gallery_choices[0], label="Search in")
            caption = gr.Markdown()
            gallery = gr.Gallery(label="Top results", columns=TOP_K, height="auto", object_fit="contain")
            gr.Examples([[q, gallery_choices[0]] for q in EXAMPLE_QUERIES], inputs=[query, where],
                        outputs=[gallery, caption], fn=on_search, run_on_click=True, cache_examples=False)
            btn.click(on_search, [query, where], [gallery, caption])
            query.submit(on_search, [query, where], [gallery, caption])

        with gr.Tab("Classify a tile"):
            with gr.Row():
                upload = gr.Image(label="Upload a Sentinel-2 tile (64×64, 10 m/px works best)", type="pil")
                verdict = gr.Markdown("Upload a tile to classify it.")
            if examples:
                gr.Examples([str(e["path"]) for e in examples], inputs=upload, label="Upper Arlington examples")
            upload.change(on_upload, upload, verdict)

        with gr.Tab("Results & limits"):
            if not tables:
                gr.Markdown("**No metrics yet.** Do not quote numbers from this screen.")
            for t in tables:
                gr.Markdown(f"### {t['title']}")
                gr.Dataframe(value=t["rows"], headers=t["headers"], interactive=False)
                gr.Markdown(f"<small>{t['note']}</small>")

            gr.Markdown("### One success, one failure (Upper Arlington, logged in T12)")
            with gr.Row():
                for e in examples:
                    r = e["pred"]
                    with gr.Column():
                        gr.Image(tile_image(e["path"], 256), label=f"{e['kind']}: {e['tile_id']}", interactive=False)
                        gr.Markdown(
                            f"**Human label:** {r['true_label']}  \n"
                            f"**Ours:** {r['predicted_label']} (top 3: {r['top3'].replace('|', ', ')})  \n"
                            f"**ResNet-18:** {r.get('resnet18_pred', '—')} (p={r.get('resnet18_prob', '—')})\n\n"
                            f"{e['why']}"
                        )

            gr.Markdown("### Where it fails: abstract questions (computed live)")
            probe_gallery = "local" if has_local else "eurosat"
            for probe in FAILURE_PROBES:
                try:
                    hits = backend.search_text(probe["query"], gallery=probe_gallery)
                    got = ", ".join(f"{h.label} ({h.score:.2f})" for h in hits)
                except Exception as e:
                    got = f"unavailable ({type(e).__name__})"
                where_txt = "Upper Arlington" if probe_gallery == "local" else "EuroSAT test"
                gr.Markdown(f"**Query:** `{probe['query']}` on {where_txt} → top {TOP_K}: {got}\n\n{probe['why']}")

        gr.Markdown(
            "<small>Frozen DINOv2 ViT-S/14 + frozen all-MiniLM-L6-v2, aligned by a closed-form "
            f"orthogonal Procrustes rotation fit on labelled EuroSAT pairs (encoders never trained). Imagery: "
            f"Copernicus Sentinel-2 (EuroSAT; Upper Arlington scene 2025-08-01). AI coding assistants were used "
            "to build this demo.</small>"
        )
    return demo


def prefetch() -> None:
    """Download MiniLM + DINOv2 so the demo runs with no internet."""
    from src.encoders import load_dinov2, load_minilm

    load_minilm(CONFIG["text"]["model_name"])
    print(f"[prefetch] cached {CONFIG['text']['model_name']}")
    load_dinov2(CONFIG["image"]["model_name"], device="cpu")
    print(f"[prefetch] cached {CONFIG['image']['model_name']}")
    print("[prefetch] done — `python app.py` now runs offline")


def main() -> None:
    parser = argparse.ArgumentParser(description="Platonic Earth demo")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--mock", action="store_true", help="Force mock mode even if artifacts exist")
    parser.add_argument("--prefetch", action="store_true", help="Download both models into the HF cache, then exit")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link (needs internet)")
    args = parser.parse_args()
    if args.prefetch:
        prefetch()
        return
    # Demo day: never wait on the Hugging Face hub; models come from the local cache (--prefetch).
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    build_ui(load_backend(REPO_ROOT, args.mock)).launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
