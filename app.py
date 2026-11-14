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

REPO_ROOT = Path(__file__).resolve().parent
TOP_K = 3

EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

DISCLAIMER = "Prototype, not a land-use determination. Do not use for permitting, taxation, or legal decisions."

EXAMPLE_QUERIES = [
    "dense forest",
    "residential neighborhood",
    "park or grassland",
    "commercial area",
    "river corridor",
    "open water",
]

# Failure probes. The Results tab runs these live through the active backend and shows what
# actually comes back, so the demo never claims a failure the model did not make.
FAILURE_PROBES = [
    {
        "query": "climate change",
        "why": (
            "“Climate change” is not a land-cover class, so there is nothing to match. The tool still "
            "returns tiles, with low scores. It cannot say “I don't know,” and A found (T12) that scores "
            "don't separate right from wrong answers either."
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


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


class RealBackend:
    """Loads J's precomputed artifacts. No fitting at demo time.

    Everything numeric goes through J's own src/ code, so the demo cannot drift from T04–T07:
    src.alignment (load_procrustes, transform_images/texts), src.encoders (load_dinov2,
    extract_image_embeddings, load_minilm, encode_texts) and src.data.get_transforms.

    Two search galleries:
    - "local": A's 50 Upper Arlington tiles (data/local_tiles/manifest.csv, T10/T11). Their DINOv2
      embeddings are computed once with A's src.local_tiles._embed_images (J's encoder + transforms)
      and cached in artifacts/embeddings/dinov2_local.npy (gitignored).
    - "eurosat": the EuroSAT test tiles that have embeddings (all 4,050 after a full-split run; the
      first N if J's --limit was used).
    Upload classification uses A's rule from T12 (agg=max: a class scores its best-matching caption),
    so it matches the logged per-image numbers.
    Text queries need MiniLM; uploads need DINOv2. Both load lazily from the local
    Hugging Face cache (run `python app.py --prefetch` once while online). `text_encoder` / `image_encoder`
    can be injected (callables → np.ndarray) for testing.
    """

    name = "REAL"

    def __init__(self, root: Path, text_encoder=None, image_encoder=None):
        from src.alignment import load_procrustes, transform_images, transform_texts
        from src.config import load_config, resolve_path

        self.root = root
        self.config = load_config(str(root / "configs" / "default.yaml"))
        art = root / "artifacts"
        self.params = load_procrustes(art / "alignment" / "procrustes.npz")
        self.t_img, self.t_txt = transform_images, transform_texts

        split = json.loads((root / "data" / "splits" / "test.json").read_text())
        X = np.load(art / "embeddings" / "dinov2_test.npy")
        if len(X) > len(split["items"]):
            raise ValueError(f"dinov2_test.npy has {len(X)} rows but test.json has only {len(split['items'])} items")
        self.items = split["items"][: len(X)]  # J's --limit takes the first N test items
        self.full_test_n = len(split["items"])
        self.eurosat_root = resolve_path(self.config["data"]["eurosat_root"], self.config)
        self.gallery = self.t_img(X, self.params)

        caps = np.load(art / "embeddings" / "minilm_captions.npy")
        index = json.loads((art / "embeddings" / "minilm_caption_index.json").read_text())
        cap_labels = [index[str(i)]["label"] for i in range(len(caps))]
        Yt = self.t_txt(caps, self.params)
        self.class_names = sorted(set(cap_labels))
        lab = np.array(cap_labels)
        self.caption_aligned = Yt
        self.caption_class_idx = np.array([self.class_names.index(c) for c in cap_labels])

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
        manifest_path = root / "data" / "local_tiles" / "manifest.csv"
        if not manifest_path.exists():
            return None
        from src.local_tiles import _embed_images, load_local_manifest

        df = load_local_manifest(manifest_path)
        cache = root / "artifacts" / "embeddings" / "dinov2_local.npy"
        ids_path = cache.with_suffix(".ids.json")
        ids = df.tile_id.tolist()
        if cache.exists() and ids_path.exists() and json.loads(ids_path.read_text()) == ids:
            X = np.load(cache)
        elif self._image_encoder is not None:  # tests: injected encoder, no caching
            X = np.concatenate([self._encode_image(Image.open(root / p)) for p in df.path])
        else:
            print("[demo] embedding 50 Upper Arlington tiles once (cached afterwards)…")
            model, device = self._dinov2_model()
            X = np.asarray(_embed_images(model, df, int(self.config["image"]["input_size"]), device), np.float32)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, X)
            ids_path.write_text(json.dumps(ids))
        return {
            "items": [
                {"tile_id": r.tile_id, "path": r.path, "label": r.label, "notes": r.notes}
                for r in df.itertuples()
            ],
            "gallery": self.t_img(X, self.params),
        }

    # -- UI API -------------------------------------------------------------
    def tile_image(self, path: Path, size: int = 192) -> Image.Image:
        try:
            return Image.open(path).convert("RGB").resize((size, size), Image.NEAREST)
        except OSError:
            return make_placeholder_tile("Residential", 0)

    def search_text(self, query: str, k: int = TOP_K, gallery: str = "local") -> list[Hit]:
        q = self.t_txt(self._encode_text(query), self.params)[0]
        if gallery == "local" and self.local is not None:
            items, G, base = self.local["items"], self.local["gallery"], self.root
        else:
            items, G, base = self.items, self.gallery, self.eurosat_root
        sims = G @ q
        top = np.argsort(-sims)[:k]
        return [Hit(self.tile_image(base / items[i]["path"]), items[i]["label"], round(float(sims[i]), 2),
                    items[i]["tile_id"]) for i in top]

    def classify_image(self, image: Image.Image) -> list[tuple[str, float]]:
        x = self.t_img(self._encode_image(image), self.params)[0]
        cap_sims = self.caption_aligned @ x
        scores = np.full(len(self.class_names), -np.inf)
        np.maximum.at(scores, self.caption_class_idx, cap_sims)  # agg=max, as in A's T12
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
    backend = RealBackend(root)
    n_local = len(backend.local["items"]) if backend.local else 0
    print(f"[demo] REAL mode — Upper Arlington N={n_local}; EuroSAT test N={len(backend.items)} of {backend.full_test_n}")
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
    if not maj:
        mj = root / "artifacts" / "metrics" / "majority.json"
        maj = json.loads(mj.read_text()) if mj.exists() else {}
    res = (_read_csv("resnet18.csv", root) or [{}])[0]
    cc = {r["method"]: r for r in _read_csv("caption_classification_test.csv", root)}
    rows = [
        ["Majority class (always guess one class)", _fmt(maj.get("accuracy")), _fmt(maj.get("macro_f1")), maj.get("test_n", "—")],
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
                "either model. It only fits one rotation. Sources: majority.csv, resnet18.csv, "
                "caption_classification_test.csv (T08, T09, T12).",
    })

    # 2) Text search: what only the rotation can do.
    ret = _read_csv("retrieval.csv", root)
    tables.append({
        "title": "2 · Text search (ResNet-18 can't do this)",
        "headers": ["Method", "Top-1", "Top-5", "Caption queries", "Tiles searched"],
        "rows": [[{"procrustes": "Procrustes rotation (ours)", "random_rotation": "Random rotation (control)",
                   "unaligned": "No rotation (control)"}.get(r["method"], r["method"]), _fmt(r.get("top1")), _fmt(r.get("top5")), r.get("n_captions", "—"),
                  r.get("n_images", "—")] for r in ret],
        "note": "Each of the 50 class captions is a search; Top-1 = the best tile has the right class. "
                "Source: retrieval.csv (T07).",
    })

    # 3) Upper Arlington, out of distribution.
    loc = _read_csv("local.csv", root)
    names = {"procrustes": "Procrustes rotation (ours)", "resnet18": "ResNet-18",
             "majority_local": "Always guess “Residential”"}
    tables.append({
        "title": "3 · Upper Arlington tiles (never used for fitting)",
        "headers": ["Method", "Accuracy (all)", "N", "Accuracy (clear-cut tiles)", "N"],
        "rows": [[names.get(r["method"], r["method"]), _fmt(r.get("accuracy")), r.get("n", "—"),
                  _fmt(r.get("accuracy_unambiguous")), r.get("n_unambiguous", "—")] for r in loc],
        "note": "31 of 50 local tiles are Residential, so guessing “Residential” already scores 62%. "
                "Both models beat that by 1 tile, and both miss almost every river tile. "
                "A small case study, not a benchmark. Source: local.csv (T12).",
    })
    return [t for t in tables if t["rows"]]


def local_examples(root: Path = REPO_ROOT) -> list[dict]:
    """A's demo picks (T13): success ua_011, failure ua_049, with their logged predictions."""
    preds = {r["tile_id"]: r for r in _read_csv("local_predictions.csv", root)}
    picks = [
        ("ua_011", "Success", "Big-box retail. Both models get it right."),
        ("ua_049", "Failure", "The Scioto River channel. At 10 m per pixel, a thin river looks like a road, so "
                              "the rotation says Highway. River→Highway is also its #1 mistake on EuroSAT "
                              "(32% of river tiles)."),
    ]
    out = []
    for tid, kind, why in picks:
        r = preds.get(tid)
        path = root / "data" / "raw" / "local_tiles" / f"{tid}.png"
        if r and path.exists():
            out.append({"tile_id": tid, "kind": kind, "why": why, "path": path, "pred": r})
    return out


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

GALLERIES = {"Upper Arlington (50 local tiles)": "local", "EuroSAT test tiles (Europe)": "eurosat"}


def build_ui(backend, root: Path = REPO_ROOT) -> gr.Blocks:
    mock_banner = (
        "> ⚠️ **MOCK MODE** — placeholder tiles and fake scores for UI rehearsal only."
        if backend.name == "MOCK" else ""
    )
    has_local = backend.name == "MOCK" or getattr(backend, "local", None) is not None
    gallery_choices = list(GALLERIES) if has_local else [list(GALLERIES)[1]]

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
            f"Top {len(hits)} tiles for **“{query}”** in *{where}*. Labels are the human labels; scores are "
            "cosine similarity, not probability."
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
            "_Upper Arlington tiles are outside the EuroSAT (Europe) training data; treat this as a hint._\n\n"
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
            gr.Examples(EXAMPLE_QUERIES, inputs=query)
            caption = gr.Markdown()
            gallery = gr.Gallery(label="Top results", columns=TOP_K, height="auto", object_fit="contain")
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
                        gr.Image(backend.tile_image(e["path"], 256) if backend.name == "REAL"
                                 else Image.open(e["path"]).resize((256, 256), Image.NEAREST),
                                 label=f"{e['kind']}: {e['tile_id']}", interactive=False)
                        gr.Markdown(
                            f"**Human label:** {r['label']}  \n"
                            f"**Ours:** {r['procrustes_pred']} (top 3: {r['procrustes_top3'].replace('|', ', ')})  \n"
                            f"**ResNet-18:** {r.get('resnet18_pred', '—')} (p={r.get('resnet18_prob', '—')})\n\n"
                            f"{e['why']}"
                        )

            gr.Markdown("### Abstract questions (computed live)")
            for probe in FAILURE_PROBES:
                try:
                    hits = backend.search_text(probe["query"], gallery="local")
                    got = ", ".join(f"{h.label} ({h.score:.2f})" for h in hits)
                except Exception as e:
                    got = f"unavailable ({type(e).__name__})"
                gr.Markdown(f"**Query:** `{probe['query']}` → top 3: {got}\n\n{probe['why']}")

        gr.Markdown(
            "<small>Frozen DINOv2 ViT-S/14 + frozen all-MiniLM-L6-v2, aligned by a closed-form "
            "orthogonal Procrustes rotation fit on EuroSAT. Imagery: Copernicus Sentinel-2 (EuroSAT; "
            "Upper Arlington scene 2025-08-01). AI coding assistants were used to build this demo.</small>"
        )
    return demo


def prefetch(root: Path = REPO_ROOT) -> None:
    """Download MiniLM + DINOv2 so the demo runs with no internet."""
    from src.config import load_config
    from src.encoders import load_dinov2, load_minilm

    cfg = load_config(str(root / "configs" / "default.yaml"))
    load_minilm(cfg["text"]["model_name"])
    print(f"[prefetch] cached {cfg['text']['model_name']}")
    load_dinov2(cfg["image"]["model_name"], device="cpu")
    print(f"[prefetch] cached {cfg['image']['model_name']}")
    print("[prefetch] done — test offline with: HF_HUB_OFFLINE=1 python app.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Platonic Earth demo")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--mock", action="store_true", help="Force mock mode even if artifacts exist")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="Repo root holding data/ and artifacts/")
    parser.add_argument("--prefetch", action="store_true", help="Download both models into the HF cache, then exit")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link (needs internet)")
    args = parser.parse_args()
    if args.prefetch:
        prefetch(args.root)
        return
    build_ui(load_backend(args.root, args.mock), args.root).launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
