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

FAILURE_EXAMPLE = {
    "query": "permanent crop",
    "expected": "PermanentCrop",
    "got": "AnnualCrop",
    "why": (
        "At 64×64 pixels, orchards and seasonal fields look similar. The difference "
        "depends on the crop lifecycle, which a single tile does not show."
    ),
}


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

    def search_text(self, query: str, k: int = TOP_K) -> list[Hit]:
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


def _transforms():
    """Use J's src.alignment transforms so fit and inference match exactly.

    Fallback mirrors plan/03 §7 (normalize → subtract train mean → rotate → normalize)
    and is only used if src/ is not importable yet.
    """
    try:
        from src.alignment import transform_images, transform_texts  # type: ignore
        return transform_images, transform_texts, "src.alignment"
    except Exception:
        def t_img(X, p):
            return _l2((_l2(X) - p["mu_x"]) @ p["W"])

        def t_txt(Y, p):
            return _l2(_l2(Y) - p["mu_y"])

        return t_img, t_txt, "app.py fallback (verify against src.alignment!)"


class RealBackend:
    """Loads precomputed artifacts (plan/INTERFACES.md §4). No fitting at demo time.

    Gallery = EuroSAT test split. Text queries need MiniLM; uploads need DINOv2.
    Both load lazily from the local Hugging Face cache, so pre-download them before demo day.
    `text_encoder` / `image_encoder` can be injected (callables → np.ndarray) for testing.
    """

    name = "REAL"

    def __init__(self, root: Path, text_encoder=None, image_encoder=None):
        self.root = root
        art = root / "artifacts"
        p = np.load(art / "alignment" / "procrustes.npz", allow_pickle=False)
        self.params = {"W": p["W"], "mu_x": p["mu_x"], "mu_y": p["mu_y"]}
        raw_cfg = None
        if "metadata" in p.files:
            raw_cfg = p["metadata"]
        elif "config" in p.files:
            raw_cfg = p["config"]
        cfg = json.loads(str(raw_cfg)) if raw_cfg is not None else {}
        self.t_img, self.t_txt, self.transform_src = _transforms()

        split = json.loads((root / "data" / "splits" / "test.json").read_text())
        self.items = split["items"]
        self.eurosat_root = root / cfg.get("eurosat_root", "data/raw/eurosat")
        X = np.load(art / "embeddings" / "dinov2_test.npy")
        self.subset_note = None
        if len(X) != len(self.items):
            if len(X) > len(self.items):
                raise ValueError(
                    f"dinov2_test.npy has {len(X)} rows but test.json has only {len(self.items)} items"
                )
            # J may have produced a stratified subset. The extraction order matches
            # the split order, so the first N test items are the matching gallery.
            self.items = self.items[: len(X)]
            self.subset_note = f"{len(X)}-tile stratified subset"
        self.gallery = self.t_img(X, self.params)

        caps = np.load(art / "embeddings" / "minilm_captions.npy")
        index = json.loads((art / "embeddings" / "minilm_caption_index.json").read_text())
        cap_labels = [index[str(i)]["label"] for i in range(len(caps))]
        Yt = self.t_txt(caps, self.params)
        self.class_names = sorted(set(cap_labels))
        lab = np.array(cap_labels)
        self.centroids = _l2(np.stack([Yt[lab == c].mean(axis=0) for c in self.class_names]))

        self._text_encoder = text_encoder
        self._image_encoder = image_encoder
        self.model_names = {
            "text": cfg.get("text_model", "sentence-transformers/all-MiniLM-L6-v2"),
            "image": cfg.get("image_model", "facebook/dinov2-small"),
        }

    # -- encoders (lazy) --------------------------------------------------
    def _encode_text(self, text: str) -> np.ndarray:
        if self._text_encoder is None:
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer(self.model_names["text"])
            self._text_encoder = lambda t: m.encode([t], convert_to_numpy=True)
        return np.asarray(self._text_encoder(text), dtype=np.float32).reshape(1, -1)

    def _encode_image(self, image: Image.Image) -> np.ndarray:
        if self._image_encoder is None:
            import torch
            from transformers import AutoImageProcessor, AutoModel
            proc = AutoImageProcessor.from_pretrained(self.model_names["image"])
            model = AutoModel.from_pretrained(self.model_names["image"]).eval()

            def enc(img):
                with torch.no_grad():
                    out = model(**proc(images=img.convert("RGB"), return_tensors="pt"))
                return out.last_hidden_state[:, 0].numpy()  # CLS, matches T04 contract

            self._image_encoder = enc
        return np.asarray(self._image_encoder(image), dtype=np.float32).reshape(1, -1)

    # -- UI API -------------------------------------------------------------
    def _tile(self, i: int) -> Image.Image:
        path = self.eurosat_root / self.items[i]["path"]
        try:
            return Image.open(path).convert("RGB").resize((192, 192), Image.NEAREST)
        except OSError:
            return make_placeholder_tile(self.items[i]["label"], i)

    def search_text(self, query: str, k: int = TOP_K) -> list[Hit]:
        q = self.t_txt(self._encode_text(query), self.params)[0]
        sims = self.gallery @ q
        top = np.argsort(-sims)[:k]
        return [Hit(self._tile(i), self.items[i]["label"], round(float(sims[i]), 2), self.items[i]["tile_id"]) for i in top]

    def classify_image(self, image: Image.Image) -> list[tuple[str, float]]:
        x = self.t_img(self._encode_image(image), self.params)[0]
        sims = self.centroids @ x
        order = np.argsort(-sims)[:TOP_K]
        return [(self.class_names[i], float(sims[i])) for i in order]


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
    except Exception as exc:
        print(f"[demo] MOCK mode — real artifacts failed to load: {type(exc).__name__}: {exc}")
        return MockBackend()
    print(f"[demo] REAL mode — gallery N={len(backend.items)}, transforms from {backend.transform_src}")
    if backend.subset_note:
        print(f"[demo] NOTE — using {backend.subset_note}")
    return backend


# ---------------------------------------------------------------------------
# Metrics (reads real CSVs from A/J when present; placeholders otherwise)
# ---------------------------------------------------------------------------

def _read_csv(name: str, root: Path = REPO_ROOT) -> list[dict]:
    path = root / "artifacts" / "metrics" / name
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _fmt(v: str | None) -> str:
    if v in (None, ""):
        return "—"
    try:
        return f"{float(v):.1%}"
    except ValueError:
        return v


def metrics_table(root: Path = REPO_ROOT) -> tuple[list[list[str]], str]:
    rows: list[list[str]] = []
    sources = []
    subset_ns: list[int] = []

    majority = _read_csv("majority.csv", root)
    resnet = _read_csv("resnet18.csv", root)
    retrieval = {r.get("method"): r for r in _read_csv("retrieval.csv", root)}

    def add(method, top1, top5, f1, n, src):
        rows.append([method, _fmt(top1), _fmt(top5), _fmt(f1), n or "—"])
        if src:
            sources.append(src)
        if n:
            try:
                subset_ns.append(int(float(n)))
            except ValueError:
                pass

    m = majority[0] if majority else {}
    add("Majority class", m.get("accuracy"), None, m.get("macro_f1"), m.get("test_n"), "majority.csv" if m else None)
    r = resnet[0] if resnet else {}
    add("ResNet-18 (supervised)", r.get("accuracy"), None, r.get("macro_f1"), r.get("test_n"), "resnet18.csv" if r else None)
    for key, label in [("unaligned", "Unaligned cosine"), ("random_rotation", "Random rotation"), ("procrustes", "Procrustes (ours)")]:
        x = retrieval.get(key, {})
        n = x.get("test_n") or x.get("n_images")
        add(label, x.get("top1"), x.get("top5"), x.get("macro_f1"), n, "retrieval.csv" if x else None)

    if sources:
        note = f"Loaded from `artifacts/metrics/` ({', '.join(sorted(set(sources)))}). Held-out EuroSAT test split."
        if subset_ns and min(subset_ns) < 4050:
            note += f" The smallest reported N is {min(subset_ns)}; this is a stratified subset, not the full 4,050-tile test split."
    else:
        note = "**No metrics yet.** Waiting on T07 (J) and T08/T09 (A). Do not quote numbers from this screen."
    return rows, note


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui(backend, root: Path = REPO_ROOT) -> gr.Blocks:
    mock_banner = (
        "> ⚠️ **MOCK MODE** — placeholder tiles and fake scores for UI rehearsal only."
        if backend.name == "MOCK" else ""
    )

    def on_search(query: str):
        query = (query or "").strip()
        if not query:
            return [], "Type a land-cover description, e.g. *dense forest*."
        try:
            hits = backend.search_text(query)
        except Exception as e:  # e.g. MiniLM not cached and no internet
            return [], f"Search failed: `{type(e).__name__}: {e}`"
        gallery = [(h.image, f"{h.label} · {h.score:.2f} · {h.tile_id}") for h in hits]
        caption = (
            f"Top {len(hits)} tiles for **“{query}”**. Scores are cosine similarity, "
            "not probability."
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
            "_Local Upper Arlington tiles are outside the EuroSAT distribution; treat this as a hint._\n\n"
            f"_{DISCLAIMER}_"
        )

    rows, note = metrics_table(root)

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
            gr.Examples(EXAMPLE_QUERIES, inputs=query)
            caption = gr.Markdown()
            gallery = gr.Gallery(label="Top results", columns=TOP_K, height="auto", object_fit="contain")
            btn.click(on_search, query, [gallery, caption])
            query.submit(on_search, query, [gallery, caption])

        with gr.Tab("Classify a tile"):
            with gr.Row():
                upload = gr.Image(label="Upload a Sentinel-2 tile", type="pil")
                verdict = gr.Markdown("Upload a tile to classify it.")
            upload.change(on_upload, upload, verdict)

        with gr.Tab("Results & limits"):
            gr.Markdown("### Baseline comparison")
            gr.Dataframe(
                value=rows,
                headers=["Method", "Top-1 / Accuracy", "Top-5", "Macro-F1", "Test N"],
                interactive=False,
            )
            gr.Markdown(note)
            gr.Markdown(
                "ResNet-18 is trained on the exact class labels, so it is expected to win at "
                "fixed-class classification. Our method answers new text queries without retraining.\n\n"
                "### Known failure\n"
                f"**Query:** `{FAILURE_EXAMPLE['query']}` → top result **{FAILURE_EXAMPLE['got']}** "
                f"(expected {FAILURE_EXAMPLE['expected']}).\n\n{FAILURE_EXAMPLE['why']}"
            )
            with gr.Row():
                gr.Image(make_placeholder_tile("PermanentCrop", 0), label="Expected: PermanentCrop", interactive=False)
                gr.Image(make_placeholder_tile("AnnualCrop", 0), label="Returned: AnnualCrop", interactive=False)

        gr.Markdown(
            "<small>Frozen DINOv2 ViT-S/14 + frozen all-MiniLM-L6-v2, aligned by a closed-form "
            "orthogonal Procrustes rotation fit on EuroSAT. Imagery: Copernicus Sentinel-2. "
            "AI coding assistants were used to build this demo.</small>"
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Platonic Earth demo")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--mock", action="store_true", help="Force mock mode even if artifacts exist")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="Repo root holding data/ and artifacts/")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link (needs internet)")
    args = parser.parse_args()
    build_ui(load_backend(args.root, args.mock), args.root).launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
