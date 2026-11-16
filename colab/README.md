# Run Platonic Earth in Google Colab

**Open:** [`colab/Platonic_Earth_backend.ipynb`](Platonic_Earth_backend.ipynb) in Colab (File → Open notebook → GitHub → `JohnPoole3rd/veritas`, branch `colab`), then choose *Runtime → Run all*.

It runs the real backend (`web/server.py`) and the website in Colab, using the **already-fitted OSM-Sentinel 1M rotation**. Nothing is trained.

| File | What it is |
|---|---|
| `artifacts/osm_1m_rotation.npz` | Orthogonal Procrustes rotation (W, mu_x, mu_y): 384×384, fitted on the RTX 5060 Ti on all 1,000,000 image–caption pairs of the 1M OSM-Sentinel build (331 regions, 117 countries). SHA-256 `358c5b94aa58708bc73d935c6ae82656c1ce05d51be9bb15e8a67ab88b1f699d`. The search benchmark (`results/search_benchmark_v1`) measured the earlier fit on 829,057 non-test pairs (sha `a47275412f7a…`), because a model fitted on every region cannot be scored on held-out places. |
| `artifacts/dinov2_test.npy`, `minilm_captions.npy`, `minilm_caption_index.json` | Precomputed EuroSAT test-image and class-caption embeddings that the backend needs to start in REAL mode |
| `prepare.py` | Verifies the rotation checksum and copies the embeddings to `artifacts/embeddings/` |
| `Platonic_Earth_backend.ipynb` | Get the code → check network → install → prepare → download the two frozen models → start the backend (`PE_ROTATION=colab/artifacts/osm_1m_rotation.npz`) → end-to-end test on a live area → open the website |

**Run it on your own computer instead:**

```sh
git clone --branch colab https://github.com/JohnPoole3rd/veritas.git && cd veritas
pip install -r requirements.txt
python colab/prepare.py && python app.py --prefetch
PE_ROTATION=colab/artifacts/osm_1m_rotation.npz python -m web.server --port 8000 --no-classic --online
# open http://localhost:8000/
```

Limits: the Colab link opens only in the notebook owner's browser, while the session is alive. A study area is limited to 400 squares of 640 m (a radius of 6 km or less).
