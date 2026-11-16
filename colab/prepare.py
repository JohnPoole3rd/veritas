"""Put the shipped, already-fitted artifacts where the website backend reads them. Nothing is trained here.

    python colab/prepare.py            # then: PE_ROTATION=colab/artifacts/osm_1m_rotation.npz python -m web.server

- colab/artifacts/osm_1m_rotation.npz : the OSM-Sentinel rotation fitted on the RTX 5060 Ti
  (829,057 image-caption pairs, all non-test rows of the 1M build; sha256 a47275412f7a...)
- dinov2_test.npy, minilm_captions.npy, minilm_caption_index.json : precomputed EuroSAT test-image and
  class-caption embeddings the backend needs to start in REAL mode (from the main pipeline).
"""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "colab" / "artifacts"
FILES = {
    "dinov2_test.npy": ROOT / "artifacts" / "embeddings" / "dinov2_test.npy",
    "minilm_captions.npy": ROOT / "artifacts" / "embeddings" / "minilm_captions.npy",
    "minilm_caption_index.json": ROOT / "artifacts" / "embeddings" / "minilm_caption_index.json",
}
ROTATION_SHA256 = "a47275412f7abf660bd5f8f55298124a7ae5781b90fe98e6576edd26cc500cde"


def main() -> None:
    rot = SRC / "osm_1m_rotation.npz"
    got = hashlib.sha256(rot.read_bytes()).hexdigest()
    if got != ROTATION_SHA256:
        sys.exit(f"osm_1m_rotation.npz checksum mismatch: {got}")
    for name, dst in FILES.items():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(SRC / name, dst)
        print(f"ok  {dst.relative_to(ROOT)}")
    print(f"ok  rotation {rot.relative_to(ROOT)} (sha256 {got[:12]}...)")


if __name__ == "__main__":
    main()
