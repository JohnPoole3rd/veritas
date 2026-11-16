"""Put the shipped, already-fitted artifacts where the website backend reads them. Nothing is trained here.

    python colab/prepare.py            # then: PE_ROTATION=colab/artifacts/osm_1m_rotation.npz python -m web.server

- colab/artifacts/osm_1m_rotation.npz : the OSM-Sentinel rotation fitted on the RTX 5060 Ti
  (all 1,000,000 image-caption pairs of the 1M build, 331 regions, 117 countries; sha256 358c5b94aa58...)
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
ROTATION_SHA256 = "358c5b94aa58708bc73d935c6ae82656c1ce05d51be9bb15e8a67ab88b1f699d"


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
