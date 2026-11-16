#!/usr/bin/env python
"""Fit one OSM-captioned rotation on every captioned chip and save the .npz.

This is the production-only counterpart to train.py:

    python experiments/osm_sentinel/train_rotation.py \
        --data ~/veritas/data/raw/osm_sentinel_1m_640 \
        --out artifacts/alignment/osm_sentinel_dinov2-small_all.npz

No test split, no validation split, and no benchmark suites are evaluated here.
Every captioned row in the manifest is used to fit the single Procrustes map.
The result is the portable artifact consumed with src.alignment.load_procrustes.

If you need to keep Upper Arlington out of the fit, pass
`--exclude-region us_columbus`; otherwise all regions are included.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
THIS = Path(__file__).resolve().parent
for path in (ROOT, THIS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train import BACKBONE, encode_captions, embed, load_rows, use_device_preprocess  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit one Procrustes rotation on all captioned OSM chips; write only a .npz."
    )
    parser.add_argument(
        "--data",
        default=str(ROOT / "data" / "raw" / "osm_sentinel_1m_640"),
        help="Dataset directory containing manifest.csv and captions.jsonl",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output .npz path (default: artifacts/alignment/osm_sentinel_<backbone>_<tag>.npz)",
    )
    parser.add_argument("--tag", default="all_data", help="Cache namespace and default output name")
    parser.add_argument("--device", default=None, help="Override autodetection (cuda/mps/xpu/cpu)")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16", "auto"])
    parser.add_argument("--batch-size", type=int, default=None, help="Default: 512 on CUDA, else 32")
    parser.add_argument("--num-workers", type=int, default=None, help="Default: min(8, CPUs) on CUDA, else 4")
    parser.add_argument("--prefetch-factor", type=int, default=None, help="Batches per worker; default 4 on CUDA, else 2")
    parser.add_argument(
        "--preprocess",
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Resize/normalize with PIL in workers (cpu) or on the device (gpu); auto = gpu on CUDA/XPU, cpu on MPS/CPU",
    )
    parser.add_argument("--compile", action="store_true", help="torch.compile the image model (opt-in)")
    parser.add_argument("--force", action="store_true", help="Recompute cached image/text embeddings")
    parser.add_argument(
        "--exclude-region",
        action="append",
        default=[],
        help="Drop every row from this region before fitting. Default: use every region.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from src.alignment import fit_procrustes, orthogonality_error, save_procrustes
    from src.config import load_config
    from src.encoders import device_summary, get_device, load_dinov2, load_minilm

    config = load_config(str(ROOT / "configs" / "default.yaml"))
    args.input_size = int(config["image"]["input_size"])
    args.device_resolved = get_device(args.device)
    on_cuda = args.device_resolved.startswith("cuda")
    if args.batch_size is None:
        args.batch_size = 512 if on_cuda else 32
    if args.num_workers is None:
        args.num_workers = min(8, os.cpu_count() or 4) if on_cuda else 4
    if args.prefetch_factor is None:
        args.prefetch_factor = 4 if on_cuda else 2

    excluded = {region for region in args.exclude_region if region}
    rows = load_rows(Path(args.data))
    if excluded:
        rows = [row for row in rows if row.get("region_id") not in excluded]
    if not rows:
        raise SystemExit("No captioned rows available to fit")

    out = Path(args.out) if args.out else (
        ROOT / "artifacts" / "alignment" / f"osm_sentinel_{BACKBONE}_{args.tag}.npz"
    )
    if out.suffix != ".npz":
        out = out.with_suffix(".npz")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"backbone={BACKBONE} device={device_summary(args.device_resolved)} "
        f"dtype={args.dtype} batch={args.batch_size} workers={args.num_workers} "
        f"preprocess={'device' if use_device_preprocess(args) else 'pil'} "
        f"rows={len(rows)} regions={len({row['region_id'] for row in rows})}",
        flush=True,
    )
    if excluded:
        print(f"excluded regions: {sorted(excluded)}", flush=True)

    model_label = config["image"]["model_name"]
    model = load_dinov2(model_label, device=args.device_resolved, compile=args.compile)
    text_model = load_minilm(config["text"]["model_name"], device=args.device_resolved)

    print(f"embedding all {len(rows)} images", flush=True)
    x = embed(model, rows, BACKBONE, "all", args)
    text_unique, text_index = encode_captions(
        text_model, [row["caption"] for row in rows], config["text"]["model_name"], args
    )
    print(
        f"fitting Procrustes: {x.shape[0]} images x {text_unique.shape[0]} unique captions",
        flush=True,
    )
    fit = fit_procrustes(
        x,
        text_unique,
        center=bool(config["alignment"]["center"]),
        y_index=text_index,
    )
    fit.update(
        backbone=BACKBONE,
        model_name=model_label,
        text_model=config["text"]["model_name"],
        data=str(Path(args.data)),
        tag=args.tag,
        fit_scope="all captioned chips (no split filter)",
        n_fit=len(rows),
        n_images=len(rows),
        n_unique_captions=int(text_unique.shape[0]),
        n_regions=int(len({row["region_id"] for row in rows})),
        n_countries=int(len({row["country"] for row in rows})),
        excluded_regions=sorted(excluded),
        input_size=args.input_size,
        device=device_summary(args.device_resolved),
        dtype=args.dtype,
        image_preprocess="device" if use_device_preprocess(args) else "pil",
        created_utc=datetime.now(timezone.utc).isoformat(),
        orthogonality_error=round(orthogonality_error(fit), 6),
    )
    save_procrustes(out, fit)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)
    print(f"sha256 {digest}", flush=True)


if __name__ == "__main__":
    main()
