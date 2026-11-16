#!/usr/bin/env python
"""Prefetch the raw OSM provider PBFs needed by the OSM-Sentinel builder.

This is deliberately separate from build_dataset.py. It resolves the provider
extracts pyrosm would use for each region, downloads unique URLs concurrently,
and seeds pyrosm's provider cache. The builder can then run with
``--require-osm-cache`` and do no OSM downloads.

Examples:
    python experiments/osm_sentinel/prefetch_osm.py --dry-run
    python experiments/osm_sentinel/prefetch_osm.py --workers 4 --per-provider 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
THIS = Path(__file__).resolve().parent
for path in (ROOT, THIS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_dataset import read_regions, region_geometry  # noqa: E402
from osm_source import (  # noqa: E402
    provider_cache_path,
    provider_for_url,
    resolve_provider_urls,
)

STATUS_VERSION = 1
CHUNK = 256 * 1024


def _urls_for_region(region: dict[str, Any], args) -> list[str]:
    _, _, _, bbox = region_geometry(region, args)
    return resolve_provider_urls(bbox, args.cache_dir)


def _load_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": STATUS_VERSION, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "regions": {}, "urls": {}, "failed": []}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        data = {}
    data.setdefault("regions", {})
    data.setdefault("urls", {})
    data.setdefault("failed", [])
    return data


def _write_status(path: Path, status: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        status["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(status, indent=1, sort_keys=True) + "\n")
        tmp.replace(path)


def _download_one(url: str, args, status: dict[str, Any], status_lock: threading.Lock) -> str:
    """Download one unique provider URL to pyrosm's expected cache path."""
    import requests

    target = provider_cache_path(args.cache_dir, url)
    if target.exists() and target.stat().st_size > 0:
        with status_lock:
            status["urls"][url] = {"file": str(target), "bytes": target.stat().st_size, "status": "cached"}
        return "cached"

    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    if not args.resume and part.exists():
        part.unlink()

    for attempt in range(args.retries + 1):
        start = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "veritas-osm-prefetch/1.0"}
        if start:
            headers["Range"] = f"bytes={start}-"
        try:
            with requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=(20, args.timeout),
            ) as response:
                if response.status_code == 416 and start:
                    os.replace(part, target)
                    return "cached"
                if response.status_code == 200:
                    # The server ignored Range; restart cleanly.
                    start = 0
                elif response.status_code != 206 and start:
                    response.raise_for_status()
                elif response.status_code not in (200, 206):
                    response.raise_for_status()

                mode = "ab" if start else "wb"
                written = start
                with part.open(mode) as handle:
                    for chunk in response.iter_content(CHUNK):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        written += len(chunk)
                expected = response.headers.get("Content-Length")
                if expected is not None and not start:
                    try:
                        if written != int(expected):
                            raise IOError(f"size mismatch: wrote {written}, expected {expected}")
                    except ValueError:
                        pass
                os.replace(part, target)
                with status_lock:
                    status["urls"][url] = {
                        "file": str(target),
                        "bytes": written,
                        "status": "complete",
                        "provider": provider_for_url(url),
                    }
                print(f"[prefetch] complete {target.name} ({written / 1e6:.1f} MB)", flush=True)
                return "complete"
        except Exception as exc:
            if attempt >= args.retries:
                with status_lock:
                    status["failed"].append({"url": url, "error": f"{type(exc).__name__}: {exc}"})
                    status["urls"][url] = {"file": str(target), "status": "failed",
                                           "error": f"{type(exc).__name__}: {exc}"}
                print(f"[prefetch] FAILED {url}: {type(exc).__name__}: {exc}", flush=True)
                return "failed"
            wait = 5 * (3 ** attempt)
            print(f"[prefetch] retry {url} in {wait}s: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(wait)
    return "failed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel OSM provider PBF prefetch")
    parser.add_argument("--regions-csv", default=str(THIS / "regions.csv"))
    parser.add_argument("--region", action="append", help="Only this region_id (repeatable)")
    parser.add_argument("--radius-km", type=float, default=None)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "raw" / "osm_sentinel_1m_640" / "_cache"))
    parser.add_argument("--status-json", default=None)
    parser.add_argument("--workers", type=int, default=4, help="Total concurrent downloads")
    parser.add_argument("--per-provider", type=int, default=3, help="Concurrent downloads per provider")
    parser.add_argument("--timeout", type=float, default=120.0, help="Read timeout seconds")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--no-resume", action="store_true", help="Discard existing .part files")
    parser.add_argument("--dry-run", action="store_true", help="Resolve URLs only; download nothing")
    args = parser.parse_args()

    if args.workers < 1 or args.per_provider < 1:
        raise SystemExit("--workers and --per-provider must be positive")
    if args.per_provider > args.workers:
        args.per_provider = args.workers

    args.cache_dir = Path(args.cache_dir)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.resume = not args.no_resume
    status_path = Path(args.status_json) if args.status_json else args.cache_dir / "prefetch_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status = _load_status(status_path)
    status_lock = threading.Lock()

    regions = read_regions(Path(args.regions_csv))
    if args.region:
        wanted = set(args.region)
        regions = [region for region in regions if region["region_id"] in wanted]
    if not regions:
        raise SystemExit("No regions selected")

    print(f"[prefetch] resolving {len(regions)} regions", flush=True)
    region_urls: dict[str, list[str]] = {}
    all_urls: set[str] = set()
    for region in regions:
        rid = region["region_id"]
        try:
            urls = _urls_for_region(region, args)
        except Exception as exc:
            print(f"[prefetch] resolve failed {rid}: {type(exc).__name__}: {exc}", flush=True)
            with status_lock:
                status["regions"][rid] = {"status": "resolve_failed",
                                          "error": f"{type(exc).__name__}: {exc}"}
            continue
        urls = list(dict.fromkeys(urls))
        region_urls[rid] = urls
        all_urls.update(urls)
        print(f"[prefetch] {rid}: {len(urls)} provider extract(s)", flush=True)
        with status_lock:
            status["regions"][rid] = {
                "urls": urls,
                "status": "ready" if urls and all(
                    (p := provider_cache_path(args.cache_dir, url)).exists() and p.stat().st_size > 0
                    for url in urls
                ) else "pending",
            }

    _write_status(status_path, status, status_lock)
    if not all_urls:
        print("[prefetch] nothing to download", flush=True)
        return

    by_provider: Counter[str] = Counter()
    for url in all_urls:
        try:
            by_provider[provider_for_url(url)] += 1
        except ValueError as exc:
            print(f"[prefetch] {exc}", flush=True)
    print(f"[prefetch] unique URLs by provider: {dict(by_provider)}", flush=True)
    if args.dry_run:
        for url in sorted(all_urls):
            print(f"[prefetch] {provider_for_url(url)} {provider_cache_path(args.cache_dir, url)} <- {url}")
        return

    provider_locks: dict[str, threading.Semaphore] = {}
    provider_locks_lock = threading.Lock()

    def provider_semaphore(provider: str) -> threading.Semaphore:
        with provider_locks_lock:
            if provider not in provider_locks:
                provider_locks[provider] = threading.Semaphore(args.per_provider)
            return provider_locks[provider]

    def run(url: str) -> tuple[str, str]:
        try:
            provider = provider_for_url(url)
        except ValueError as exc:
            return url, f"failed: {exc}"
        with provider_semaphore(provider):
            return url, _download_one(url, args, status, status_lock)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="osm-prefetch") as pool:
        futures = {pool.submit(run, url): url for url in sorted(all_urls)}
        for future in as_completed(futures):
            url, outcome = future.result()
            print(f"[prefetch] {outcome}: {url}", flush=True)
            _write_status(status_path, status, status_lock)

    elapsed = time.perf_counter() - started
    failed = [item["url"] for item in status.get("failed", [])]
    print(f"[prefetch] done in {elapsed:.1f}s; failed={len(failed)}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
