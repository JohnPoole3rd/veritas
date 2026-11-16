import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("pyrosm")

from experiments.osm_sentinel import osm_source, prefetch_osm
from experiments.osm_sentinel.osm_source import (
    provider_cache_name,
    provider_cache_path,
    provider_for_url,
    require_provider_cache,
    resolve_provider_urls,
)

GEO = "https://download.geofabrik.de/north-america/us/ohio-latest.osm.pbf"
BBB = "https://download.bbbike.org/osm/bbbike/Berlin/Berlin.osm.pbf"
MOV = "https://osm.download.movisda.io/admin/US-OH-latest.osm.pbf"


def test_provider_cache_names_match_pyrosm():
    assert provider_for_url(GEO) == "geofabrik"
    assert provider_cache_name(GEO) == "geofabrik_ohio-latest.osm.pbf"
    assert provider_cache_name(BBB) == "bbbike_Berlin.osm.pbf"
    assert provider_cache_name(MOV) == "movisda_US-OH-latest.osm.pbf"
    with pytest.raises(ValueError):
        provider_for_url("https://example.com/foo.osm.pbf")


def test_resolve_provider_urls_normalizes_single_and_list(monkeypatch):
    import pyrosm

    monkeypatch.setattr(pyrosm, "get_data_by_bbox", lambda *a, **k: GEO)
    assert resolve_provider_urls((0, 0, 1, 1)) == [GEO]

    monkeypatch.setattr(pyrosm, "get_data_by_bbox", lambda *a, **k: [GEO, BBB])
    assert resolve_provider_urls((0, 0, 1, 1)) == [GEO, BBB]


def test_require_provider_cache_checks_all_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(osm_source, "resolve_provider_urls", lambda *a, **k: [GEO, BBB])
    geo = provider_cache_path(tmp_path, GEO)
    bbb = provider_cache_path(tmp_path, BBB)
    geo.parent.mkdir(parents=True)
    geo.write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="bbbike"):
        require_provider_cache((0, 0, 1, 1), tmp_path)
    bbb.write_bytes(b"y")
    assert require_provider_cache((0, 0, 1, 1), tmp_path) == [geo, bbb]


class _FakeResponse:
    def __init__(self, chunks, status=200):
        self.status_code = status
        self.headers = {"Content-Length": str(sum(len(c) for c in chunks))}
        self._chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_content(self, chunk_size):
        yield from self._chunks

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


def test_download_one_writes_expected_pyrosm_path(tmp_path, monkeypatch):
    import requests

    chunks = [b"abc", b"def"]
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(chunks))
    args = SimpleNamespace(cache_dir=tmp_path, resume=True, retries=0, timeout=10)
    status = {"urls": {}, "failed": []}
    outcome = prefetch_osm._download_one(GEO, args, status, threading.Lock())
    target = provider_cache_path(tmp_path, GEO)
    assert outcome == "complete"
    assert target.read_bytes() == b"abcdef"
    assert status["urls"][GEO]["status"] == "complete"
    assert not target.with_name(target.name + ".part").exists()


def test_download_one_skips_existing_file(tmp_path, monkeypatch):
    import requests

    target = provider_cache_path(tmp_path, GEO)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"already")
    called = []
    monkeypatch.setattr(requests, "get", lambda *a, **k: called.append(1))
    args = SimpleNamespace(cache_dir=tmp_path, resume=True, retries=0, timeout=10)
    status = {"urls": {}, "failed": []}
    assert prefetch_osm._download_one(GEO, args, status, threading.Lock()) == "cached"
    assert not called
