import zipfile
from collections import Counter
from types import SimpleNamespace

import pandas as pd
import pytest

from aibi.collectors import Frame
from aibi.collectors.diffusiondb import DiffusionDBCollector, DiskSpaceLow
from aibi.models import Corpus
from aibi.pipeline import fetch_one
from aibi.storage import Store

from .test_core import rec, scene


def fake_metadata(parts=(1, 2, 3), per_part=4):
    rows = []
    for p in parts:
        for i in range(per_part):
            rows.append(dict(image_name=f"p{p}-{i}.png", prompt=f"prompt {p}-{i}", part_id=p, seed=1, step=50, cfg=7.0,
                             sampler=8, width=128, height=128, user_name="u" if i != 3 else None,
                             timestamp=pd.Timestamp("2022-08-10", tz="UTC"), image_nsfw=0.1, prompt_nsfw=0.1))
    return pd.DataFrame(rows)


def make_collector(tmp_path, limit, **params):
    frame = Frame(frame_id="G-test", source="diffusiondb", corpus=Corpus.GENERATION, limit=limit, seed=3,
                  params={"mode": "parts", "cache_dir": str(tmp_path / "cache"), **params})
    c = DiffusionDBCollector(frame, session=SimpleNamespace())
    c._metadata = lambda: fake_metadata()
    downloads = []

    def fake_download(url, dest):
        part = int(url.rsplit("part-", 1)[1].split(".")[0])
        downloads.append(part)
        with zipfile.ZipFile(dest, "w") as z:
            for i in range(4):
                z.writestr(f"p{part}-{i}.png", scene(part * 10 + i))
            z.writestr(f"part-{part:06d}.json", "{}")

    c._download = fake_download
    c.part_order = lambda: [3, 1, 2] + [p for p in range(4, 2001)]
    return c, downloads


def test_parts_mode_takes_whole_parts_and_cleans_up(tmp_path):
    c, downloads = make_collector(tmp_path, limit=100)
    store = Store(tmp_path / "data")
    counts = Counter()
    recs = []
    for r in c.collect():
        fetch_one(c, r, store, counts)  # consumer fetches each record as it is yielded, like the pipeline
        recs.append(r)
    assert downloads == [3, 1, 2]
    assert len(recs) == 9  # 3 parts x 3 eligible (one deleted-by-user row per part excluded)
    assert counts["fetched"] == 9 and counts.get("fetch_errors", 0) == 0
    assert recs[0].source.query == "cluster=part-000003"
    assert all(r.image.sha256 and r.image.local_path for r in recs)
    assert not list((tmp_path / "cache" / "parts").glob("*.zip"))  # zips deleted after use


def test_parts_mode_respects_limit(tmp_path):
    c, downloads = make_collector(tmp_path, limit=4)
    store, counts = Store(tmp_path / "data"), Counter()
    recs = [r for r in c.collect() if fetch_one(c, r, store, counts) is None]
    assert len(recs) == 4 and downloads == [3, 1]


def test_disk_guard(tmp_path, monkeypatch):
    c, _ = make_collector(tmp_path, limit=100, min_free_gb=10**6)
    with pytest.raises(DiskSpaceLow):
        next(iter(c.collect()))


def test_store_appends_without_rewrite(tmp_path, monkeypatch):
    store = Store(tmp_path)
    store.upsert([rec("a"), rec("b")])
    monkeypatch.setattr(store, "write", lambda *a, **k: (_ for _ in ()).throw(AssertionError("rewrite")))
    store.upsert([rec("c")])  # new id: append only
    assert [r.record_id for r in store.iter_records()] == ["a", "b", "c"]
    monkeypatch.undo()
    store.upsert([rec("a")])  # existing id: merge path
    assert sorted(store.load("reddit")) == ["a", "b", "c"]


def test_failed_part_is_skipped_and_recorded(tmp_path):
    c, downloads = make_collector(tmp_path, limit=100)
    real = c._download

    def flaky(url, dest):
        if "part-000001" in url:
            raise ConnectionResetError("reset by peer")
        real(url, dest)

    c._download = flaky
    store, counts = Store(tmp_path / "data"), Counter()
    recs = [r for r in c.collect() if fetch_one(c, r, store, counts) is None]
    assert len(recs) == 6
    assert c.notes["parts_used"] == ["000003", "000002"] and "000001" in c.notes["parts_skipped"]


def test_download_retries_and_cleans_partial(tmp_path, monkeypatch):
    import aibi.collectors.diffusiondb as mod

    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    calls = []

    class Resp:
        def raise_for_status(self):
            pass

        def iter_content(self, n):
            yield b"partial"
            if len(calls) < 3:
                raise ConnectionResetError("reset")
            yield b"-done"

    frame = Frame(frame_id="t", source="diffusiondb", corpus=Corpus.GENERATION, params={"cache_dir": str(tmp_path)})
    c = DiffusionDBCollector(frame, session=SimpleNamespace(get=lambda *a, **k: calls.append(1) or Resp()))
    dest = tmp_path / "x.zip"
    c._download("u", dest)
    assert dest.read_bytes() == b"partial-done" and len(calls) == 3

    class Bad(Resp):
        def iter_content(self, n):
            yield b"x"
            raise ConnectionResetError("reset")

    c.session = SimpleNamespace(get=lambda *a, **k: Bad())
    with pytest.raises(ConnectionResetError):
        c._download("u", tmp_path / "y.zip")
    assert not list(tmp_path.glob("*.part"))


def test_exclude_parts_continues_order(tmp_path):
    c, downloads = make_collector(tmp_path, limit=100, exclude_parts=["000003", 2])
    store, counts = Store(tmp_path / "data"), Counter()
    recs = [r for r in c.collect() if fetch_one(c, r, store, counts) is None]
    assert downloads == [1] and len(recs) == 3
