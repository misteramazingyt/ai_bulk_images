"""JSONL record store + content-addressed image cache + run manifests.

Adequate for pilot scale (<~1M records). Swap for Parquet/DuckDB later while
keeping the ImageRecord schema as the interchange format."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Iterator

from .models import ImageRecord


class Store:
    def __init__(self, root: str | Path = "data"):
        self.root = Path(root)
        self._id_cache: dict[str, set[str]] = {}
        (self.root / "records").mkdir(parents=True, exist_ok=True)
        (self.root / "images").mkdir(parents=True, exist_ok=True)
        (self.root / "runs").mkdir(parents=True, exist_ok=True)

    def _path(self, platform: str) -> Path:
        return self.root / "records" / f"{platform}.jsonl"

    def platforms(self) -> list[str]:
        return sorted(p.stem for p in (self.root / "records").glob("*.jsonl"))

    def iter_records(self, platform: str | None = None) -> Iterator[ImageRecord]:
        for plat in [platform] if platform else self.platforms():
            path = self._path(plat)
            if not path.exists():
                continue
            with path.open() as f:
                for line in f:
                    if line.strip():
                        yield ImageRecord.model_validate_json(line)

    def load(self, platform: str) -> dict[str, ImageRecord]:
        out: dict[str, ImageRecord] = {}
        for rec in self.iter_records(platform):
            if rec.record_id in out:
                out[rec.record_id].merge(rec)
            else:
                out[rec.record_id] = rec
        return out

    def _ids(self, platform: str) -> set[str]:
        """Record ids on disk (cached; kept current by upsert/write)."""
        if platform not in self._id_cache:
            ids = set()
            path = self._path(platform)
            if path.exists():
                with path.open() as f:
                    for line in f:
                        if line.strip():
                            ids.add(json.loads(line)["record_id"])
            self._id_cache[platform] = ids
        return self._id_cache[platform]

    def upsert(self, records: Iterable[ImageRecord]) -> int:
        """Append new records; merge (full rewrite) only when an id already exists."""
        by_platform: dict[str, list[ImageRecord]] = {}
        for r in records:
            by_platform.setdefault(r.source.platform, []).append(r)
        n = 0
        for plat, recs in by_platform.items():
            ids = self._ids(plat)
            if any(r.record_id in ids for r in recs) or len({r.record_id for r in recs}) < len(recs):
                existing = self.load(plat)
                for r in recs:
                    if r.record_id in existing:
                        existing[r.record_id].merge(r)
                    else:
                        existing[r.record_id] = r
                self.write(plat, existing.values())
            else:
                with self._path(plat).open("a") as f:
                    for r in recs:
                        f.write(r.model_dump_json(exclude_none=True) + "\n")
                ids.update(r.record_id for r in recs)
            n += len(recs)
        return n

    def write(self, platform: str, records: Iterable[ImageRecord]) -> None:
        path = self._path(platform)
        tmp = path.with_suffix(".jsonl.tmp")
        with tmp.open("w") as f:
            for r in records:
                f.write(r.model_dump_json(exclude_none=True) + "\n")
        os.replace(tmp, path)
        self._id_cache.pop(platform, None)

    def cache_image(self, sha: str, raw: bytes, ext: str = "") -> str:
        d = self.root / "images" / sha[:2]
        d.mkdir(exist_ok=True)
        p = d / (sha + ext)
        if not p.exists():
            p.write_bytes(raw)
        return str(p)

    def write_manifest(self, run_id: str, manifest: dict) -> Path:
        p = self.root / "runs" / f"{run_id}.json"
        p.write_text(json.dumps(manifest, indent=2, default=str))
        return p


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]
