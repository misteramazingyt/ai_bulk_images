"""Pipeline stages: collect -> fetch (bytes, hashes, embedded provenance) -> cluster."""

from __future__ import annotations

import logging
import platform as _platform
import uuid
from collections import Counter
from typing import Any

from . import __version__
from .collectors import REGISTRY, Frame
from .hashing import cluster, perceptual_hashes, sha256
from .http import PoliteSession
from .models import Corpus, ImageContent, utcnow
from .provenance import inspect_bytes
from .storage import Store, config_hash

log = logging.getLogger("aibi")
EXT = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif"}


def make_session(cfg: dict) -> PoliteSession:
    http = cfg.get("http", {})
    return PoliteSession(user_agent=http.get("user_agent") or PoliteSession().s.headers["User-Agent"],
                         min_interval=float(http.get("min_interval_seconds", 1.0)),
                         per_host_interval=http.get("per_host_interval_seconds", {}))


def frames_from_config(cfg: dict) -> list[Frame]:
    return [Frame(frame_id=f["frame_id"], source=f["source"], corpus=Corpus(f["corpus"]),
                  params=f.get("params", {}), limit=int(f.get("limit", 100)), seed=int(f.get("seed", 0)))
            for f in cfg.get("frames", [])]


def collect(cfg: dict, store: Store, frame_ids: list[str] | None = None, fetch: bool = True) -> dict[str, Any]:
    run_id = utcnow().strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    session = make_session(cfg)
    manifest: dict[str, Any] = {
        "run_id": run_id, "started_at": utcnow(), "aibi_version": __version__, "python": _platform.python_version(),
        "config_hash": config_hash(cfg), "frames": [], "errors": [],
    }
    for frame in frames_from_config(cfg):
        if frame_ids and frame.frame_id not in frame_ids:
            continue
        collector = REGISTRY[frame.source](frame, session=session, run_id=run_id)
        counts: Counter = Counter()
        batch = []
        try:
            for rec in collector.collect():
                counts["collected"] += 1
                if fetch and rec.image.media_url and not rec.image.sha256:
                    fetch_one(collector, rec, store, counts)
                batch.append(rec)
                if len(batch) >= 200:
                    store.upsert(batch)
                    batch = []
        except Exception as e:  # record and continue with other frames
            log.exception("frame %s failed", frame.frame_id)
            manifest["errors"].append({"frame_id": frame.frame_id, "error": repr(e)})
        store.upsert(batch)
        manifest["frames"].append({"frame_id": frame.frame_id, "source": frame.source, "corpus": frame.corpus.value,
                                   "params": frame.params, "limit": frame.limit, "seed": frame.seed,
                                   "collector_version": collector.version, "counts": dict(counts),
                                   **({"notes": collector.notes} if getattr(collector, "notes", None) else {})})
    manifest["finished_at"] = utcnow()
    store.write_manifest(run_id, manifest)
    return manifest


def fetch_one(collector, rec, store: Store, counts: Counter) -> None:
    try:
        raw = collector.fetch_bytes(rec.image.media_url)
        ph, dh, w, h, mime = perceptual_hashes(raw)
        sha = sha256(raw)
        local = store.cache_image(sha, raw, EXT.get(mime or "", "")) if rec.rights.may_cache_bytes else None
        rec.image = ImageContent(media_url=rec.image.media_url, sha256=sha, phash=ph, dhash=dh, mime=mime,
                                 width=w, height=h, n_bytes=len(raw), local_path=local, fetched_at=utcnow())
        for ev in inspect_bytes(raw):
            rec.add_evidence(ev)
        counts["fetched"] += 1
    except Exception as e:
        rec.image.fetch_error = repr(e)[:300]
        counts["fetch_errors"] += 1


def run_clustering(store: Store, max_distance: int) -> dict[str, int]:
    all_recs = {p: store.load(p) for p in store.platforms()}
    triples = [(r.record_id, r.image.sha256, r.image.phash) for recs in all_recs.values() for r in recs.values()]
    assignment = cluster(triples, max_distance=max_distance)
    for plat, recs in all_recs.items():
        for r in recs.values():
            r.cluster_id = assignment.get(r.record_id)
        store.write(plat, recs.values())
    sizes = Counter(assignment.values())
    return {"records": len(assignment), "clusters": len(sizes), "multi_member_clusters": sum(v > 1 for v in sizes.values())}
