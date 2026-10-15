"""Content hashing and near-duplicate clustering (METHODOLOGY §5.4)."""

from __future__ import annotations

import hashlib
import io
from collections import defaultdict
from typing import Iterable

import imagehash
from PIL import Image


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def perceptual_hashes(raw: bytes) -> tuple[str, str, int, int, str | None]:
    """Return (phash_hex, dhash_hex, width, height, mime)."""
    img = Image.open(io.BytesIO(raw))
    img.load()
    mime = Image.MIME.get(img.format) if img.format else None
    w, h = img.size
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return str(imagehash.phash(img)), str(imagehash.dhash(img)), w, h, mime


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def cluster(items: Iterable[tuple[str, str | None, str | None]], max_distance: int = 8) -> dict[str, str]:
    """Single-linkage clustering of (record_id, sha256, phash) triples.

    Exact SHA-256 matches always merge; pHashes merge at Hamming <= max_distance.
    Candidate pairs come from splitting the 64 bits into max_distance+1 bands:
    by pigeonhole, any pair within the threshold agrees exactly on >= 1 band,
    so the search is exact. Returns record_id -> cluster_id.
    """
    items = list(items)
    uf = _UnionFind()
    by_sha: dict[str, str] = {}
    hashed: list[tuple[str, int]] = []
    for rid, sha, ph in items:
        uf.find(rid)
        if sha:
            if sha in by_sha:
                uf.union(rid, by_sha[sha])
            else:
                by_sha[sha] = rid
        if ph:
            hashed.append((rid, int(ph, 16)))

    n_bands = max_distance + 1
    edges = [round(64 * k / n_bands) for k in range(n_bands + 1)]
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, (_, h) in enumerate(hashed):
        for band in range(n_bands):
            lo, hi = edges[band], edges[band + 1]
            buckets[(band, (h >> lo) & ((1 << (hi - lo)) - 1))].append(idx)
    seen: set[tuple[int, int]] = set()
    for members in buckets.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pair = (members[i], members[j])
                if pair in seen:
                    continue
                seen.add(pair)
                a, b = hashed[pair[0]], hashed[pair[1]]
                if bin(a[1] ^ b[1]).count("1") <= max_distance:
                    uf.union(a[0], b[0])

    roots = {rid: uf.find(rid) for rid, _, _ in items}
    # Stable, opaque cluster ids derived from the root record id.
    return {rid: "cl_" + hashlib.sha1(root.encode()).hexdigest()[:12] for rid, root in roots.items()}
