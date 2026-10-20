"""DiffusionDB 2M (Wang et al. 2023; CC0-1.0) - generation-side corpus G.

Reads the 2M-row metadata.parquet and selects images in one of these modes:

* `random` / `prescreen` / `prescreen+random`: image-level samples (optionally
  codebook-lexicon pre-screened, METHODOLOGY §2.2). Each image is read out of
  its remote part-zip with HTTP Range requests: ~3 s/image, fine for pilots.
* `parts`: a cluster sample. Whole part-zips (1,000 images each, ~600 MB) are
  chosen at random with the frame seed and downloaded in full, and every
  eligible image in each part is taken. The next part downloads in the
  background while the current one is processed. Throughput is ~10-20k
  images/hour. Parts were filled in generation order, so images within a part
  are temporally clustered; analyses must treat part_id as the sampling
  cluster.
"""

from __future__ import annotations

import io
import logging
import random
import re
import shutil
import threading
import time
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import pandas as pd

from .. import codebook as cbmod
from ..http import HTTPRangeFile
from ..models import Criterion, DetectionEvidence, ImageRecord, PostInfo, Rights
from .base import Collector

log = logging.getLogger("aibi.diffusiondb")
HF = "https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main"
N_PARTS = 2000
SAMPLERS = {1: "ddim", 2: "plms", 3: "k_euler", 4: "k_euler_ancestral", 5: "k_heun",
            6: "k_dpm_2", 7: "k_dpm_2_ancestral", 8: "k_lms", 9: "others"}
_LOCATOR = re.compile(r"^hf-zip://diffusiondb/part-(\d{6})/(.+)$")


class DiskSpaceLow(RuntimeError):
    pass


class DiffusionDBCollector(Collector):
    name = "diffusiondb"
    platform = "diffusiondb"
    version = "0.2.0"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.cache_dir = Path(self.frame.params.get("cache_dir", "data/cache"))
        self._local_parts: dict[str, Path] = {}
        self.notes: dict = {"parts_used": [], "parts_skipped": {}}

    def _metadata(self) -> pd.DataFrame:
        cache = self.cache_dir / "diffusiondb_metadata.parquet"
        if not cache.exists():
            cache.parent.mkdir(parents=True, exist_ok=True)
            self._download(f"{HF}/metadata.parquet", cache)
        return pd.read_parquet(cache)

    def _download(self, url: str, dest: Path, attempts: int = 3) -> None:
        """Stream to a temp file; retry mid-transfer failures; never leave partial files."""
        tmp = dest.with_suffix(dest.suffix + ".part")
        for attempt in range(1, attempts + 1):
            try:
                r = self.session.get(url, stream=True, timeout=600)
                r.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                tmp.replace(dest)
                return
            except Exception as e:
                tmp.unlink(missing_ok=True)
                if attempt == attempts:
                    raise
                log.warning("download %s failed (attempt %d/%d): %r - retrying", url, attempt, attempts, e)
                time.sleep(5 * attempt)

    def _eligible(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["user_name"].notna()]  # rows with None were deleted by their creators
        nsfw_max = float(self.frame.params.get("max_nsfw", 0.5))
        return df[(df["image_nsfw"] < nsfw_max) & (df["prompt_nsfw"] < nsfw_max)]

    def _record(self, row, stratum: str) -> ImageRecord:
        name = row["image_name"]
        ts = row["timestamp"]
        return ImageRecord(
            record_id=f"diffusiondb:{name}:0",
            source=self.source_kwargs(name, None, stratum),
            post=PostInfo(created_at=None if pd.isna(ts) else ts.to_pydatetime(), text=row["prompt"],
                          community=f"part-{int(row['part_id']):06d}", nsfw=bool(row["image_nsfw"] >= 0.5)),
            image={"media_url": f"hf-zip://diffusiondb/part-{int(row['part_id']):06d}/{name}",
                   "width": int(row["width"]), "height": int(row["height"])},
            rights=Rights(license="CC0-1.0", terms_basis="DiffusionDB CC0-1.0 dedication",
                          may_cache_bytes=True, may_redistribute_bytes=True),
            generator="Stable Diffusion (v1.x, Discord beta, 2022)",
            prompt=row["prompt"],
            ai_evidence=[DetectionEvidence(criterion=Criterion.DATASET, method="dataset:poloclub/diffusiondb",
                                           value=f"seed={row['seed']};steps={row['step']};cfg={row['cfg']};"
                                                 f"sampler={SAMPLERS.get(int(row['sampler']), '?')}")],
        )

    def collect(self) -> Iterator[ImageRecord]:
        df = self._eligible(self._metadata())
        mode = self.frame.params.get("mode", "prescreen+random")
        if mode == "parts":
            yield from self._collect_parts(df)
            return

        rng = random.Random(self.frame.seed)
        selected: list[tuple[str, pd.Series]] = []
        if "prescreen" in mode:
            cb = cbmod.load()
            terms = sorted({t.lower() for c in cb.visual_motifs for t in c.lexicon})
            pattern = r"\b(?:" + "|".join(re.escape(t) for t in terms) + r")\b"
            hits = df[df["prompt"].str.lower().str.contains(pattern, regex=True, na=False)]
            n = self.frame.limit if mode == "prescreen" else self.frame.limit // 2
            idx = rng.sample(range(len(hits)), min(n, len(hits)))
            selected += [("stratum=prescreen", hits.iloc[i]) for i in idx]
        if "random" in mode:
            n = self.frame.limit - len(selected)
            idx = rng.sample(range(len(df)), min(n, len(df)))
            selected += [("stratum=random", df.iloc[i]) for i in idx]
        for stratum, row in selected:
            yield self._record(row, stratum)

    # --- cluster sampling by whole part ---------------------------------
    def part_order(self) -> list[int]:
        parts = list(range(1, N_PARTS + 1))
        random.Random(self.frame.seed).shuffle(parts)
        return parts

    def _check_disk(self) -> None:
        min_free = float(self.frame.params.get("min_free_gb", 4)) * 1e9
        free = shutil.disk_usage(self.cache_dir).free
        if free < min_free:
            raise DiskSpaceLow(f"only {free / 1e9:.1f} GB free (< {min_free / 1e9:.0f} GB); stopping")

    def _collect_parts(self, df: pd.DataFrame) -> Iterator[ImageRecord]:
        parts_dir = self.cache_dir / "parts"
        parts_dir.mkdir(parents=True, exist_ok=True)
        by_part = {int(k): g for k, g in df.groupby("part_id")}
        # exclude_parts lets a continuation run resume the same seeded order
        # without re-collecting parts an earlier run already took.
        excluded = {int(p) for p in self.frame.params.get("exclude_parts", [])}
        order = [p for p in self.part_order() if p in by_part and p not in excluded]

        def fetch(part: int, box: dict) -> None:
            dest = parts_dir / f"part-{part:06d}.zip"
            try:
                if not dest.exists():
                    self._download(f"{HF}/images/part-{part:06d}.zip", dest)
                box["path"] = dest
            except Exception as e:  # surfaced to the consumer below
                box["error"] = e

        n = 0
        prefetch: tuple[int, threading.Thread, dict] | None = None
        for i, part in enumerate(order):
            if n >= self.frame.limit:
                break
            try:
                self._check_disk()
            except DiskSpaceLow as e:  # planned stop, not a failure
                log.info("%s", e)
                self.notes["stopped"] = str(e)
                break
            if prefetch is None:
                box: dict = {}
                t = threading.Thread(target=fetch, args=(part, box), daemon=True)
                t.start()
                prefetch = (part, t, box)
            _, t, box = prefetch
            t.join()
            prefetch = None
            if i + 1 < len(order) and n + len(by_part[part]) < self.frame.limit:
                nbox: dict = {}
                nt = threading.Thread(target=fetch, args=(order[i + 1], nbox), daemon=True)
                nt.start()
                prefetch = (order[i + 1], nt, nbox)
            if "error" in box:
                log.error("part %06d download failed: %r - skipping", part, box["error"])
                self.notes["parts_skipped"][f"{part:06d}"] = repr(box["error"])[:300]
                continue
            key = f"{part:06d}"
            self.notes["parts_used"].append(key)
            self._local_parts[key] = box["path"]
            self._zip.cache_clear()
            log.info("part %06d: %d eligible images (%d collected so far)", part, len(by_part[part]), n)
            for _, row in by_part[part].iterrows():
                if n >= self.frame.limit:
                    break
                yield self._record(row, f"cluster=part-{key}")
                n += 1
            # Everything from this part has been yielded and fetched by the consumer.
            self._local_parts.pop(key, None)
            self._zip.cache_clear()
            if not self.frame.params.get("keep_zips", False):
                box["path"].unlink(missing_ok=True)
        if prefetch is not None:  # stopped with a download in flight
            prefetch[1].join()
            if "path" in prefetch[2] and not self.frame.params.get("keep_zips", False):
                prefetch[2]["path"].unlink(missing_ok=True)

    @lru_cache(maxsize=8)
    def _zip(self, part: str) -> zipfile.ZipFile:
        local = self._local_parts.get(part)
        if local is not None:
            return zipfile.ZipFile(local)
        return zipfile.ZipFile(io.BufferedReader(HTTPRangeFile(self.session, f"{HF}/images/part-{part}.zip"), 1 << 16))

    def fetch_bytes(self, media_url: str) -> bytes:
        m = _LOCATOR.match(media_url)
        if not m:
            return super().fetch_bytes(media_url)
        return self._zip(m.group(1)).read(m.group(2))
