"""DiffusionDB 2M (Wang et al. 2023; CC0-1.0) - generation-side corpus G.

Reads the 2M-row metadata.parquet, applies an optional codebook-lexicon
pre-screen plus a seeded random comparison sample (METHODOLOGY §2.2), and
reads individual PNGs out of the remote part-zips via HTTP Range requests.
"""

from __future__ import annotations

import random
import re
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import pandas as pd

from .. import codebook as cbmod
from ..http import HTTPRangeFile
from ..models import Criterion, DetectionEvidence, ImageRecord, PostInfo, Rights
from .base import Collector

HF = "https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main"
SAMPLERS = {1: "ddim", 2: "plms", 3: "k_euler", 4: "k_euler_ancestral", 5: "k_heun",
            6: "k_dpm_2", 7: "k_dpm_2_ancestral", 8: "k_lms", 9: "others"}
_LOCATOR = re.compile(r"^hf-zip://diffusiondb/part-(\d{6})/(.+)$")


class DiffusionDBCollector(Collector):
    name = "diffusiondb"
    platform = "diffusiondb"

    def _metadata(self) -> pd.DataFrame:
        cache = Path(self.frame.params.get("cache_dir", "data/cache")) / "diffusiondb_metadata.parquet"
        if not cache.exists():
            cache.parent.mkdir(parents=True, exist_ok=True)
            r = self.session.get(f"{HF}/metadata.parquet", stream=True, timeout=600)
            r.raise_for_status()
            with cache.open("wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        return pd.read_parquet(cache)

    def collect(self) -> Iterator[ImageRecord]:
        p = self.frame.params
        df = self._metadata()
        df = df[df["user_name"].notna()]  # rows with None were deleted by their creators
        nsfw_max = float(p.get("max_nsfw", 0.5))
        df = df[(df["image_nsfw"] < nsfw_max) & (df["prompt_nsfw"] < nsfw_max)]

        rng = random.Random(self.frame.seed)
        mode = p.get("mode", "prescreen+random")  # "random" | "prescreen" | "prescreen+random"
        selected: list[tuple[str, pd.Series]] = []
        if "prescreen" in mode:
            cb = cbmod.load()
            terms = sorted({t.lower() for c in cb.visual_motifs for t in c.lexicon})
            pattern = r"\b(?:" + "|".join(re.escape(t) for t in terms) + r")\b"
            hits = df[df["prompt"].str.lower().str.contains(pattern, regex=True, na=False)]
            n = self.frame.limit if mode == "prescreen" else self.frame.limit // 2
            idx = rng.sample(range(len(hits)), min(n, len(hits)))
            selected += [("prescreen", hits.iloc[i]) for i in idx]
        if "random" in mode:
            n = self.frame.limit - len(selected)
            idx = rng.sample(range(len(df)), min(n, len(df)))
            selected += [("random", df.iloc[i]) for i in idx]

        for stratum, row in selected:
            name = row["image_name"]
            ts = row["timestamp"]
            yield ImageRecord(
                record_id=f"diffusiondb:{name}:0",
                source=self.source_kwargs(name, None, f"stratum={stratum}"),
                post=PostInfo(created_at=None if pd.isna(ts) else ts.to_pydatetime(), text=row["prompt"],
                              nsfw=bool(row["image_nsfw"] >= 0.5)),
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

    @lru_cache(maxsize=8)
    def _zip(self, part: str) -> zipfile.ZipFile:
        import io

        return zipfile.ZipFile(io.BufferedReader(HTTPRangeFile(self.session, f"{HF}/images/part-{part}.zip"), 1 << 16))

    def fetch_bytes(self, media_url: str) -> bytes:
        m = _LOCATOR.match(media_url)
        if not m:
            return super().fetch_bytes(media_url)
        return self._zip(m.group(1)).read(m.group(2))
