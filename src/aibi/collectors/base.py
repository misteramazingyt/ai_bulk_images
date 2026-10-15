from __future__ import annotations

import hashlib
import hmac
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator

from ..http import PoliteSession
from ..models import Corpus, ImageRecord


def pseudonymize(author_id: str | None, salt: str | None = None) -> str | None:
    """Keyed hash of an author identifier (METHODOLOGY §7). The salt must be
    kept secret and constant for the life of the project (env AIBI_SALT)."""
    if not author_id:
        return None
    key = (salt or os.environ.get("AIBI_SALT") or "").encode()
    if not key:
        raise RuntimeError("Set AIBI_SALT (a long random secret) before collecting author data")
    return "u_" + hmac.new(key, author_id.encode(), hashlib.sha256).hexdigest()[:24]


@dataclass
class Frame:
    """A sampling frame (METHODOLOGY §2.1), loaded from config.yaml."""

    frame_id: str
    source: str
    corpus: Corpus
    params: dict[str, Any] = field(default_factory=dict)
    limit: int = 100
    seed: int = 0


class Collector(ABC):
    name: str = "base"
    version: str = "0.1.0"
    platform: str = "unknown"

    def __init__(self, frame: Frame, session: PoliteSession | None = None, run_id: str | None = None):
        self.frame = frame
        self.session = session or PoliteSession()
        self.run_id = run_id

    @abstractmethod
    def collect(self) -> Iterator[ImageRecord]:
        """Yield records with media_url set; bytes are fetched by the fetch stage."""

    def fetch_bytes(self, media_url: str) -> bytes:
        r = self.session.get(media_url)
        r.raise_for_status()
        return r.content

    def source_kwargs(self, native_id: str, post_url: str | None, query: str | None) -> dict:
        return dict(
            platform=self.platform,
            corpus=self.frame.corpus,
            sampling_frame=self.frame.frame_id,
            collector=self.name,
            collector_version=self.version,
            native_id=native_id,
            post_url=post_url,
            query=query,
            run_id=self.run_id,
        )
