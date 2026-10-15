"""Canonical data schema. `schema/image_record.schema.json` is generated from
these models (`aibi schema`); see docs/DATA_SCHEMA.md for field semantics."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "0.1.0"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Corpus(str, Enum):
    GENERATION = "G"
    CIRCULATION = "C"
    VERIFIED_MISUSE = "V"


class Criterion(str, Enum):
    """AI-generation evidence codes (METHODOLOGY §3)."""

    C2PA = "C2PA"
    META = "META"
    DATASET = "DATASET"
    FACTCHECK = "FACTCHECK"
    PLATFORM = "PLATFORM"
    DECLARED = "DECLARED"
    DETECTOR = "DETECTOR"


class AIStatus(str, Enum):
    CONFIRMED = "confirmed"
    PROBABLE = "probable"
    POSSIBLE = "possible"
    UNVERIFIED = "unverified"


CRITERION_STATUS: dict[Criterion, AIStatus] = {
    Criterion.C2PA: AIStatus.CONFIRMED,
    Criterion.META: AIStatus.CONFIRMED,
    Criterion.DATASET: AIStatus.CONFIRMED,
    Criterion.FACTCHECK: AIStatus.CONFIRMED,
    Criterion.PLATFORM: AIStatus.PROBABLE,
    Criterion.DECLARED: AIStatus.PROBABLE,
    Criterion.DETECTOR: AIStatus.POSSIBLE,
}
_STATUS_RANK = [AIStatus.CONFIRMED, AIStatus.PROBABLE, AIStatus.POSSIBLE, AIStatus.UNVERIFIED]


class ImageContent(BaseModel):
    media_url: str | None = Field(None, description="URL (or hf-zip:// locator) the bytes were fetched from")
    sha256: str | None = None
    phash: str | None = Field(None, description="64-bit perceptual hash, hex")
    dhash: str | None = Field(None, description="64-bit difference hash, hex")
    mime: str | None = None
    width: int | None = None
    height: int | None = None
    n_bytes: int | None = None
    local_path: str | None = Field(None, description="Content-addressed cache path, only if caching is permitted")
    fetched_at: datetime | None = None
    fetch_error: str | None = None


class SourceInfo(BaseModel):
    platform: str = Field(..., description="reddit | bluesky | wikimedia_commons | diffusiondb | factcheck_claimreview | ...")
    corpus: Corpus
    sampling_frame: str
    collector: str
    collector_version: str
    native_id: str
    post_url: str | None = None
    query: str | None = Field(None, description="Query / subreddit / category / filter that matched this item")
    collected_at: datetime = Field(default_factory=utcnow)
    run_id: str | None = None


class PostInfo(BaseModel):
    created_at: datetime | None = None
    text: str | None = Field(None, description="Caption/title/prompt; may be omitted where terms require")
    alt_text: str | None = None
    language: str | None = None
    hashtags: list[str] = Field(default_factory=list)
    author_pseudonym: str | None = Field(None, description="HMAC-SHA256(author id, project salt); never the raw handle")
    community: str | None = Field(None, description="Subreddit, feed, category")
    platform_labels: list[str] = Field(default_factory=list)
    nsfw: bool | None = None


class StandardEngagement(BaseModel):
    reactions: int | None = None
    amplifications: int | None = None
    replies: int | None = None
    views: int | None = None
    audience: int | None = None

    @property
    def total(self) -> int | None:
        parts = [self.reactions, self.amplifications, self.replies]
        if all(p is None for p in parts):
            return None
        return sum(p or 0 for p in parts)


class EngagementSnapshot(BaseModel):
    observed_at: datetime = Field(default_factory=utcnow)
    age_hours: float | None = None
    raw: dict[str, Any] = Field(default_factory=dict)
    standard: StandardEngagement = Field(default_factory=StandardEngagement)


class DetectionEvidence(BaseModel):
    criterion: Criterion
    method: str = Field(..., description="e.g. 'png_text_chunk:parameters', 'subreddit_rule:r/midjourney'")
    value: str | None = None
    confidence: float | None = Field(None, ge=0, le=1)
    iptc_digital_source_type: str | None = None
    recorded_at: datetime = Field(default_factory=utcnow)


class MotifCode(BaseModel):
    code_id: str = Field(..., description="Codebook id, e.g. VM06 or PT22")
    present: bool
    score: float | None = Field(None, description="Model score before thresholding")
    coder_id: str
    coder_type: Literal["human", "model", "consensus"]
    codebook_version: str
    validated: bool = Field(False, description="True only for model coders that passed METHODOLOGY §4.3 for this code")
    note: str | None = None
    coded_at: datetime = Field(default_factory=utcnow)


class AffectCode(BaseModel):
    coder_id: str
    coder_type: Literal["human", "model", "consensus"]
    valence: float = Field(..., ge=1, le=9)
    arousal: float = Field(..., ge=1, le=9)
    dominance: float | None = Field(None, ge=1, le=9)
    primary_emotion: str | None = None
    secondary_emotion: str | None = None
    codebook_version: str
    validated: bool = Field(False, description="True only for model coders whose affect ratings passed calibration")
    coded_at: datetime = Field(default_factory=utcnow)


class Rights(BaseModel):
    license: str | None = None
    terms_basis: str = Field(..., description="Why collection is permitted, e.g. 'CC0-1.0', 'Reddit Data API Terms'")
    may_cache_bytes: bool = False
    may_redistribute_bytes: bool = False


class ImageRecord(BaseModel):
    schema_version: str = SCHEMA_VERSION
    record_id: str = Field(..., description="'<platform>:<native_id>:<image_index>'")
    image_index: int = 0
    source: SourceInfo
    post: PostInfo = Field(default_factory=PostInfo)
    image: ImageContent = Field(default_factory=ImageContent)
    rights: Rights
    generator: str | None = Field(None, description="Generator name if known (e.g. 'Stable Diffusion v1')")
    prompt: str | None = None
    ai_evidence: list[DetectionEvidence] = Field(default_factory=list)
    engagement: list[EngagementSnapshot] = Field(default_factory=list)
    motif_codes: list[MotifCode] = Field(default_factory=list)
    affect_codes: list[AffectCode] = Field(default_factory=list)
    cluster_id: str | None = None
    exclusion: str | None = Field(None, description="If set, record is excluded from analysis (reason)")

    @property
    def ai_status(self) -> AIStatus:
        statuses = {CRITERION_STATUS[e.criterion] for e in self.ai_evidence}
        for s in _STATUS_RANK:
            if s in statuses:
                return s
        return AIStatus.UNVERIFIED

    def add_evidence(self, ev: DetectionEvidence) -> None:
        key = (ev.criterion, ev.method, ev.value)
        if key not in {(e.criterion, e.method, e.value) for e in self.ai_evidence}:
            self.ai_evidence.append(ev)

    def merge(self, other: "ImageRecord") -> None:
        """Merge a newer observation of the same record: append-only lists."""
        assert other.record_id == self.record_id
        for ev in other.ai_evidence:
            self.add_evidence(ev)
        self.engagement.extend(other.engagement)
        self.motif_codes.extend(other.motif_codes)
        self.affect_codes.extend(other.affect_codes)
        if other.image.sha256 and not self.image.sha256:
            self.image = other.image
        for field in ("cluster_id", "exclusion", "generator", "prompt"):
            if getattr(other, field) is not None:
                setattr(self, field, getattr(other, field))
