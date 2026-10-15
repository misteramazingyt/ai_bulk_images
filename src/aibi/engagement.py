"""Engagement standardization and normalization (METHODOLOGY §5)."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

from .models import EngagementSnapshot, ImageRecord, StandardEngagement


def _i(x: Any) -> int | None:
    return None if x is None else int(x)


def standardize(platform: str, raw: dict[str, Any]) -> StandardEngagement:
    """Map raw platform metrics onto the common schema (METHODOLOGY table 5.1)."""
    if platform == "reddit":
        return StandardEngagement(
            reactions=_i(raw.get("score")),
            amplifications=_i(raw.get("num_crossposts")),
            replies=_i(raw.get("num_comments")),
            audience=_i(raw.get("subreddit_subscribers")),
        )
    if platform == "bluesky":
        rep, quo = raw.get("repostCount"), raw.get("quoteCount")
        amp = None if rep is None and quo is None else (rep or 0) + (quo or 0)
        return StandardEngagement(
            reactions=_i(raw.get("likeCount")),
            amplifications=amp,
            replies=_i(raw.get("replyCount")),
            audience=_i(raw.get("followersCount")),
        )
    if platform == "wikimedia_commons":
        return StandardEngagement(views=_i(raw.get("pageviews")))
    return StandardEngagement()


def snapshot(platform: str, raw: dict[str, Any], created_at: datetime | None, observed_at: datetime) -> EngagementSnapshot:
    age = (observed_at - created_at).total_seconds() / 3600 if created_at else None
    return EngagementSnapshot(observed_at=observed_at, age_hours=age, raw=raw, standard=standardize(platform, raw))


def primary_snapshot(rec: ImageRecord, min_age_hours: float = 168) -> EngagementSnapshot | None:
    """Earliest snapshot at or beyond the standard observation age (T+7d)."""
    eligible = [s for s in rec.engagement if s.age_hours is not None and s.age_hours >= min_age_hours]
    return min(eligible, key=lambda s: s.age_hours) if eligible else None


@dataclass
class NormalizedEngagement:
    record_id: str
    stratum: str
    total: int
    rate: float | None
    percentile: float
    log_z: float | None


def _stratum(rec: ImageRecord, with_community: bool) -> str:
    month = rec.post.created_at.strftime("%Y-%m") if rec.post.created_at else "unknown"
    community = (rec.post.community or "-") if with_community else "*"
    return f"{rec.source.platform}|{community}|{month}"


def percentile_rank(values: list[float]) -> list[float]:
    """Mid-rank percentile (0-100), ties share the average rank."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return [100.0 * (r + 0.5) / n for r in ranks]


def normalize(
    records: Iterable[ImageRecord], min_age_hours: float = 168, min_stratum_size: int = 30
) -> dict[str, NormalizedEngagement]:
    rows: list[tuple[ImageRecord, StandardEngagement]] = []
    for rec in records:
        snap = primary_snapshot(rec, min_age_hours)
        if snap and snap.standard.total is not None:
            rows.append((rec, snap.standard))

    fine: dict[str, list[int]] = defaultdict(list)
    for idx, (rec, _) in enumerate(rows):
        fine[_stratum(rec, True)].append(idx)
    assignment: dict[int, str] = {}
    for key, idxs in fine.items():
        for idx in idxs:
            assignment[idx] = key if len(idxs) >= min_stratum_size else _stratum(rows[idx][0], False)
    strata: dict[str, list[int]] = defaultdict(list)
    for idx, key in assignment.items():
        strata[key].append(idx)

    out: dict[str, NormalizedEngagement] = {}
    for key, idxs in strata.items():
        totals = [rows[i][1].total for i in idxs]
        pct = percentile_rank([float(t) for t in totals])
        logs = [math.log1p(max(t, 0)) for t in totals]
        mean = sum(logs) / len(logs)
        sd = math.sqrt(sum((v - mean) ** 2 for v in logs) / (len(logs) - 1)) if len(logs) > 1 else 0.0
        for i, p, lg in zip(idxs, pct, logs):
            rec, std = rows[i]
            out[rec.record_id] = NormalizedEngagement(
                record_id=rec.record_id,
                stratum=key,
                total=std.total,
                rate=(std.total / std.audience) if std.audience else None,
                percentile=p,
                log_z=((lg - mean) / sd) if sd > 0 else None,
            )
    return out
