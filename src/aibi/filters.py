"""Filter 1 (motif / affect) and Filter 2 (spread / engagement) - METHODOLOGY §4.4, §5.4."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from statistics import median
from typing import Iterable

from .codebook import Codebook
from .engagement import NormalizedEngagement, normalize, primary_snapshot
from .models import AIStatus, ImageRecord

ANALYTIC_STATUSES = {AIStatus.CONFIRMED, AIStatus.PROBABLE}
SOCIAL_PLATFORMS = {"reddit", "bluesky"}


def consensus_motifs(rec: ImageRecord, cb: Codebook) -> set[str]:
    """Codes present by consensus: an adjudicated 'consensus' code wins; else
    majority of human coders; else validated model codes. Unvalidated model
    codes never count."""
    by_code: dict[str, list] = defaultdict(list)
    for c in rec.motif_codes:
        by_code[c.code_id].append(c)
    present = set()
    for cid, codes in by_code.items():
        code = cb.codes.get(cid)
        if code is None or (code.layer == "persuasion_technique" and code.modality == "text"):
            continue
        cons = [c for c in codes if c.coder_type == "consensus"]
        humans = [c for c in codes if c.coder_type == "human"]
        models = [c for c in codes if c.coder_type == "model" and c.validated]
        if cons:
            ok = cons[-1].present
        elif humans:
            ok = sum(c.present for c in humans) * 2 > len(humans)
        elif models:
            ok = models[-1].present
        else:
            ok = False
        if ok:
            present.add(cid)
    return present


def consensus_affect(rec: ImageRecord) -> tuple[float, float] | None:
    cons = [a for a in rec.affect_codes if a.coder_type == "consensus"]
    pool = cons or [a for a in rec.affect_codes if a.coder_type == "human"]
    if not pool:
        return None
    return median(a.valence for a in pool), median(a.arousal for a in pool)


def is_high_affect(valence: float, arousal: float, cfg: dict) -> bool:
    return arousal >= cfg.get("arousal_min", 7) or (
        arousal >= cfg.get("alt_arousal_min", 6) and abs(valence - 5) >= cfg.get("alt_valence_distance_min", 3)
    )


@dataclass
class FilterRow:
    record_id: str
    platform: str
    corpus: str
    sampling_frame: str
    ai_status: str
    cluster_id: str | None
    motifs: str
    n_motifs: int
    n_techniques: int
    valence: float | None
    arousal: float | None
    high_affect: bool | None
    pai: float | None
    passes_filter1: bool | None
    engagement_total: int | None
    engagement_percentile: float | None
    engagement_rate: float | None
    stratum: str | None
    cluster_n_posts: int
    cluster_n_platforms: int
    cluster_max_percentile: float | None
    passes_filter2: bool
    post_url: str | None


def evaluate(records: Iterable[ImageRecord], cb: Codebook) -> list[FilterRow]:
    f = cb.filters
    recs = [r for r in records if r.exclusion is None and r.ai_status in ANALYTIC_STATUSES]
    norm: dict[str, NormalizedEngagement] = normalize(
        recs, f.get("primary_snapshot_min_age_hours", 168), f.get("min_stratum_size", 30))

    clusters: dict[str, list[ImageRecord]] = defaultdict(list)
    for r in recs:
        clusters[r.cluster_id or r.record_id].append(r)
    cstats = {}
    for cid, members in clusters.items():
        social = [m for m in members if m.source.platform in SOCIAL_PLATFORMS and primary_snapshot(m, f.get("primary_snapshot_min_age_hours", 168))]
        pcts = [norm[m.record_id].percentile for m in members if m.record_id in norm]
        cstats[cid] = (len(members), len({m.source.platform for m in members}), max(pcts) if pcts else None, bool(social))

    rows = []
    for r in recs:
        motifs = consensus_motifs(r, cb)
        n_vm = sum(cb.codes[m].layer == "visual_motif" for m in motifs)
        n_pt = len(motifs) - n_vm
        aff = consensus_affect(r)
        high = is_high_affect(*aff, f.get("high_affect", {})) if aff else None
        pai = (len(motifs) + (aff[1] - 1) / 8 + abs(aff[0] - 5) / 4) if aff else None
        f1 = (bool(motifs) and bool(high)) if aff else None
        n_posts, n_plat, max_pct, has_social = cstats[r.cluster_id or r.record_id]
        f2 = has_social and ((max_pct is not None and max_pct >= f.get("high_engagement_percentile", 90))
                             or n_plat >= f.get("min_platforms_for_spread", 2))
        ne = norm.get(r.record_id)
        rows.append(FilterRow(
            record_id=r.record_id, platform=r.source.platform, corpus=r.source.corpus.value,
            sampling_frame=r.source.sampling_frame, ai_status=r.ai_status.value, cluster_id=r.cluster_id,
            motifs=";".join(sorted(motifs)), n_motifs=n_vm, n_techniques=n_pt,
            valence=aff[0] if aff else None, arousal=aff[1] if aff else None, high_affect=high, pai=pai,
            passes_filter1=f1, engagement_total=ne.total if ne else None,
            engagement_percentile=ne.percentile if ne else None, engagement_rate=ne.rate if ne else None,
            stratum=ne.stratum if ne else None, cluster_n_posts=n_posts, cluster_n_platforms=n_plat,
            cluster_max_percentile=max_pct, passes_filter2=f2, post_url=r.source.post_url,
        ))
    return rows


def rows_to_dicts(rows: list[FilterRow]) -> list[dict]:
    return [asdict(r) for r in rows]
