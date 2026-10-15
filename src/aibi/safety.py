"""Harmful-content handling applied after any coder runs (ETHICS_AND_LEGAL §4).

Flags never delete records: they delete cached bytes and mark the record so
it is excluded and queued for human review / reporting.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .models import ImageRecord, utcnow

log = logging.getLogger("aibi.safety")


def drop_bytes(rec: ImageRecord, delete_file: bool = True) -> None:
    if rec.image.local_path and delete_file:
        try:
            Path(rec.image.local_path).unlink(missing_ok=True)
        except OSError as e:  # pragma: no cover
            log.error("could not delete %s: %s", rec.image.local_path, e)
        rec.image.local_path = None
    rec.rights.may_cache_bytes = False
    rec.rights.may_redistribute_bytes = False


def apply_safety_flag(rec: ImageRecord, flag: str, source: str) -> None:
    if flag == "possible_minor_sexualization":
        drop_bytes(rec)
        rec.exclusion = "csam_suspected_removed"
        log.critical("CSAM suspected (%s) for %s - bytes deleted; follow ETHICS_AND_LEGAL §4 (report URL: %s)",
                     source, rec.record_id, rec.source.post_url or rec.image.media_url)
    elif flag == "sexualized_real_person":
        drop_bytes(rec)  # VM19 handling: hashes + URL only
        log.warning("sexualized real person flagged (%s) for %s - bytes deleted", source, rec.record_id)


def tombstone(rec: ImageRecord, reason: str, delete_file: bool = True) -> None:
    """Strip user content from a record deleted at source, keeping an audit stub.
    Pass delete_file=False when another live record shares the cached bytes."""
    drop_bytes(rec, delete_file)
    rec.post = type(rec.post)(created_at=rec.post.created_at, community=rec.post.community)
    rec.image = type(rec.image)()
    rec.prompt = None
    rec.engagement = []
    rec.motif_codes = []
    rec.affect_codes = []
    rec.cluster_id = None
    rec.exclusion = f"{reason}@{utcnow().date().isoformat()}"
