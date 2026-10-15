"""Human coding round-trip: export a coding sheet (CSV), import coded sheets.

Sheet columns: record_id, post_url, local_path, <code ids...>, valence,
arousal, dominance, primary_emotion, secondary_emotion, note. Motif cells take
1 / 0 / blank (blank = not coded). One sheet per coder.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Iterable

from ..codebook import Codebook
from ..models import AffectCode, ImageRecord, MotifCode

AFFECT_COLS = ["valence", "arousal", "dominance", "primary_emotion", "secondary_emotion"]


def export_sheet(records: Iterable[ImageRecord], cb: Codebook, path: str | Path, sample: int | None = None, seed: int = 0) -> int:
    recs = [r for r in records if r.exclusion is None]
    if sample is not None and sample < len(recs):
        recs = random.Random(seed).sample(recs, sample)
    cols = ["record_id", "post_url", "local_path", *cb.codes.keys(), *AFFECT_COLS, "note"]
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in recs:
            w.writerow({"record_id": r.record_id, "post_url": r.source.post_url or r.image.media_url,
                        "local_path": r.image.local_path or ""})
    return len(recs)


def import_sheet(path: str | Path, cb: Codebook, coder_id: str) -> dict[str, tuple[list[MotifCode], list[AffectCode]]]:
    out: dict[str, tuple[list[MotifCode], list[AffectCode]]] = {}
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            motifs = []
            for cid in cb.codes:
                v = (row.get(cid) or "").strip()
                if v in ("0", "1"):
                    motifs.append(MotifCode(code_id=cid, present=v == "1", coder_id=coder_id, coder_type="human",
                                            codebook_version=cb.version, validated=True, note=row.get("note") or None))
            affect = []
            if (row.get("valence") or "").strip() and (row.get("arousal") or "").strip():
                emo = (row.get("primary_emotion") or "").strip() or None
                if emo and emo not in cb.emotions:
                    raise ValueError(f"{row['record_id']}: emotion '{emo}' not in codebook")
                affect.append(AffectCode(coder_id=coder_id, coder_type="human", valence=float(row["valence"]),
                                         arousal=float(row["arousal"]),
                                         dominance=float(row["dominance"]) if (row.get("dominance") or "").strip() else None,
                                         primary_emotion=emo,
                                         secondary_emotion=(row.get("secondary_emotion") or "").strip() or None,
                                         codebook_version=cb.version))
            out[row["record_id"]] = (motifs, affect)
    return out
