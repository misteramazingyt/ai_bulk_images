"""Model-coder validation against human gold labels (METHODOLOGY §4.3).

Gold = adjudicated consensus code if present, else human majority. Records
with gold labels are split deterministically (seeded hash of record_id) into
calibration and test halves. Per code: threshold is chosen on calibration,
metrics are reported on test, and `validated` requires test F1 >= min_f1 and
at least min_test_positives gold positives in the test half. Affect is
validated with interval Krippendorff's alpha (model vs. human median) on the
test half.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Iterable

from .codebook import Codebook
from .models import ImageRecord
from .reliability import auroc, binary_metrics, calibrate_threshold, krippendorff_alpha


def gold_motifs(rec: ImageRecord) -> dict[str, bool]:
    by_code: dict[str, list] = defaultdict(list)
    for c in rec.motif_codes:
        if c.coder_type in ("human", "consensus"):
            by_code[c.code_id].append(c)
    out = {}
    for cid, codes in by_code.items():
        cons = [c for c in codes if c.coder_type == "consensus"]
        if cons:
            out[cid] = cons[-1].present
        else:
            votes = [c.present for c in codes]
            if sum(votes) * 2 != len(votes):  # skip ties
                out[cid] = sum(votes) * 2 > len(votes)
    return out


def gold_affect(rec: ImageRecord) -> tuple[float, float] | None:
    cons = [a for a in rec.affect_codes if a.coder_type == "consensus"]
    pool = cons or [a for a in rec.affect_codes if a.coder_type == "human"]
    if not pool:
        return None
    return median(a.valence for a in pool), median(a.arousal for a in pool)


def split(record_id: str, seed: int) -> str:
    h = int(hashlib.sha256(f"{seed}:{record_id}".encode()).hexdigest(), 16)
    return "calibration" if h % 2 == 0 else "test"


def _nan_to_none(x: float) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(x, 4)


def calibrate(records: Iterable[ImageRecord], cb: Codebook, coder_id: str, seed: int = 0,
              rule: str = "max_f1", min_precision: float = 0.80, min_f1: float = 0.70,
              min_test_positives: int = 10) -> dict:
    data: dict[str, dict[str, list]] = defaultdict(lambda: {"calibration": [], "test": []})
    affect_pairs: list[tuple[float, float, float, float]] = []
    for rec in records:
        gold = gold_motifs(rec)
        model = {}
        for c in rec.motif_codes:
            if c.coder_id == coder_id and c.score is not None:
                model[c.code_id] = c.score
        part = split(rec.record_id, seed)
        for cid in gold.keys() & model.keys():
            data[cid][part].append((gold[cid], model[cid]))
        ga = gold_affect(rec)
        ma = [a for a in rec.affect_codes if a.coder_id == coder_id]
        if ga and ma and part == "test":
            affect_pairs.append((ga[0], ma[-1].valence, ga[1], ma[-1].arousal))

    report: dict = {"coder_id": coder_id, "codebook_version": cb.version, "seed": seed, "rule": rule,
                    "criteria": {"min_f1": min_f1, "min_test_positives": min_test_positives}, "codes": {}}
    for cid in sorted(data):
        cal, test = data[cid]["calibration"], data[cid]["test"]
        if not cal or not test or not any(y for y, _ in cal):
            report["codes"][cid] = {"validated": False, "reason": "insufficient gold labels",
                                    "n_calibration": len(cal), "n_test": len(test)}
            continue
        chosen = calibrate_threshold([y for y, _ in cal], [s for _, s in cal], rule=rule, min_precision=min_precision)
        m = binary_metrics([y for y, _ in test], [s for _, s in test], chosen.threshold)
        n_pos = sum(y for y, _ in test)
        ok = m.f1 >= min_f1 and n_pos >= min_test_positives
        report["codes"][cid] = {
            "threshold": chosen.threshold, "validated": ok,
            "reason": None if ok else ("test F1 below minimum" if m.f1 < min_f1 else "too few test positives"),
            "test": {"precision": round(m.precision, 4), "recall": round(m.recall, 4), "f1": round(m.f1, 4),
                     "auroc": _nan_to_none(auroc([y for y, _ in test], [s for _, s in test])),
                     "tp": m.tp, "fp": m.fp, "fn": m.fn, "tn": m.tn, "n_positive": n_pos},
            "n_calibration": len(cal), "n_test": len(test),
        }
    if len(affect_pairs) >= 2:
        a_val = krippendorff_alpha([[g, m] for g, m, _, _ in affect_pairs], "interval")
        a_aro = krippendorff_alpha([[g, m] for _, _, g, m in affect_pairs], "interval")
        report["affect"] = {"n_test": len(affect_pairs), "alpha_valence": _nan_to_none(a_val),
                            "alpha_arousal": _nan_to_none(a_aro),
                            "validated": bool(a_val >= 0.8 and a_aro >= 0.8)}
    else:
        report["affect"] = {"n_test": len(affect_pairs), "validated": False}
    return report


def write_thresholds(report: dict, path: str | Path) -> None:
    path = Path(path)
    data = json.loads(path.read_text()) if path.exists() else {}
    data[report["coder_id"]] = {cid: {"threshold": r.get("threshold", 0.5), "validated": r["validated"]}
                                for cid, r in report["codes"].items()}
    data[report["coder_id"]]["_affect"] = {"validated": report["affect"]["validated"]}
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def apply_thresholds(records: Iterable[ImageRecord], thresholds: dict) -> int:
    """Re-threshold and set `validated` on existing model codes from thresholds.json."""
    n = 0
    for rec in records:
        for c in rec.motif_codes:
            t = thresholds.get(c.coder_id, {}).get(c.code_id)
            if c.coder_type == "model" and t and c.score is not None:
                c.present, c.validated = c.score >= t["threshold"], bool(t["validated"])
                n += 1
        for a in rec.affect_codes:
            t = thresholds.get(a.coder_id, {}).get("_affect")
            if a.coder_type == "model" and t:
                a.validated = bool(t["validated"])
    return n
