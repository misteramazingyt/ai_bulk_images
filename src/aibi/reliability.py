"""Inter-coder reliability (Krippendorff's alpha) and model-coder validation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import permutations
from typing import Hashable, Iterable, Literal, Sequence

Level = Literal["nominal", "ordinal", "interval", "ratio"]


def krippendorff_alpha(units: Iterable[Sequence[Hashable | None]], level: Level = "nominal") -> float:
    """Krippendorff's alpha for a units x coders table; None = missing.

    Implements the coincidence-matrix formulation (Krippendorff 2004, ch. 11).
    Units with fewer than two pairable values are ignored.
    """
    coincidence: dict[tuple, float] = defaultdict(float)
    for unit in units:
        vals = [v for v in unit if v is not None]
        m = len(vals)
        if m < 2:
            continue
        for a, b in permutations(range(m), 2):
            coincidence[(vals[a], vals[b])] += 1.0 / (m - 1)

    n_c: Counter = Counter()
    for (c, _k), w in coincidence.items():
        n_c[c] += w
    n = sum(n_c.values())
    if n <= 1:
        return float("nan")
    cats = sorted(n_c, key=lambda x: (str(type(x)), x))

    def delta(c, k) -> float:
        if level == "nominal":
            return 0.0 if c == k else 1.0
        if level == "interval":
            return float(c - k) ** 2
        if level == "ratio":
            return ((c - k) / (c + k)) ** 2 if (c + k) else 0.0
        # ordinal
        lo, hi = sorted((cats.index(c), cats.index(k)))
        s = sum(n_c[cats[g]] for g in range(lo, hi + 1)) - (n_c[cats[lo]] + n_c[cats[hi]]) / 2
        return s**2

    d_o = sum(w * delta(c, k) for (c, k), w in coincidence.items())
    d_e = sum(n_c[c] * n_c[k] * delta(c, k) for c in cats for k in cats) / (n - 1)
    if d_e == 0:
        return 1.0
    return 1.0 - d_o / d_e


def interpret_alpha(alpha: float) -> str:
    if alpha >= 0.800:
        return "reliable"
    if alpha >= 0.667:
        return "tentative"
    return "unreliable"


@dataclass
class BinaryMetrics:
    threshold: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int


def binary_metrics(y_true: Sequence[bool], scores: Sequence[float], threshold: float) -> BinaryMetrics:
    tp = fp = fn = tn = 0
    for y, s in zip(y_true, scores):
        pred = s >= threshold
        tp += pred and y
        fp += pred and not y
        fn += (not pred) and y
        tn += (not pred) and (not y)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return BinaryMetrics(threshold, p, r, f1, tp, fp, fn, tn)


def auroc(y_true: Sequence[bool], scores: Sequence[float]) -> float:
    pos = [s for y, s in zip(y_true, scores) if y]
    neg = [s for y, s in zip(y_true, scores) if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def calibrate_threshold(
    y_true: Sequence[bool], scores: Sequence[float], rule: Literal["max_f1", "precision"] = "max_f1", min_precision: float = 0.80
) -> BinaryMetrics:
    """Pick a threshold on a calibration split (METHODOLOGY §4.3)."""
    candidates = sorted(set(scores))
    results = [binary_metrics(y_true, scores, t) for t in candidates]
    if rule == "precision":
        ok = [m for m in results if m.precision >= min_precision and m.tp > 0]
        if ok:
            return min(ok, key=lambda m: m.threshold)
    return max(results, key=lambda m: (m.f1, -m.threshold))
