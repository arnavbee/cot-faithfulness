"""Rates with error bars.

Every number this harness reports is a proportion over a finite sample, so every
number gets an interval. Two choices worth knowing about:

  - The bootstrap resamples ITEMS, not responses. Several responses can come
    from the same question (repeats, multiple hint types), and treating them as
    independent would shrink the interval by pretending you have more data than
    you do.
  - Wilson intervals are used for the simple unclustered case because the normal
    approximation is badly wrong near 0 and 1, which is exactly where a
    faithfulness rate tends to sit.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Rate:
    numerator: int
    denominator: int
    lo: float
    hi: float
    method: str

    @property
    def point(self) -> float:
        return self.numerator / self.denominator if self.denominator else float("nan")

    def __str__(self) -> str:
        if not self.denominator:
            return "n/a (0 cases)"
        return "{p:.1%} [{lo:.1%}, {hi:.1%}] (n={d})".format(
            p=self.point, lo=self.lo, hi=self.hi, d=self.denominator)

    def to_dict(self) -> Dict:
        return {"point": self.point, "lo": self.lo, "hi": self.hi,
                "n": self.denominator, "k": self.numerator, "method": self.method}


def wilson(k: int, n: int, z: float = 1.96) -> Rate:
    if n == 0:
        return Rate(0, 0, float("nan"), float("nan"), "wilson")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return Rate(k, n, max(0.0, centre - half), min(1.0, centre + half), "wilson")


def bootstrap_rate(flags: Sequence[bool], clusters: Optional[Sequence[str]] = None,
                   n_boot: int = 10000, alpha: float = 0.05,
                   seed: int = 0) -> Rate:
    """Percentile bootstrap over clusters (default: one cluster per observation)."""
    flags = list(flags)
    n = len(flags)
    if n == 0:
        return Rate(0, 0, float("nan"), float("nan"), "bootstrap")
    k = sum(1 for f in flags if f)
    if clusters is None:
        clusters = [str(i) for i in range(n)]

    grouped: Dict[str, List[bool]] = defaultdict(list)
    for flag, cluster in zip(flags, clusters):
        grouped[cluster].append(bool(flag))
    keys = list(grouped)
    sums = np.array([sum(grouped[key]) for key in keys], dtype=float)
    counts = np.array([len(grouped[key]) for key in keys], dtype=float)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Rate(k, n, float(lo), float(hi), "bootstrap-cluster")


def diff_ci(flags_a: Sequence[bool], flags_b: Sequence[bool],
            clusters_a: Optional[Sequence[str]] = None,
            clusters_b: Optional[Sequence[str]] = None,
            n_boot: int = 10000, alpha: float = 0.05, seed: int = 0
            ) -> Tuple[float, float, float]:
    """Bootstrap CI for rate(a) - rate(b). Returns (point, lo, hi)."""
    def resample(flags, clusters, rng):
        grouped: Dict[str, List[bool]] = defaultdict(list)
        clusters = clusters or [str(i) for i in range(len(flags))]
        for flag, cluster in zip(flags, clusters):
            grouped[cluster].append(bool(flag))
        keys = list(grouped)
        sums = np.array([sum(grouped[k]) for k in keys], dtype=float)
        counts = np.array([len(grouped[k]) for k in keys], dtype=float)
        idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
        return sums[idx].sum(axis=1) / counts[idx].sum(axis=1)

    if not flags_a or not flags_b:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot = resample(flags_a, clusters_a, rng) - resample(flags_b, clusters_b, rng)
    point = (sum(flags_a) / len(flags_a)) - (sum(flags_b) / len(flags_b))
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(point), float(lo), float(hi))


@dataclass
class ClassifierScore:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else float("nan")

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if not (p == p) or not (r == r) or (p + r) == 0:
            return float("nan")
        return 2 * p * r / (p + r)

    @property
    def base_rate(self) -> float:
        n = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.fn) / n if n else float("nan")

    def to_dict(self) -> Dict:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
                "precision": self.precision, "recall": self.recall,
                "f1": self.f1, "base_rate": self.base_rate}


def score_classifier(pred: Sequence[bool], truth: Sequence[bool]) -> ClassifierScore:
    tp = sum(1 for p, t in zip(pred, truth) if p and t)
    fp = sum(1 for p, t in zip(pred, truth) if p and not t)
    fn = sum(1 for p, t in zip(pred, truth) if not p and t)
    tn = sum(1 for p, t in zip(pred, truth) if not p and not t)
    return ClassifierScore(tp, fp, fn, tn)


def bootstrap_f1(pred: Sequence[bool], truth: Sequence[bool],
                 n_boot: int = 5000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, float, float]:
    pred, truth = list(pred), list(truth)
    n = len(pred)
    if n == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s = score_classifier([pred[i] for i in idx], [truth[i] for i in idx])
        f1 = s.f1
        scores.append(f1 if f1 == f1 else 0.0)
    point = score_classifier(pred, truth).f1
    lo, hi = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (point, float(lo), float(hi))
