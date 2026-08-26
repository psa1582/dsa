from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def stable_topk(values: np.ndarray, k: int) -> np.ndarray:
    """Return descending top-k indices with index as a deterministic tie-breaker."""
    x = np.asarray(values)
    if x.ndim != 1:
        raise ValueError(f"expected a vector, got {x.shape}")
    if not 0 < k <= x.size:
        raise ValueError(f"k={k} is invalid for {x.size} values")
    # ``argpartition`` is unstable at the cutoff.  Sorting its arbitrary tie
    # subset afterwards does not provide a deterministic global-index tie-break.
    cutoff = np.partition(x, x.size - k)[x.size - k]
    greater = np.flatnonzero(x > cutoff)
    equal = np.flatnonzero(x == cutoff)
    candidate = np.concatenate((greater, equal[: k - greater.size]))
    return candidate[np.lexsort((candidate, -x[candidate]))]


def topk_recall(predicted: np.ndarray, target: np.ndarray) -> float:
    predicted = np.asarray(predicted)
    target = np.asarray(target)
    if target.size == 0:
        return 1.0
    return float(np.intersect1d(predicted, target, assume_unique=False).size / target.size)


def normalized_recall_lift(recall: float, k: int, length: int) -> float:
    random_recall = min(1.0, k / length)
    if random_recall == 1.0:
        return 1.0
    return float((recall - random_recall) / (1.0 - random_recall))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size != y.size or x.size < 2:
        return math.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def _average_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    sorted_x = x[order]
    ranks = np.empty(x.size, dtype=np.float64)
    start = 0
    while start < x.size:
        end = start + 1
        while end < x.size and sorted_x[end] == sorted_x[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size != y.size:
        raise ValueError("vectors must have equal length")
    return pearson(_average_ranks(x), _average_ranks(y))


def distribution(values: Iterable[float]) -> dict[str, float]:
    x = np.asarray(list(values), dtype=np.float64)
    if not x.size:
        return {name: math.nan for name in ("mean", "p5", "p25", "median", "p75", "p95")}
    return {
        "mean": float(np.mean(x)),
        "p5": float(np.percentile(x, 5)),
        "p25": float(np.percentile(x, 25)),
        "median": float(np.median(x)),
        "p75": float(np.percentile(x, 75)),
        "p95": float(np.percentile(x, 95)),
    }
