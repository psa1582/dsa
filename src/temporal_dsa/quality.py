from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import normalized_recall_lift, stable_topk, topk_recall


@dataclass(frozen=True)
class QualityGate:
    passed: bool
    layer_pass_fraction: float
    failed_layers: list[int]
    reason: str


def evaluate_quality_rows(
    teacher_scores: np.ndarray,
    indexer_scores: np.ndarray,
    *,
    layer: int,
    workload: str,
    k_values: list[int],
) -> pd.DataFrame:
    teacher = np.asarray(teacher_scores)
    indexer = np.asarray(indexer_scores)
    if teacher.shape != indexer.shape or teacher.ndim != 2:
        raise ValueError("teacher and indexer scores must both be [queries,keys]")
    rows = []
    for query in range(teacher.shape[0]):
        for k in k_values:
            if k > teacher.shape[1]:
                continue
            recall = topk_recall(stable_topk(indexer[query], k), stable_topk(teacher[query], k))
            rows.append(
                {
                    "layer": layer,
                    "workload": workload,
                    "query": query,
                    "context_length": teacher.shape[1],
                    "k": k,
                    "recall": recall,
                    "normalized_lift": normalized_recall_lift(recall, k, teacher.shape[1]),
                }
            )
    return pd.DataFrame(rows)


def apply_quality_gate(
    rows: pd.DataFrame,
    *,
    gate_k: int = 512,
    min_median_normalized_lift: float = 0.20,
    min_layer_pass_fraction: float = 0.75,
) -> QualityGate:
    required = {"layer", "k", "normalized_lift"}
    if not required.issubset(rows.columns):
        raise ValueError(f"quality rows are missing {sorted(required - set(rows.columns))}")
    selected = rows[rows["k"] == gate_k]
    if selected.empty:
        return QualityGate(False, 0.0, [], f"no Recall@{gate_k} observations")
    medians = selected.groupby("layer")["normalized_lift"].median()
    failed = [int(layer) for layer, value in medians.items() if value < min_median_normalized_lift]
    fraction = float((medians >= min_median_normalized_lift).mean())
    passed = fraction >= min_layer_pass_fraction
    reason = (
        f"{fraction:.1%} of layers pass normalized Recall@{gate_k} lift "
        f">= {min_median_normalized_lift:.2f}"
    )
    return QualityGate(passed, fraction, failed, reason)

