from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from temporal_dsa.metrics import stable_topk

K = 2048
P_VALUES = (1, 2, 4, 8, 16)
ADMISSION_LANES = (1, 2, 4)
RANK_BINS = (
    ("rank_2049_2304", 2049, 2304),
    ("rank_2305_4096", 2305, 4096),
    ("rank_4097_8192", 4097, 8192),
    ("rank_8193_16384", 8193, 16384),
    ("rank_16385_plus", 16385, np.iinfo(np.int32).max),
)


def phase(step: int, transitions: int) -> str:
    fraction = step / max(1, transitions)
    if fraction <= 1 / 3:
        return "early"
    if fraction <= 2 / 3:
        return "middle"
    return "late"


def fifo_depth(flags: np.ndarray, scores_per_cycle: int, lanes: int) -> tuple[int, int]:
    padding = (-flags.size) % scores_per_cycle
    if padding:
        flags = np.pad(flags, (0, padding), constant_values=False)
    arrivals = flags.reshape(-1, scores_per_cycle).sum(axis=1, dtype=np.int32)
    cumulative = np.cumsum(arrivals - lanes, dtype=np.int32)
    prior_minimum = np.minimum.accumulate(
        np.concatenate((np.zeros(1, dtype=np.int32), cumulative[:-1]))
    )
    queue = cumulative - np.minimum(prior_minimum, cumulative)
    return int(queue.max(initial=0)), int(queue[-1])


def summarize(frame: pd.DataFrame, columns: list[str], group_type: str) -> list[dict]:
    groups: Any = [((), frame)] if not columns else frame.groupby(columns, dropna=False)
    rows = []
    for key, group in groups:
        if columns and not isinstance(key, tuple):
            key = (key,)
        values = dict(zip(columns, key if columns else ()))
        rows.append(
            {
                "group_type": group_type,
                "context": values.get("context"),
                "layer": values.get("layer"),
                "phase": values.get("phase"),
                "transitions": len(group),
                "overlap_mean": group.overlap.mean(),
                "overlap_median": group.overlap.median(),
                "overlap_p5": group.overlap.quantile(0.05),
                "overlap_p1": group.overlap.quantile(0.01),
                "new_entries_mean": group.new_entries.mean(),
                "new_entries_p95": group.new_entries.quantile(0.95),
                "new_entries_p99": group.new_entries.quantile(0.99),
                "new_entries_max": group.new_entries.max(),
                "warm_admission_upper_mean": group.warm_admission_upper.mean(),
                "warm_admission_upper_p95": group.warm_admission_upper.quantile(0.95),
                "warm_admission_upper_p99": group.warm_admission_upper.quantile(0.99),
                "warm_admission_upper_max": group.warm_admission_upper.max(),
                "seed_to_final_threshold_gap_mean": group.seed_to_final_threshold_gap.mean(),
                "unsafe_previous_threshold_misses_total": group.unsafe_previous_threshold_misses.sum(),
                "new_token_entries_total": group.new_token_entries.sum(),
            }
        )
    return rows


def rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "new_token_no_previous_rank"
    if rank <= K:
        return "rank_le_2048_audit"
    for name, lower, upper in RANK_BINS:
        if lower <= rank <= upper:
            return name
    raise AssertionError(rank)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full64 Top-2048 temporal churn analysis")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inventory = pd.read_csv(args.inventory)
    inventory = inventory[inventory.split.isin(["validation", "test"])].copy()
    rows: list[dict] = []
    fifo_rows: list[dict] = []
    rank_counts: dict[tuple[str, int | None, int | None], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    for trace_index, trace in enumerate(inventory.itertuples(), start=1):
        trace_path = Path(trace.trace_file)
        if not trace_path.is_absolute():
            trace_path = args.repo / trace_path
        with np.load(trace_path, allow_pickle=False) as payload:
            scores = payload["scores"].astype(np.float32)
            lengths = payload["lengths"].astype(np.int32)
        context = int(trace.base_context_length)
        layer = int(trace.layer)
        transitions = scores.shape[0] - 1
        for step in range(1, scores.shape[0]):
            previous_length = int(lengths[step - 1])
            current_length = int(lengths[step])
            previous = scores[step - 1, :previous_length]
            current = scores[step, :current_length]
            previous_topk = stable_topk(previous, K)
            current_topk = stable_topk(current, K)
            previous_mask = np.zeros(current_length, dtype=bool)
            previous_mask[previous_topk] = True
            newly_active = current_topk[~previous_mask[current_topk]]
            overlap_count = K - newly_active.size

            previous_order = np.lexsort((np.arange(previous_length), -previous))
            previous_rank = np.empty(previous_length, dtype=np.int32)
            previous_rank[previous_order] = np.arange(1, previous_length + 1)
            for token in newly_active:
                rank = int(previous_rank[token]) if token < previous_length else None
                bucket = rank_bucket(rank)
                for group in [
                    ("overall", None, None),
                    ("context", context, None),
                    ("context_layer", context, layer),
                ]:
                    rank_counts[group][bucket] += 1

            seed_scores = current[previous_topk]
            seed_tau = float(seed_scores.min())
            worst_seed = int(previous_topk[seed_scores == seed_tau].max())
            indices = np.arange(current_length)
            initial_qualifier = (current > seed_tau) | (
                (current == seed_tau) & (indices < worst_seed)
            )
            initial_qualifier[previous_topk] = False
            warm_upper = int(initial_qualifier.sum())
            final_tau = float(current[current_topk[-1]])
            previous_tau = float(previous[previous_topk[-1]])
            unsafe_misses = int(
                np.count_nonzero(
                    (~previous_mask[current_topk]) & (current[current_topk] <= previous_tau)
                )
            )
            row = {
                "split": trace.split,
                "prompt_id": trace.prompt_id,
                "workload": trace.workload,
                "context": context,
                "layer": layer,
                "step": step,
                "phase": phase(step, transitions),
                "previous_length": previous_length,
                "current_length": current_length,
                "overlap_count": overlap_count,
                "overlap": overlap_count / K,
                "new_entries": int(newly_active.size),
                "new_token_entries": int(np.count_nonzero(newly_active >= previous_length)),
                "previous_threshold": previous_tau,
                "warm_seed_current_threshold": seed_tau,
                "final_current_threshold": final_tau,
                "seed_to_final_threshold_gap": final_tau - seed_tau,
                "warm_admission_upper": warm_upper,
                "warm_upper_per_final_insertion": warm_upper / max(1, newly_active.size),
                "unsafe_previous_threshold_misses": unsafe_misses,
                "trace_file": str(trace.trace_file),
            }
            rows.append(row)
            for p in P_VALUES:
                for lanes in ADMISSION_LANES:
                    maximum, final_depth = fifo_depth(initial_qualifier, p, lanes)
                    fifo_rows.append(
                        {
                            "context": context,
                            "layer": layer,
                            "step": step,
                            "scores_per_cycle": p,
                            "admission_lanes": lanes,
                            "max_fifo_depth": maximum,
                            "final_fifo_depth": final_depth,
                        }
                    )
        print(f"[{trace_index}/{len(inventory)}] {trace_path.name}", flush=True)

    churn = pd.DataFrame(rows)
    churn.to_csv(args.output / "topk_churn_rows.csv", index=False)
    summary_rows = []
    for columns, name in [
        ([], "overall"),
        (["context"], "context"),
        (["layer"], "layer"),
        (["context", "layer"], "context_layer"),
        (["context", "phase"], "context_phase"),
    ]:
        summary_rows.extend(summarize(churn, columns, name))
    pd.DataFrame(summary_rows).to_csv(args.output / "topk_churn_summary.csv", index=False)

    rank_rows = []
    bucket_order = [
        "rank_le_2048_audit",
        *[value[0] for value in RANK_BINS],
        "new_token_no_previous_rank",
    ]
    for (group_type, context, layer), counts in rank_counts.items():
        total = sum(counts.values())
        for bucket in bucket_order:
            count = counts.get(bucket, 0)
            rank_rows.append(
                {
                    "group_type": group_type,
                    "context": context,
                    "layer": layer,
                    "previous_rank_bucket": bucket,
                    "new_entry_count": count,
                    "fraction": count / max(1, total),
                }
            )
    pd.DataFrame(rank_rows).to_csv(args.output / "new_entry_previous_rank_histogram.csv", index=False)

    fifo = pd.DataFrame(fifo_rows)
    fifo_summary = (
        fifo.groupby(["context", "scores_per_cycle", "admission_lanes"])
        .agg(
            transitions=("step", "size"),
            max_fifo_depth_mean=("max_fifo_depth", "mean"),
            max_fifo_depth_p95=("max_fifo_depth", lambda x: x.quantile(0.95)),
            max_fifo_depth_p99=("max_fifo_depth", lambda x: x.quantile(0.99)),
            max_fifo_depth_max=("max_fifo_depth", "max"),
            final_fifo_depth_p99=("final_fifo_depth", lambda x: x.quantile(0.99)),
            final_fifo_depth_max=("final_fifo_depth", "max"),
        )
        .reset_index()
    )
    fifo_summary.to_csv(args.output / "warm_start_fifo_summary.csv", index=False)


if __name__ == "__main__":
    main()
