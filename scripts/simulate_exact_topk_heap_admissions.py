from __future__ import annotations

import argparse
from pathlib import Path

import numba
import numpy as np
import pandas as pd

from temporal_dsa.metrics import stable_topk

K = 2048
P_VALUES = (1, 2, 4, 8, 16)
ADMISSION_LANES = (1, 2, 4)


@numba.njit(cache=True)
def worse(score_a: float, index_a: int, score_b: float, index_b: int) -> bool:
    return score_a < score_b or (score_a == score_b and index_a > index_b)


@numba.njit(cache=True)
def better(score_a: float, index_a: int, score_b: float, index_b: int) -> bool:
    return score_a > score_b or (score_a == score_b and index_a < index_b)


@numba.njit(cache=True)
def sift_down(scores: np.ndarray, indices: np.ndarray, start: int, size: int) -> None:
    parent = start
    while True:
        left = parent * 2 + 1
        if left >= size:
            return
        right = left + 1
        child = left
        if right < size and worse(
            scores[right], indices[right], scores[left], indices[left]
        ):
            child = right
        if not worse(scores[child], indices[child], scores[parent], indices[parent]):
            return
        scores[parent], scores[child] = scores[child], scores[parent]
        indices[parent], indices[child] = indices[child], indices[parent]
        parent = child


@numba.njit(cache=True)
def exact_heap_scan(
    values: np.ndarray, initial_indices: np.ndarray
) -> tuple[int, np.ndarray, np.ndarray]:
    heap_indices = initial_indices.copy()
    heap_scores = values[heap_indices].copy()
    for position in range(heap_indices.size // 2 - 1, -1, -1):
        sift_down(heap_scores, heap_indices, position, heap_indices.size)
    initial_mask = np.zeros(values.size, dtype=np.uint8)
    for token in initial_indices:
        initial_mask[token] = 1
    admissions = np.zeros(values.size, dtype=np.uint8)
    count = 0
    for token in range(values.size):
        if initial_mask[token]:
            continue
        value = values[token]
        if better(value, token, heap_scores[0], heap_indices[0]):
            heap_scores[0] = value
            heap_indices[0] = token
            sift_down(heap_scores, heap_indices, 0, heap_indices.size)
            admissions[token] = 1
            count += 1
    return count, admissions, heap_indices


def fifo_depth(flags: np.ndarray, scores_per_cycle: int, lanes: int) -> tuple[int, int]:
    padding = (-flags.size) % scores_per_cycle
    if padding:
        flags = np.pad(flags, (0, padding), constant_values=0)
    arrivals = flags.reshape(-1, scores_per_cycle).sum(axis=1, dtype=np.int32)
    cumulative = np.cumsum(arrivals - lanes, dtype=np.int32)
    prior_minimum = np.minimum.accumulate(
        np.concatenate((np.zeros(1, dtype=np.int32), cumulative[:-1]))
    )
    queue = cumulative - np.minimum(prior_minimum, cumulative)
    return int(queue.max(initial=0)), int(queue[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact cold/warm Top-K heap admission replay")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inventory = pd.read_csv(args.inventory)
    inventory = inventory[inventory.split.isin(["validation", "test"])]
    rows: list[dict] = []
    fifo_rows: list[dict] = []

    # Compile outside the timed trace loop.
    exact_heap_scan(np.arange(K + 1, dtype=np.float32), np.arange(K, dtype=np.int64))
    for trace_index, trace in enumerate(inventory.itertuples(), start=1):
        path = Path(trace.trace_file)
        if not path.is_absolute():
            path = args.repo / path
        with np.load(path, allow_pickle=False) as payload:
            scores = payload["scores"].astype(np.float32)
            lengths = payload["lengths"].astype(np.int32)
        for step in range(1, scores.shape[0]):
            previous = scores[step - 1, : int(lengths[step - 1])]
            current = scores[step, : int(lengths[step])]
            previous_topk = stable_topk(previous, K).astype(np.int64)
            expected = stable_topk(current, K)
            cold_initial = np.arange(K, dtype=np.int64)
            cold_count, cold_flags, cold_heap = exact_heap_scan(current, cold_initial)
            warm_count, warm_flags, warm_heap = exact_heap_scan(current, previous_topk)
            if not np.array_equal(np.sort(cold_heap), np.sort(expected)):
                raise RuntimeError(f"cold heap exactness failure: {path} step {step}")
            if not np.array_equal(np.sort(warm_heap), np.sort(expected)):
                raise RuntimeError(f"warm heap exactness failure: {path} step {step}")
            rows.append(
                {
                    "split": trace.split,
                    "prompt_id": trace.prompt_id,
                    "workload": trace.workload,
                    "context": int(trace.base_context_length),
                    "layer": int(trace.layer),
                    "step": step,
                    "current_length": current.size,
                    "cold_heap_admissions": cold_count,
                    "warm_heap_admissions": warm_count,
                    "admission_reduction": 1.0 - warm_count / max(1, cold_count),
                    "exact_match": True,
                }
            )
            for p in P_VALUES:
                for lanes in ADMISSION_LANES:
                    for start_mode, flags in [("cold", cold_flags), ("warm", warm_flags)]:
                        maximum, final_depth = fifo_depth(flags, p, lanes)
                        fifo_rows.append(
                            {
                                "context": int(trace.base_context_length),
                                "layer": int(trace.layer),
                                "step": step,
                                "start_mode": start_mode,
                                "scores_per_cycle": p,
                                "admission_lanes": lanes,
                                "max_fifo_depth": maximum,
                                "final_fifo_depth": final_depth,
                            }
                        )
        print(f"[{trace_index}/{len(inventory)}] {path.name}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "exact_heap_admission_rows.csv", index=False)
    summary = (
        frame.groupby("context")
        .agg(
            transitions=("step", "size"),
            cold_admissions_mean=("cold_heap_admissions", "mean"),
            cold_admissions_p95=("cold_heap_admissions", lambda x: x.quantile(0.95)),
            cold_admissions_p99=("cold_heap_admissions", lambda x: x.quantile(0.99)),
            cold_admissions_max=("cold_heap_admissions", "max"),
            warm_admissions_mean=("warm_heap_admissions", "mean"),
            warm_admissions_p95=("warm_heap_admissions", lambda x: x.quantile(0.95)),
            warm_admissions_p99=("warm_heap_admissions", lambda x: x.quantile(0.99)),
            warm_admissions_max=("warm_heap_admissions", "max"),
            admission_reduction_mean=("admission_reduction", "mean"),
            admission_reduction_p5=("admission_reduction", lambda x: x.quantile(0.05)),
            exact_match_rate=("exact_match", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(args.output / "exact_heap_admission_summary.csv", index=False)
    fifo = pd.DataFrame(fifo_rows)
    fifo_summary = (
        fifo.groupby(["context", "start_mode", "scores_per_cycle", "admission_lanes"])
        .agg(
            transitions=("step", "size"),
            max_fifo_depth_mean=("max_fifo_depth", "mean"),
            max_fifo_depth_p95=("max_fifo_depth", lambda x: x.quantile(0.95)),
            max_fifo_depth_p99=("max_fifo_depth", lambda x: x.quantile(0.99)),
            max_fifo_depth_max=("max_fifo_depth", "max"),
            final_fifo_depth_p99=("final_fifo_depth", lambda x: x.quantile(0.99)),
        )
        .reset_index()
    )
    fifo_summary.to_csv(args.output / "exact_heap_fifo_summary.csv", index=False)


if __name__ == "__main__":
    main()
