from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .metrics import pearson, spearman, stable_topk, topk_recall
from .trace import ScoreTrace

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised by the pure-Python CI path
    njit = None


if njit is not None:
    @njit(cache=True)
    def _worse(score_a: float, index_a: int, score_b: float, index_b: int) -> bool:
        return score_a < score_b or (score_a == score_b and index_a > index_b)


    @njit(cache=True)
    def _sift_down(heap_scores: np.ndarray, heap_indices: np.ndarray, root: int) -> None:
        size = heap_scores.size
        while True:
            left = root * 2 + 1
            if left >= size:
                return
            worst = left
            right = left + 1
            if right < size and _worse(
                heap_scores[right], heap_indices[right], heap_scores[left], heap_indices[left]
            ):
                worst = right
            if not _worse(
                heap_scores[worst], heap_indices[worst], heap_scores[root], heap_indices[root]
            ):
                return
            heap_scores[root], heap_scores[worst] = heap_scores[worst], heap_scores[root]
            heap_indices[root], heap_indices[worst] = heap_indices[worst], heap_indices[root]
            root = worst


    @njit(cache=True)
    def _compiled_filter(
        current: np.ndarray,
        previous_topk: np.ndarray,
        true_topk: np.ndarray,
        upper: np.ndarray,
        order: np.ndarray,
        block_size: int,
        fixed_threshold: float,
        dynamic: bool,
    ) -> tuple[int, int, int, int, float]:
        length = current.size
        k = true_topk.size
        seeded = np.zeros(length, dtype=np.uint8)
        true_mask = np.zeros(length, dtype=np.uint8)
        true_block = np.zeros(upper.size, dtype=np.uint8)
        for key in true_topk:
            true_mask[key] = 1
            true_block[key // block_size] = 1

        heap_scores = np.empty(k, dtype=np.float64)
        heap_indices = np.empty(k, dtype=np.int64)
        scored = 0
        true_evaluated = 0
        last_true_position = 0
        for position in range(k):
            key = previous_topk[position]
            seeded[key] = 1
            heap_scores[position] = current[key]
            heap_indices[position] = key
            scored += 1
            if true_mask[key]:
                true_evaluated += 1
                last_true_position = scored
        for root in range(k // 2 - 1, -1, -1):
            _sift_down(heap_scores, heap_indices, root)

        threshold = heap_scores[0] if dynamic else fixed_threshold
        pruned = 0
        false_cold = 0
        for block_id in order:
            if upper[block_id] < threshold:
                pruned += 1
                false_cold += true_block[block_id]
                continue
            start = block_id * block_size
            end = min(length, start + block_size)
            for key in range(start, end):
                if seeded[key]:
                    continue
                scored += 1
                if true_mask[key]:
                    true_evaluated += 1
                    last_true_position = scored
                if dynamic and (
                    current[key] > heap_scores[0]
                    or (current[key] == heap_scores[0] and key < heap_indices[0])
                ):
                    heap_scores[0] = current[key]
                    heap_indices[0] = key
                    _sift_down(heap_scores, heap_indices, 0)
                    threshold = heap_scores[0]
        discovery = 1.0 if true_evaluated < k else last_true_position / length
        return scored, pruned, false_cold, true_evaluated, discovery
else:
    _compiled_filter = None


@dataclass(frozen=True)
class ReplayResult:
    method: str
    scan_order: str
    step: int
    layer: int
    workload: str
    prompt_id: str
    context_length: int
    k: int
    block_size: int
    gamma_sigma: float
    gamma: float
    tau_seed: float
    tau_final: float
    threshold_ratio: float
    threshold_gap: float
    qk_scored: int
    qk_reduction: float
    blocks_pruned: int
    block_pruning_rate: float
    false_cold_blocks: int
    false_cold_rate: float
    recall: float
    exact_match: bool
    discovery_fraction: float
    key_bytes: int
    metadata_bytes: int
    net_byte_reduction: float


def _block_slices(length: int, block_size: int) -> list[slice]:
    return [slice(start, min(length, start + block_size)) for start in range(0, length, block_size)]


def _block_max(values: np.ndarray, blocks: list[slice]) -> np.ndarray:
    return np.asarray([np.max(values[block]) for block in blocks], dtype=np.float64)


def _scan_order(
    name: str,
    upper: np.ndarray,
    current_max: np.ndarray,
    previous_hot: np.ndarray,
) -> np.ndarray:
    address = np.arange(upper.size)
    if name == "address":
        return address
    if name == "previous_hot":
        # Stable sort: hot blocks first, then high U, then address.
        return np.lexsort((address, -upper, -previous_hot.astype(np.int8)))
    if name == "upper_hot":
        return np.lexsort((address, -upper))
    if name == "oracle_current":
        return np.lexsort((address, -current_max))
    raise ValueError(f"unknown scan order: {name}")


def _run_filter(
    *,
    current: np.ndarray,
    previous_topk: np.ndarray,
    true_topk: np.ndarray,
    blocks: list[slice],
    upper: np.ndarray,
    fixed_threshold: float | None,
    order_name: str,
    current_max: np.ndarray,
    method: str,
) -> tuple[np.ndarray, list[int], int, float]:
    length = current.size
    evaluated = np.zeros(length, dtype=bool)
    evaluation_position = np.full(length, length, dtype=np.int64)
    scored = 0

    # Previous top-k is an explicit seed: score these keys before block replay.
    for key in previous_topk:
        key = int(key)
        if key < length and not evaluated[key]:
            evaluated[key] = True
            scored += 1
            evaluation_position[key] = scored

    threshold = (
        float(fixed_threshold)
        if fixed_threshold is not None
        else float(np.min(current[previous_topk]))
    )
    previous_hot_blocks = np.zeros(len(blocks), dtype=bool)
    previous_hot_blocks[np.minimum(previous_topk // (blocks[0].stop - blocks[0].start), len(blocks) - 1)] = True
    order = _scan_order(order_name, upper, current_max, previous_hot_blocks)
    pruned: list[int] = []

    for block_id in order:
        block_id = int(block_id)
        if upper[block_id] < threshold:
            pruned.append(block_id)
            continue
        block = blocks[block_id]
        unseen = np.flatnonzero(~evaluated[block]) + block.start
        if unseen.size:
            evaluation_position[unseen] = np.arange(scored + 1, scored + unseen.size + 1)
            evaluated[unseen] = True
            scored += int(unseen.size)
        if fixed_threshold is None and np.count_nonzero(evaluated) >= true_topk.size:
            threshold = float(np.partition(current[evaluated], -true_topk.size)[-true_topk.size])

    discovery = float(np.max(evaluation_position[true_topk], initial=0) / length)
    return evaluated, pruned, scored, discovery


def replay_transition(
    previous: np.ndarray,
    current: np.ndarray,
    *,
    k: int,
    block_size: int,
    gamma_sigma: float,
    method: str,
    scan_order: str,
    step: int = 0,
    layer: int = 0,
    workload: str = "synthetic",
    prompt_id: str = "0",
    key_bytes: int = 256,
    metadata_bytes_per_block: int = 8,
    _previous_topk: np.ndarray | None = None,
    _true_topk: np.ndarray | None = None,
    _block_cache: tuple[list[slice], np.ndarray, np.ndarray, float] | None = None,
    _absolute_gamma: float | None = None,
) -> ReplayResult:
    previous = np.asarray(previous, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    if previous.ndim != 1 or current.ndim != 1 or previous.size > current.size:
        raise ValueError("expected 1-D scores with previous length <= current length")
    if not 0 < k <= previous.size:
        raise ValueError("k must fit the previous step so it can seed replay")

    if _block_cache is None:
        blocks = _block_slices(current.size, block_size)
        previous_blocks = _block_slices(previous.size, block_size)
        current_max = _block_max(current, blocks)
        previous_max = np.full(len(blocks), -math.inf, dtype=np.float64)
        previous_max[: len(previous_blocks)] = _block_max(previous, previous_blocks)
        sigma = float(np.std(current[: previous.size] - previous))
    else:
        blocks, current_max, previous_max, sigma = _block_cache
    gamma = gamma_sigma * sigma if _absolute_gamma is None else _absolute_gamma
    upper = previous_max + gamma
    # A partial tail block contains at least one unseen key and is never pruned.
    if previous.size < current.size:
        first_new_block = previous.size // block_size
        upper[first_new_block:] = math.inf

    previous_topk = stable_topk(previous, k) if _previous_topk is None else _previous_topk
    true_topk = stable_topk(current, k) if _true_topk is None else _true_topk
    tau_seed = float(np.min(current[previous_topk]))
    tau_final = float(np.min(current[true_topk]))
    fast_metrics: tuple[int, int, int, int, float] | None = None

    if method != "full" and _compiled_filter is not None:
        previous_hot_blocks = np.zeros(len(blocks), dtype=bool)
        previous_hot_blocks[previous_topk // block_size] = True
        if method == "static":
            scan_order = "address"
            filter_upper = upper
            fixed_threshold = tau_seed
            dynamic_threshold = False
        elif method == "static_hot_first":
            scan_order = "previous_hot"
            filter_upper = upper
            fixed_threshold = tau_seed
            dynamic_threshold = False
        elif method == "dynamic":
            filter_upper = upper
            fixed_threshold = tau_seed
            dynamic_threshold = True
        elif method == "oracle":
            scan_order = "oracle_current"
            filter_upper = current_max
            fixed_threshold = tau_seed
            dynamic_threshold = True
        else:
            raise ValueError(f"unknown method: {method}")
        order = _scan_order(scan_order, filter_upper, current_max, previous_hot_blocks)
        fast_metrics = _compiled_filter(
            current,
            previous_topk.astype(np.int64),
            true_topk.astype(np.int64),
            filter_upper,
            order.astype(np.int64),
            block_size,
            fixed_threshold,
            dynamic_threshold,
        )

    if method == "full":
        evaluated = np.ones(current.size, dtype=bool)
        pruned: list[int] = []
        qk_scored = current.size
        discovery = 1.0
    elif fast_metrics is not None:
        qk_scored, pruned_count, pruned_true, true_evaluated, discovery = fast_metrics
        evaluated = None
        pruned = []
    elif method in {"static", "static_hot_first"}:
        order = "address" if method == "static" else "previous_hot"
        evaluated, pruned, qk_scored, discovery = _run_filter(
            current=current,
            previous_topk=previous_topk,
            true_topk=true_topk,
            blocks=blocks,
            upper=upper,
            fixed_threshold=tau_seed,
            order_name=order,
            current_max=current_max,
            method=method,
        )
        scan_order = order
    elif method == "dynamic":
        evaluated, pruned, qk_scored, discovery = _run_filter(
            current=current,
            previous_topk=previous_topk,
            true_topk=true_topk,
            blocks=blocks,
            upper=upper,
            fixed_threshold=None,
            order_name=scan_order,
            current_max=current_max,
            method=method,
        )
    elif method == "oracle":
        evaluated, pruned, qk_scored, discovery = _run_filter(
            current=current,
            previous_topk=previous_topk,
            true_topk=true_topk,
            blocks=blocks,
            upper=current_max,
            fixed_threshold=None,
            order_name="oracle_current",
            current_max=current_max,
            method=method,
        )
        scan_order = "oracle_current"
    else:
        raise ValueError(f"unknown method: {method}")

    if fast_metrics is None:
        pruned_count = len(pruned)
        true_blocks = np.zeros(len(blocks), dtype=bool)
        true_blocks[true_topk // block_size] = True
        pruned_true = int(true_blocks[np.asarray(pruned, dtype=np.int64)].sum()) if pruned else 0
        true_evaluated = int(evaluated[true_topk].sum())
    recall = true_evaluated / k
    exact = true_evaluated == k
    metadata_bytes = len(blocks) * metadata_bytes_per_block
    bytes_read = qk_scored * key_bytes + metadata_bytes
    full_bytes = current.size * key_bytes
    return ReplayResult(
        method=method,
        scan_order=scan_order,
        step=step,
        layer=layer,
        workload=workload,
        prompt_id=prompt_id,
        context_length=current.size,
        k=k,
        block_size=block_size,
        gamma_sigma=gamma_sigma,
        gamma=gamma,
        tau_seed=tau_seed,
        tau_final=tau_final,
        threshold_ratio=tau_seed / tau_final if tau_final != 0 else math.nan,
        threshold_gap=tau_final - tau_seed,
        qk_scored=qk_scored,
        qk_reduction=1.0 - qk_scored / current.size,
        blocks_pruned=pruned_count,
        block_pruning_rate=pruned_count / len(blocks),
        false_cold_blocks=pruned_true,
        false_cold_rate=pruned_true / pruned_count if pruned_count else 0.0,
        recall=recall,
        exact_match=exact,
        discovery_fraction=discovery,
        key_bytes=key_bytes,
        metadata_bytes=metadata_bytes,
        net_byte_reduction=1.0 - bytes_read / full_bytes,
    )


def temporal_stats(trace: ScoreTrace, k_values: list[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for step in range(1, trace.scores.shape[0]):
        previous = trace.row(step - 1)
        current = trace.row(step)
        common = min(previous.size, current.size)
        base = {
            "step": step,
            "layer": trace.layer,
            "workload": trace.workload,
            "prompt_id": trace.prompt_id,
            "context_length": current.size,
            "score_pearson": pearson(previous[:common], current[:common]),
            "score_spearman": spearman(previous[:common], current[:common]),
            "delta_mean": float(np.mean(current[:common] - previous[:common])),
            "delta_std": float(np.std(current[:common] - previous[:common])),
            "delta_p99_abs": float(np.percentile(np.abs(current[:common] - previous[:common]), 99)),
        }
        for k in k_values:
            if k > common:
                continue
            previous_topk = stable_topk(previous[:common], k)
            current_topk = stable_topk(current[:common], k)
            rows.append({**base, "k": k, "topk_overlap": topk_recall(previous_topk, current_topk)})
    return pd.DataFrame(rows)


def block_temporal_stats(
    trace: ScoreTrace, block_sizes: list[int], k_values: list[int]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cold_streak: dict[tuple[int, int, int], int] = {}
    for step in range(1, trace.scores.shape[0]):
        previous = trace.row(step - 1)
        current = trace.row(step)
        common = min(previous.size, current.size)
        for block_size in block_sizes:
            blocks = _block_slices(common, block_size)
            prev_max = _block_max(previous[:common], blocks)
            curr_max = _block_max(current[:common], blocks)
            for k in k_values:
                if k > common:
                    continue
                threshold = float(np.partition(current[:common], -k)[-k])
                streak_values = np.zeros(len(blocks), dtype=np.int32)
                for block_id, new in enumerate(curr_max):
                    key = (block_size, k, block_id)
                    cold_streak[key] = cold_streak.get(key, 0) + 1 if new < threshold else 0
                    streak_values[block_id] = cold_streak[key]
                delta = curr_max - prev_max
                rows.append(
                    {
                        "step": step,
                        "layer": trace.layer,
                        "workload": trace.workload,
                        "prompt_id": trace.prompt_id,
                        "context_length": current.size,
                        "k": k,
                        "block_size": block_size,
                        "topk_threshold": threshold,
                        "delta_max_mean": float(np.mean(delta)),
                        "delta_max_p5": float(np.percentile(delta, 5)),
                        "delta_max_median": float(np.median(delta)),
                        "delta_max_p95": float(np.percentile(delta, 95)),
                        "cold_fraction_1": float(np.mean(streak_values >= 1)),
                        "cold_fraction_2": float(np.mean(streak_values >= 2)),
                        "cold_fraction_4": float(np.mean(streak_values >= 4)),
                        "cold_fraction_8": float(np.mean(streak_values >= 8)),
                    }
                )
    return pd.DataFrame(rows)


def replay_trace(
    trace: ScoreTrace,
    *,
    k_values: list[int],
    block_sizes: list[int],
    gamma_sigma_values: list[float],
    key_bytes: int = 256,
    metadata_bytes_per_block: int = 8,
    aggregate: bool = True,
    absolute_margins: dict[int, float] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for step in range(1, trace.scores.shape[0]):
        previous = trace.row(step - 1)
        current = trace.row(step)
        sigma = float(np.std(current[: previous.size] - previous))
        block_caches = {}
        for cached_block_size in block_sizes:
            cached_blocks = _block_slices(current.size, cached_block_size)
            previous_blocks = _block_slices(previous.size, cached_block_size)
            cached_current_max = _block_max(current, cached_blocks)
            cached_previous_max = np.full(len(cached_blocks), -math.inf, dtype=np.float64)
            cached_previous_max[: len(previous_blocks)] = _block_max(previous, previous_blocks)
            block_caches[cached_block_size] = (
                cached_blocks,
                cached_current_max,
                cached_previous_max,
                sigma,
            )
        for k in k_values:
            if k > previous.size:
                continue
            previous_topk = stable_topk(previous, k)
            true_topk = stable_topk(current, k)
            for block_size in block_sizes:
                block_cache = block_caches[block_size]
                invariant_variants = [("full", "address"), ("oracle", "oracle_current")]
                for method, scan_order in invariant_variants:
                    result = replay_transition(
                        previous,
                        current,
                        k=k,
                        block_size=block_size,
                        gamma_sigma=0.0,
                        method=method,
                        scan_order=scan_order,
                        step=step,
                        layer=trace.layer,
                        workload=trace.workload,
                        prompt_id=trace.prompt_id,
                        key_bytes=key_bytes,
                        metadata_bytes_per_block=metadata_bytes_per_block,
                        _previous_topk=previous_topk,
                        _true_topk=true_topk,
                        _block_cache=block_cache,
                    )
                    rows.append(asdict(result))
                for gamma_sigma in gamma_sigma_values:
                    variants = [
                        ("static", "address"),
                        ("static_hot_first", "previous_hot"),
                        ("dynamic", "address"),
                        ("dynamic", "previous_hot"),
                        ("dynamic", "upper_hot"),
                        ("dynamic", "oracle_current"),
                    ]
                    for method, scan_order in variants:
                        result = replay_transition(
                            previous,
                            current,
                            k=k,
                            block_size=block_size,
                            gamma_sigma=gamma_sigma,
                            method=method,
                            scan_order=scan_order,
                            step=step,
                            layer=trace.layer,
                            workload=trace.workload,
                            prompt_id=trace.prompt_id,
                            key_bytes=key_bytes,
                            metadata_bytes_per_block=metadata_bytes_per_block,
                            _previous_topk=previous_topk,
                            _true_topk=true_topk,
                            _block_cache=block_cache,
                        )
                        rows.append(asdict(result))
                if absolute_margins and block_size in absolute_margins:
                    calibrated_gamma = float(absolute_margins[block_size])
                    calibrated_variants = [
                        ("static", "address"),
                        ("static_hot_first", "previous_hot"),
                        ("dynamic", "address"),
                        ("dynamic", "previous_hot"),
                        ("dynamic", "upper_hot"),
                    ]
                    for method, scan_order in calibrated_variants:
                        result = replay_transition(
                            previous,
                            current,
                            k=k,
                            block_size=block_size,
                            gamma_sigma=-1.0,
                            method=method,
                            scan_order=scan_order,
                            step=step,
                            layer=trace.layer,
                            workload=trace.workload,
                            prompt_id=trace.prompt_id,
                            key_bytes=key_bytes,
                            metadata_bytes_per_block=metadata_bytes_per_block,
                            _previous_topk=previous_topk,
                            _true_topk=true_topk,
                            _block_cache=block_cache,
                            _absolute_gamma=calibrated_gamma,
                        )
                        rows.append(asdict(result))
    frame = pd.DataFrame(rows)
    if not aggregate or frame.empty:
        return frame
    group = [
        "layer",
        "workload",
        "prompt_id",
        "k",
        "block_size",
        "gamma_sigma",
        "method",
        "scan_order",
    ]
    return frame.groupby(group, as_index=False).agg(
        transition_count=("step", "count"),
        context_length=("context_length", "median"),
        gamma=("gamma", "median"),
        tau_seed=("tau_seed", "median"),
        tau_final=("tau_final", "median"),
        threshold_ratio=("threshold_ratio", "median"),
        threshold_gap=("threshold_gap", "median"),
        qk_scored=("qk_scored", "median"),
        qk_reduction=("qk_reduction", "median"),
        qk_reduction_p5=("qk_reduction", lambda x: x.quantile(0.05)),
        qk_reduction_p95=("qk_reduction", lambda x: x.quantile(0.95)),
        blocks_pruned=("blocks_pruned", "median"),
        block_pruning_rate=("block_pruning_rate", "median"),
        false_cold_blocks=("false_cold_blocks", "sum"),
        false_cold_rate=("false_cold_rate", "mean"),
        recall=("recall", "mean"),
        exact_match=("exact_match", "mean"),
        discovery_fraction=("discovery_fraction", "median"),
        key_bytes=("key_bytes", "first"),
        metadata_bytes=("metadata_bytes", "median"),
        net_byte_reduction=("net_byte_reduction", "median"),
    )
