from __future__ import annotations

import math
import heapq
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable

import numpy as np
import pandas as pd

from .metrics import stable_topk
from .trace import ScoreTrace


@dataclass(frozen=True)
class ApproxPolicy:
    """Runtime-legal temporal policy used by the approximate pilot.

    ``gamma`` is an absolute score margin calibrated on validation traces.  In
    own-trajectory mode only refreshed block metadata is updated; skipped
    blocks retain stale values and get older.
    """

    name: str
    gamma: float = 0.0
    cold_streak: int = 1
    dynamic_threshold: bool = False
    order: str = "address"
    refresh_interval: int | None = None
    age_cap: int | None = None
    recent_window: int = 0
    risk_model: str = "last_max"
    ema_alpha: float = 0.9
    volatility_lambda: float = 0.0
    age_slope: float = 0.0
    fallback_threshold: float | None = None

    def with_gamma(self, gamma: float) -> "ApproxPolicy":
        return replace(self, gamma=float(gamma))


@dataclass
class ApproxState:
    topk: np.ndarray
    baseline_topk: np.ndarray
    last_max: np.ndarray
    ema: np.ndarray
    volatility: np.ndarray
    age: np.ndarray
    cold_streak: np.ndarray


def _block_max(values: np.ndarray, block_size: int) -> np.ndarray:
    starts = np.arange(0, values.size, block_size)
    return np.maximum.reduceat(values, starts).astype(np.float32, copy=False)


def _extend(values: np.ndarray, size: int, fill: float | int) -> np.ndarray:
    if values.size >= size:
        return values[:size].copy()
    return np.pad(values, (0, size - values.size), constant_values=fill)


def _topk_from_indices(scores: np.ndarray, indices: np.ndarray, k: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size < k:
        raise RuntimeError(f"policy scored {indices.size} keys, fewer than K={k}")
    local = stable_topk(scores[indices], k)
    return indices[local]


def _mass_ratio(scores: np.ndarray, approximate: np.ndarray, baseline: np.ndarray) -> float:
    shift = float(np.max(scores[baseline]))
    numerator = np.exp(scores[approximate].astype(np.float64) - shift).sum()
    denominator = np.exp(scores[baseline].astype(np.float64) - shift).sum()
    return float(numerator / max(denominator, 1e-300))


def _retention(approximate: np.ndarray, ranked_baseline: np.ndarray, count: int) -> float:
    count = min(count, ranked_baseline.size)
    return float(np.isin(ranked_baseline[:count], approximate, assume_unique=True).sum() / count)


def _metadata_bytes_per_block(policy: ApproxPolicy) -> int:
    # Packed hardware model: fp32 max, uint8 streak/age/bucket.  EMA policies
    # add fp32 mean and volatility.  Round to a practical aligned size.
    raw = 4 + 1 + 1 + (1 if policy.order.startswith("bucket") else 0)
    if policy.risk_model == "ema":
        raw += 8
    return int(math.ceil(raw / 4) * 4)


def _order_blocks(policy: ApproxPolicy, risk: np.ndarray) -> np.ndarray:
    address = np.arange(risk.size, dtype=np.int64)
    finite = np.nan_to_num(risk, nan=-np.inf)
    if policy.order == "address":
        return address
    if policy.order in {"previous_max", "predicted_risk"}:
        return np.lexsort((address, -finite))
    if policy.order.startswith("bucket"):
        bucket_count = int(policy.order.removeprefix("bucket"))
        rank = np.empty(risk.size, dtype=np.int64)
        sorted_ids = np.lexsort((address, -finite))
        rank[sorted_ids] = np.arange(risk.size)
        bucket = np.minimum(bucket_count - 1, rank * bucket_count // max(1, risk.size))
        return np.lexsort((address, bucket))
    if policy.order == "oracle_current":
        raise ValueError("oracle order must be supplied with current block maxima")
    raise ValueError(f"unknown block order: {policy.order}")


def initialize_state(scores: np.ndarray, *, k: int, block_size: int) -> ApproxState:
    scores = np.asarray(scores, dtype=np.float32)
    topk = stable_topk(scores, k)
    maxima = _block_max(scores, block_size)
    hot = np.zeros(maxima.size, dtype=bool)
    hot[topk // block_size] = True
    return ApproxState(
        topk=topk,
        baseline_topk=topk.copy(),
        last_max=maxima,
        ema=maxima.copy(),
        volatility=np.zeros(maxima.size, dtype=np.float32),
        age=np.zeros(maxima.size, dtype=np.int16),
        cold_streak=(~hot).astype(np.int16),
    )


def replay_step(
    state: ApproxState,
    current_scores: np.ndarray,
    *,
    policy: ApproxPolicy,
    k: int,
    block_size: int,
    step: int,
    history_mode: str = "own",
    previous_length: int | None = None,
    fallback_value: float | None = None,
) -> tuple[ApproxState, dict[str, Any], dict[str, np.ndarray]]:
    """Replay one transition without leaking skipped current block scores."""

    if history_mode not in {"own", "teacher"}:
        raise ValueError("history_mode must be 'own' or 'teacher'")
    scores = np.asarray(current_scores, dtype=np.float32)
    length = scores.size
    if k > length:
        raise ValueError("K cannot exceed the current trace length")
    previous_length = length if previous_length is None else int(previous_length)
    block_count = (length + block_size - 1) // block_size
    last_max = _extend(state.last_max, block_count, -math.inf).astype(np.float32)
    ema = _extend(state.ema, block_count, -math.inf).astype(np.float32)
    volatility = _extend(state.volatility, block_count, 0.0).astype(np.float32)
    age = _extend(state.age, block_count, 0).astype(np.int16)
    cold_streak = _extend(state.cold_streak, block_count, 1).astype(np.int16)

    baseline = stable_topk(scores, k)
    seed = state.baseline_topk if history_mode == "teacher" else state.topk
    seed = np.asarray(seed[seed < length], dtype=np.int64)
    if seed.size != k or np.unique(seed).size != k:
        raise RuntimeError(f"previous Top-K seed has {seed.size} unique keys, expected {k}")

    evaluated = np.zeros(length, dtype=bool)
    evaluated[seed] = True
    tau_seed = float(np.min(scores[seed]))
    full_refresh = bool(
        policy.name == "full"
        or (policy.refresh_interval and step % policy.refresh_interval == 0)
        or (
            policy.fallback_threshold is not None
            and fallback_value is not None
            and fallback_value > policy.fallback_threshold
        )
    )

    current_max = _block_max(scores, block_size)
    if policy.risk_model == "last_max":
        risk = last_max.astype(np.float64) + policy.gamma + policy.age_slope * age
    elif policy.risk_model == "ema":
        risk = (
            ema.astype(np.float64)
            + policy.volatility_lambda * volatility
            + policy.gamma
            + policy.age_slope * age
        )
    else:
        raise ValueError(f"unknown risk model: {policy.risk_model}")

    if policy.order == "oracle_current":
        address = np.arange(block_count, dtype=np.int64)
        order = np.lexsort((address, -current_max))
    else:
        order = _order_blocks(policy, risk)

    threshold = tau_seed
    running_heap: list[tuple[float, int]] | None = None
    if policy.dynamic_threshold:
        # A larger key index loses deterministic ties, hence ``-key`` in the
        # min-heap's second tuple position.
        running_heap = [(float(scores[key]), -int(key)) for key in seed]
        heapq.heapify(running_heap)
    refreshed = np.zeros(block_count, dtype=bool)
    skipped = np.zeros(block_count, dtype=bool)
    new_block = np.zeros(block_count, dtype=bool)
    if previous_length < length:
        new_block[previous_length // block_size :] = True
    recent_block = np.zeros(block_count, dtype=bool)
    if policy.recent_window:
        recent_start = max(0, length - policy.recent_window)
        recent_block[recent_start // block_size :] = True

    for block_id in order:
        block_id = int(block_id)
        start = block_id * block_size
        stop = min(length, start + block_size)
        force = bool(
            full_refresh
            or new_block[block_id]
            or recent_block[block_id]
            or (policy.age_cap is not None and age[block_id] >= policy.age_cap)
            or cold_streak[block_id] < policy.cold_streak
        )
        can_skip = not force and risk[block_id] < threshold
        if can_skip:
            skipped[block_id] = True
            continue
        unseen = np.flatnonzero(~evaluated[start:stop]) + start
        evaluated[start:stop] = True
        refreshed[block_id] = True
        if running_heap is not None:
            if unseen.size:
                candidate_scores = scores[unseen]
                possible = unseen[candidate_scores >= running_heap[0][0]]
                for key in possible:
                    item = (float(scores[key]), -int(key))
                    if item > running_heap[0]:
                        heapq.heapreplace(running_heap, item)
                threshold = running_heap[0][0]

    scored = np.flatnonzero(evaluated)
    if running_heap is None:
        approximate = _topk_from_indices(scores, scored, k)
    else:
        approximate = np.asarray(
            [-item[1] for item in sorted(running_heap, reverse=True)], dtype=np.int64
        )
    approx_mask = np.zeros(length, dtype=bool)
    approx_mask[approximate] = True
    base_mask = np.zeros(length, dtype=bool)
    base_mask[baseline] = True
    retained = int(base_mask[approximate].sum())

    # State update is deliberately limited to refreshed blocks in own mode.
    new_last = last_max.copy()
    new_ema = ema.copy()
    new_volatility = volatility.copy()
    new_age = np.minimum(age.astype(np.int32) + 1, np.iinfo(np.int16).max).astype(np.int16)
    for block_id in np.flatnonzero(refreshed):
        observed = float(current_max[block_id])
        previous_mean = float(ema[block_id]) if np.isfinite(ema[block_id]) else observed
        new_last[block_id] = observed
        new_ema[block_id] = (
            policy.ema_alpha * observed + (1.0 - policy.ema_alpha) * previous_mean
        )
        new_volatility[block_id] = (
            policy.ema_alpha * abs(observed - previous_mean)
            + (1.0 - policy.ema_alpha) * volatility[block_id]
        )
        new_age[block_id] = 0
    if history_mode == "teacher":
        new_last = current_max.copy()
        new_ema = current_max.copy()
        new_volatility.fill(0.0)
        new_age.fill(0)

    hot = np.zeros(block_count, dtype=bool)
    hot[(baseline if history_mode == "teacher" else approximate) // block_size] = True
    next_cold = np.where(hot, 0, np.minimum(cold_streak.astype(np.int32) + 1, 32767))
    next_state = ApproxState(
        topk=approximate,
        baseline_topk=baseline,
        last_max=new_last,
        ema=new_ema,
        volatility=new_volatility,
        age=new_age,
        cold_streak=next_cold.astype(np.int16),
    )

    previous_baseline = state.baseline_topk
    newly_hot = np.setdiff1d(baseline, previous_baseline, assume_unique=False)
    new_missed = int((~approx_mask[newly_hot]).sum()) if newly_hot.size else 0
    missed_rank = np.flatnonzero(~approx_mask[baseline]) + 1
    seed_blocks = np.unique(seed // block_size)
    metadata_bpb = _metadata_bytes_per_block(policy)
    metadata_bytes = block_count * metadata_bpb
    metrics: dict[str, Any] = {
        "step": step,
        "context_length": length,
        "history_mode": history_mode,
        "policy": policy.name,
        "gamma": policy.gamma,
        "k": k,
        "block_size": block_size,
        "qk_scored": int(scored.size),
        "qk_reduction": float(1.0 - scored.size / length),
        "exact_match": bool(np.array_equal(np.sort(approximate), np.sort(baseline))),
        "recall": float(retained / k),
        "top128_recall": _retention(approximate, baseline, 128),
        "top256_recall": _retention(approximate, baseline, 256),
        "top512_recall": _retention(approximate, baseline, 512),
        "top1024_recall": _retention(approximate, baseline, 1024),
        "top2048_recall": _retention(approximate, baseline, 2048),
        "index_mass_ratio": _mass_ratio(scores, approximate, baseline),
        "newly_hot_count": int(newly_hot.size),
        "newly_hot_missed": new_missed,
        "newly_hot_miss_rate": float(new_missed / newly_hot.size) if newly_hot.size else 0.0,
        "missed_count": int(missed_rank.size),
        "worst_missed_rank": int(missed_rank.min()) if missed_rank.size else 0,
        "tau_seed": tau_seed,
        "tau_final": float(np.min(scores[approximate])),
        "full_refresh": full_refresh,
        "fallback": bool(
            policy.fallback_threshold is not None
            and fallback_value is not None
            and fallback_value > policy.fallback_threshold
        ),
        "blocks_refreshed": int(refreshed.sum()),
        "blocks_skipped": int(skipped.sum()),
        "seed_blocks_touched": int(seed_blocks.size),
        "seed_block_fraction": float(seed_blocks.size / block_count),
        "metadata_bytes_per_block": metadata_bpb,
        "metadata_bytes": metadata_bytes,
        "ideal_bf16_bytes": int(scored.size * 256 + metadata_bytes),
        "ideal_fp8_bytes": int(scored.size * 128 + metadata_bytes),
        "net_bf16_reduction": float(1.0 - (scored.size * 256 + metadata_bytes) / (length * 256)),
        "net_fp8_reduction": float(1.0 - (scored.size * 128 + metadata_bytes) / (length * 128)),
    }
    for physical_block in (16, 32, 64, 128):
        touched = np.unique(scored // physical_block).size
        full_blocks = (length + physical_block - 1) // physical_block
        metrics[f"physical_b{physical_block}_blocks"] = int(touched)
        metrics[f"physical_b{physical_block}_reduction"] = float(1.0 - touched / full_blocks)
    for cache_entries in (0, 256, 512, 1024, 2048):
        cached = seed[: min(cache_entries, seed.size)]
        uncached = evaluated.copy()
        uncached[cached] = False
        hbm_keys = int(uncached.sum())
        metrics[f"cache{cache_entries}_hbm_keys"] = hbm_keys
        metrics[f"cache{cache_entries}_net_bf16_reduction"] = float(
            1.0 - (hbm_keys * 256 + metadata_bytes) / (length * 256)
        )

    details = {
        "approximate": approximate,
        "baseline": baseline,
        "evaluated": scored,
        "missed_ranks": missed_rank.astype(np.int32),
    }
    return next_state, metrics, details


def replay_approx_trace(
    trace: ScoreTrace,
    *,
    policy: ApproxPolicy,
    k: int = 2048,
    block_size: int = 64,
    history_mode: str = "own",
    fallback_values: np.ndarray | None = None,
    detail_callback: Callable[[int, dict[str, np.ndarray]], None] | None = None,
    max_transitions: int | None = None,
) -> pd.DataFrame:
    steps = trace.scores.shape[0]
    if steps < 2:
        return pd.DataFrame()
    state = initialize_state(trace.row(0), k=k, block_size=block_size)
    rows: list[dict[str, Any]] = []
    stop = steps if max_transitions is None else min(steps, max_transitions + 1)
    for step in range(1, stop):
        state, metrics, details = replay_step(
            state,
            trace.row(step),
            policy=policy,
            k=k,
            block_size=block_size,
            step=step,
            history_mode=history_mode,
            previous_length=int(trace.lengths[step - 1]),
            fallback_value=None if fallback_values is None else float(fallback_values[step]),
        )
        metrics.update(
            {
                "layer": trace.layer,
                "workload": trace.workload,
                "prompt_id": trace.prompt_id,
            }
        )
        rows.append(metrics)
        if detail_callback is not None:
            detail_callback(step, details)
    return pd.DataFrame(rows)


def policy_dict(policy: ApproxPolicy) -> dict[str, Any]:
    return asdict(policy)


def positive_block_delta_quantiles(
    traces: list[ScoreTrace],
    *,
    block_size: int,
    quantiles: list[float],
) -> dict[int, dict[str, float]]:
    """Calibrate per-layer margins using validation traces only."""

    by_layer: dict[int, list[np.ndarray]] = {}
    for trace in traces:
        pieces = by_layer.setdefault(trace.layer, [])
        for step in range(1, trace.scores.shape[0]):
            previous = trace.row(step - 1)
            current = trace.row(step)
            common = min(previous.size, current.size)
            prev_max = _block_max(previous[:common], block_size)
            curr_max = _block_max(current[:common], block_size)
            pieces.append(np.maximum(0.0, curr_max - prev_max))
    result: dict[int, dict[str, float]] = {}
    for layer, pieces in by_layer.items():
        values = np.concatenate(pieces).astype(np.float64)
        result[layer] = {
            f"p{q:g}": float(np.percentile(values, q)) for q in quantiles
        }
        result[layer]["max"] = float(np.max(values))
    return result
