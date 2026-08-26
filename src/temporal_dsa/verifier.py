from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from .approx import ApproxPolicy, ApproxState, initialize_state, replay_step
from .metrics import stable_topk
from .trace import ScoreTrace


@dataclass(frozen=True)
class VerifierConfig:
    """Runtime-legal current-query candidate generator.

    ``score_ratio`` is the verifier MAC cost per cold token relative to one
    full 64-head x 128-D sidecar score.  It is deliberately explicit so cost
    accounting cannot confuse verifier scan work with candidate reranking.
    """

    name: str
    path: str
    width: int
    score_ratio: float
    block_size: int = 64
    rescue_fraction: float = 0.05
    sketch_bytes_per_token: float = 0.0
    full_key_bytes_per_token: int = 256
    metadata_bytes_per_block: int = 4
    retain_candidate_keys: bool = False

    def __post_init__(self) -> None:
        if self.path not in {"head", "dim"}:
            raise ValueError("path must be 'head' or 'dim'")
        if self.width <= 0:
            raise ValueError("width must be positive")
        if not 0 <= self.rescue_fraction <= 1:
            raise ValueError("rescue_fraction must be in [0, 1]")
        if not 0 <= self.score_ratio <= 1:
            raise ValueError("score_ratio must be in [0, 1]")


def _extend(values: np.ndarray, size: int, fill: float | int) -> np.ndarray:
    if values.size >= size:
        return values[:size].copy()
    return np.pad(values, (0, size - values.size), constant_values=fill)


def _block_max_masked(
    values: np.ndarray, available: np.ndarray, block_size: int
) -> tuple[np.ndarray, np.ndarray]:
    block_count = (values.size + block_size - 1) // block_size
    maxima = np.full(block_count, -np.inf, dtype=np.float32)
    cold_blocks = np.zeros(block_count, dtype=bool)
    for block_id in range(block_count):
        start = block_id * block_size
        stop = min(values.size, start + block_size)
        mask = available[start:stop]
        if mask.any():
            cold_blocks[block_id] = True
            maxima[block_id] = float(np.max(values[start:stop][mask]))
    return maxima, cold_blocks


def _rank_blocks(scores: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    block_ids = np.flatnonzero(eligible)
    if not block_ids.size:
        return block_ids.astype(np.int64)
    # Higher score wins and lower block address deterministically wins ties.
    return block_ids[np.lexsort((block_ids, -scores[block_ids]))].astype(np.int64)


def _average_precision(ranking: np.ndarray, positives: np.ndarray) -> float:
    positive_set = set(int(value) for value in np.asarray(positives).tolist())
    if not positive_set:
        return 1.0
    hits = 0
    precision_sum = 0.0
    for rank, block_id in enumerate(ranking, start=1):
        if int(block_id) in positive_set:
            hits += 1
            precision_sum += hits / rank
    return float(precision_sum / len(positive_set))


def _retention(approximate: np.ndarray, ranked_baseline: np.ndarray, count: int) -> float:
    count = min(count, ranked_baseline.size)
    if count == 0:
        return 1.0
    return float(np.isin(ranked_baseline[:count], approximate, assume_unique=True).sum() / count)


def _mass_ratio(scores: np.ndarray, approximate: np.ndarray, baseline: np.ndarray) -> float:
    shift = float(np.max(scores[baseline]))
    numerator = np.exp(scores[approximate].astype(np.float64) - shift).sum()
    denominator = np.exp(scores[baseline].astype(np.float64) - shift).sum()
    return float(numerator / max(denominator, 1e-300))


def apply_rescue_to_state(
    previous: ApproxState,
    temporal_next: ApproxState,
    scores: np.ndarray,
    evaluated: np.ndarray,
    rescued_blocks: np.ndarray,
    *,
    policy: ApproxPolicy,
    k: int,
    block_size: int,
) -> tuple[ApproxState, np.ndarray, np.ndarray]:
    """Rerank the exact candidate union and update only rescued metadata.

    The temporal transition has already updated blocks it refreshed.  This
    function observes current full scores only for the selected rescue blocks;
    every other skipped block keeps its stale state and increments age.
    """

    scores = np.asarray(scores, dtype=np.float32)
    exact = np.zeros(scores.size, dtype=bool)
    exact[np.asarray(evaluated, dtype=np.int64)] = True
    for block_id in np.asarray(rescued_blocks, dtype=np.int64):
        start = int(block_id) * block_size
        exact[start : min(scores.size, start + block_size)] = True
    exact_ids = np.flatnonzero(exact)
    if exact_ids.size < k:
        raise RuntimeError(f"candidate union has {exact_ids.size} keys, fewer than K={k}")
    approximate = exact_ids[stable_topk(scores[exact_ids], k)]

    block_count = (scores.size + block_size - 1) // block_size
    last_max = _extend(temporal_next.last_max, block_count, -math.inf).astype(np.float32)
    ema = _extend(temporal_next.ema, block_count, -math.inf).astype(np.float32)
    volatility = _extend(temporal_next.volatility, block_count, 0.0).astype(np.float32)
    age = _extend(temporal_next.age, block_count, 0).astype(np.int16)
    previous_ema = _extend(previous.ema, block_count, -math.inf).astype(np.float32)
    previous_volatility = _extend(previous.volatility, block_count, 0.0).astype(np.float32)
    previous_cold = _extend(previous.cold_streak, block_count, 1).astype(np.int16)

    for block_id in np.asarray(rescued_blocks, dtype=np.int64):
        start = int(block_id) * block_size
        stop = min(scores.size, start + block_size)
        observed = float(np.max(scores[start:stop]))
        old_mean = float(previous_ema[block_id])
        if not np.isfinite(old_mean):
            old_mean = observed
        last_max[block_id] = observed
        ema[block_id] = policy.ema_alpha * observed + (1.0 - policy.ema_alpha) * old_mean
        volatility[block_id] = (
            policy.ema_alpha * abs(observed - old_mean)
            + (1.0 - policy.ema_alpha) * previous_volatility[block_id]
        )
        age[block_id] = 0

    hot = np.zeros(block_count, dtype=bool)
    hot[approximate // block_size] = True
    cold_streak = np.where(
        hot, 0, np.minimum(previous_cold.astype(np.int32) + 1, 32767)
    ).astype(np.int16)
    state = ApproxState(
        topk=approximate,
        baseline_topk=temporal_next.baseline_topk,
        last_max=last_max,
        ema=ema,
        volatility=volatility,
        age=age,
        cold_streak=cold_streak,
    )
    return state, approximate, exact_ids


def verifier_step(
    state: ApproxState,
    current_scores: np.ndarray,
    verifier_scores: np.ndarray,
    *,
    policy: ApproxPolicy,
    config: VerifierConfig,
    k: int,
    step: int,
    previous_length: int | None = None,
    tail_critical_blocks: np.ndarray | None = None,
    promotion_threshold: float | None = None,
) -> tuple[ApproxState, dict[str, Any], dict[str, np.ndarray]]:
    """Run temporal filtering, current-query verification and exact rerank."""

    scores = np.asarray(current_scores, dtype=np.float32)
    verifier_scores = np.asarray(verifier_scores[: scores.size], dtype=np.float32)
    if verifier_scores.shape != scores.shape:
        raise ValueError("verifier score length must match current score length")
    previous_state = state
    temporal_next, temporal_metrics, temporal_detail = replay_step(
        state,
        scores,
        policy=policy,
        k=k,
        block_size=config.block_size,
        step=step,
        history_mode="own",
        previous_length=previous_length,
    )
    evaluated_mask = np.zeros(scores.size, dtype=bool)
    evaluated_mask[temporal_detail["evaluated"]] = True
    cold_mask = ~evaluated_mask
    block_scores, cold_blocks = _block_max_masked(
        verifier_scores, cold_mask, config.block_size
    )
    ranking = _rank_blocks(block_scores, cold_blocks)
    if promotion_threshold is None:
        rescue_count = min(
            ranking.size,
            int(math.ceil(config.rescue_fraction * int(cold_blocks.sum()))),
        )
        rescued_blocks = ranking[:rescue_count]
        promotion_policy = "global_budget"
    else:
        rescued_blocks = ranking[block_scores[ranking] >= float(promotion_threshold)]
        promotion_policy = "fixed_threshold"
    state, approximate, exact_ids = apply_rescue_to_state(
        previous_state,
        temporal_next,
        scores,
        temporal_detail["evaluated"],
        rescued_blocks,
        policy=policy,
        k=k,
        block_size=config.block_size,
    )

    baseline = temporal_detail["baseline"]
    base_mask = np.zeros(scores.size, dtype=bool)
    base_mask[baseline] = True
    approx_mask = np.zeros(scores.size, dtype=bool)
    approx_mask[approximate] = True
    newly_active = baseline[cold_mask[baseline]]
    newly_active_blocks = np.unique(newly_active // config.block_size)
    rescued_token_mask = np.isin(newly_active, exact_ids, assume_unique=False)
    rescued_positive_blocks = np.intersect1d(
        rescued_blocks, newly_active_blocks, assume_unique=False
    )
    tail_blocks = np.asarray(
        [] if tail_critical_blocks is None else tail_critical_blocks, dtype=np.int64
    )
    rescued_tail = np.intersect1d(rescued_blocks, tail_blocks, assume_unique=False)

    temporal_exact = int(temporal_detail["evaluated"].size)
    rescue_new = int(exact_ids.size - temporal_exact)
    verifier_tokens = int(cold_mask.sum())
    full_equivalent = temporal_exact + rescue_new + verifier_tokens * config.score_ratio
    full_work = float(scores.size)
    block_count = (scores.size + config.block_size - 1) // config.block_size
    metadata_bytes = block_count * config.metadata_bytes_per_block
    temporal_key_bytes = temporal_exact * config.full_key_bytes_per_token
    if config.path == "head":
        verifier_key_bytes = verifier_tokens * config.full_key_bytes_per_token
        reread_key_bytes = 0 if config.retain_candidate_keys else rescue_new * config.full_key_bytes_per_token
        sketch_bytes = 0.0
    else:
        verifier_key_bytes = 0.0
        sketch_bytes = verifier_tokens * config.sketch_bytes_per_token
        reread_key_bytes = rescue_new * config.full_key_bytes_per_token
    total_key_bytes = (
        temporal_key_bytes + verifier_key_bytes + sketch_bytes + reread_key_bytes + metadata_bytes
    )
    baseline_key_bytes = scores.size * config.full_key_bytes_per_token

    physical_reductions: dict[int, float] = {}
    exact_mask = np.zeros(scores.size, dtype=bool)
    exact_mask[exact_ids] = True
    for physical_block in (32, 64, 128):
        full_blocks = (scores.size + physical_block - 1) // physical_block
        baseline_physical = full_blocks * physical_block * config.full_key_bytes_per_token
        if config.path == "head":
            # Temporal exact and verifier-cold sets partition the prefix, so a
            # full-dimensional K stream touches every physical block at least once.
            full_read = baseline_physical
            reread_blocks = 0
            if not config.retain_candidate_keys and rescued_blocks.size:
                rescue_token_mask = np.zeros(scores.size, dtype=bool)
                for block_id in rescued_blocks:
                    start = int(block_id) * config.block_size
                    rescue_token_mask[start : min(scores.size, start + config.block_size)] = True
                reread_blocks = np.unique(np.flatnonzero(rescue_token_mask) // physical_block).size
            physical_bytes = (
                full_read
                + reread_blocks * physical_block * config.full_key_bytes_per_token
                + metadata_bytes
            )
        else:
            full_key_blocks = np.unique(np.flatnonzero(exact_mask) // physical_block).size
            cold_blocks_physical = np.unique(np.flatnonzero(cold_mask) // physical_block).size
            physical_bytes = (
                full_key_blocks * physical_block * config.full_key_bytes_per_token
                + cold_blocks_physical * physical_block * config.sketch_bytes_per_token
                + metadata_bytes
            )
        physical_reductions[physical_block] = float(1.0 - physical_bytes / baseline_physical)

    metrics: dict[str, Any] = {
        "step": step,
        "context_length": scores.size,
        "policy": policy.name,
        "verifier": config.name,
        "path": config.path,
        "width": config.width,
        "block_size": config.block_size,
        "rescue_fraction": config.rescue_fraction,
        "promotion_policy": promotion_policy,
        "promotion_threshold": (
            math.nan if promotion_threshold is None else float(promotion_threshold)
        ),
        "promotion_fraction_of_cold": float(
            rescued_blocks.size / max(1, int(cold_blocks.sum()))
        ),
        "temporal_exact_tokens": temporal_exact,
        "cold_tokens_scanned": verifier_tokens,
        "cold_blocks_scanned": int(cold_blocks.sum()),
        "rescue_blocks": int(rescued_blocks.size),
        "rescue_exact_tokens": rescue_new,
        "candidate_union_tokens": int(exact_ids.size),
        "duplicate_exact_tokens": 0,
        "qk_full_equivalent": float(full_equivalent),
        "net_qk_reduction": float(1.0 - full_equivalent / full_work),
        "exact_match": bool(np.array_equal(np.sort(approximate), np.sort(baseline))),
        "recall": float(base_mask[approximate].sum() / k),
        "top128_recall": _retention(approximate, baseline, 128),
        "top256_recall": _retention(approximate, baseline, 256),
        "top512_recall": _retention(approximate, baseline, 512),
        "top1024_recall": _retention(approximate, baseline, 1024),
        "top2048_recall": _retention(approximate, baseline, 2048),
        "index_mass_ratio": _mass_ratio(scores, approximate, baseline),
        "newly_active_tokens": int(newly_active.size),
        "newly_active_token_recall": float(rescued_token_mask.mean()) if newly_active.size else 1.0,
        "newly_active_blocks": int(newly_active_blocks.size),
        "newly_active_block_recall": (
            float(rescued_positive_blocks.size / newly_active_blocks.size)
            if newly_active_blocks.size
            else 1.0
        ),
        "detector_ap": _average_precision(ranking, newly_active_blocks),
        "rescue_precision": (
            float(rescued_positive_blocks.size / rescued_blocks.size)
            if rescued_blocks.size
            else 1.0
        ),
        "tail_critical_blocks": int(tail_blocks.size),
        "tail_critical_block_recall": (
            float(rescued_tail.size / tail_blocks.size) if tail_blocks.size else 1.0
        ),
        "temporal_key_bytes": float(temporal_key_bytes),
        "verifier_key_bytes": float(verifier_key_bytes),
        "sketch_bytes": float(sketch_bytes),
        "rescue_reread_key_bytes": float(reread_key_bytes),
        "metadata_bytes": int(metadata_bytes),
        "total_key_bytes": float(total_key_bytes),
        "ideal_key_byte_reduction": float(1.0 - total_key_bytes / baseline_key_bytes),
        "physical_key_byte_reduction": physical_reductions[64],
        "physical_b32_key_byte_reduction": physical_reductions[32],
        "physical_b64_key_byte_reduction": physical_reductions[64],
        "physical_b128_key_byte_reduction": physical_reductions[128],
    }
    details = {
        "approximate": approximate,
        "baseline": baseline,
        "temporal_approximate": temporal_detail["approximate"],
        "temporal_evaluated": temporal_detail["evaluated"],
        "cold": np.flatnonzero(cold_mask),
        "block_ranking": ranking,
        "rescued_blocks": rescued_blocks,
        "exact_candidates": exact_ids,
        "newly_active": newly_active,
        "newly_active_blocks": newly_active_blocks,
    }
    return state, metrics, details


def replay_verifier_trace(
    trace: ScoreTrace,
    verifier_scores: np.ndarray | Callable[[int, int], np.ndarray],
    *,
    policy: ApproxPolicy,
    config: VerifierConfig,
    k: int = 2048,
    max_transitions: int | None = None,
    tail_labels: dict[int, np.ndarray] | None = None,
    detail_callback: Callable[[int, dict[str, np.ndarray]], None] | None = None,
    promotion_threshold: float | Callable[[int, int], float] | None = None,
) -> pd.DataFrame:
    """Replay one trace with precomputed or lazily generated verifier scores."""

    state = initialize_state(trace.row(0), k=k, block_size=config.block_size)
    stop = trace.scores.shape[0]
    if max_transitions is not None:
        stop = min(stop, max_transitions + 1)
    rows: list[dict[str, Any]] = []
    for step in range(1, stop):
        length = int(trace.lengths[step])
        partial = (
            verifier_scores(step, length)
            if callable(verifier_scores)
            else verifier_scores[step, :length]
        )
        threshold = (
            promotion_threshold(step, length)
            if callable(promotion_threshold)
            else promotion_threshold
        )
        state, metrics, details = verifier_step(
            state,
            trace.row(step),
            partial,
            policy=policy,
            config=config,
            k=k,
            step=step,
            previous_length=int(trace.lengths[step - 1]),
            tail_critical_blocks=None if tail_labels is None else tail_labels.get(step),
            promotion_threshold=threshold,
        )
        metrics.update(
            {
                "layer": trace.layer,
                "workload": trace.workload,
                "prompt_id": trace.prompt_id,
                "base_context_length": int(trace.lengths[0]) - 1,
            }
        )
        rows.append(metrics)
        if detail_callback is not None:
            detail_callback(step, details)
    return pd.DataFrame(rows)
