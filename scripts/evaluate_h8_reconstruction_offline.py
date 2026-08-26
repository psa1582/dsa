from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from temporal_dsa.approx import ApproxPolicy, ApproxState, initialize_state, replay_step
from temporal_dsa.metrics import stable_topk


K = 2048
BLOCK_SIZE = 64
LAYERS = [2, 5, 8, 11, 14, 17, 21, 25]
PURE_METHODS = [
    "raw_h8",
    "global_affine_h8",
    "layer_affine_h8",
    "mass_normalized_h8",
    "mass_affine_h8",
    "prev_full",
    "t1_global",
    "t1_layer",
    "t2_residual_layer",
    "t3_delta_layer",
    "t4_margin_layer",
]
HYBRID_METHODS = [
    "raw_h8",
    "layer_affine_h8",
    "mass_affine_h8",
    "t1_layer",
    "t2_residual_layer",
    "t3_delta_layer",
    "t4_margin_layer",
]


class LeastSquares:
    def __init__(self, width: int) -> None:
        self.xtx = np.zeros((width, width), dtype=np.float64)
        self.xty = np.zeros(width, dtype=np.float64)
        self.count = 0

    def update(self, columns: list[np.ndarray], target: np.ndarray) -> None:
        x = np.column_stack([np.asarray(value, dtype=np.float64) for value in columns])
        y = np.asarray(target, dtype=np.float64)
        self.xtx += x.T @ x
        self.xty += x.T @ y
        self.count += y.size

    def solve(self) -> np.ndarray:
        ridge = np.eye(self.xtx.shape[0], dtype=np.float64) * 1e-12
        ridge[-1, -1] = 0.0
        return np.linalg.lstsq(self.xtx + ridge, self.xty, rcond=None)[0]


class PairStats:
    def __init__(self) -> None:
        self.count = 0
        self.sx = self.sy = self.sxx = self.syy = self.sxy = 0.0
        self.sample_x: list[np.ndarray] = []
        self.sample_y: list[np.ndarray] = []

    def update(self, x: np.ndarray, y: np.ndarray, sample_count: int = 16) -> None:
        x64 = np.asarray(x, dtype=np.float64)
        y64 = np.asarray(y, dtype=np.float64)
        if not x64.size:
            return
        self.count += x64.size
        self.sx += float(x64.sum())
        self.sy += float(y64.sum())
        self.sxx += float(x64 @ x64)
        self.syy += float(y64 @ y64)
        self.sxy += float(x64 @ y64)
        take = min(sample_count, x64.size)
        ids = np.linspace(0, x64.size - 1, take, dtype=np.int64)
        self.sample_x.append(x64[ids].astype(np.float32))
        self.sample_y.append(y64[ids].astype(np.float32))

    def result(self) -> tuple[float, float, int]:
        n = self.count
        numerator = n * self.sxy - self.sx * self.sy
        denominator = math.sqrt(
            max(0.0, n * self.sxx - self.sx * self.sx)
            * max(0.0, n * self.syy - self.sy * self.sy)
        )
        pearson = numerator / denominator if denominator else math.nan
        x = np.concatenate(self.sample_x) if self.sample_x else np.empty(0)
        y = np.concatenate(self.sample_y) if self.sample_y else np.empty(0)
        if x.size > 1:
            xr = pd.Series(x).rank(method="average").to_numpy()
            yr = pd.Series(y).rank(method="average").to_numpy()
            spearman = float(np.corrcoef(xr, yr)[0, 1])
        else:
            spearman = math.nan
        return float(pearson), spearman, int(x.size)


class ScalarStats:
    def __init__(self) -> None:
        self.count = 0
        self.total = self.total2 = 0.0
        self.samples: list[np.ndarray] = []

    def update(self, values: np.ndarray, sample_count: int = 32) -> None:
        values = np.asarray(values, dtype=np.float64)
        if not values.size:
            return
        self.count += values.size
        self.total += float(values.sum())
        self.total2 += float(values @ values)
        take = min(sample_count, values.size)
        ids = np.linspace(0, values.size - 1, take, dtype=np.int64)
        self.samples.append(values[ids].astype(np.float32))

    def row(self, condition: str) -> dict[str, Any]:
        sample = np.concatenate(self.samples) if self.samples else np.empty(0)
        mean = self.total / max(1, self.count)
        variance = max(0.0, self.total2 / max(1, self.count) - mean * mean)
        quantiles = np.quantile(sample, [0.5, 0.9, 0.95, 0.99, 0.999]) if sample.size else [math.nan] * 5
        return {
            "condition": condition,
            "observations": self.count,
            "quantile_sample_count": int(sample.size),
            "mean": mean,
            "std": math.sqrt(variance),
            "p50": quantiles[0],
            "p90": quantiles[1],
            "p95": quantiles[2],
            "p99": quantiles[3],
            "p99_9": quantiles[4],
        }


def family_policy(family: str, *, name: str, gamma: float) -> ApproxPolicy:
    common: dict[str, Any] = {"name": name, "gamma": gamma}
    if family == "static":
        return ApproxPolicy(**common)
    if family == "dynamic_address":
        return ApproxPolicy(**common, dynamic_threshold=True)
    if family == "dynamic_previous_max":
        return ApproxPolicy(**common, dynamic_threshold=True, order="previous_max")
    if family in {"dynamic_bucket8", "dynamic_bucket16"}:
        return ApproxPolicy(**common, dynamic_threshold=True, order=family.removeprefix("dynamic_"))
    if family in {"streak2_bucket8", "streak4_bucket8"}:
        streak = int(family.removeprefix("streak").split("_", 1)[0])
        return ApproxPolicy(**common, dynamic_threshold=True, order="bucket8", cold_streak=streak)
    if family == "ema_bucket8_age8":
        return ApproxPolicy(
            **common,
            dynamic_threshold=True,
            order="bucket8",
            risk_model="ema",
            ema_alpha=0.9,
            volatility_lambda=2.0,
            age_cap=8,
        )
    raise ValueError(f"unknown family: {family}")


def build_policy(selection: dict[str, Any], layer: int) -> ApproxPolicy:
    spec = next(row for row in selection["policies"] if row["selection_role"] == "Aggressive")
    scales = {int(key): float(value) for key, value in selection["gamma_scale"].items()}
    policy = family_policy(
        str(spec["family"]),
        name=str(spec["config_id"]),
        gamma=float(spec["multiplier"]) * scales[layer],
    )
    return replace(policy, **spec.get("overrides", {}))


def cache_path(root: Path, split: str, trace_file: str) -> Path:
    return root / split / Path(trace_file).name


def load_arrays(row: Any, cache_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    with np.load(row.trace_file, allow_pickle=False) as trace:
        full = trace["scores"].astype(np.float32)
        lengths = trace["lengths"].astype(np.int32)
        metadata = json.loads(str(trace["metadata"].item()))
    with np.load(cache_path(cache_root, row.split, row.trace_file), allow_pickle=False) as cached:
        h8 = cached["h8"].astype(np.float32)
        rho = cached["rho"].astype(np.float32)
    if h8.shape != full.shape or rho.shape != lengths.shape:
        raise RuntimeError(f"cache mismatch for {row.trace_file}")
    return full, h8, rho, lengths, metadata


def fit_coefficients(inventory: pd.DataFrame, cache_root: Path) -> dict[str, Any]:
    calibration = inventory[inventory.split.eq("calibration")]
    fits: dict[str, LeastSquares] = {
        "affine_global": LeastSquares(2),
        "mass_affine_global": LeastSquares(2),
        "t1_global": LeastSquares(3),
        "t2_global": LeastSquares(3),
        "t3_global": LeastSquares(1),
    }
    for layer in LAYERS:
        for name, width in [
            ("affine", 2),
            ("mass_affine", 2),
            ("t1", 3),
            ("t2", 3),
            ("t3", 1),
        ]:
            fits[f"{name}_layer_{layer}"] = LeastSquares(width)
    band_values: dict[int, list[float]] = {layer: [] for layer in LAYERS}

    for row in calibration.itertuples():
        full, h8, rho, lengths, _ = load_arrays(row, cache_root)
        layer = int(row.layer)
        for step in range(1, len(lengths)):
            common = int(lengths[step - 1])
            current = full[step, :common]
            previous = full[step - 1, :common]
            current_h8 = h8[step, :common]
            previous_h8 = h8[step - 1, :common]
            normalized = current_h8 / max(float(rho[step]), 1e-6)
            ones = np.ones(common, dtype=np.float32)
            residual = current - current_h8
            delta_h8 = current_h8 - previous_h8
            delta_full = current - previous
            fits["affine_global"].update([current_h8, ones], current)
            fits[f"affine_layer_{layer}"].update([current_h8, ones], current)
            fits["mass_affine_global"].update([normalized, ones], current)
            fits[f"mass_affine_layer_{layer}"].update([normalized, ones], current)
            fits["t1_global"].update([current_h8, previous, ones], current)
            fits[f"t1_layer_{layer}"].update([current_h8, previous, ones], current)
            fits["t2_global"].update([current_h8, previous, ones], residual)
            fits[f"t2_layer_{layer}"].update([current_h8, previous, ones], residual)
            fits["t3_global"].update([delta_h8], delta_full)
            fits[f"t3_layer_{layer}"].update([delta_h8], delta_full)
            ranked = stable_topk(previous, min(4096, common))
            if ranked.size > K:
                band_values[layer].append(float(previous[ranked[K - 1]] - previous[ranked[-1]]))

    margin_band = {layer: float(np.median(values)) for layer, values in band_values.items()}
    margin_fits: dict[tuple[int, int], LeastSquares] = {
        (layer, bucket): LeastSquares(3) for layer in LAYERS for bucket in range(4)
    }
    for row in calibration.itertuples():
        full, h8, _, lengths, _ = load_arrays(row, cache_root)
        layer = int(row.layer)
        for step in range(1, len(lengths)):
            common = int(lengths[step - 1])
            current = full[step, :common]
            previous = full[step - 1, :common]
            current_h8 = h8[step, :common]
            tau = float(previous[stable_topk(previous, K)[-1]])
            bucket = margin_bucket(previous, tau, margin_band[layer])
            for bucket_id in range(4):
                mask = bucket == bucket_id
                if mask.any():
                    margin_fits[(layer, bucket_id)].update(
                        [current_h8[mask], previous[mask], np.ones(mask.sum())], current[mask]
                    )

    coefficients = {name: fit.solve().tolist() for name, fit in fits.items()}
    counts = {name: fit.count for name, fit in fits.items()}
    coefficients["margin_band"] = {str(key): value for key, value in margin_band.items()}
    coefficients["t4_margin"] = {
        str(layer): {
            str(bucket): margin_fits[(layer, bucket)].solve().tolist()
            for bucket in range(4)
        }
        for layer in LAYERS
    }
    counts["t4_margin"] = {
        str(layer): {str(bucket): margin_fits[(layer, bucket)].count for bucket in range(4)}
        for layer in LAYERS
    }
    coefficients["sample_counts"] = counts
    coefficients["fit_split"] = "calibration_only"
    return coefficients


def margin_bucket(previous: np.ndarray, tau: float, band: float) -> np.ndarray:
    margin = previous - tau
    result = np.full(previous.shape, 3, dtype=np.int8)
    result[(margin >= -band) & (margin < 0)] = 2
    result[(margin >= 0) & (margin <= band)] = 1
    result[margin > band] = 0
    return result


def coef(coefficients: dict[str, Any], name: str, layer: int | None = None) -> np.ndarray:
    key = name if layer is None else f"{name}_layer_{layer}"
    return np.asarray(coefficients[key], dtype=np.float64)


def predict(
    method: str,
    step: int,
    length: int,
    full: np.ndarray,
    h8: np.ndarray,
    rho: np.ndarray,
    layer: int,
    coefficients: dict[str, Any],
) -> np.ndarray:
    current_h8 = h8[step, :length].astype(np.float64)
    common = min(length, int(full.shape[1]), int(length - 1))
    previous = full[step - 1, :common].astype(np.float64)
    previous_h8 = h8[step - 1, :common].astype(np.float64)
    fallback = coef(coefficients, "affine", layer)
    output = fallback[0] * current_h8 + fallback[1]
    if method == "raw_h8":
        return current_h8.astype(np.float32)
    if method == "global_affine_h8":
        value = coef(coefficients, "affine_global")
        return (value[0] * current_h8 + value[1]).astype(np.float32)
    if method == "layer_affine_h8":
        return output.astype(np.float32)
    if method == "mass_normalized_h8":
        return (current_h8 / max(float(rho[step]), 1e-6)).astype(np.float32)
    if method == "mass_affine_h8":
        value = coef(coefficients, "mass_affine", layer)
        normalized = current_h8 / max(float(rho[step]), 1e-6)
        return (value[0] * normalized + value[1]).astype(np.float32)
    if method == "prev_full":
        output[:] = -np.inf
        output[:common] = previous
        return output.astype(np.float32)
    if method in {"t1_global", "t1_layer"}:
        value = coef(coefficients, "t1_global" if method == "t1_global" else "t1", None if method == "t1_global" else layer)
        output[:common] = value[0] * current_h8[:common] + value[1] * previous + value[2]
        return output.astype(np.float32)
    if method == "t2_residual_layer":
        value = coef(coefficients, "t2", layer)
        output[:common] = current_h8[:common] + (
            value[0] * current_h8[:common] + value[1] * previous + value[2]
        )
        return output.astype(np.float32)
    if method == "t3_delta_layer":
        value = coef(coefficients, "t3", layer)
        output[:common] = previous + value[0] * (current_h8[:common] - previous_h8)
        return output.astype(np.float32)
    if method == "t4_margin_layer":
        tau = float(previous[stable_topk(previous, K)[-1]])
        band = float(coefficients["margin_band"][str(layer)])
        buckets = margin_bucket(previous, tau, band)
        for bucket_id in range(4):
            mask = buckets == bucket_id
            value = np.asarray(coefficients["t4_margin"][str(layer)][str(bucket_id)])
            output[:common][mask] = (
                value[0] * current_h8[:common][mask]
                + value[1] * previous[mask]
                + value[2]
            )
        return output.astype(np.float32)
    raise ValueError(method)


def update_hybrid_state(
    previous: ApproxState,
    temporal_next: ApproxState,
    selected: np.ndarray,
    baseline: np.ndarray,
) -> ApproxState:
    block_count = temporal_next.last_max.size
    prior_streak = np.pad(
        previous.cold_streak,
        (0, max(0, block_count - previous.cold_streak.size)),
        constant_values=1,
    )[:block_count]
    hot = np.zeros(block_count, dtype=bool)
    hot[selected // BLOCK_SIZE] = True
    temporal_next.topk = selected
    temporal_next.baseline_topk = baseline
    temporal_next.cold_streak = np.where(
        hot, 0, np.minimum(prior_streak.astype(np.int32) + 1, 32767)
    ).astype(np.int16)
    return temporal_next


def selection_metrics(
    selected: np.ndarray,
    baseline: np.ndarray,
    full: np.ndarray,
    teacher: np.ndarray | None,
) -> dict[str, Any]:
    baseline_mask = np.isin(baseline, selected, assume_unique=True)
    selected_mask = np.isin(selected, baseline, assume_unique=True)
    missed = baseline[~baseline_mask]
    false_positive = selected[~selected_mask]
    tau = float(full[baseline[-1]])
    shift = float(np.max(full[baseline]))
    weights = np.exp(full[baseline].astype(np.float64) - shift)
    weighted_recall = float(weights[baseline_mask].sum() / max(weights.sum(), 1e-300))
    result = {
        "top128_recall": float(np.isin(baseline[:128], selected).mean()),
        "top512_recall": float(np.isin(baseline[:512], selected).mean()),
        "top1024_recall": float(np.isin(baseline[:1024], selected).mean()),
        "top2048_recall": float(baseline_mask.mean()),
        "top2048_precision": float(selected_mask.mean()),
        "exact_top2048_match": bool(not missed.size and not false_positive.size),
        "weighted_recall": weighted_recall,
        "false_negative_count": int(missed.size),
        "false_positive_count": int(false_positive.size),
        "worst_false_negative_score": float(np.max(full[missed])) if missed.size else math.nan,
        "worst_false_negative_threshold_gap": float(np.max(full[missed] - tau)) if missed.size else 0.0,
    }
    if teacher is not None:
        denominator = float(np.asarray(teacher[baseline], dtype=np.float64).sum())
        result["teacher_attention_mass_ratio"] = float(
            np.asarray(teacher[selected], dtype=np.float64).sum() / max(denominator, 1e-300)
        )
    else:
        result["teacher_attention_mass_ratio"] = math.nan
    return result


def details_path(root: Path, trace_stem: str) -> Path:
    return root / f"{trace_stem}__Aggressive__head_dynamic_abs_w_w8_b64_threshold_r0.1.npz"


def rescue_cost_map(path: Path) -> dict[tuple, float]:
    frame = pd.read_csv(path)
    frame = frame[
        frame.policy_role.eq("Aggressive")
        & frame.head_scheme.eq("dynamic_abs_w")
        & frame.promotion_target.eq(0.10)
    ]
    return {
        (
            row.prompt_id,
            int(row.base_context_length),
            int(row.layer),
            int(row.step),
        ): float(row.net_qk_reduction)
        for row in frame.itertuples()
    }


def load_teacher(metadata: dict[str, Any]) -> np.ndarray | None:
    capture = torch.load(metadata["source_capture"], map_location="cpu", weights_only=True)
    teacher = capture.get("teacher_probabilities")
    return None if teacher is None else teacher.float().numpy()


def evaluate_split(
    inventory: pd.DataFrame,
    split: str,
    cache_root: Path,
    coefficients: dict[str, Any],
    selection: dict[str, Any],
    detail_root: Path,
    rescue_costs: dict[tuple, float],
    *,
    hybrid_methods: list[str],
    include_pure: bool,
    include_reference: bool,
    diagnostic_methods: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, PairStats], dict[str, ScalarStats], dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    newly_rows: list[dict[str, Any]] = []
    pair_stats: dict[str, PairStats] = {}
    residual_stats: dict[str, ScalarStats] = {}
    scatter: dict[str, list[np.ndarray]] = {key: [] for key in ["h8", "full", "previous", "residual", "newly_active"]}
    diag_methods = diagnostic_methods or set()
    subset = inventory[inventory.split.eq(split)]
    for trace_index, item in enumerate(subset.itertuples(), start=1):
        full, h8, rho, lengths, metadata = load_arrays(item, cache_root)
        teacher = load_teacher(metadata)
        layer = int(item.layer)
        policy = build_policy(selection, layer)
        states = {
            method: initialize_state(full[0, : int(lengths[0])], k=K, block_size=BLOCK_SIZE)
            for method in hybrid_methods
        }
        detail_file = details_path(detail_root, Path(item.trace_file).stem)
        with np.load(detail_file, allow_pickle=True) as details:
            rescue_steps = details["steps"].astype(np.int64)
            rescue_approximate = details["approximate"].astype(np.int64)
            rescue_baseline = details["baseline"].astype(np.int64)
        if not np.array_equal(rescue_steps, np.arange(1, len(lengths))):
            raise RuntimeError(f"rescue detail step mismatch: {detail_file}")

        previous_oracle = stable_topk(full[0, : int(lengths[0])], K)
        for step in range(1, len(lengths)):
            length = int(lengths[step])
            common = int(lengths[step - 1])
            current_full = full[step, :length]
            current_teacher = None if teacher is None else teacher[step, :length]
            oracle = stable_topk(current_full, K)
            predictions = {
                method: predict(method, step, length, full, h8, rho, layer, coefficients)
                for method in PURE_METHODS
            }
            selected_by_name: dict[str, np.ndarray] = {}
            if include_pure:
                for method, predicted in predictions.items():
                    selected = stable_topk(predicted, K)
                    selected_by_name[f"pure_{method}"] = selected
                    metric = selection_metrics(selected, oracle, current_full, current_teacher)
                    rows.append(
                        {
                            "split": split,
                            "scope": "pure_no_h56",
                            "method": method,
                            "prompt_id": item.prompt_id,
                            "workload": item.workload,
                            "layer": layer,
                            "base_context_length": int(item.base_context_length),
                            "step": step,
                            "context_length": length,
                            "qk_reduction": 1.0 if method == "prev_full" else 0.875,
                            "temporal_exact_fraction": 0.0,
                            **metric,
                        }
                    )

            best_evaluated: np.ndarray | None = None
            best_prediction: np.ndarray | None = None
            for method in hybrid_methods:
                prior = states[method]
                temporal_next, _, temporal_detail = replay_step(
                    prior,
                    current_full,
                    policy=policy,
                    k=K,
                    block_size=BLOCK_SIZE,
                    step=step,
                    history_mode="own",
                    previous_length=common,
                )
                evaluated = temporal_detail["evaluated"]
                hybrid = predictions[method].copy()
                hybrid[evaluated] = current_full[evaluated]
                selected = stable_topk(hybrid, K)
                states[method] = update_hybrid_state(prior, temporal_next, selected, oracle)
                name = f"hybrid_{method}"
                selected_by_name[name] = selected
                exact_fraction = evaluated.size / length
                qk_reduction = 1.0 - (
                    evaluated.size * 64 + (length - evaluated.size) * 8
                ) / (length * 64)
                metric = selection_metrics(selected, oracle, current_full, current_teacher)
                rows.append(
                    {
                        "split": split,
                        "scope": "hybrid_temporal_hot_full_cold_h8",
                        "method": method,
                        "prompt_id": item.prompt_id,
                        "workload": item.workload,
                        "layer": layer,
                        "base_context_length": int(item.base_context_length),
                        "step": step,
                        "context_length": length,
                        "qk_reduction": qk_reduction,
                        "temporal_exact_fraction": exact_fraction,
                        **metric,
                    }
                )
                if name in diag_methods:
                    best_evaluated = evaluated
                    best_prediction = hybrid

            if include_reference:
                rescue_ids = rescue_approximate[step - 1]
                if not np.array_equal(rescue_baseline[step - 1], oracle):
                    raise RuntimeError("stored rescue oracle selection changed")
                rows.append(
                    {
                        "split": split,
                        "scope": "reference",
                        "method": "existing_h8_10pct_h56_rescue",
                        "prompt_id": item.prompt_id,
                        "workload": item.workload,
                        "layer": layer,
                        "base_context_length": int(item.base_context_length),
                        "step": step,
                        "context_length": length,
                        "qk_reduction": rescue_costs[(item.prompt_id, int(item.base_context_length), layer, step)],
                        "temporal_exact_fraction": math.nan,
                        **selection_metrics(rescue_ids, oracle, current_full, current_teacher),
                    }
                )
                rows.append(
                    {
                        "split": split,
                        "scope": "reference",
                        "method": "full64_oracle",
                        "prompt_id": item.prompt_id,
                        "workload": item.workload,
                        "layer": layer,
                        "base_context_length": int(item.base_context_length),
                        "step": step,
                        "context_length": length,
                        "qk_reduction": 0.0,
                        "temporal_exact_fraction": 1.0,
                        **selection_metrics(oracle, oracle, current_full, current_teacher),
                    }
                )

            newly = np.setdiff1d(oracle, previous_oracle, assume_unique=False)
            if split == "test" and best_evaluated is not None and best_prediction is not None:
                evaluated_mask = np.zeros(length, dtype=bool)
                evaluated_mask[best_evaluated] = True
                conditions = {
                    "all": np.arange(common, dtype=np.int64),
                    "temporal_hot": np.flatnonzero(evaluated_mask[:common]),
                    "temporal_cold": np.flatnonzero(~evaluated_mask[:common]),
                    "newly_active": newly[newly < common],
                }
                ranked_near = stable_topk(current_full[:common], min(2560, common))
                conditions["near_top2048_threshold"] = ranked_near[1792:min(2304, ranked_near.size)]
                residual = current_full[:common] - h8[step, :common]
                previous = full[step - 1, :common]
                pairs = {
                    "h8_vs_full": (h8[step, :common], current_full[:common]),
                    "previous_full_vs_current_full": (previous, current_full[:common]),
                    "h8_vs_h56_residual": (h8[step, :common], residual),
                    "previous_full_vs_h56_residual": (previous, residual),
                }
                for condition, ids in conditions.items():
                    residual_stats.setdefault(condition, ScalarStats()).update(residual[ids])
                    for pair_name, (x, y) in pairs.items():
                        pair_stats.setdefault(f"{condition}::{pair_name}", PairStats()).update(x[ids], y[ids])
                sample_ids = np.linspace(0, common - 1, min(16, common), dtype=np.int64)
                scatter["h8"].append(h8[step, sample_ids].astype(np.float32))
                scatter["full"].append(current_full[sample_ids].astype(np.float32))
                scatter["previous"].append(previous[sample_ids].astype(np.float32))
                scatter["residual"].append(residual[sample_ids].astype(np.float32))
                scatter["newly_active"].append(np.zeros(sample_ids.size, dtype=np.int8))
                new_common = newly[newly < common]
                if new_common.size:
                    take_new = new_common[: min(16, new_common.size)]
                    scatter["h8"].append(h8[step, take_new].astype(np.float32))
                    scatter["full"].append(current_full[take_new].astype(np.float32))
                    scatter["previous"].append(previous[take_new].astype(np.float32))
                    scatter["residual"].append(residual[take_new].astype(np.float32))
                    scatter["newly_active"].append(np.ones(take_new.size, dtype=np.int8))

                cold = np.flatnonzero(~evaluated_mask[:common])
                cold_h8_q10 = float(np.quantile(h8[step, cold], 0.10)) if cold.size else -math.inf
                cold_residual_q90 = float(np.quantile(residual[cold], 0.90)) if cold.size else math.inf
                for name in sorted(diag_methods | {"pure_raw_h8"}):
                    selected = selected_by_name[name]
                    method_prediction = (
                        best_prediction if name.startswith("hybrid_") else predictions[name.removeprefix("pure_")]
                    )
                    new_cold = new_common[~evaluated_mask[new_common]]
                    missed = new_common[~np.isin(new_common, selected)]
                    missed_cold = new_cold[~np.isin(new_cold, selected)]
                    if new_cold.size:
                        sorted_cold = np.sort(method_prediction[cold].astype(np.float64))[::-1]
                        ranks = np.searchsorted(-sorted_cold, -method_prediction[new_cold], side="left") + 1
                    else:
                        ranks = np.empty(0, dtype=np.int64)
                    failure = new_common[
                        (h8[step, new_common] <= cold_h8_q10)
                        & (residual[new_common] >= cold_residual_q90)
                    ]
                    tau = float(current_full[oracle[-1]])
                    newly_rows.append(
                        {
                            "method": name,
                            "prompt_id": item.prompt_id,
                            "workload": item.workload,
                            "layer": layer,
                            "base_context_length": int(item.base_context_length),
                            "step": step,
                            "newly_active_tokens": int(new_common.size),
                            "newly_active_blocks": int(np.unique(new_common // BLOCK_SIZE).size),
                            "newly_active_recall": float(np.isin(new_common, selected).mean()) if new_common.size else 1.0,
                            "newly_active_cold_tokens": int(new_cold.size),
                            "newly_active_cold_recall": float(np.isin(new_cold, selected).mean()) if new_cold.size else 1.0,
                            "false_negative_rate": float(missed.size / max(1, new_common.size)),
                            "false_positive_rate": float((~np.isin(selected, oracle)).sum() / K),
                            "cold_rank_p50": float(np.median(ranks)) if ranks.size else math.nan,
                            "cold_rank_p95": float(np.quantile(ranks, 0.95)) if ranks.size else math.nan,
                            "score_error_mean": float(np.mean(method_prediction[new_common] - current_full[new_common])) if new_common.size else 0.0,
                            "score_error_p95_abs": float(np.quantile(np.abs(method_prediction[new_common] - current_full[new_common]), 0.95)) if new_common.size else 0.0,
                            "missed_full64_margin_p50": float(np.median(current_full[missed] - tau)) if missed.size else 0.0,
                            "missed_full64_margin_max": float(np.max(current_full[missed] - tau)) if missed.size else 0.0,
                            "missed_cold_full64_margin_max": float(np.max(current_full[missed_cold] - tau)) if missed_cold.size else 0.0,
                            "small_h8_large_h56_cases": int(failure.size),
                            "small_h8_large_h56_recall": float(np.isin(failure, selected).mean()) if failure.size else 1.0,
                        }
                    )
            previous_oracle = oracle
        print(f"[{split} {trace_index}/{len(subset)}] {item.trace_file}", flush=True)

    scatter_arrays = {
        key: np.concatenate(value) if value else np.empty(0, dtype=np.float32)
        for key, value in scatter.items()
    }
    return pd.DataFrame(rows), pd.DataFrame(newly_rows), pair_stats, residual_stats, scatter_arrays


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["split", "scope", "method"], as_index=False).agg(
        observations=("step", "size"),
        qk_reduction=("qk_reduction", "mean"),
        temporal_exact_fraction=("temporal_exact_fraction", "mean"),
        top2048_recall=("top2048_recall", "mean"),
        top2048_precision=("top2048_precision", "mean"),
        exact_top2048_match_rate=("exact_top2048_match", "mean"),
        top128_recall=("top128_recall", "mean"),
        top512_recall=("top512_recall", "mean"),
        top1024_recall=("top1024_recall", "mean"),
        weighted_recall=("weighted_recall", "mean"),
        worst_false_negative_score=("worst_false_negative_score", "max"),
        worst_false_negative_threshold_gap=("worst_false_negative_threshold_gap", "max"),
        teacher_attention_mass_ratio=("teacher_attention_mass_ratio", "mean"),
    )


def choose(summary: pd.DataFrame, scope: str, candidates: list[str]) -> str:
    subset = summary[
        summary.split.eq("validation")
        & summary.scope.eq(scope)
        & summary.method.isin(candidates)
    ].sort_values(["top2048_recall", "weighted_recall", "method"], ascending=[False, False, True])
    if subset.empty:
        raise RuntimeError(f"no validation candidates for {scope}")
    return str(subset.iloc[0].method)


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict CPU-only H8 full-score reconstruction replay")
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--detail-root", type=Path, required=True)
    parser.add_argument("--rescue-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError("CUDA must be hidden for the strict offline replay")
    started = time.perf_counter()
    inventory = pd.read_csv(args.inventory)
    selection = json.loads(args.selection.read_text())
    coefficients = fit_coefficients(inventory, args.cache_root)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "coefficients.json").write_text(
        json.dumps(coefficients, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    costs = rescue_cost_map(args.rescue_rows)

    validation_pure, _, _, _, _ = evaluate_split(
        inventory,
        "validation",
        args.cache_root,
        coefficients,
        selection,
        args.detail_root,
        costs,
        hybrid_methods=[],
        include_pure=True,
        include_reference=True,
    )
    validation_pure_summary = summarize(validation_pure)
    best_pure_affine = choose(
        validation_pure_summary, "pure_no_h56", ["layer_affine_h8", "mass_affine_h8"]
    )
    best_pure_temporal = choose(
        validation_pure_summary,
        "pure_no_h56",
        ["t1_layer", "t2_residual_layer", "t3_delta_layer", "t4_margin_layer"],
    )
    chosen_hybrids = list(dict.fromkeys(["raw_h8", best_pure_affine, best_pure_temporal]))
    validation_hybrid, _, _, _, _ = evaluate_split(
        inventory,
        "validation",
        args.cache_root,
        coefficients,
        selection,
        args.detail_root,
        costs,
        hybrid_methods=chosen_hybrids,
        include_pure=False,
        include_reference=False,
    )
    validation = pd.concat([validation_pure, validation_hybrid], ignore_index=True)
    validation_summary = summarize(validation)
    best_hybrid_temporal = best_pure_temporal
    selected = {
        "best_pure_affine": best_pure_affine,
        "best_pure_temporal": best_pure_temporal,
        "best_hybrid_temporal": best_hybrid_temporal,
        "selection_split": "validation",
        "tie_break": "top2048_recall, weighted_recall, method name",
    }
    (args.output / "selected_methods.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    diagnostic_methods = {
        f"pure_{best_pure_temporal}",
        f"hybrid_{best_hybrid_temporal}",
    }
    test, newly, pair_stats, scalar_stats, scatter = evaluate_split(
        inventory,
        "test",
        args.cache_root,
        coefficients,
        selection,
        args.detail_root,
        costs,
        hybrid_methods=chosen_hybrids,
        include_pure=True,
        include_reference=True,
        diagnostic_methods=diagnostic_methods,
    )
    raw = pd.concat([validation, test], ignore_index=True)
    raw.to_csv(args.output / "selection_rows.csv", index=False)
    summary = summarize(raw)
    summary.to_csv(args.output / "selection_summary.csv", index=False)
    raw.groupby(
        ["split", "scope", "method", "base_context_length", "layer", "workload"],
        as_index=False,
    ).agg(
        observations=("step", "size"),
        qk_reduction=("qk_reduction", "mean"),
        top2048_recall=("top2048_recall", "mean"),
        top128_recall=("top128_recall", "mean"),
        top512_recall=("top512_recall", "mean"),
        weighted_recall=("weighted_recall", "mean"),
        teacher_attention_mass_ratio=("teacher_attention_mass_ratio", "mean"),
    ).to_csv(args.output / "selection_breakdown.csv", index=False)
    newly.to_csv(args.output / "newly_active_rows.csv", index=False)
    if not newly.empty:
        newly.groupby("method", as_index=False).agg(
            observations=("step", "size"),
            newly_active_tokens=("newly_active_tokens", "sum"),
            newly_active_blocks=("newly_active_blocks", "sum"),
            newly_active_recall=("newly_active_recall", "mean"),
            newly_active_cold_tokens=("newly_active_cold_tokens", "sum"),
            newly_active_cold_recall=("newly_active_cold_recall", "mean"),
            false_negative_rate=("false_negative_rate", "mean"),
            false_positive_rate=("false_positive_rate", "mean"),
            cold_rank_p50=("cold_rank_p50", "median"),
            cold_rank_p95=("cold_rank_p95", lambda x: x.quantile(0.95)),
            score_error_mean=("score_error_mean", "mean"),
            score_error_p95_abs=("score_error_p95_abs", lambda x: x.quantile(0.95)),
            missed_full64_margin_max=("missed_full64_margin_max", "max"),
            small_h8_large_h56_cases=("small_h8_large_h56_cases", "sum"),
            small_h8_large_h56_recall=("small_h8_large_h56_recall", "mean"),
        ).to_csv(args.output / "newly_active_summary.csv", index=False)

    correlation_rows = []
    for key, stats in sorted(pair_stats.items()):
        condition, pair = key.split("::", 1)
        pearson, spearman, sample_count = stats.result()
        correlation_rows.append(
            {
                "condition": condition,
                "pair": pair,
                "observations": stats.count,
                "pearson": pearson,
                "spearman": spearman,
                "spearman_sample_count": sample_count,
            }
        )
    pd.DataFrame(correlation_rows).to_csv(args.output / "correlation_results.csv", index=False)
    pd.DataFrame(
        [stats.row(condition) for condition, stats in sorted(scalar_stats.items())]
    ).to_csv(args.output / "residual_statistics.csv", index=False)
    np.savez_compressed(args.output / "diagnostic_samples.npz", **scatter)
    audit = {
        "command": " ".join(__import__("sys").argv),
        "elapsed_seconds": time.perf_counter() - started,
        "device": "cpu",
        "gpu_used": False,
        "model_inference_run": False,
        "oracle_current_full_used_as_predictor": False,
        "oracle_current_full_used_for_evaluation_only": True,
        "calibration_trace_count": int(inventory.split.eq("calibration").sum()),
        "validation_trace_count": int(inventory.split.eq("validation").sum()),
        "test_trace_count": int(inventory.split.eq("test").sum()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "mla_output_status": "GPU FOLLOW-UP REQUIRED; values/projection not stored in offline score trace",
        "logit_ppl_status": "GPU FOLLOW-UP REQUIRED; model logits not stored for new selections",
    }
    (args.output / "offline_replay_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(selected, indent=2, sort_keys=True))
    print(summary[summary.split.eq("test")].to_string(index=False))
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
