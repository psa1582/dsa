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

from temporal_dsa.approx import ApproxPolicy
from temporal_dsa.metrics import stable_topk
from temporal_dsa.trace import ScoreTrace, load_trace
from temporal_dsa.verifier import VerifierConfig, replay_verifier_trace
from temporal_dsa.verifier_scoring import (
    dynamic_head_indices,
    load_sidecar_encoded,
    score_head_sparse,
)


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
        return ApproxPolicy(
            **common, dynamic_threshold=True, order="bucket8", cold_streak=streak
        )
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


def build_policy(spec: dict[str, Any], layer: int, scales: dict[int, float]) -> ApproxPolicy:
    policy = family_policy(
        str(spec["family"]),
        name=str(spec["config_id"]),
        gamma=float(spec["multiplier"]) * scales[layer],
    )
    return replace(policy, **spec.get("overrides", {}))


def discover(roots: list[Path]) -> list[Path]:
    paths: set[Path] = set()
    for root in roots:
        paths.update(root.rglob("*.npz"))
    if not paths:
        raise FileNotFoundError("no score traces")
    return sorted(paths)


def verifier_config(name: str) -> VerifierConfig:
    return VerifierConfig(
        name=name,
        path="head",
        width=8,
        score_ratio=8 / 64,
        block_size=64,
        rescue_fraction=0.1,
        retain_candidate_keys=True,
    )


def encode(trace: ScoreTrace, checkpoint_root: Path):
    checkpoint = checkpoint_root / f"layer_{trace.layer:02d}.safetensors"
    queries, weights, keys, _ = load_sidecar_encoded(
        trace.metadata["source_capture"], checkpoint, trace.lengths, device="cuda:0"
    )
    return queries, weights, keys


def fixed_head_sets(
    entries: list[tuple[ScoreTrace, torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    layer: int,
    seed: int,
) -> dict[str, np.ndarray]:
    weight_sum = np.zeros(64, dtype=np.float64)
    transition_sum = np.zeros(64, dtype=np.float64)
    tail_sum = np.zeros(64, dtype=np.float64)
    count = 0
    transition_weight = 0.0
    tail_weight = 0.0
    for trace, _, weights, _ in entries:
        values = weights.detach().float().abs().cpu().numpy()
        weight_sum += values.sum(axis=0)
        count += values.shape[0]
        for step in range(1, values.shape[0]):
            previous = stable_topk(trace.row(step - 1), min(2048, int(trace.lengths[step - 1])))
            current = stable_topk(trace.row(step), min(2048, int(trace.lengths[step])))
            transition = max(1, int((~np.isin(current, previous)).sum()))
            previous_tail = previous[: min(128, previous.size)]
            current_tail = current[: min(128, current.size)]
            tail = max(1, int((~np.isin(current_tail, previous_tail)).sum()))
            transition_sum += values[step] * transition
            tail_sum += values[step] * tail
            transition_weight += transition
            tail_weight += tail

    def top8(values: np.ndarray) -> np.ndarray:
        return np.argsort(-values, kind="stable")[:8].astype(np.int64)

    sets = {
        "fixed_avg_abs_w": top8(weight_sum / max(1, count)),
        "fixed_transition_aware": top8(transition_sum / max(1.0, transition_weight)),
        "fixed_tail_aware": top8(tail_sum / max(1.0, tail_weight)),
    }
    for random_seed in range(5):
        rng = np.random.default_rng(seed + layer * 1009 + random_seed)
        sets[f"fixed_random_seed{random_seed}"] = np.sort(
            rng.choice(64, 8, replace=False)
        ).astype(np.int64)
    return sets


def head_scores(
    queries: torch.Tensor,
    weights: torch.Tensor,
    keys: torch.Tensor,
    scheme: str,
    fixed_ids: dict[str, np.ndarray],
) -> np.ndarray:
    if scheme == "dynamic_abs_w":
        ids = dynamic_head_indices(weights, 8, "high_weight")
    else:
        ids = fixed_ids[scheme]
    return score_head_sparse(queries, weights, keys, ids).cpu().numpy()


def raw_block_maxima(scores: np.ndarray, lengths: np.ndarray, block_size: int = 64) -> np.ndarray:
    maxima = []
    for step in range(1, scores.shape[0]):
        length = int(lengths[step])
        for start in range(0, length, block_size):
            maxima.append(float(np.max(scores[step, start : min(length, start + block_size)])))
    return np.asarray(maxima, dtype=np.float32)


def achieved_rate(frames: list[pd.DataFrame]) -> float:
    rescue = sum(int(frame.rescue_blocks.sum()) for frame in frames)
    cold = sum(int(frame.cold_blocks_scanned.sum()) for frame in frames)
    return rescue / max(1, cold)


def calibrate_threshold(
    entries: list[tuple[ScoreTrace, np.ndarray]],
    *,
    policy: ApproxPolicy,
    target: float,
    name: str,
) -> tuple[float, float]:
    config = verifier_config(name)
    cold_maxima: list[float] = []
    for trace, scores in entries:
        detail_rows: dict[int, dict[str, np.ndarray]] = {}

        def collect(step: int, values: dict[str, np.ndarray]) -> None:
            detail_rows[step] = values

        replay_verifier_trace(
            trace,
            scores,
            policy=policy,
            config=config,
            promotion_threshold=math.inf,
            detail_callback=collect,
        )
        for step, details in detail_rows.items():
            cold = details["cold"]
            if not cold.size:
                continue
            block_ids = np.unique(cold // config.block_size)
            row = scores[step]
            for block_id in block_ids:
                ids = cold[cold // config.block_size == block_id]
                cold_maxima.append(float(np.max(row[ids])))
    if not cold_maxima:
        threshold = math.inf
    else:
        # Validation-only cutoff.  Held-out current-query distributions are never inspected.
        threshold = float(np.quantile(np.asarray(cold_maxima), 1.0 - target))
    frames = [
        replay_verifier_trace(
            trace,
            scores,
            policy=policy,
            config=config,
            promotion_threshold=threshold,
        )
        for trace, scores in entries
    ]
    return threshold, achieved_rate(frames)


def save_details(path: Path, details: dict[int, dict[str, np.ndarray]]) -> None:
    steps = np.asarray(sorted(details), dtype=np.int16)
    np.savez_compressed(
        path,
        steps=steps,
        approximate=np.stack([details[int(step)]["approximate"] for step in steps]),
        baseline=np.stack([details[int(step)]["baseline"] for step in steps]),
        rescued_blocks=np.asarray(
            [details[int(step)]["rescued_blocks"] for step in steps], dtype=object
        ),
    )


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(["head_scheme", "promotion_target"], sort=False):
        head_scheme, promotion_target = keys
        cold_blocks = group.cold_blocks_scanned.sum()
        newly = int(group.newly_active_tokens.sum())
        newly_blocks = int(group.newly_active_blocks.sum())
        rows.append(
            {
                "head_scheme": head_scheme,
                "promotion_policy": "validation_fixed_threshold",
                "promotion_target": promotion_target,
                "observations": len(group),
                "actual_promotion_rate": float(group.rescue_blocks.sum() / max(1, cold_blocks)),
                "net_qk_reduction_mean": group.net_qk_reduction.mean(),
                "net_qk_reduction_median": group.net_qk_reduction.median(),
                "candidate_fraction_mean": (group.candidate_union_tokens / group.context_length).mean(),
                "top128_recall": group.top128_recall.mean(),
                "top512_recall": group.top512_recall.mean(),
                "top2048_recall": group.top2048_recall.mean(),
                "index_mass_ratio_mean": group.index_mass_ratio.mean(),
                "newly_active_token_recall": float(
                    (group.newly_active_token_recall * group.newly_active_tokens).sum()
                    / max(1, newly)
                ),
                "newly_active_block_recall": float(
                    (group.newly_active_block_recall * group.newly_active_blocks).sum()
                    / max(1, newly_blocks)
                ),
                "physical_key_byte_reduction_median": group.physical_key_byte_reduction.median(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-calibrated threshold H8 replay")
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--heldout-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1582)
    args = parser.parse_args()

    if torch.cuda.device_count() != 2:
        raise RuntimeError("expected CUDA_VISIBLE_DEVICES=0,1")
    gpu_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("L40" in name for name in gpu_names):
        raise RuntimeError(f"L40 guard rejected {gpu_names}")
    selection = json.loads(args.selection.read_text())
    spec = next(row for row in selection["policies"] if row["selection_role"] == "Aggressive")
    scales = {int(key): float(value) for key, value in selection["gamma_scale"].items()}
    validation_paths = discover([args.validation_root])
    heldout_paths = discover(args.heldout_roots)
    layers = sorted({load_trace(path).layer for path in validation_paths})
    args.output.mkdir(parents=True, exist_ok=True)
    detail_root = args.output / "selection_details"
    detail_root.mkdir(exist_ok=True)
    thresholds: dict[str, dict[str, dict[str, float]]] = {}
    head_ids: dict[str, dict[str, list[int]]] = {}
    calibration_rows = []
    started = time.perf_counter()

    for layer in layers:
        entries = []
        for path in validation_paths:
            trace = load_trace(path)
            if trace.layer != layer:
                continue
            queries, weights, keys = encode(trace, args.checkpoint_root)
            entries.append((trace, queries, weights, keys))
        fixed = fixed_head_sets(entries, layer=layer, seed=args.seed)
        head_ids[str(layer)] = {name: ids.tolist() for name, ids in fixed.items()}
        schemes = ["dynamic_abs_w", *fixed]
        thresholds[str(layer)] = {}
        policy = build_policy(spec, layer, scales)
        for scheme in schemes:
            scored = [
                (trace, head_scores(queries, weights, keys, scheme, fixed))
                for trace, queries, weights, keys in entries
            ]
            targets = (0.05, 0.10, 0.15, 0.20) if scheme == "dynamic_abs_w" else (0.10,)
            thresholds[str(layer)][scheme] = {}
            for target in targets:
                name = f"head_{scheme}_w8_b64_threshold_r{target:g}"
                threshold, actual = calibrate_threshold(
                    scored, policy=policy, target=target, name=name
                )
                thresholds[str(layer)][scheme][str(target)] = threshold
                calibration_rows.append(
                    {
                        "layer": layer,
                        "head_scheme": scheme,
                        "promotion_target": target,
                        "threshold": threshold,
                        "validation_actual_promotion_rate": actual,
                        "validation_trace_count": len(scored),
                    }
                )
        del entries
        torch.cuda.empty_cache()
        print(f"calibrated layer {layer}", flush=True)

    # Persist validation-only decisions before the longer held-out pass.
    calibration_payload = {
        "promotion_policy": "validation_fixed_threshold",
        "kernel_visible_before_launch": True,
        "oracle_leakage": False,
        "thresholds_by_layer_scheme_rate": thresholds,
        "fixed_head_ids_by_layer": head_ids,
    }
    (args.output / "thresholds.json").write_text(
        json.dumps(calibration_payload, indent=2) + "\n"
    )
    pd.DataFrame(calibration_rows).to_csv(
        args.output / "threshold_calibration.csv", index=False
    )

    all_rows: list[pd.DataFrame] = []
    for index, path in enumerate(heldout_paths, start=1):
        trace = load_trace(path)
        queries, weights, keys = encode(trace, args.checkpoint_root)
        fixed = {name: np.asarray(ids) for name, ids in head_ids[str(trace.layer)].items()}
        policy = build_policy(spec, trace.layer, scales)
        schemes = ["dynamic_abs_w", *fixed]
        for scheme in schemes:
            partial = head_scores(queries, weights, keys, scheme, fixed)
            targets = (0.05, 0.10, 0.15, 0.20) if scheme == "dynamic_abs_w" else (0.10,)
            for target in targets:
                name = f"head_{scheme}_w8_b64_threshold_r{target:g}"
                config = verifier_config(name)
                detail_rows: dict[int, dict[str, np.ndarray]] = {}

                def collect(step: int, values: dict[str, np.ndarray]) -> None:
                    detail_rows[step] = values

                frame = replay_verifier_trace(
                    trace,
                    partial,
                    policy=policy,
                    config=config,
                    promotion_threshold=thresholds[str(trace.layer)][scheme][str(target)],
                    detail_callback=collect if target == 0.10 else None,
                )
                frame["policy_role"] = "Aggressive"
                frame["head_scheme"] = scheme
                frame["promotion_target"] = target
                all_rows.append(frame)
                if detail_rows:
                    save_details(
                        detail_root / f"{path.stem}__Aggressive__{name}.npz", detail_rows
                    )
        del queries, weights, keys
        torch.cuda.empty_cache()
        print(f"[{index}/{len(heldout_paths)}] {path.name}", flush=True)

    raw = pd.concat(all_rows, ignore_index=True)
    raw.to_csv(args.output / "threshold_replay_rows.csv", index=False)
    summary = aggregate(raw)
    summary.to_csv(args.output / "promotion_quality.csv", index=False)
    fixed_summary = summary[summary.promotion_target.eq(0.10)].copy()
    fixed_summary.to_csv(args.output / "fixed_vs_dynamic_heads.csv", index=False)
    audit = {
        "command": " ".join(__import__("sys").argv),
        "elapsed_seconds": time.perf_counter() - started,
        "validation_trace_count": len(validation_paths),
        "heldout_trace_count": len(heldout_paths),
        "layers": layers,
        "gpus": gpu_names,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "threshold_uses_current_distribution": False,
        "fixed_head_selection_uses_validation_only": True,
    }
    (args.output / "threshold_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
