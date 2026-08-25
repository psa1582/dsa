from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
from typing import Any

import pandas as pd

from temporal_dsa.approx import ApproxPolicy, replay_approx_trace
from temporal_dsa.trace import load_trace


def family_policy(family: str, *, name: str, gamma: float) -> ApproxPolicy:
    common: dict[str, Any] = {"name": name, "gamma": gamma}
    if family == "static":
        return ApproxPolicy(**common)
    if family == "dynamic_address":
        return ApproxPolicy(**common, dynamic_threshold=True)
    if family == "dynamic_previous_max":
        return ApproxPolicy(**common, dynamic_threshold=True, order="previous_max")
    if family in {"dynamic_bucket8", "dynamic_bucket16"}:
        return ApproxPolicy(
            **common, dynamic_threshold=True, order=family.removeprefix("dynamic_")
        )
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
    if family == "full":
        return ApproxPolicy(name=name)
    raise ValueError(f"unknown family: {family}")


def build_policy(spec: dict[str, Any], layer: int, scales: dict[int, float]) -> ApproxPolicy:
    gamma = float(spec.get("multiplier", 0.0)) * float(scales.get(layer, 0.0))
    policy = family_policy(str(spec["family"]), name=str(spec["config_id"]), gamma=gamma)
    return replace(policy, **spec.get("overrides", {}))


def worker(
    payload: tuple[str, list[dict[str, Any]], dict[int, float], int, int, str]
) -> pd.DataFrame:
    path_string, specs, scales, k, block_size, history_mode = payload
    trace = load_trace(path_string)
    frames = []
    for spec in specs:
        policy = build_policy(spec, trace.layer, scales)
        frame = replay_approx_trace(
            trace,
            policy=policy,
            k=k,
            block_size=block_size,
            history_mode=history_mode,
        )
        frame["config_id"] = spec["config_id"]
        frame["family"] = spec["family"]
        frame["repair"] = spec.get("repair", "none")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, group in frame.groupby("config_id", sort=False):
        newly_hot = group.newly_hot_count.sum()
        row = {
            "policy": policy,
            "history_mode": group.history_mode.iloc[0],
            "family": group.family.iloc[0],
            "repair": group.repair.iloc[0],
            "transitions": len(group),
            "qk_reduction_mean": group.qk_reduction.mean(),
            "qk_reduction_p5": group.qk_reduction.quantile(0.05),
            "qk_reduction_p10": group.qk_reduction.quantile(0.10),
            "qk_reduction_median": group.qk_reduction.median(),
            "qk_reduction_p95": group.qk_reduction.quantile(0.95),
            "exact_match": group.exact_match.mean(),
            "recall_mean": group.recall.mean(),
            "recall_p1": group.recall.quantile(0.01),
            "recall_p5": group.recall.quantile(0.05),
            "recall_median": group.recall.median(),
            "recall_worst": group.recall.min(),
            "top128_recall": group.top128_recall.mean(),
            "top256_recall": group.top256_recall.mean(),
            "top512_recall": group.top512_recall.mean(),
            "top1024_recall": group.top1024_recall.mean(),
            "top2048_recall": group.top2048_recall.mean(),
            "index_mass_ratio_mean": group.index_mass_ratio.mean(),
            "index_mass_ratio_p5": group.index_mass_ratio.quantile(0.05),
            "newly_hot_miss_rate": group.newly_hot_missed.sum() / newly_hot if newly_hot else 0,
            "net_bf16_reduction_median": group.net_bf16_reduction.median(),
            "net_fp8_reduction_median": group.net_fp8_reduction.median(),
            "seed_block_fraction_median": group.seed_block_fraction.median(),
            "metadata_bytes_per_block": group.metadata_bytes_per_block.iloc[0],
            "full_refresh_rate": group.full_refresh.mean(),
        }
        for block in (16, 32, 64, 128):
            row[f"physical_b{block}_reduction_median"] = group[
                f"physical_b{block}_reduction"
            ].median()
        for cache in (0, 256, 512, 1024, 2048):
            row[f"cache{cache}_net_bf16_reduction_median"] = group[
                f"cache{cache}_net_bf16_reduction"
            ].median()
        rows.append(row)
    return pd.DataFrame(rows)


def choose(summary: pd.DataFrame, low: float, high: float, target: float) -> pd.Series:
    candidates = summary[
        summary.qk_reduction_median.between(low, high) & summary.policy.ne("full")
    ]
    if candidates.empty:
        candidates = summary[summary.policy.ne("full")].assign(
            distance=(summary.qk_reduction_median - target).abs()
        ).sort_values("distance").head(8)
    return candidates.sort_values(
        ["index_mass_ratio_mean", "top128_recall", "recall_mean", "qk_reduction_median"],
        ascending=[False, False, False, False],
    ).iloc[0]


def deduplicate(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for spec in specs:
        if spec["config_id"] not in seen:
            result.append(spec)
            seen.add(spec["config_id"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Predefined temporal repair ablations")
    parser.add_argument("--phase-a", type=Path, required=True)
    parser.add_argument("--heldout-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    selection = json.loads((args.phase_a / "selected_policies.json").read_text())
    scales = {int(key): float(value) for key, value in selection["gamma_scale"].items()}
    k, block_size = int(selection["k"]), int(selection["block_size"])
    calibration = pd.read_csv(args.phase_a / "target_reduction_calibration.csv")
    specs = [{"config_id": "full", "family": "full", "multiplier": 0.0}]
    for row in calibration.itertuples(index=False):
        specs.append(
            {
                "config_id": str(row.config_id),
                "family": str(row.family),
                "multiplier": float(row.multiplier),
                "repair": "none",
            }
        )
    bases = [
        spec
        for spec in selection["policies"]
        if spec["family"] in {"streak2_bucket8", "static", "dynamic_bucket16"}
    ]
    for base in bases:
        for interval in (2, 4, 8, 16):
            specs.append(
                {
                    **base,
                    "config_id": f"{base['config_id']}_refresh{interval}",
                    "overrides": {"refresh_interval": interval},
                    "repair": f"refresh{interval}",
                }
            )
    for base in [spec for spec in bases if spec["family"] != "static"]:
        for age in (2, 4, 8, 16):
            specs.append(
                {
                    **base,
                    "config_id": f"{base['config_id']}_age{age}",
                    "overrides": {"age_cap": age},
                    "repair": f"age{age}",
                }
            )
        for window in (128, 256, 512, 1024):
            specs.append(
                {
                    **base,
                    "config_id": f"{base['config_id']}_recent{window}",
                    "overrides": {"recent_window": window},
                    "repair": f"recent{window}",
                }
            )
    specs = deduplicate(specs)
    paths = sorted({path for root in args.heldout_roots for path in root.rglob("*.npz")})
    args.output.mkdir(parents=True, exist_ok=True)
    own_path = args.output / "repair_rows_own.csv"
    if args.reuse and own_path.exists():
        own = pd.read_csv(own_path)
    else:
        payloads = [(str(path), specs, scales, k, block_size, "own") for path in paths]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            own_pieces = list(executor.map(worker, payloads))
        own = pd.concat(own_pieces, ignore_index=True)
        own.to_csv(own_path, index=False)
    summary = aggregate(own)
    summary.to_csv(args.output / "repair_summary_own.csv", index=False)

    chosen_rows = {
        "Safe": choose(summary, 0.10, 0.20, 0.15),
        "Balanced": choose(summary, 0.25, 0.35, 0.30),
        "Aggressive": choose(summary, 0.40, 0.50, 0.45),
    }
    by_id = {spec["config_id"]: spec for spec in specs}
    chosen_specs = [by_id[str(row.policy)] for row in chosen_rows.values()]
    teacher_specs = deduplicate(
        [{"config_id": "full", "family": "full", "multiplier": 0.0}, *chosen_specs]
    )
    teacher_path = args.output / "repair_rows_teacher.csv"
    if args.reuse and teacher_path.exists():
        teacher = pd.read_csv(teacher_path)
    else:
        teacher_payloads = [
            (str(path), teacher_specs, scales, k, block_size, "teacher") for path in paths
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            teacher_pieces = list(executor.map(worker, teacher_payloads))
        teacher = pd.concat(teacher_pieces, ignore_index=True)
        teacher.to_csv(teacher_path, index=False)
    combined_summary = pd.concat([summary, aggregate(teacher)], ignore_index=True)
    combined_summary.to_csv(args.output / "policy_pareto_repaired.csv", index=False)

    phase_b = {
        "selection_basis": "Phase-A held-out Pareto selection; policy parameters remain validation-calibrated or prespecified repair values",
        "k": k,
        "block_size": block_size,
        "gamma_scale": {str(key): value for key, value in scales.items()},
        "policies": [
            {**by_id[str(row.policy)], "selection_role": role}
            for role, row in chosen_rows.items()
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output / "selected_phase_b_policies.json").write_text(
        json.dumps(phase_b, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    own.groupby(
        ["config_id", "layer", "base_context_length", "workload"], as_index=False
    ).agg(
        qk_reduction=("qk_reduction", "median"),
        recall=("recall", "mean"),
        top128_recall=("top128_recall", "mean"),
        top512_recall=("top512_recall", "mean"),
        top2048_recall=("top2048_recall", "mean"),
        index_mass_ratio=("index_mass_ratio", "mean"),
        newly_hot_miss_rate=("newly_hot_miss_rate", "mean"),
    ).rename(columns={"config_id": "policy"}).to_csv(
        args.output / "rank_recall_repaired.csv", index=False
    )
    print(combined_summary.to_string(index=False))
    print(json.dumps(phase_b, indent=2))


if __name__ == "__main__":
    main()
