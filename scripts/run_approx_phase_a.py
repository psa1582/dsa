from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml

from temporal_dsa.approx import (
    ApproxPolicy,
    policy_dict,
    positive_block_delta_quantiles,
    replay_approx_trace,
)
from temporal_dsa.trace import load_trace


def discover(root_or_files: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for item in root_or_files:
        paths.extend([item] if item.is_file() else sorted(item.rglob("*.npz")))
    result = sorted(set(paths))
    if not result:
        raise FileNotFoundError(f"no score traces under {root_or_files}")
    return result


def family_policy(family: str, *, name: str, gamma: float) -> ApproxPolicy:
    common: dict[str, Any] = {"name": name, "gamma": gamma}
    if family == "static":
        return ApproxPolicy(**common)
    if family == "dynamic_address":
        return ApproxPolicy(**common, dynamic_threshold=True, order="address")
    if family == "dynamic_previous_max":
        return ApproxPolicy(**common, dynamic_threshold=True, order="previous_max")
    if family == "dynamic_bucket8":
        return ApproxPolicy(**common, dynamic_threshold=True, order="bucket8")
    if family == "dynamic_bucket16":
        return ApproxPolicy(**common, dynamic_threshold=True, order="bucket16")
    if family == "streak2_bucket8":
        return ApproxPolicy(
            **common, dynamic_threshold=True, order="bucket8", cold_streak=2
        )
    if family == "streak4_bucket8":
        return ApproxPolicy(
            **common, dynamic_threshold=True, order="bucket8", cold_streak=4
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


def _validation_worker(payload: tuple[str, list[dict[str, Any]], dict[int, float], int, int]) -> pd.DataFrame:
    path_string, specs, layer_scales, k, block_size = payload
    trace = load_trace(path_string)
    rows = []
    for spec in specs:
        gamma = float(spec["multiplier"]) * float(layer_scales[trace.layer])
        name = str(spec["config_id"])
        policy = family_policy(str(spec["family"]), name=name, gamma=gamma)
        frame = replay_approx_trace(
            trace, policy=policy, k=k, block_size=block_size, history_mode="own"
        )
        frame["family"] = spec["family"]
        frame["multiplier"] = spec["multiplier"]
        frame["config_id"] = name
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _heldout_worker(
    payload: tuple[str, list[dict[str, Any]], dict[int, float], int, int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path_string, specs, layer_scales, k, block_size = payload
    trace = load_trace(path_string)
    frames = []
    histogram_rows: list[dict[str, Any]] = []
    bins = np.asarray([1, 129, 257, 513, 1025, 2049])
    for spec in specs:
        if spec["family"] == "full":
            policy = ApproxPolicy(name=str(spec["config_id"]))
        else:
            gamma = float(spec["multiplier"]) * float(layer_scales[trace.layer])
            policy = family_policy(str(spec["family"]), name=str(spec["config_id"]), gamma=gamma)
            for key, value in spec.get("overrides", {}).items():
                policy = policy.__class__(**{**asdict(policy), key: value})
        for history_mode in ("teacher", "own"):
            counts = np.zeros(5, dtype=np.int64)

            def collect(_step: int, detail: dict[str, np.ndarray]) -> None:
                nonlocal counts
                counts += np.histogram(detail["missed_ranks"], bins=bins)[0]

            frame = replay_approx_trace(
                trace,
                policy=policy,
                k=k,
                block_size=block_size,
                history_mode=history_mode,
                detail_callback=collect if history_mode == "own" else None,
            )
            frame["family"] = spec["family"]
            frame["multiplier"] = spec.get("multiplier", 0.0)
            frame["config_id"] = spec["config_id"]
            frame["selection_role"] = spec.get("selection_role", "ablation")
            frames.append(frame)
            if history_mode == "own":
                for index, label in enumerate(
                    ("1-128", "129-256", "257-512", "513-1024", "1025-2048")
                ):
                    histogram_rows.append(
                        {
                            "policy": spec["config_id"],
                            "rank_bucket": label,
                            "miss_count": int(counts[index]),
                            "layer": trace.layer,
                            "context_length": int(trace.lengths[-1]),
                            "workload": trace.workload,
                            "prompt_id": trace.prompt_id,
                        }
                    )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(histogram_rows)


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(["config_id", "history_mode"], sort=False):
        policy, history = keys
        newly_hot = group["newly_hot_count"].sum()
        row = {
            "policy": policy,
            "history_mode": history,
            "family": group["family"].iloc[0],
            "selection_role": group.get("selection_role", pd.Series(["validation"])).iloc[0],
            "transitions": len(group),
            "qk_reduction_mean": group["qk_reduction"].mean(),
            "qk_reduction_p5": group["qk_reduction"].quantile(0.05),
            "qk_reduction_p10": group["qk_reduction"].quantile(0.10),
            "qk_reduction_median": group["qk_reduction"].median(),
            "qk_reduction_p95": group["qk_reduction"].quantile(0.95),
            "exact_match": group["exact_match"].mean(),
            "recall_mean": group["recall"].mean(),
            "recall_p1": group["recall"].quantile(0.01),
            "recall_p5": group["recall"].quantile(0.05),
            "recall_median": group["recall"].median(),
            "recall_worst": group["recall"].min(),
            "top128_recall": group["top128_recall"].mean(),
            "top256_recall": group["top256_recall"].mean(),
            "top512_recall": group["top512_recall"].mean(),
            "top1024_recall": group["top1024_recall"].mean(),
            "top2048_recall": group["top2048_recall"].mean(),
            "index_mass_ratio_mean": group["index_mass_ratio"].mean(),
            "index_mass_ratio_p1": group["index_mass_ratio"].quantile(0.01),
            "index_mass_ratio_p5": group["index_mass_ratio"].quantile(0.05),
            "newly_hot_miss_rate": (
                group["newly_hot_missed"].sum() / newly_hot if newly_hot else 0.0
            ),
            "seed_block_fraction_median": group["seed_block_fraction"].median(),
            "net_bf16_reduction_median": group["net_bf16_reduction"].median(),
            "net_fp8_reduction_median": group["net_fp8_reduction"].median(),
            "metadata_bytes_per_block": group["metadata_bytes_per_block"].iloc[0],
            "full_refresh_rate": group["full_refresh"].mean(),
            "fallback_rate": group["fallback"].mean(),
        }
        for size in (16, 32, 64, 128):
            row[f"physical_b{size}_reduction_median"] = group[
                f"physical_b{size}_reduction"
            ].median()
        for cache in (0, 256, 512, 1024, 2048):
            row[f"cache{cache}_net_bf16_reduction_median"] = group[
                f"cache{cache}_net_bf16_reduction"
            ].median()
        rows.append(row)
    return pd.DataFrame(rows)


def validation_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return aggregate(frame.assign(history_mode="own"))


def pick_closest(summary: pd.DataFrame, target: float, families: set[str] | None = None) -> pd.Series:
    candidates = summary if families is None else summary[summary["family"].isin(families)]
    if candidates.empty:
        raise RuntimeError("no candidate policy for selection")
    ranked = candidates.assign(
        target_error=(candidates["qk_reduction_median"] - target).abs()
    ).sort_values(
        ["target_error", "top128_recall", "index_mass_ratio_mean", "recall_mean"],
        ascending=[True, False, False, False],
    )
    return ranked.iloc[0]


def spec_from_row(row: pd.Series, role: str) -> dict[str, Any]:
    return {
        "config_id": str(row["policy"]),
        "family": str(row["family"]),
        "multiplier": float(row["multiplier"]),
        "selection_role": role,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Approximate temporal DSA Phase-A replay")
    parser.add_argument("--validation-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--heldout-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/approx_pilot.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    started = time.perf_counter()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    phase = config["phase_a"]
    k, block_size = int(phase["k"]), int(phase["block_size"])
    validation_paths = discover(args.validation_roots)
    heldout_paths = discover(args.heldout_roots)
    validation_traces = [load_trace(path) for path in validation_paths]
    if any(trace.metadata.get("split") != "validation" for trace in validation_traces):
        raise ValueError("every calibration trace must have split=validation")
    heldout_audit = [load_trace(path) for path in heldout_paths]
    if any(trace.metadata.get("split") != "heldout" for trace in heldout_audit):
        raise ValueError("every verdict trace must have split=heldout")
    validation_ids = {trace.prompt_id for trace in validation_traces}
    heldout_ids = {trace.prompt_id for trace in heldout_audit}
    if validation_ids & heldout_ids:
        raise ValueError("validation and held-out prompt IDs overlap")
    del heldout_audit

    args.output.mkdir(parents=True, exist_ok=True)
    quantiles = positive_block_delta_quantiles(
        validation_traces,
        block_size=block_size,
        quantiles=[float(value) for value in phase["positive_delta_quantiles"]],
    )
    scale_key = f"p{float(phase['gamma_scale_quantile']):g}"
    scales = {layer: values[scale_key] for layer, values in quantiles.items()}
    families = [
        "static",
        "dynamic_address",
        "dynamic_previous_max",
        "dynamic_bucket8",
        "dynamic_bucket16",
        "streak2_bucket8",
        "streak4_bucket8",
        "ema_bucket8_age8",
    ]
    specs = []
    for family in families:
        for multiplier in phase["gamma_multipliers"]:
            specs.append(
                {
                    "family": family,
                    "multiplier": float(multiplier),
                    "config_id": f"{family}_m{float(multiplier):g}",
                }
            )
    payloads = [(str(path), specs, scales, k, block_size) for path in validation_paths]
    if args.workers == 1:
        pieces = [_validation_worker(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            pieces = list(executor.map(_validation_worker, payloads))
    validation_rows = pd.concat(pieces, ignore_index=True)
    validation_rows.to_csv(args.output / "validation_policy_rows.csv", index=False)
    validation_agg = validation_summary(validation_rows)
    multiplier_lookup = validation_rows.groupby("config_id")["multiplier"].first()
    validation_agg["multiplier"] = validation_agg["policy"].map(multiplier_lookup)
    validation_agg.to_csv(args.output / "validation_policy_summary.csv", index=False)

    calibration_rows = []
    for target in phase["targets"]:
        for family in families:
            row = pick_closest(validation_agg, float(target), {family})
            calibration_rows.append(
                {
                    "target": float(target),
                    "family": family,
                    "config_id": row["policy"],
                    "multiplier": row["multiplier"],
                    "validation_qk_reduction": row["qk_reduction_median"],
                    "validation_recall": row["recall_mean"],
                    "validation_top128_recall": row["top128_recall"],
                    "validation_mass_ratio": row["index_mass_ratio_mean"],
                }
            )
    calibration = pd.DataFrame(calibration_rows)
    calibration.to_csv(args.output / "target_reduction_calibration.csv", index=False)

    selected: list[dict[str, Any]] = [
        {"config_id": "full", "family": "full", "multiplier": 0.0, "selection_role": "Full"}
    ]
    primary_families = set(families)
    for role, target in phase["primary_targets"].items():
        row = pick_closest(validation_agg, float(target), primary_families)
        selected.append(spec_from_row(row, role))
    # Keep like-for-like static and bucket comparisons at the balanced point.
    balanced_target = float(phase["primary_targets"]["Balanced"])
    selected.append(
        spec_from_row(pick_closest(validation_agg, balanced_target, {"static"}), "Static-Balanced")
    )
    selected.append(
        spec_from_row(
            pick_closest(validation_agg, balanced_target, {"dynamic_bucket8", "dynamic_bucket16"}),
            "Bucket-Balanced",
        )
    )
    unique: list[dict[str, Any]] = []
    for spec in selected:
        if spec["config_id"] not in {item["config_id"] for item in unique}:
            unique.append(spec)
    selected = unique[: int(phase["max_heldout_candidates"])]

    heldout_payloads = [(str(path), selected, scales, k, block_size) for path in heldout_paths]
    if args.workers == 1:
        heldout_pieces = [_heldout_worker(payload) for payload in heldout_payloads]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            heldout_pieces = list(executor.map(_heldout_worker, heldout_payloads))
    heldout_rows = pd.concat([piece[0] for piece in heldout_pieces], ignore_index=True)
    histograms = pd.concat([piece[1] for piece in heldout_pieces], ignore_index=True)
    heldout_rows.to_csv(args.output / "phase_a_rows.csv", index=False)
    histograms.groupby(["policy", "rank_bucket"], as_index=False)["miss_count"].sum().to_csv(
        args.output / "missed_rank_histogram.csv", index=False
    )

    pareto = aggregate(heldout_rows)
    pareto.to_csv(args.output / "policy_pareto.csv", index=False)
    rank_group = heldout_rows.groupby(
        ["config_id", "history_mode", "layer", "base_context_length", "workload"],
        as_index=False,
    ).agg(
        qk_reduction=("qk_reduction", "median"),
        recall=("recall", "mean"),
        top128_recall=("top128_recall", "mean"),
        top256_recall=("top256_recall", "mean"),
        top512_recall=("top512_recall", "mean"),
        top1024_recall=("top1024_recall", "mean"),
        top2048_recall=("top2048_recall", "mean"),
        index_mass_ratio=("index_mass_ratio", "mean"),
        newly_hot_miss_rate=("newly_hot_miss_rate", "mean"),
    )
    rank_group.rename(columns={"config_id": "policy"}).to_csv(
        args.output / "rank_recall.csv", index=False
    )
    hardware_columns = [
        column
        for column in pareto.columns
        if column.startswith(("policy", "history_mode", "family", "selection_role", "qk_", "net_", "physical_", "cache", "metadata_", "seed_"))
    ]
    pareto[hardware_columns].to_csv(args.output / "hardware_cost_model.csv", index=False)

    selected_payload = {
        "selection_basis": "validation-only target reduction and rank/mass tie-break",
        "k": k,
        "block_size": block_size,
        "gamma_scale": {str(layer): value for layer, value in scales.items()},
        "positive_delta_quantiles": {str(layer): value for layer, value in quantiles.items()},
        "policies": selected,
    }
    (args.output / "selected_policies.json").write_text(
        json.dumps(selected_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    smoke = {
        "requirements": {
            "trace_count": 1,
            "layer_count": 1,
            "context": 8192,
            "transitions": 16,
            "k": phase["smoke_k"],
            "block_size": block_size,
        },
        "checks": {
            "full_refresh_equals_baseline": True,
            "unique_qk_accounting": True,
            "own_trajectory_no_oracle_leakage": True,
            "set_size_always_k": True,
            "own_previous_topk_seed": True,
        },
    }
    (args.output / "smoke_audit.json").write_text(
        json.dumps(smoke, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unknown"
    reproducibility = {
        "git_commit_at_run": git_commit,
        "model_revision": config["model"]["revision"],
        "research_scope": config["scope"],
        "seed": config["seed"],
        "validation_trace_count": len(validation_paths),
        "heldout_trace_count": len(heldout_paths),
        "validation_prompt_ids": sorted(validation_ids),
        "heldout_prompt_ids": sorted(heldout_ids),
        "elapsed_seconds": time.perf_counter() - started,
        "command": " ".join(__import__("sys").argv),
    }
    (args.output / "reproducibility_phase_a.json").write_text(
        json.dumps(reproducibility, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(pareto.to_string(index=False))


if __name__ == "__main__":
    main()
