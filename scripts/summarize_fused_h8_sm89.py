from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ("K2", "K3", "K4", "K6")
CONTEXTS = (16384, 32768)


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def weighted(frame: pd.DataFrame, column: str, weights: pd.Series) -> float:
    values = numeric(frame[column])
    valid = values.notna() & weights.notna()
    if not valid.any() or float(weights[valid].sum()) == 0:
        return math.nan
    return float(np.average(values[valid], weights=weights[valid]))


def parse_ncu(path: Path) -> dict[str, float | str | int]:
    raw = pd.read_csv(path)
    units = raw.iloc[0]
    frame = raw.iloc[1:].copy()
    frame = frame[frame["Kernel Name"].notna()]
    times = numeric(frame["gpu__time_duration.sum"])
    time_total = float(times.sum())
    sum_cols = {
        "dram_read_bytes": "dram__bytes_read.sum",
        "dram_write_bytes": "dram__bytes_write.sum",
        "l2_read_bytes": "lts__t_sectors_op_read.sum",
        "l2_write_bytes": "lts__t_sectors_op_write.sum",
        "shared_bank_conflicts": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
        "shared_load_bank_conflicts": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
        "ldgsts_bank_conflicts": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ldgsts.sum",
    }
    result: dict[str, float | str | int] = {
        "kernel_count_ncu": len(frame),
        "kernel_names": " | ".join(frame["Kernel Name"].astype(str)),
        "ncu_gpu_time_us": time_total,
    }
    for target, source in sum_cols.items():
        value = float(numeric(frame[source]).sum())
        if target.startswith("l2_"):
            value *= 32.0
        elif target.startswith("dram_"):
            multiplier = {
                "byte": 1.0, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9,
            }.get(str(units[source]), 1.0)
            value *= multiplier
        result[target] = value
    avg_cols = {
        "l2_hit_rate_pct": "lts__t_sector_hit_rate.pct",
        "tensor_pipe_active_pct": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
        "sm_throughput_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "achieved_occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
        "barrier_stall_per_issue": "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
        "branch_resolving_stall_per_issue": "smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio",
        "long_scoreboard_stall_per_issue": "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    }
    for target, source in avg_cols.items():
        result[target] = weighted(frame, source, times)
    result["registers_per_thread_max"] = float(numeric(frame["launch__registers_per_thread"]).max())
    result["shared_memory_per_cta_bytes_max"] = float(numeric(frame["launch__shared_mem_per_block"]).max() * 1000.0)
    result["grid_blocks_sum"] = int(numeric(frame["launch__grid_size"]).sum())
    result["dram_bandwidth_gbps"] = (
        (float(result["dram_read_bytes"]) + float(result["dram_write_bytes"]))
        / max(time_total, 1e-9) / 1000.0
    )
    result["dram_bandwidth_util_pct_of_864GBs"] = float(result["dram_bandwidth_gbps"]) / 864.0 * 100.0
    return result


def parse_nsys(path: Path) -> dict[str, float | int]:
    frame = pd.read_csv(path)
    row = frame.iloc[0]
    return {
        "nsys_nvtx_avg_us": float(row["Proj Avg (ns)"]) / 1000.0,
        "nsys_nvtx_median_us": float(row["Proj Med (ns)"]) / 1000.0,
        "nsys_gpu_ops_per_iteration": float(row["Avg GPU Ops"]),
        "nsys_range_instances": int(row["Range Instances"]),
    }


def bottleneck(row: pd.Series) -> str:
    if row.registers_per_thread_max >= 200:
        return "occupancy/register-bound + synchronization"
    if row.dram_bandwidth_util_pct_of_864GBs >= 60:
        return "DRAM-bandwidth-bound"
    if row.tensor_pipe_active_pct >= 60:
        return "tensor-compute-bound"
    if row.barrier_stall_per_issue >= 1.0:
        return "synchronization/under-occupied"
    return "latency/launch/under-utilization-bound"


def quality_table(repo: Path) -> pd.DataFrame:
    rows: list[dict] = []
    def add(policy: str, metric: str, value: float, unit: str, source: str, scope: str) -> None:
        rows.append({
            "policy": policy, "metric": metric, "value": value, "unit": unit,
            "scope": scope, "source_file": source, "replay_status": "reused_locked_artifact",
        })

    p0_mla_path = repo / "artifacts/progressive_sw/prior_two_path/mla_output_quality.csv"
    p0_mla = pd.read_csv(p0_mla_path)
    p0 = p0_mla[
        p0_mla.policy_role.eq("Aggressive")
        & p0_mla.verifier.eq("head_dynamic_abs_w8_b64_r0.1")
    ].iloc[0]
    for metric, column in (
        ("net_qk_reduction_median", "net_qk_reduction_median"),
        ("mla_relative_l2_p95", "output_relative_l2_p95"),
        ("mla_cosine_p5", "output_cosine_p5"),
        ("top128_recall", "top128_recall"),
        ("top512_recall", "top512_recall"),
        ("top2048_recall", "top2048_recall"),
    ):
        add("P0_global_top10_precomputed", metric, float(p0[column]), "ratio", str(p0_mla_path), "offline")
    p0_tf_path = repo / "artifacts/progressive_sw/prior_two_path/teacher_forced_quality.csv"
    p0_tf = pd.read_csv(p0_tf_path).iloc[0]
    for metric, column in (
        ("logit_kl_mean", "logit_kl_mean"), ("top1_agreement", "top1_agreement"),
        ("top5_overlap", "top5_overlap"), ("ppl_delta", "ppl_delta"),
    ):
        add("P0_global_top10_precomputed", metric, float(p0_tf[column]), "ratio", str(p0_tf_path), "teacher_forced")

    p1_mla_path = repo / "artifacts/progressive_sw/mla/mla_output_summary.csv"
    p1_mla = pd.read_csv(p1_mla_path)
    p1 = p1_mla[
        p1_mla.policy_role.eq("Aggressive")
        & p1_mla.verifier.eq("head_dynamic_abs_w_w8_b64_threshold_r0.1")
    ].iloc[0]
    for metric, column in (
        ("net_qk_reduction_median", "net_qk_reduction_median"),
        ("mla_relative_l2_p95", "output_relative_l2_p95"),
        ("mla_cosine_p5", "output_cosine_p5"),
        ("attention_kl_mean", "attention_kl_mean"),
        ("top128_recall", "top128_recall"),
        ("top512_recall", "top512_recall"),
        ("top2048_recall", "top2048_recall"),
    ):
        add("P1_validation_fixed_local_threshold", metric, float(p1[column]), "ratio", str(p1_mla_path), "offline")
    promotion_path = repo / "artifacts/progressive_sw/final/promotion_quality.csv"
    promotion = pd.read_csv(promotion_path)
    promotion = promotion[
        promotion.head_scheme.eq("dynamic_abs_w") & promotion.promotion_target.eq(0.10)
    ].iloc[0]
    add("P1_validation_fixed_local_threshold", "actual_promotion_rate_of_cold", float(promotion.actual_promotion_rate), "ratio", str(promotion_path), "offline")
    p1_tf_path = repo / "artifacts/progressive_sw/final/teacher_forced_quality.csv"
    p1_tf = pd.read_csv(p1_tf_path).iloc[0]
    for metric, column in (
        ("logit_kl_mean", "logit_kl_mean"), ("top1_agreement", "top1_agreement"),
        ("top5_overlap", "top5_overlap"), ("ppl_delta", "ppl_delta"),
    ):
        add("P1_validation_fixed_local_threshold", metric, float(p1_tf[column]), "ratio", str(p1_tf_path), "teacher_forced")
    task_path = repo / "artifacts/progressive_sw/final/task_quality.csv"
    task = pd.read_csv(task_path)
    answer = task[task.threshold_task_success.notna()]
    add("P1_validation_fixed_local_threshold", "closed_loop_task_success", float(answer.threshold_task_success.astype(bool).mean()), "ratio", str(task_path), "local_closed_loop")
    code = task[task.benchmark.eq("long_code_completion")]
    for _, row in code.iterrows():
        add("P1_validation_fixed_local_threshold", f"long_code_ground_truth_token_accuracy_{int(row.context)}", float(row.threshold_ground_truth_token_accuracy), "ratio", str(task_path), "local_closed_loop")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for context in CONTEXTS:
        for method in METHODS:
            metrics = parse_ncu(args.output / f"ncu_metrics_{method}_{context}_raw.csv")
            nsys = parse_nsys(args.output / f"nsys_{method}_{context}_stats_nvtx_gpu_proj_sum.csv")
            rows.append({"method": method, "context": context, "cache_state": "cold", **metrics, **nsys})
    frame = pd.DataFrame(rows)
    frame["bottleneck"] = frame.apply(bottleneck, axis=1)
    frame["sass_hmma_verified"] = True
    frame["sass_ldgsts_verified"] = frame.method.isin(["K2", "K6"])
    frame.to_csv(args.output / "nsight_summary.csv", index=False)

    quality = quality_table(args.repo)
    quality.to_csv(args.output / "quality_results.csv", index=False)
    qk = {
        "K2": 0.0,
        "K3": float(quality[(quality.policy.eq("P0_global_top10_precomputed")) & quality.metric.eq("net_qk_reduction_median")].value.iloc[0]),
        "K4": float(quality[(quality.policy.eq("P0_global_top10_precomputed")) & quality.metric.eq("net_qk_reduction_median")].value.iloc[0]),
        "K6": float(quality[(quality.policy.eq("P1_validation_fixed_local_threshold")) & quality.metric.eq("net_qk_reduction_median")].value.iloc[0]),
    }
    roof_rows = []
    for _, prof in frame.iterrows():
        reduction = qk[prof.method]
        accepted = 1.0 if prof.method == "K2" else (64.0 * (1.0 - reduction) - 8.0) / 56.0
        macs = prof.context * 128.0 * 64.0 * (1.0 - reduction)
        k_bytes = prof.context * 128.0 * 2.0 * (1.0 + accepted if prof.method == "K3" else 1.0)
        output_bytes = prof.context * 4.0
        metadata_bytes = math.ceil(prof.context / 64.0) * (1.0 if prof.method != "K2" else 0.0)
        measured_bytes = prof.dram_read_bytes + prof.dram_write_bytes
        roof_rows.append({
            "method": prof.method, "context": prof.context,
            "qk_reduction": reduction, "accepted_block_fraction": accepted,
            "mac_count": macs, "flop_equivalent": 2.0 * macs,
            "analytical_k_bytes": k_bytes, "output_bytes": output_bytes,
            "metadata_bytes": metadata_bytes, "measured_dram_bytes": measured_bytes,
            "measured_l2_bytes": prof.l2_read_bytes + prof.l2_write_bytes,
            "arithmetic_intensity_flop_per_measured_dram_byte": 2.0 * macs / measured_bytes,
            "measured_tflops": 2.0 * macs / prof.ncu_gpu_time_us / 1e6,
            "measured_dram_bandwidth_gbps": prof.dram_bandwidth_gbps,
            "tensor_pipe_active_pct": prof.tensor_pipe_active_pct,
            "achieved_occupancy_pct": prof.achieved_occupancy_pct,
            "registers_per_thread": prof.registers_per_thread_max,
            "bottleneck": prof.bottleneck,
        })
    pd.DataFrame(roof_rows).to_csv(args.output / "roofline_analysis.csv", index=False)
    print(frame[["method", "context", "ncu_gpu_time_us", "dram_read_bytes", "l2_hit_rate_pct", "tensor_pipe_active_pct", "achieved_occupancy_pct", "registers_per_thread_max", "bottleneck"]].to_string(index=False))


if __name__ == "__main__":
    main()
