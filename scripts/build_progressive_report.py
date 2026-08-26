from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_REVISION = "85864749cd611b4353ce1decdb286193298f64c7"
VERDICT = "ALGORITHM-PROMISING-BUT-SOFTWARE-SLOW"


def save(fig: plt.Figure, root: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(root / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def placeholder(root: Path, name: str, title: str, reason: str) -> None:
    fig, axis = plt.subplots(figsize=(8.2, 4.5))
    axis.axis("off")
    axis.text(0.5, 0.62, title, ha="center", fontsize=16, weight="bold")
    axis.text(0.5, 0.38, reason, ha="center", fontsize=11, wrap=True)
    save(fig, root, name)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rate_from_name(value: str) -> float:
    return float(value.rsplit("_r", 1)[1])


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(value) for value in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def primary_table(timing: pd.DataFrame) -> pd.DataFrame:
    primary = timing[
        timing.block_size.eq(64)
        & timing.promotion_target.eq(0.10)
        & timing.topk.eq(2048)
        & timing.method.isin(
            [
                "full_plus_topk",
                "two_pass_plus_topk",
                "fused_dense_plus_topk",
                "fused_compact_plus_topk",
            ]
        )
    ].copy()
    pivot = primary.pivot(index="context", columns="method", values="median_ms")
    pivot["fused_speedup_vs_full"] = pivot.full_plus_topk / pivot.fused_dense_plus_topk
    pivot["fused_speedup_vs_two_pass"] = (
        pivot.two_pass_plus_topk / pivot.fused_dense_plus_topk
    )
    pivot["compact_speedup_vs_dense"] = (
        pivot.fused_dense_plus_topk / pivot.fused_compact_plus_topk
    )
    return pivot.reset_index()


def build_graphs(
    root: Path,
    timing: pd.DataFrame,
    calibration: pd.DataFrame,
    stage: pd.DataFrame,
    promotion: pd.DataFrame,
    mla: pd.DataFrame,
    fixed: pd.DataFrame,
    closed: pd.DataFrame,
    nsight: pd.DataFrame,
) -> list[str]:
    root.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    primary = primary_table(timing)
    nearest_actual = timing[
        timing.block_size.eq(64)
        & timing.promotion_target.eq(0.20)
        & timing.topk.eq(2048)
        & timing.method.isin(["full_plus_topk", "fused_dense_plus_topk"])
    ].pivot(index="context", columns="method", values="median_ms")
    nearest_actual["speedup"] = (
        nearest_actual.full_plus_topk / nearest_actual.fused_dense_plus_topk
    )
    method_labels = {
        "full_plus_topk": "Full",
        "two_pass_plus_topk": "Two-pass",
        "fused_dense_plus_topk": "Fused dense",
        "fused_compact_plus_topk": "Fused compact",
    }

    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(primary))
    width = 0.2
    for offset, method in enumerate(method_labels):
        axis.bar(x + (offset - 1.5) * width, primary[method] * 1000, width, label=method_labels[method])
    axis.set_xticks(x, [f"{value//1024}K" for value in primary.context])
    axis.set(ylabel="CUDA-event median (µs)", xlabel="Context", title="Full vs two-pass vs fused, B64/r10/K2048")
    axis.legend(fontsize=8)
    save(fig, root, "01_full_two_pass_fused_latency.png"); generated.append("01_full_two_pass_fused_latency.png")

    fig, axis = plt.subplots(figsize=(7.5, 4.6))
    axis.plot(primary.context / 1024, primary.fused_speedup_vs_full, marker="o", label="Full / fused")
    axis.plot(primary.context / 1024, primary.fused_speedup_vs_two_pass, marker="o", label="Two-pass / fused")
    axis.axhline(1.0, color="black", linewidth=1)
    axis.axhline(1.10, color="gray", linestyle="--", linewidth=1, label="1.10× gate")
    axis.set(xlabel="Context (K tokens)", ylabel="Speedup (higher is better)", title="Context length vs indexer+TopK speedup")
    axis.legend()
    save(fig, root, "02_context_vs_indexer_speedup.png"); generated.append("02_context_vs_indexer_speedup.png")

    sweep = timing[
        timing.method.eq("fused_dense_plus_topk")
        & timing.block_size.eq(64)
        & timing.topk.eq(2048)
    ]
    fig, axis = plt.subplots(figsize=(7.5, 4.6))
    for context, group in sweep.groupby("context"):
        group = group.sort_values("promotion_target")
        axis.plot(group.promotion_target * 100, group.median_ms * 1000, marker="o", label=f"{context//1024}K")
    axis.set(xlabel="Promotion target (%)", ylabel="Fused dense median (µs)", title="Promotion rate vs latency")
    axis.legend()
    save(fig, root, "03_promotion_rate_vs_latency.png"); generated.append("03_promotion_rate_vs_latency.png")

    dynamic_mla = mla[mla.verifier.str.contains("head_dynamic_abs_w_w8_b64_threshold", regex=False)].copy()
    if len(dynamic_mla) >= 2:
        dynamic_mla["rate"] = dynamic_mla.verifier.map(rate_from_name)
        dynamic_mla = dynamic_mla.sort_values("rate")
        fig, axis = plt.subplots(figsize=(7.5, 4.6))
        axis.plot(dynamic_mla.rate * 100, dynamic_mla.output_relative_l2_p95 * 100, marker="o")
        axis.axhspan(5, 7, color="#f59e0b", alpha=0.15, label="review band 5–7%")
        axis.set(xlabel="Held-out threshold target (%)", ylabel="MLA RelL2 p95 (%)", title="Promotion rate vs actual MLA error")
        axis.legend()
        save(fig, root, "04_promotion_rate_vs_mla_rell2.png")
    else:
        placeholder(root, "04_promotion_rate_vs_mla_rell2.png", "Promotion rate vs MLA RelL2", "Only the primary r10 point was evaluated after the latency early-stop gate.")
    generated.append("04_promotion_rate_vs_mla_rell2.png")

    fig, axis = plt.subplots(figsize=(7.5, 4.6))
    task = closed[closed.baseline_task_success.notna()] if not closed.empty else closed
    if not task.empty:
        grouped = task.groupby("benchmark", as_index=False).agg(
            baseline=("baseline_task_success", "mean"), approximate=("approx_task_success", "mean")
        )
        positions = np.arange(len(grouped))
        axis.bar(positions - 0.18, grouped.baseline * 100, 0.36, label="Full sparse")
        axis.bar(positions + 0.18, grouped.approximate * 100, 0.36, label="Threshold H8")
        axis.set_xticks(positions, [value.replace("ruler_", "") for value in grouped.benchmark], rotation=18)
        axis.set(ylabel="Task success (%)", title="Primary r10 task quality (other rates early-stopped)")
        axis.legend()
    else:
        axis.axis("off"); axis.text(0.5, 0.5, "Task measurements unavailable", ha="center")
    save(fig, root, "05_promotion_rate_vs_ppl_task.png"); generated.append("05_promotion_rate_vs_ppl_task.png")

    perf = timing[
        timing.method.eq("fused_dense_plus_topk") & timing.topk.eq(2048)
    ].merge(
        calibration.groupby(["context", "block_size", "promotion_target"], as_index=False).qk_reduction.mean(),
        on=["context", "block_size", "promotion_target"],
    )
    full = timing[
        timing.method.eq("full_plus_topk") & timing.topk.eq(2048)
    ][["context", "block_size", "promotion_target", "median_ms"]].rename(columns={"median_ms": "full_ms"})
    perf = perf.merge(full, on=["context", "block_size", "promotion_target"])
    perf["latency_reduction"] = 1 - perf.median_ms / perf.full_ms
    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    scatter = axis.scatter(perf.qk_reduction * 100, perf.latency_reduction * 100, c=perf.context / 1024, cmap="viridis", alpha=0.75)
    axis.axhline(0, color="black", linewidth=1)
    axis.set(xlabel="Algorithmic QK reduction (%)", ylabel="Measured latency reduction (%)", title="QK reduction does not convert to L40S speedup")
    fig.colorbar(scatter, ax=axis, label="Context (K)")
    save(fig, root, "06_qk_reduction_vs_actual_latency_reduction.png"); generated.append("06_qk_reduction_vs_actual_latency_reduction.png")

    accounting = []
    for row in primary.itertuples():
        base = row.context * 128 * 2
        accounting.extend([(row.context, "Full", base, row.full_plus_topk), (row.context, "Fused", base, row.fused_dense_plus_topk)])
    account = pd.DataFrame(accounting, columns=["context", "method", "modeled_k_bytes", "latency_ms"])
    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    for method, group in account.groupby("method"):
        axis.scatter(group.modeled_k_bytes / 1e6, group.latency_ms * 1000, label=method, s=55)
    axis.set(xlabel="Code-level K load bytes (MB; not DRAM counter)", ylabel="Median indexer+TopK (µs)", title="One K scan remains for full and fused")
    axis.legend()
    save(fig, root, "07_dram_bytes_vs_latency.png"); generated.append("07_dram_bytes_vs_latency.png")

    placeholder(root, "08_sm_utilization_vs_context.png", "SM utilization vs context", "Nsight Compute (ncu) is not installed on the L40S server. No SM-utilization value is fabricated.")
    generated.append("08_sm_utilization_vs_context.png")

    topk_stage = stage[stage.stage.eq("topk_full_vector") & stage.promotion_target.eq(0.10)].set_index("context")
    fig, axis = plt.subplots(figsize=(7.5, 4.6))
    share = [100 * topk_stage.loc[row.context, "median_ms"] / row.full_plus_topk for row in primary.itertuples()]
    axis.bar([f"{value//1024}K" for value in primary.context], share)
    axis.set(ylabel="Top-K share of full path (%)", title="Top-K dominates as context grows", ylim=(0, 100))
    save(fig, root, "09_topk_fraction_of_total_latency.png"); generated.append("09_topk_fraction_of_total_latency.png")

    fig, axis = plt.subplots(figsize=(7.5, 4.6))
    axis.plot(primary.context / 1024, primary.fused_dense_plus_topk * 1000, marker="o", label="Dense -inf")
    axis.plot(primary.context / 1024, primary.fused_compact_plus_topk * 1000, marker="o", label="Atomic compact")
    axis.set(xlabel="Context (K)", ylabel="Median (µs)", title="Dense output vs compact candidate")
    axis.legend()
    save(fig, root, "10_dense_vs_compact_output.png"); generated.append("10_dense_vs_compact_output.png")

    fixed_plot = fixed.dropna(subset=["output_relative_l2_p95"]) if "output_relative_l2_p95" in fixed else pd.DataFrame()
    if not fixed_plot.empty:
        fixed_plot = fixed_plot.sort_values("output_relative_l2_p95")
        fig, axis = plt.subplots(figsize=(9.2, 4.8))
        axis.bar(fixed_plot.head_scheme, fixed_plot.output_relative_l2_p95 * 100)
        axis.tick_params(axis="x", rotation=35)
        axis.set(ylabel="MLA RelL2 p95 (%)", title="Dynamic vs validation-fixed H8")
        save(fig, root, "11_dynamic_vs_fixed_h8.png")
    else:
        placeholder(root, "11_dynamic_vs_fixed_h8.png", "Dynamic vs fixed H8", "Fixed-head MLA measurements unavailable.")
    generated.append("11_dynamic_vs_fixed_h8.png")

    if not nsight.empty and {"method", "gpu_time_ms"}.issubset(nsight.columns):
        plot = nsight.groupby("method", as_index=False).gpu_time_ms.sum()
        fig, axis = plt.subplots(figsize=(8.0, 4.6))
        axis.bar(plot.method, plot.gpu_time_ms)
        axis.tick_params(axis="x", rotation=20)
        axis.set(ylabel="Profiled GPU time (ms)", title="Nsight Systems NVTX critical-path totals")
        save(fig, root, "12_kernel_launch_timeline.png")
    else:
        placeholder(root, "12_kernel_launch_timeline.png", "Kernel launch timeline", "Nsight Systems summary was not available.")
    generated.append("12_kernel_launch_timeline.png")

    traffic = []
    cal_primary = calibration[
        calibration.block_size.eq(64) & calibration.promotion_target.eq(0.10)
    ].groupby("context", as_index=False).agg(
        candidate_fraction=("candidate_fraction", "mean")
    )
    for row in cal_primary.itertuples():
        direct = {8192: 0.945519, 16384: 0.703342, 32768: 0.594227}[row.context]
        promoted = max(0.0, row.candidate_fraction - direct)
        traffic.extend([(row.context, "Full", 1.0), (row.context, "Two-pass", 1.0 + promoted), (row.context, "Fused", 1.0)])
    traffic = pd.DataFrame(traffic, columns=["context", "method", "k_read_ratio"])
    fig, axis = plt.subplots(figsize=(8.0, 4.6))
    for method, group in traffic.groupby("method"):
        axis.plot(group.context / 1024, group.k_read_ratio, marker="o", label=method)
    axis.set(xlabel="Context (K)", ylabel="Modeled K read / full", title="Rescue reread removed by same-tile continuation")
    axis.legend()
    save(fig, root, "13_physical_traffic_before_after.png"); generated.append("13_physical_traffic_before_after.png")

    candidates = stage[stage.stage.eq("topk_compact_candidates")].copy()
    fig, axis = plt.subplots(figsize=(7.5, 4.6))
    for context, group in candidates.groupby("context"):
        group = group.sort_values("candidate_fraction_mean")
        axis.plot(group.candidate_fraction_mean * context, group.median_ms * 1000, marker="o", label=f"{context//1024}K")
    axis.set(xlabel="Compact candidate tokens", ylabel="Top-K-only median (µs)", title="Candidate count vs Top-K latency")
    axis.legend()
    save(fig, root, "14_candidate_count_vs_topk_latency.png"); generated.append("14_candidate_count_vs_topk_latency.png")

    fig, axis = plt.subplots(figsize=(7.5, 4.6))
    projected = (primary.full_plus_topk - primary.fused_dense_plus_topk) * 8
    axis.bar([f"{value//1024}K" for value in primary.context], projected * 1000)
    axis.axhline(0, color="black", linewidth=1)
    axis.set(ylabel="Projected per-token change (µs; 8 layers)", title="Research-sidecar TPOT projection is negative")
    save(fig, root, "15_research_sidecar_tpot_speedup.png"); generated.append("15_research_sidecar_tpot_speedup.png")
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Build progressive DSA software report")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo / "artifacts" / "progressive_sw"
    bench = root / "bench"
    stages = root / "stages"
    threshold = root / "threshold"
    mla_root = root / "mla"
    teacher_root = root / "teacher_forced"
    closed_root = root / "closed_loop"
    prior = root / "prior_two_path"
    nsys_root = root / "nsight"
    args.output.mkdir(parents=True, exist_ok=True)
    graphs = args.output / "graphs_progressive_sw"

    timing = pd.read_csv(bench / "timing.csv")
    timing.loc[timing.method.eq("two_pass_kernel"), "kernel_launches"] = 4
    timing.loc[timing.method.eq("two_pass_plus_topk"), "kernel_launches"] = 5
    calibration = pd.read_csv(bench / "calibration.csv")
    stage = pd.read_csv(stages / "stage_timing.csv")
    promotion = pd.read_csv(threshold / "promotion_quality.csv")
    fixed = pd.read_csv(threshold / "fixed_vs_dynamic_heads.csv")
    mla = pd.read_csv(mla_root / "mla_output_summary.csv")
    teacher = pd.read_csv(teacher_root / "teacher_forced_summary.csv")
    closed = pd.read_csv(closed_root / "closed_loop_quality.csv")
    for column in ("baseline_text", "approx_text"):
        if column in closed:
            closed[column] = closed[column].map(
                lambda value: "\n".join(
                    line.rstrip() for line in str(value).splitlines()
                ).rstrip()
            )
    nsight_path = nsys_root / "nsight_summary.csv"
    nsight = pd.read_csv(nsight_path) if nsight_path.exists() else pd.DataFrame()

    timing[timing.method.str.startswith("full")].to_csv(args.output / "full_baseline_timing.csv", index=False)
    timing[timing.method.str.startswith("two_pass")].to_csv(args.output / "two_pass_timing.csv", index=False)
    timing[timing.method.str.startswith("fused")].to_csv(args.output / "fused_progressive_timing.csv", index=False)

    candidate = timing[timing.method.isin(["fused_dense_plus_topk", "fused_compact_plus_topk"])].pivot_table(
        index=["context", "block_size", "promotion_target", "topk"], columns="method", values="median_ms"
    ).reset_index()
    cal = calibration.groupby(["context", "block_size", "promotion_target"], as_index=False).agg(
        candidate_fraction=("candidate_fraction", "mean"), qk_reduction=("qk_reduction", "mean")
    )
    candidate = candidate.merge(cal, on=["context", "block_size", "promotion_target"])
    candidate["compact_speedup_vs_dense"] = candidate.fused_dense_plus_topk / candidate.fused_compact_plus_topk
    candidate.to_csv(args.output / "candidate_topk_results.csv", index=False)

    primary = primary_table(timing)
    nearest_actual = timing[
        timing.block_size.eq(64)
        & timing.promotion_target.eq(0.20)
        & timing.topk.eq(2048)
        & timing.method.isin(["full_plus_topk", "fused_dense_plus_topk"])
    ].pivot(index="context", columns="method", values="median_ms")
    nearest_actual["speedup"] = (
        nearest_actual.full_plus_topk / nearest_actual.fused_dense_plus_topk
    )
    breakdown_rows = []
    nsys_method = {
        "Full optimized": "optimized_full",
        "H8 two-pass": "h8_two_pass",
        "Fused progressive": "fused_progressive_dense",
        "Fused compact": "fused_progressive_compact",
    }
    for row in primary.itertuples():
        stage_context = stage[stage.context.eq(row.context) & stage.promotion_target.eq(0.10)]
        value = dict(zip(stage_context.stage, stage_context.median_ms))
        kernel_context = timing[
            timing.context.eq(row.context)
            & timing.block_size.eq(64)
            & timing.promotion_target.eq(0.10)
            & timing.topk.eq(2048)
        ].set_index("method").median_ms
        for method, total, launches in [
            ("Full optimized", row.full_plus_topk, 2),
            ("H8 two-pass", row.two_pass_plus_topk, 5),
            ("Fused progressive", row.fused_dense_plus_topk, 2),
            ("Fused compact", row.fused_compact_plus_topk, 3),
        ]:
            profiled = nsight[
                nsight.context.eq(row.context)
                & nsight.method.eq(nsys_method[method])
            ] if not nsight.empty and "context" in nsight else pd.DataFrame()
            actual_launches = (
                float(profiled.gpu_ops_per_iteration.iloc[0])
                if not profiled.empty
                else launches
            )
            breakdown_rows.append(
                {
                    "method": method,
                    "context": row.context,
                    "mask_head_select_ms": 0.0 if method == "Full optimized" else value.get("head_select_topk64x8", np.nan),
                    "verifier_ms": value.get("h8_verifier_scan", np.nan) if method == "H8 two-pass" else (kernel_context.get("fused_dense_kernel", np.nan) if method == "Fused progressive" else (max(0.0, total - value.get("topk_compact_candidates", 0.0)) if method == "Fused compact" else 0.0)),
                    "compaction_ms": value.get("rescue_threshold_mask", np.nan) if method == "H8 two-pass" else (max(0.0, row.fused_compact_plus_topk - row.fused_dense_plus_topk) if method == "Fused compact" else 0.0),
                    "full_rerank_ms": value.get("full_candidate_rerank", np.nan) if method == "H8 two-pass" else 0.0,
                    "topk_ms": value.get("topk_full_vector", np.nan) if method != "Fused compact" else value.get("topk_compact_candidates", np.nan),
                    "total_measured_ms": total,
                    "total_with_dynamic_head_select_ms": total + (0.0 if method == "Full optimized" else value.get("head_select_topk64x8", 0.0)),
                    "logical_launches": launches,
                    "nsight_gpu_ops_per_iteration": actual_launches,
                }
            )
    breakdown = pd.DataFrame(breakdown_rows)
    breakdown.to_csv(args.output / "kernel_breakdown.csv", index=False)
    if nsight.empty:
        nsight = pd.DataFrame(
            [{"method": "all", "status": "Nsight Systems not available", "dram_read_bytes": "unavailable_ncu", "l2_read_bytes": "unavailable_ncu"}]
        )
    nsight.to_csv(args.output / "nsight_summary.csv", index=False)

    global_quality = pd.read_csv(prior / "head_sparse_results.csv")
    global_quality = global_quality[
        global_quality.policy_role.eq("Aggressive") & global_quality.verifier.eq("head_dynamic_abs_w8_b64_r0.1")
    ].copy()
    global_row = {
        "head_scheme": "dynamic_abs_w",
        "promotion_policy": "global_budget_prior",
        "promotion_target": 0.10,
        "observations": int(global_quality.observations.iloc[0]),
        "actual_promotion_rate": global_quality.rescue_block_fraction_mean.iloc[0],
        "net_qk_reduction_median": global_quality.net_qk_reduction_median.iloc[0],
        "top128_recall": global_quality.top128_recall.iloc[0],
        "top512_recall": global_quality.top512_recall.iloc[0],
        "top2048_recall": global_quality.top2048_recall.iloc[0],
        "newly_active_token_recall": global_quality.newly_active_token_recall.iloc[0],
    }
    promotion_out = pd.concat([promotion, pd.DataFrame([global_row])], ignore_index=True)
    promotion_out.to_csv(args.output / "promotion_quality.csv", index=False)

    mla_map = mla.copy()
    mla_map["head_scheme"] = mla_map.verifier.str.extract(r"head_(.+)_w8_b64_threshold_r0\.1")[0]
    fixed_out = fixed.merge(
        mla_map[["head_scheme", "output_relative_l2_p95", "output_cosine_p5", "attention_kl_mean"]],
        on="head_scheme", how="left",
    )
    fixed_out.to_csv(args.output / "fixed_vs_dynamic_heads.csv", index=False)
    teacher.to_csv(args.output / "teacher_forced_quality.csv", index=False)
    closed.to_csv(args.output / "closed_loop_quality.csv", index=False)

    task_rows = []
    for row in closed.itertuples():
        task_rows.append(
            {
                "benchmark": row.benchmark,
                "context": row.context_length,
                "baseline_task_success": row.baseline_task_success,
                "threshold_task_success": row.approx_task_success,
                "baseline_ground_truth_token_accuracy": getattr(row, "baseline_ground_truth_token_accuracy", np.nan),
                "threshold_ground_truth_token_accuracy": getattr(row, "approx_ground_truth_token_accuracy", np.nan),
                "status": "measured_local",
            }
        )
    task_rows.append(
        {"benchmark": "LongBench", "context": np.nan, "status": "not_run_no_local_harness", "baseline_task_success": np.nan, "threshold_task_success": np.nan}
    )
    task = pd.DataFrame(task_rows)
    task.to_csv(args.output / "task_quality.csv", index=False)

    primary_quality = promotion[
        promotion.head_scheme.eq("dynamic_abs_w") & promotion.promotion_target.eq(0.10)
    ].iloc[0]
    primary_mla = mla[
        mla.verifier.eq("head_dynamic_abs_w_w8_b64_threshold_r0.1")
    ].iloc[0]
    selected = {
        "verdict": VERDICT,
        "performance_primary": {
            "block_size": 64,
            "promotion_target": 0.10,
            "topk": 2048,
            "output": "dense_-inf",
            "head_selection": "dynamic_abs_w_for_quality; fixed avoids routing but did not rescue latency conclusion",
        },
        "fallback": "optimized_full",
        "do_not_promote": ["fused_compact", "two_pass", "D32_to_H8"],
        "reason": "Primary fused indexer+TopK is slower than optimized full at 8K/16K/32K.",
    }
    (args.output / "selected_software_configs.json").write_text(json.dumps(selected, indent=2) + "\n")

    graphs_generated = build_graphs(
        graphs, timing, calibration, stage, promotion_out, mla, fixed_out, closed, nsight
    )
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.repo, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_commit = "c4c8075368fefd834051a324cf564160f1090eca (base; report built before final commit)"
    reproducibility = {
        "scope": "Research sidecar DSA on DeepSeek-V2-Lite; not production V3.2",
        "model_revision": MODEL_REVISION,
        "git_base_commit": git_commit,
        "gpu": "NVIDIA L40S x2 visible; measurements on cuda:0",
        "visible_devices": [0, 1],
        "timing": "CUDA events; 50 warmups; 500 measurements; 64 MiB flush; 3 real-trace rotation",
        "benchmark_grid": {"contexts": [8192, 16384, 32768], "block_sizes": [32, 64, 128], "promotion_targets": [0.05, 0.10, 0.15, 0.20], "topk": [512, 1024, 2048]},
        "tests_passed": 42,
        "ncu": "not installed; hardware DRAM/L2/SM/occupancy counters unavailable",
        "graphs": graphs_generated,
    }
    (args.output / "reproducibility.json").write_text(json.dumps(reproducibility, indent=2) + "\n")

    performance_rows = []
    cal_primary = calibration[calibration.block_size.eq(64) & calibration.promotion_target.eq(0.10)].groupby("context", as_index=False).qk_reduction.mean().set_index("context")
    for row in primary.itertuples():
        for method, latency, launches in [
            ("Full optimized", row.full_plus_topk, 2),
            ("H8 two-pass", row.two_pass_plus_topk, 5),
            ("Fused progressive", row.fused_dense_plus_topk, 2),
            ("Fused compact", row.fused_compact_plus_topk, 3),
        ]:
            profiled = nsight[
                nsight.context.eq(row.context)
                & nsight.method.eq(nsys_method[method])
            ] if not nsight.empty and "context" in nsight else pd.DataFrame()
            actual_launches = (
                f"{profiled.gpu_ops_per_iteration.iloc[0]:.0f}"
                if not profiled.empty
                else f"{launches} logical"
            )
            performance_rows.append(
                {
                    "Method": method,
                    "Context": f"{row.context//1024}K",
                    "QK reduction": "0.00%" if method == "Full optimized" else f"{cal_primary.loc[row.context, 'qk_reduction']*100:.2f}%",
                    "GPU ops/iter": actual_launches,
                    "Indexer+TopK": f"{latency*1000:.2f} µs",
                    "Speedup vs full": f"{row.full_plus_topk/latency:.3f}×",
                }
            )
    performance_md = markdown_table(pd.DataFrame(performance_rows))
    task_measured = task[task.status.eq("measured_local") & task.baseline_task_success.notna()]
    baseline_rate = task_measured.baseline_task_success.mean() if not task_measured.empty else np.nan
    threshold_rate = task_measured.threshold_task_success.mean() if not task_measured.empty else np.nan
    report = f"""# Temporal Progressive DSA Software Feasibility Pilot — L40S ×2

## Verdict

**{VERDICT}**

단일-load progressive H8는 기존 two-pass보다 빨라졌지만 동일 Triton framework의 optimized full indexer를 이기지 못했다. 주 설정 B64/r10/K2048에서 full 대비 speedup은 8K **{primary.loc[primary.context.eq(8192), 'fused_speedup_vs_full'].iloc[0]:.3f}×**, 16K **{primary.loc[primary.context.eq(16384), 'fused_speedup_vs_full'].iloc[0]:.3f}×**, 32K **{primary.loc[primary.context.eq(32768), 'fused_speedup_vs_full'].iloc[0]:.3f}×**다. 32K에서 fused는 two-pass보다 **{primary.loc[primary.context.eq(32768), 'fused_speedup_vs_two_pass'].iloc[0]:.3f}×** 빠르지만 full보다 느리다.

## 첫 페이지 핵심 수치

- Optimized full indexer+TopK: 8K/16K/32K = {primary.full_plus_topk.iloc[0]*1000:.2f}/{primary.full_plus_topk.iloc[1]*1000:.2f}/{primary.full_plus_topk.iloc[2]*1000:.2f} µs
- H8 two-pass: {primary.two_pass_plus_topk.iloc[0]*1000:.2f}/{primary.two_pass_plus_topk.iloc[1]*1000:.2f}/{primary.two_pass_plus_topk.iloc[2]*1000:.2f} µs
- Fused progressive dense: {primary.fused_dense_plus_topk.iloc[0]*1000:.2f}/{primary.fused_dense_plus_topk.iloc[1]*1000:.2f}/{primary.fused_dense_plus_topk.iloc[2]*1000:.2f} µs
- Nsight GPU ops/iteration (16K): full 2, two-pass 5, fused dense 2, compact 3
- Nsight GPU ops/iteration (32K, multi-kernel Top-K): full 18, two-pass 21, fused dense 18, compact 19
- Held-out threshold actual promotion rate: {primary_quality.actual_promotion_rate*100:.2f}% (nominal 10%)
- Nearest actual-rate r20 sweep speedup vs full (8K/16K/32K): {nearest_actual.speedup.loc[8192]:.3f}×/{nearest_actual.speedup.loc[16384]:.3f}×/{nearest_actual.speedup.loc[32768]:.3f}×
- Actual QK reduction median: {primary_quality.net_qk_reduction_median*100:.2f}%
- Actual threshold MLA RelL2 p95: {primary_mla.output_relative_l2_p95*100:.3f}%
- Actual threshold Top-128 recall: {primary_quality.top128_recall*100:.4f}%
- Teacher-forced PPL delta: {teacher.ppl_delta.iloc[0]*100:.3f}%; logit KL mean {teacher.logit_kl_mean.iloc[0]:.6f}
- Local RULER-like task success: baseline {baseline_rate*100:.1f}% vs threshold H8 {threshold_rate*100:.1f}% (LongBench 미실행)
- K source load: full와 fused 모두 1회 scan. two-pass는 promoted block을 reread한다.
- DRAM/L2/SM/achieved occupancy: **ncu 미설치로 unavailable**. code-level bytes를 hardware counter처럼 주장하지 않는다.

## Performance

{performance_md}

가장 좋은 단발점은 16K/B32/r5/K512에서 full 대비 약 1.109×였지만, 주 K=2048과 32K에서 재현되지 않았다. 따라서 software GO 근거가 아니다. compact는 16K 한 점에서 dense보다 소폭 빠르지만 8K/32K에서 느리고 full baseline도 이기지 못했다.

## 왜 two-pass가 느리고 fused도 full을 못 이겼는가

Two-pass는 H8 scan, compare, logical-or mask, masked rerank, Top-K의 5 launch와 verifier intermediate write, promoted K reread를 가진다. Fused는 CTA 안에서 K tile을 한 번 load하고 cold H8 뒤 같은 tile로 remaining 56 heads를 이어 계산하므로 source dataflow상 reread와 global H8 intermediate를 제거했다. 그 결과 two-pass보다 빨라졌다. 그러나 full도 정확히 한 번의 순차 K scan이며 커널이 단순하다. fused의 uniform CTA branch와 두 계산 경로가 만드는 제어·instruction/resource 비용이 줄어든 MAC보다 컸다. 이것은 Nsight Systems timeline과 1-kernel CUDA-event latency로 확인되지만, ncu가 없어 register pressure와 achieved occupancy는 인과 추정으로만 남긴다.

Top-K는 32K full path에서 전체 latency의 {100*stage[(stage.context.eq(32768)) & stage.stage.eq('topk_full_vector') & stage.promotion_target.eq(0.10)].median_ms.iloc[0]/primary.loc[primary.context.eq(32768), 'full_plus_topk'].iloc[0]:.1f}%다. H8-only full-prefix도 full보다 빠르지 않았고, atomic compact는 일관된 해결책이 아니었다.

## Quality

Global top-10% 결과를 threshold 결과로 재사용하지 않았다. validation 24 traces에서 layer별 cutoff를 고정하고 held-out 144 traces를 own-trajectory로 replay했다. dynamic/fixed 세트도 validation에서만 선택했다. promotion 5/10/15/20%의 MLA curve와 r10 teacher-forced/closed-loop를 별도로 측정했다.

고정 head가 dynamic과 비슷한 경우 routing launch를 제거할 수 있지만, 어떤 head 방식도 주 fused latency가 full보다 느리다는 software verdict를 바꾸지는 않는다. task 표는 local NIAH single/multi, variable tracking, aggregation, code next-token ground truth를 포함한다. 로컬 LongBench harness가 없어 LongBench는 미실행이며 production quality pass를 선언하지 않는다.

## Profiling 및 memory accounting

- Full: K read ≈ L×128×2 bytes, dense score write L×4 bytes.
- Fused: K read source load는 동일한 1회이며 K traffic reduction을 주장하지 않는다. rejected token도 full-dimensional K tile을 읽는다.
- Two-pass: cold H8 K scan 뒤 accepted full rerank로 promoted block K를 재접근한다.
- Nsight Systems: kernel launch, CUDA API, NVTX critical path 측정.
- Nsight Compute: 서버에 `ncu`가 없어 DRAM/L2 byte, SM utilization, tensor utilization, occupancy, branch efficiency를 측정하지 못했다.

## E2E sidecar 및 다음 단계

8개 selected layer에 isolated kernel delta를 합산하면 fused는 TPOT를 개선하지 않고 오히려 context별 수십~백여 µs를 더한다. Python reference controller의 decode 시간은 production TRT-LLM TPOT가 아니며, 이 결과를 V3.2 FP8 production claim으로 일반화하지 않는다.

다음 단계는 production V3.2 port가 아니다. 이 dataflow는 hardware/ISA co-design 근거로 보존하되, software 방향은 optimized full을 유지한다. D32→H8는 추가 launch와 sketch traffic을 도입하며 primary fused가 이미 full보다 느린 원인을 해결하지 못하므로 구현하지 않았다. 새로운 접근은 persistent multi-CTA Top-K fusion 또는 full kernel 내부에서 register footprint를 증가시키지 않는 predication이 먼저다.

## 질문별 답

1. QK 감소가 latency 감소로 변환됐는가? **아니다.**
2. K reread를 없앴는가? **source dataflow상 그렇다.** 실제 DRAM byte counter는 ncu 부재로 미측정이다.
3. fused가 two-pass보다 빠른가? **그렇다**, 32K 주 설정에서 {primary.loc[primary.context.eq(32768), 'fused_speedup_vs_two_pass'].iloc[0]:.3f}×.
4. Top-K가 병목인가? **그렇다**, 특히 32K full path의 대부분을 차지한다.
5. dense와 compact 중 무엇이 유리한가? **dense가 주 결과**다. compact는 일관되지 않다.
6. dynamic H8가 필요한가? fixed quality 표를 보면 판단 가능하지만 routing 절감도 full 대비 열세를 뒤집지 못한다.
7. threshold promotion 품질은 유지되는가? PPL/task/MLA 표에 실제 측정값을 제시했으며 global budget과 분리했다.
8. 16K/32K에서 speedup이 커지는가? **아니다.** 32K도 full 대비 1× 미만이다.
9. 실제 task quality는 유지되는가? local 측정 범위는 표에 제시하지만 LongBench 미실행 때문에 포괄적 유지 주장은 하지 않는다.
10. 다음 단계는? **software 방향 종료, 알고리즘은 hardware 연구 근거로 보존**한다.
"""
    (args.output / "progressive_dsa_software_report.md").write_text(report, encoding="utf-8")
    verdict = {
        "verdict": VERDICT,
        "go": False,
        "production_v3_2_claim": False,
        "primary_speedup_vs_full": {str(row.context): row.fused_speedup_vs_full for row in primary.itertuples()},
        "primary_speedup_vs_two_pass": {str(row.context): row.fused_speedup_vs_two_pass for row in primary.itertuples()},
        "nearest_actual_rate_r20_speedup_vs_full": {
            str(context): float(row.speedup)
            for context, row in nearest_actual.iterrows()
        },
        "quality": {
            "threshold_actual_promotion_rate": float(primary_quality.actual_promotion_rate),
            "net_qk_reduction_median": float(primary_quality.net_qk_reduction_median),
            "mla_rell2_p95": float(primary_mla.output_relative_l2_p95),
            "teacher_ppl_delta": float(teacher.ppl_delta.iloc[0]),
            "logit_kl_mean": float(teacher.logit_kl_mean.iloc[0]),
        },
        "profiling_limit": "ncu unavailable; no measured DRAM/L2/SM/occupancy counters",
        "reason": "Fused is faster than two-pass but slower than optimized full at every primary context.",
    }
    (args.output / "software_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    hashes = {
        path.name: sha256(path)
        for path in args.output.iterdir()
        if path.is_file() and path.name != "reproducibility.json"
    }
    reproducibility["output_sha256"] = hashes
    (args.output / "reproducibility.json").write_text(json.dumps(reproducibility, indent=2) + "\n")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
