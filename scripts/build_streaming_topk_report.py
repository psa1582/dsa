from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FINAL_CSVS = {
    "gpu_bottleneck_decomposition.csv": ("hardware", "gpu_bottleneck_decomposition.csv"),
    "gpu_decomposition_repeat_audit.csv": ("hardware", "gpu_decomposition_repeat_audit.csv"),
    "gpu_topk_timeline.csv": ("analysis", "gpu_topk_timeline.csv"),
    "gpu_topk_operation_summary.csv": ("analysis", "gpu_topk_operation_summary.csv"),
    "topk_churn_summary.csv": ("churn", "topk_churn_summary.csv"),
    "new_entry_previous_rank_histogram.csv": ("churn", "new_entry_previous_rank_histogram.csv"),
    "exact_heap_admission_summary.csv": ("heap", "exact_heap_admission_summary.csv"),
    "exact_heap_fifo_summary.csv": ("heap", "exact_heap_fifo_summary.csv"),
    "architecture_a_sweep.csv": ("hardware", "architecture_a_sweep.csv"),
    "throughput_sensitivity.csv": ("hardware", "throughput_sensitivity.csv"),
    "hardware_performance_comparison.csv": ("hardware", "hardware_performance_comparison.csv"),
    "data_movement_analysis.csv": ("hardware", "data_movement_analysis.csv"),
    "storage_cost.csv": ("hardware", "storage_cost.csv"),
    "pcie_offload_analysis.csv": ("hardware", "pcie_offload_analysis.csv"),
}

COLORS = {
    "blue": "#2563EB",
    "orange": "#EA580C",
    "green": "#16A34A",
    "purple": "#7C3AED",
    "red": "#DC2626",
    "gray": "#6B7280",
    "dark": "#111827",
}


def save_figure(fig: plt.Figure, output: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(output / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def context_label(context: int) -> str:
    return f"{context // 1024}K"


def make_h8_summary(progressive_root: Path, h8_root: Path) -> pd.DataFrame:
    promotion = pd.read_csv(progressive_root / "final" / "promotion_quality.csv")
    p = promotion[
        promotion.head_scheme.eq("dynamic_abs_w")
        & promotion.promotion_policy.eq("validation_fixed_threshold")
        & promotion.promotion_target.eq(0.1)
    ].iloc[0]
    mla = pd.read_csv(progressive_root / "mla" / "mla_output_summary.csv")
    m = mla[
        mla.policy_role.eq("Aggressive")
        & mla.verifier.eq("head_dynamic_abs_w_w8_b64_threshold_r0.1")
    ].iloc[0]
    teacher = pd.read_csv(progressive_root / "final" / "teacher_forced_quality.csv").iloc[0]
    global_h8 = pd.read_csv(h8_root / "final" / "main_results.csv")
    g = global_h8[global_h8.method.eq("Existing H8 + 10% H56 rescue")].iloc[0]
    rows = [
        {
            "method": "Temporal progressive H8, dynamic_abs_w, r0.1",
            "classification": "APPROXIMATE",
            "candidate_fraction": p.candidate_fraction_mean,
            "qk_reduction_median": p.net_qk_reduction_median,
            "top128_recall": p.top128_recall,
            "top512_recall": p.top512_recall,
            "top2048_recall": p.top2048_recall,
            "newly_active_token_recall": p.newly_active_token_recall,
            "mla_relative_l2_p95": m.output_relative_l2_p95,
            "mla_relative_l2_p99": m.output_relative_l2_p99,
            "mla_cosine_p5": m.output_cosine_p5,
            "logit_kl_mean": teacher.logit_kl_mean,
            "ppl_delta": teacher.ppl_delta,
            "scope_note": "locked sidecar replay; all contexts/layers",
        },
        {
            "method": "Existing H8 + 10% H56 rescue",
            "classification": "APPROXIMATE",
            "candidate_fraction": np.nan,
            "qk_reduction_median": g.qk_reduction,
            "top128_recall": g.top128_recall,
            "top512_recall": g.top512_recall,
            "top2048_recall": g.top2048_recall,
            "newly_active_token_recall": np.nan,
            "mla_relative_l2_p95": g.mla_relative_l2_p95,
            "mla_relative_l2_p99": g.mla_relative_l2_p99,
            "mla_cosine_p5": g.mla_cosine_p5,
            "logit_kl_mean": g.logit_kl_mean,
            "ppl_delta": g.ppl_delta,
            "scope_note": "existing held-out first-64-step GPU replay",
        },
    ]
    return pd.DataFrame(rows)


def copy_tables(analysis: Path, output: Path, h8: pd.DataFrame) -> dict[str, pd.DataFrame]:
    roots = {
        "analysis": analysis,
        "hardware": analysis / "hardware",
        "churn": analysis / "churn",
        "heap": analysis / "heap",
    }
    tables: dict[str, pd.DataFrame] = {}
    for final_name, (root_key, source_name) in FINAL_CSVS.items():
        source = roots[root_key] / source_name
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, output / final_name)
        tables[final_name] = pd.read_csv(source)
    h8.to_csv(output / "h8_secondary_extension.csv", index=False)
    tables["h8_secondary_extension.csv"] = h8
    return tables


def make_figures(tables: dict[str, pd.DataFrame], output: Path) -> None:
    gpu = tables["gpu_bottleneck_decomposition.csv"].copy()
    gpu["label"] = gpu.context.astype(int).map(context_label)
    x = np.arange(len(gpu))

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.bar(x, 1000 * gpu.score_ms, label="Full64 score only", color=COLORS["blue"])
    ax.bar(
        x,
        1000 * gpu.topk_ms,
        bottom=1000 * gpu.score_ms,
        label="Stock Top-K only",
        color=COLORS["orange"],
    )
    ax.plot(x, 1000 * gpu.combined_ms, "o--", color=COLORS["dark"], label="Measured/projected combined")
    ax.set_xticks(x, gpu.label)
    ax.set(xlabel="Context length", ylabel="Latency (us)", title="Figure 1 — Full64 score and stock Top-K latency")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(fontsize=8)
    ax.text(0.99, 0.02, "Isolated stages are non-additive.\n64K is a 2x analytical projection.", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, bbox={"facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.9})
    save_figure(fig, output, "figure_1_latency_breakdown.png")

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(x, 100 * gpu.topk_fraction_of_isolated_sum, "o-", color=COLORS["orange"], label="Top-K / isolated sum")
    ax.plot(x, 100 * gpu.score_fraction_of_isolated_sum, "o-", color=COLORS["blue"], label="Score / isolated sum")
    ax.plot(x, 100 * gpu.topk_fraction_of_combined, "s--", color=COLORS["purple"], label="Top-K / combined")
    ax.set_xticks(x, gpu.label)
    ax.set(xlabel="Context length", ylabel="Fraction (%)", ylim=(0, 105), title="Figure 2 — Context-dependent Top-K fraction")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8)
    save_figure(fig, output, "figure_2_topk_fraction.png")

    churn_all = tables["topk_churn_summary.csv"]
    churn = churn_all[churn_all.group_type.eq("context")].sort_values("context")
    cx = np.arange(len(churn))
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.plot(cx, churn.overlap_mean, "o-", color=COLORS["blue"], label="Mean")
    ax.plot(cx, churn.overlap_median, "s--", color=COLORS["green"], label="Median")
    ax.fill_between(cx, churn.overlap_p5, churn.overlap_mean, color=COLORS["blue"], alpha=0.14, label="p5 to mean")
    ax.scatter(cx, churn.overlap_p1, marker="x", color=COLORS["red"], label="p1")
    ax.set_xticks(cx, churn.context.astype(int).map(context_label))
    ax.set(xlabel="Context length", ylabel="Previous/current Top-2048 overlap", ylim=(0.3, 1.02), title="Figure 3 — Exact Top-2048 temporal overlap")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8)
    save_figure(fig, output, "figure_3_topk_overlap.png")

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.bar(cx - 0.22, churn.new_entries_mean, width=0.22, color=COLORS["blue"], label="Mean")
    ax.bar(cx, churn.new_entries_p95, width=0.22, color=COLORS["orange"], label="p95")
    ax.bar(cx + 0.22, churn.new_entries_p99, width=0.22, color=COLORS["red"], label="p99")
    ax.set_xticks(cx, churn.context.astype(int).map(context_label))
    ax.set(xlabel="Context length", ylabel="New Top-K entries / step", title="Figure 4 — Top-2048 insertions per decode step")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(fontsize=8)
    save_figure(fig, output, "figure_4_new_entries.png")

    timeline = tables["gpu_topk_timeline.csv"].copy()
    category_colors = {
        "score": COLORS["blue"],
        "topk_gather": COLORS["green"],
        "topk_radix": COLORS["orange"],
        "topk_count": COLORS["purple"],
        "topk_scan": "#0891B2",
        "topk_aux": COLORS["gray"],
        "memset": COLORS["red"],
    }
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 6.4), sharex=False)
    for ax, context in zip(axes, [16384, 32768]):
        rows = timeline[timeline.context.eq(context)].sort_values("operation_order")
        start = 0.0
        for row in rows.itertuples():
            color = category_colors.get(row.category, COLORS["gray"])
            duration = row.duration_us_first_iteration
            ax.barh([0], [duration], left=[start], height=0.48, color=color, edgecolor="white")
            if duration >= 1.1:
                ax.text(start + duration / 2, 0, row.short_name, ha="center", va="center", fontsize=6, rotation=90 if duration < 4 else 0)
            start += duration
        ax.set_yticks([])
        ax.set_title(f"{context_label(context)} — {len(rows)} GPU operations; active duration {start:.1f} us", loc="left", fontsize=10)
        ax.set_xlabel("Cumulative active GPU duration (us; launch gaps omitted)")
        ax.grid(axis="x", alpha=0.18)
    handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in category_colors.values()]
    fig.legend(handles, list(category_colors), loc="lower center", ncol=7, fontsize=7)
    fig.suptitle("Figure 5 — Nsight operation timeline: 16K vs 32K", fontsize=13)
    fig.subplots_adjust(bottom=0.14, hspace=0.62)
    fig.savefig(output / "figure_5_gpu_topk_timeline.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    perf = tables["hardware_performance_comparison.csv"]
    systems = ["FPGA conservative HW-A", "FPGA conservative HW-B", "FPGA optimistic HW-B", "ASIC HW-B"]
    fig, ax = plt.subplots(figsize=(9.2, 5.5))
    width = 0.16
    ax.bar(x - 2 * width, 1000 * gpu.combined_ms, width, label="GPU baseline", color=COLORS["dark"])
    for idx, system in enumerate(systems):
        rows = perf[perf.system.eq(system)].sort_values("context")
        color = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["purple"]][idx]
        ax.bar(x + (idx - 1) * width, rows.latency_us, width, label=system, color=color)
    ax.set_xticks(x, gpu.label)
    ax.set_yscale("log")
    ax.set(xlabel="Context length", ylabel="Latency (us, log scale)", title="Figure 6 — Optimized GPU vs exact streaming hardware model")
    ax.grid(axis="y", which="both", alpha=0.20)
    ax.legend(fontsize=7, ncol=2)
    ax.text(0.99, 0.98, "FPGA optimistic P=2 exceeds a conservative\n1 BF16 MAC/DSP envelope; ASIC is pre-floorplan.", transform=ax.transAxes, ha="right", va="top", fontsize=8, bbox={"facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.9})
    save_figure(fig, output, "figure_6_gpu_vs_exact_hw.png")

    sweep = tables["throughput_sensitivity.csv"]
    resource = sweep[
        sweep.context.eq(32768)
        & sweep["mode"].eq("warm")
        & sweep.requested_admission_lanes.eq(2)
        & sweep.candidate_sram_banks.eq(4)
        & sweep.frequency_mhz.eq(400)
        & sweep.memory_bandwidth_gbps.eq(512)
    ].sort_values("scores_per_cycle").drop_duplicates("scores_per_cycle")
    fig, ax1 = plt.subplots(figsize=(8.2, 5.2))
    ax1.plot(resource.scores_per_cycle, resource.full64_bf16_mac_lanes, "o-", color=COLORS["blue"], label="BF16 MAC lanes")
    ax1.plot(resource.scores_per_cycle, resource.total_comparator_units * 256, "s--", color=COLORS["purple"], label="Comparator units x256")
    ax1.axhline(14352, color=COLORS["red"], linestyle=":", label="14,352-DSP reference")
    ax1.set(xlabel="Score production rate P (scores/cycle)", ylabel="Compute resource proxy", title="Figure 7 — Throughput vs compute/SRAM resource pressure")
    ax1.set_xticks(resource.scores_per_cycle)
    ax1.grid(alpha=0.22)
    ax2 = ax1.twinx()
    ax2.plot(resource.scores_per_cycle, resource.onchip_sram_bytes / 1024, "D-", color=COLORS["green"], label="Selector SRAM")
    ax2.set_ylabel("Selector SRAM (KiB)", color=COLORS["green"])
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], fontsize=8, loc="upper left")
    save_figure(fig, output, "figure_7_throughput_resources.png")

    movement = tables["data_movement_analysis.csv"]
    gpu_move = movement[movement.system.str.startswith("GPU")].sort_values("context")
    dense_write = gpu_move.dense_score_write_bytes / 1024
    read_low = gpu_move.selection_score_read_bytes_lower / 1024
    read_extra = (gpu_move.selection_score_read_bytes_upper - gpu_move.selection_score_read_bytes_lower) / 1024
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.bar(x - 0.18, dense_write, width=0.36, label="GPU dense score write", color=COLORS["blue"])
    ax.bar(x - 0.18, read_low, width=0.36, bottom=dense_write, label="GPU mandatory selection read", color=COLORS["orange"])
    ax.bar(x - 0.18, read_extra, width=0.36, bottom=dense_write + read_low, label="32K+ signature upper-pass increment", color=COLORS["red"], alpha=0.7)
    ax.bar(x + 0.18, np.zeros(len(x)), width=0.36, label="Streaming HW dense-score traffic = 0", color=COLORS["green"])
    ax.set_xticks(x, gpu.label)
    ax.set(xlabel="Context length", ylabel="Dense-score path traffic estimate (KiB)", title="Figure 8 — Dense score materialization eliminated by fusion")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(fontsize=8)
    ax.text(0.01, 0.98, "CODE-LEVEL / ANALYTICAL ESTIMATE; not DRAM-counter data. Indexer-K reads are excluded here.", transform=ax.transAxes, va="top", fontsize=8)
    save_figure(fig, output, "figure_8_offchip_score_traffic.png")


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


def report_text(tables: dict[str, pd.DataFrame]) -> str:
    gpu = tables["gpu_bottleneck_decomposition.csv"].sort_values("context")
    timeline = tables["gpu_topk_timeline.csv"]
    churn_all = tables["topk_churn_summary.csv"]
    churn = churn_all[churn_all.group_type.eq("context")].sort_values("context")
    heap = tables["exact_heap_admission_summary.csv"].sort_values("context")
    arch = tables["architecture_a_sweep.csv"]
    perf = tables["hardware_performance_comparison.csv"]
    movement = tables["data_movement_analysis.csv"]
    pcie = tables["pcie_offload_analysis.csv"]
    h8 = tables["h8_secondary_extension.csv"]

    g32 = gpu[gpu.context.eq(32768)].iloc[0]
    c32 = churn[churn.context.eq(32768)].iloc[0]
    fpga_a32 = perf[(perf.system.eq("FPGA conservative HW-A")) & perf.context.eq(32768)].iloc[0]
    fpga_b32 = perf[(perf.system.eq("FPGA conservative HW-B")) & perf.context.eq(32768)].iloc[0]
    fpga_b64 = perf[(perf.system.eq("FPGA conservative HW-B")) & perf.context.eq(65536)].iloc[0]
    fpga_opt32 = perf[(perf.system.eq("FPGA optimistic HW-B")) & perf.context.eq(32768)].iloc[0]
    asic32 = perf[(perf.system.eq("ASIC HW-B")) & perf.context.eq(32768)].iloc[0]
    asic64 = perf[(perf.system.eq("ASIC HW-B")) & perf.context.eq(65536)].iloc[0]
    h8_progressive = h8.iloc[0]
    h8_rescue = h8.iloc[1]
    move32_gpu = movement[(movement.context.eq(32768)) & movement.system.str.startswith("GPU")].iloc[0]
    move32_hw = movement[(movement.context.eq(32768)) & movement.system.str.startswith("Fused")].iloc[0]
    traffic_lower = 1 - move32_hw.total_offchip_bytes_lower / move32_gpu.total_offchip_bytes_lower
    traffic_upper = 1 - move32_hw.total_offchip_bytes_upper / move32_gpu.total_offchip_bytes_upper

    lines: list[str] = []
    add = lines.append
    add("# DSA Streaming Top-K Hardware Feasibility")
    add("")
    add("## 1. Executive Verdict")
    add("")
    add("**Verdict: PROMISING. Final recommendation: BUILD MORE DETAILED CYCLE MODEL FIRST.**")
    add("")
    add("This verdict is for a DSA-specific, fused Full64-score + exact Top-2048 stream, not for a stand-alone sorting accelerator and not for the approximate H8 path. The evidence supports continued architecture work, but it does not yet support RTL sign-off.")
    add("")
    add("1. **Does the bottleneck change with context?** The GPU dispatch changes sharply at 32K, and the normalized isolated-stage Top-K share rises from 72.3% at 8K to 85.0% at 32K. However, there is no measured score-to-Top-K crossover: stock Top-K is already the larger isolated stage at 8K.")
    add("2. **Where does Top-K become dominant?** At or below the smallest measured point, 8K. At 32K, Top-K-only time is 73.1% of the measured combined latency; isolated stages are non-additive.")
    add("3. **Why are there 18 GPU operations at 32K?** Nsight resolves them as 1 Full64 scorer kernel, 15 Top-K CUDA kernels, and 2 CUDA memsets. The Top-K path uses workspace initialization, four radix threshold-refinement passes, four within-K count passes, a Kth-count pass, two scan initializations, two scans, and a final gather. They are not 18 sorting kernels.")
    add("4. **Can exact streaming Top-K remove the bottleneck?** Architecturally yes: it eliminates dense score materialization and overlaps selection with score production. Quantitatively, the conservative FPGA point reaches only 1.062x at 32K and 0.914x at projected 64K, while a P=2 FPGA point reaches 2.124x but exceeds a conservative one-BF16-MAC-per-DSP envelope. The pre-floorplan ASIC point is 5.309x/4.567x at 32K/64K. This is promising, not proven.")
    add("5. **Does previous Top-K warm-start help?** Yes for activity: exact heap admissions fall by 70.5% on average at 32K, with exact-match rate 1.0 over all 18,288 transitions. It does not reduce the mandatory N-score scan and therefore does not improve the modeled P=1 scorer-bound latency.")
    add("6. **FPGA Top-K-only offload?** NO-GO. PCIe payload and round-trip latency are not consistently faster than the GPU Top-K, retain dense score materialization, and add two synchronization boundaries.")
    add("7. **Fused Indexer+TopK hardware?** Worth a detailed model for ASIC and possibly a more favorable FPGA DSP packing study. Conservative FPGA mapping is not yet an RTL GO.")
    add("8. **Start RTL now?** No. First close the heap hazard, banking, DSP packing, timing, HBM scheduling, and power/floorplan uncertainties in a more detailed cycle model.")
    add("")
    add("Primary evidence uses 144 real Full64 score traces (6 sequences x 3 contexts x 8 layers, 128 steps each) and the optimized Triton scorer plus stock `torch.topk(K=2048)`. The 64K values are explicitly analytical because no real 64K sidecar capture exists. No 128K timing was generated: there is no real 128K trace and expensive new model inference was outside this pilot. These DeepSeek-V2-Lite sidecar results must not be presented as production DeepSeek-V3.2 measurements.")
    add("")
    add("## 2. GPU Bottleneck Characterization")
    add("")
    add("### Repository and evidence audit")
    add("")
    add("| Item | Exact location / finding |")
    add("|---|---|")
    add("| Full64 Lightning Indexer | `src/temporal_dsa/progressive_kernel.py`: `_full_score_kernel`, `full_scores_triton` |")
    add("| Current Top-K | `scripts/benchmark_progressive_dsa.py`: `torch.topk(..., sorted=False)` in `launch_full` |")
    add("| Timing harnesses | `scripts/benchmark_progressive_dsa.py`, `scripts/benchmark_progressive_stages.py`, and new `scripts/benchmark_full64_topk_decomposition.py` |")
    add("| Nsight harness/parser | `scripts/profile_progressive_dsa.py`, `scripts/summarize_progressive_nsight.py`, new `scripts/extract_full64_topk_nsight.py` |")
    add("| Full64 traces | `artifacts/pilot/scores_a/traces`, `artifacts/pilot/scores_b/traces`; inventory in `artifacts/h8_reconstruction/final/trace_inventory.csv` |")
    add("| Prior H8/temporal artifacts | `artifacts/progressive_sw/final`, `artifacts/h8_reconstruction/final`, `artifacts/fused_h8_sm89/run_20260826_sm89` |")
    add("| Context dispatch | No repository-level custom branch was found around `torch.topk`; the 16K/32K divergence occurs inside the ATen/CUDA implementation selected for shape K=2048 |")
    add("")
    add("### Locked bottleneck decomposition")
    add("")
    add("| Context | Full64 score only (us) | Top-K only (us) | Combined (us) | Score fraction* | Top-K fraction* | GPU ops | Status |")
    add("|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in gpu.itertuples():
        ops_text = "n/a" if pd.isna(row.gpu_ops) else str(int(row.gpu_ops))
        add(f"| {context_label(int(row.context))} | {1000*row.score_ms:.3f} | {1000*row.topk_ms:.3f} | {1000*row.combined_ms:.3f} | {pct(row.score_fraction_of_isolated_sum)} | {pct(row.topk_fraction_of_isolated_sum)} | {ops_text} | {row.measurement_status} |")
    add("")
    add("\\* Fractions are normalized over `T_score + T_topk` so the two columns sum to 100%. Independent CUDA-event stage measurements are not additive: cache state, allocator/workspace behavior, launch scheduling, and measurement boundaries differ. Relative to measured combined latency, Top-K is 85.4%, 91.9%, and 73.1% at 8K, 16K, and 32K. The 32K 73.1% observation is preserved, but it must not be combined with the normalized table fractions.")
    add("")
    add("The same scorer and stock Top-K were rerun twice on the currently free L40S device with three real captures per context, 100 warmups, 1,000 measurements, and a 64 MiB cache flush. Score-only ranges were 21.50–22.82 us at 8K, 22.62–23.68 us at 16K, and 22.62–24.61 us at 32K; combined ranges were 56.32, 63.49, and 101.38–103.09 us. This repeat audit is retained in `gpu_decomposition_repeat_audit.csv`, while the locked baseline remains the performance reference for continuity. The protocol sensitivity is a weakness, not silently averaged away.")
    add("")
    add("![Figure 1](figure_1_latency_breakdown.png)")
    add("")
    add("![Figure 2](figure_2_topk_fraction.png)")
    add("")
    add("### Exact 16K and 32K operation path")
    add("")
    add("At 16K the first profiled iteration consists of `_full_score_kernel` (4.512 us) followed by `gatherTopK` (55.840 us). At 32K, the first iteration is:")
    add("")
    add("| # | Type | Stage | Operation | Duration (us) |")
    add("|---:|---|---|---|---:|")
    for row in timeline[timeline.context.eq(32768)].sort_values("operation_order").itertuples():
        add(f"| {int(row.operation_order)} | {row.operation_type} | {row.stage} | `{row.short_name}` | {row.duration_us_first_iteration:.3f} |")
    add("")
    add("The aggregated 32K Top-K kernel+memset active duration is about 30.2 us, whereas the Nsight NVTX wall interval is 140.66 us and the locked CUDA-event combined result is 87.04 us. These are different protocols: the active sum excludes launch gaps and CPU scheduling; NVTX includes them; CUDA events use the locked benchmark boundary. No memcpy was recorded in the profiled range.")
    add("")
    add("![Figure 5](figure_5_gpu_topk_timeline.png)")
    add("")
    add("PyTorch's CUDA radix-selection source describes iterative bit refinement and explicitly notes as many as 16 float32 passes in the generic routine; the observed shape-specific path uses four refinement rounds plus count/scan/gather machinery. This explains the GPU synchronization structure without calling it a full sort: https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/SortingRadixSelect.cuh")
    add("")
    add("### Data movement")
    add("")
    add("`ncu` was unavailable on the L40S server, so every byte count below is **CODE-LEVEL / ANALYTICAL ESTIMATE**, not measured DRAM traffic. The lower bound includes one dense score write and one selection read. For the 32K signature upper estimate, four radix, four within-K-count, and one gather score pass are counted; partial accesses and workspace traffic make this a signature-based bound, not a counter result.")
    add("")
    add("| Context | GPU dense write | GPU select read lower / upper | Score passes lower / signature upper | GPU final result | HW dense-score traffic | HW final IDs |")
    add("|---:|---:|---:|---:|---:|---:|---:|")
    for context in gpu.context.astype(int):
        g = movement[(movement.context.eq(context)) & movement.system.str.startswith("GPU")].iloc[0]
        h = movement[(movement.context.eq(context)) & movement.system.str.startswith("Fused")].iloc[0]
        add(f"| {context_label(context)} | {g.dense_score_write_bytes/1024:.0f} KiB | {g.selection_score_read_bytes_lower/1024:.0f} / {g.selection_score_read_bytes_upper/1024:.0f} KiB | {int(g.full_or_partial_score_passes_lower)} / {int(g.full_or_partial_score_passes_signature_upper)} | {g.final_result_bytes/1024:.0f} KiB | {h.dense_score_write_bytes:.0f} B | {h.final_result_bytes/1024:.0f} KiB |")
    add("")
    add(f"At 32K, eliminating the score materialization/select traffic cuts the modeled total off-chip bytes by {pct(traffic_lower)} under the mandatory lower bound and {pct(traffic_upper)} under the multi-pass signature upper estimate. The percentage is modest because the unavoidable BF16 indexer-K stream is 8 MiB per layer and dominates total bytes. Dense-score traffic itself is eliminated 100%.")
    add("")
    add("Input token IDs are implicit array indices until the final gather. The GPU output is 2,048 FP32 values plus 2,048 INT64 indices (24 KiB); the hardware interface returns only 2,048 INT32 IDs (8 KiB) because scores remain internal. Nsight exposes the 32K 4-byte and 128-byte workspace memsets and the fill/count/scan kernels, but not a reliable total intermediate allocation or DRAM byte count. Intermediate workspace traffic is therefore listed as unknown rather than fabricated.")
    add("")
    add("![Figure 8](figure_8_offchip_score_traffic.png)")
    add("")
    add("## 3. Top-K Temporal Churn")
    add("")
    add("All 18,288 adjacent-step transitions were replayed from the real Full64 scalar traces using stable exact Top-2048 sets.")
    add("")
    add("| Context | Transitions | Overlap mean | Median | p5 | p1 | New mean | p95 | p99 | Max |")
    add("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in churn.itertuples():
        add(f"| {context_label(int(row.context))} | {int(row.transitions):,} | {pct(row.overlap_mean)} | {pct(row.overlap_median)} | {pct(row.overlap_p5)} | {pct(row.overlap_p1)} | {row.new_entries_mean:.1f} | {row.new_entries_p95:.1f} | {row.new_entries_p99:.1f} | {int(row.new_entries_max)} |")
    add("")
    add("Overlap declines materially with context. At 32K, layer 17 is worst (62.0% mean overlap) and layer 2 is best (69.5%). Early/middle/late differences are small: 65.4%, 65.5%, and 66.1% mean overlap, so decode phase alone does not yield an easy exact shortcut.")
    add("")
    add("![Figure 3](figure_3_topk_overlap.png)")
    add("")
    add("![Figure 4](figure_4_new_entries.png)")
    add("")
    add("The previous-rank tail is decisive. Among 32K entries newly entering current Top-2048: 12.31% came from ranks 2049–2304, 50.35% from 2305–4096, 29.56% from 4097–8192, 6.89% from 8193–16384, 0.75% from 16385+, and 0.14% were newly appended tokens without a prior rank. Thus 37.20% came from beyond rank 4096 and 7.64% from beyond rank 8192 or were new. A small neighborhood around previous Top-K cannot be exact.")
    add("")
    add("Using the previous threshold as a hard discard would have missed 3,216,180 current-TopK occurrences across the replay. Previous state is therefore an ordering and initialization hint only.")
    add("")
    add("## 4. Exact Streaming Top-K Architecture")
    add("")
    add("The fused datapath is: BF16 indexer-K stream -> Full64 MAC/reduction -> scalar score -> threshold comparator -> candidate FIFO -> banked exact min-heap/candidate manager -> final Top-2048 ID SRAM. The N scalar scores are never written to off-chip memory. Score production and selection overlap, giving `T_fused ~= max(T_score_hw, T_topk_hw) + T_drain` rather than the GPU's serialized `T_score + T_topk`.")
    add("")
    add("### Rejected exact baseline: chunk-local Top-r + merge")
    add("")
    add("Architecture-A sweeps B={32,64,128,256} and r={8,16,32,64}. Because K=2048 exceeds every chunk size, an arbitrary chunk may contribute all B items to the global Top-2048. Worst-case exactness therefore requires r>=B. Every r<B point is explicitly **APPROXIMATE**; exact points retain every item and offer no candidate reduction. A naive 2048-way insertion network is also rejected because its 2048 comparisons per arriving score are not area/timing credible.")
    add("")
    exact_count = int(arch.worst_case_exact.sum())
    add(f"The CSV contains {len(arch):,} sweep rows, of which {exact_count:,} are exact only by retaining the entire chunk. The primary exact model instead uses a threshold-guided, pipelined K-entry min-heap with 11 logical heap levels, candidate FIFO, and banked SRAM.")
    add("")
    add("### Cycle/event model")
    add("")
    add("The simulator sweeps P={1,2,4,8,16} scores/cycle, admission lanes={1,2,4}, SRAM banks={2,4,8}, 250/400/500/1000 MHz, and 256/512/1024/2048 GB/s. Compute cycles are `ceil(N/P)`; K-stream memory cycles are `ceil(256N/BW * f)`; admission cycles use trace-measured mean admissions divided by effective lanes; drain uses the p99 FIFO depth plus 11 heap levels. The selector must accept every score; warm start never skips scoring.")
    add("")
    add("The crucial model assumption is one admitted candidate per cycle per lane in a pipelined heap. Same-address SRAM hazards, threshold feedback latency, multi-bank arbitration, tie handling, and worst-case burst admission are not RTL-validated. This is the main reason to build a deeper cycle model before RTL.")
    add("")
    add("## 5. Previous-TopK Warm-Start Architecture")
    add("")
    add("HW-B seeds the exact K-entry state with current scores of the previous exact Top-2048 IDs, then streams every remaining current score. The previous Kth threshold orders work but is never used as a correctness certificate. The final stable Top-2048 set matched the reference on every replayed transition.")
    add("")
    add("| Context | Cold admissions mean | Warm admissions mean | Mean reduction | Warm p95 | Warm p99 | Warm max | Exact match |")
    add("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in heap.itertuples():
        add(f"| {context_label(int(row.context))} | {row.cold_admissions_mean:.1f} | {row.warm_admissions_mean:.1f} | {pct(row.admission_reduction_mean)} | {row.warm_admissions_p95:.0f} | {row.warm_admissions_p99:.0f} | {int(row.warm_admissions_max)} | {row.exact_match_rate:.3f} |")
    add("")
    add("At 32K, warm-start admissions fall from 4,834 to 1,402 on average. At P=16 and four admission lanes the modeled p99 residual FIFO depth is 468 entries (about 3.7 KiB at 8 bytes/entry), versus 2,593 cold. This reduces candidate SRAM writes, merges, and switching activity. It does not change the N/P streaming floor, so cold and warm have identical latency at scorer- or memory-bound selected points.")
    add("")
    add("Conceptually this resembles temporal warm-start in GVR, but the proposed hardware differs in where it acts: GVR predicts/refines a threshold on a GPU and finishes with exact shared-memory selection, whereas HW-B uses previous state to initialize an always-on fused score/selection pipeline. GVR reports 1–2 global threshold passes followed by exact verification/final selection: https://arxiv.org/abs/2604.22312 and https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog21_Temporal_Correlation_Meets_Sparse_Attention.md")
    add("")
    add("## 6. FPGA Feasibility")
    add("")
    add("The reference envelope is a generic modern high-end FPGA, not a claim about one exact SKU. AMD's Versal Premium selection guide lists a high-end envelope up to 14,352 DSPs, 174 Mb BRAM, and 717 Mb URAM; VP1902 itself lists 6,864 DSPs, so the numerical model must not be labeled a VP1902 implementation. Sources: https://docs.amd.com/api/khub/documents/4V3OO2hrA~S52y3qLcexSw/content and https://www.amd.com/content/dam/amd/en/documents/products/adaptive-socs-and-fpgas/versal/2118851-versal-premium-vp1902-product-brief.pdf")
    add("")
    add("| FPGA point | P | Admission lanes | Clock | K bandwidth | 32K latency | 32K speedup | 64K latency | 64K speedup | BF16 MAC lanes | DSP fraction* | Selector SRAM |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in ["FPGA conservative HW-A", "FPGA conservative HW-B", "FPGA optimistic HW-B"]:
        p32 = perf[(perf.system.eq(name)) & perf.context.eq(32768)].iloc[0]
        p64 = perf[(perf.system.eq(name)) & perf.context.eq(65536)].iloc[0]
        add(f"| {name} | {int(p32.scores_per_cycle)} | {int(p32.effective_admission_lanes)} | {int(p32.frequency_mhz)} MHz | {int(p32.memory_bandwidth_gbps)} GB/s | {p32.latency_us:.3f} us | {p32.speedup_vs_gpu:.3f}x | {p64.latency_us:.3f} us | {p64.speedup_vs_gpu:.3f}x | {int(p32.full64_bf16_mac_lanes):,} | {pct(p32.dsp58_fraction_if_1_bf16_mac_per_dsp)} | {p32.onchip_sram_bytes/1024:.1f} KiB |")
    add("")
    add("\\* Conservative accounting assumes one BF16 MAC lane per DSP. P=2 needs 16,384 lanes, or 114.2% of the 14,352-DSP envelope, so the attractive optimistic point is resource-infeasible under that mapping. Packing two BF16 MACs per DSP or using LUT arithmetic could change the conclusion, but neither is claimed without synthesis. P=1 consumes 8,192 MAC lanes (57.1%) and is resource-plausible, yet its modeled speedup is weak and becomes <1x against the projected 64K GPU baseline.")
    add("")
    add("Selector state itself is modest: 16 KiB theoretical FP32+INT32 Top-2048, 32 KiB double-buffered heap, 8 KiB previous IDs, FIFO/pipeline scratch, and about 43 KiB in the selected warm model. Provision 64–96 KiB for ECC, banking, metadata, and burst margin. The dominant challenge is Full64 MAC density and K-stream bandwidth, not Top-K SRAM.")
    add("")
    add("At 64–96 KiB (0.52–0.79 Mb), selector storage is below 0.5% of the cited 174 Mb BRAM envelope and can be split across BRAM/URAM banks. The simplified P=1 selector count is 12 comparator units (one stream threshold plus 11 heap levels); the P=2 point uses 24. This is a logical count, not a LUT estimate—LUT use, fanout, and 400 MHz closure require synthesis.")
    add("")
    add("**FPGA decision: Top-K-only NO-GO; fused Indexer+TopK NO-GO for immediate RTL under conservative one-MAC/DSP mapping. Continue only with a DSP-packing/HBM-aware detailed model.**")
    add("")
    add("![Figure 7](figure_7_throughput_resources.png)")
    add("")
    add("## 7. ASIC Feasibility")
    add("")
    add("The abstract ASIC point uses P=2, two admission lanes, four SRAM banks, 1 GHz, and 1,024 GB/s local/HBM bandwidth. It needs 16,384 BF16 MAC lanes, 24 threshold/heap comparator units in the simplified count, and about 43.4 KiB selector SRAM. The model predicts 16.395 us at 32K (5.309x) and 32.779 us at projected 64K (4.567x). Cold and warm latency are equal because score streaming dominates; warm start remains valuable as an activity/energy optimization.")
    add("")
    add("This is not an area, power, timing, or floorplan result. A credible ASIC next model must include MAC-tree wiring, ReLU/head weighting, K-tile buffering, HBM command efficiency, heap feedback timing, bank conflicts, clock crossings, output ordering, and sparse-MLA handoff. **Decision: GO for floorplan-quality cycle/area/power modeling, not RTL sign-off.**")
    add("")
    add("![Figure 6](figure_6_gpu_vs_exact_hw.png)")
    add("")
    add("## 8. PCIe Offload Analysis")
    add("")
    add("The naive GPU-score -> PCIe -> FPGA-TopK -> PCIe -> GPU path transfers 40, 72, 136, and 264 KiB at 8K/16K/32K/64K. The model uses effective 25 GB/s Gen4 x16 or 50 GB/s Gen5 x16, fixed round trips of 20/40/80 us, and a generous P=16, 400 MHz selector.")
    add("")
    add("| Context | Payload | Gen4 40-us total / speedup | Gen5 40-us total / speedup |")
    add("|---:|---:|---:|---:|")
    for context in gpu.context.astype(int):
        r4 = pcie[(pcie.context.eq(context)) & pcie.pcie.eq("Gen4 x16") & pcie.fixed_roundtrip_latency_us.eq(40)].iloc[0]
        r5 = pcie[(pcie.context.eq(context)) & pcie.pcie.eq("Gen5 x16") & pcie.fixed_roundtrip_latency_us.eq(40)].iloc[0]
        payload = r4.gpu_to_fpga_score_bytes + r4.fpga_to_gpu_index_bytes
        add(f"| {context_label(context)} | {payload/1024:.0f} KiB | {r4.modeled_total_us:.2f} us / {r4.topk_speedup:.2f}x | {r5.modeled_total_us:.2f} us / {r5.topk_speedup:.2f}x |")
    add("")
    add("Some optimistic long-context points exceed 1x, but the result is not consistent across contexts and 64K uses a projected GPU baseline. More importantly, the design preserves the GPU dense-score write, introduces transport in both directions, and requires two cross-device synchronization points per layer. **TOP-K-ONLY FPGA OFFLOAD = NO-GO.**")
    add("")
    add("## 9. Comparison with Current H8 Progressive Software")
    add("")
    add("The locked 32K result remains: optimized Full64 + stock Top-K = 87.04 us; fused progressive H8 + stock Top-K = 104.61 us; speedup = 0.832x. At this context the stock Top-K itself is 63.584 us and dispatches the same 17-operation Top-K subpath. Reducing QK work did not eliminate selection, and the H8 kernel added temporal masking, head reduction/control, and less regular work. The measured result therefore invalidates a simple 'fewer MACs means faster' argument.")
    add("")
    add("The dedicated fused architecture changes the question: it removes the serialized dense-score/stock-TopK boundary and overlaps exact selection with Full64 scoring. The old analytical 1.571x ceiling is not used as measured performance anywhere in this report.")
    add("")
    add("## 10. Secondary Temporal+H8 Incremental Extension")
    add("")
    add("This section is explicitly **APPROXIMATE** and is not part of the exact accelerator GO decision.")
    add("")
    add("| Approximate method | Candidate fraction | QK reduction | Top-128 | Top-512 | Top-2048 | New-active recall | MLA RelL2 p95 / p99 | PPL delta |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    add(f"| Temporal progressive H8 r0.1 | {pct(h8_progressive.candidate_fraction)} | {pct(h8_progressive.qk_reduction_median)} | {pct(h8_progressive.top128_recall, 3)} | {pct(h8_progressive.top512_recall, 3)} | {pct(h8_progressive.top2048_recall, 3)} | {pct(h8_progressive.newly_active_token_recall)} | {h8_progressive.mla_relative_l2_p95:.4f} / {h8_progressive.mla_relative_l2_p99:.4f} | {h8_progressive.ppl_delta:+.4f} |")
    add(f"| Existing H8 + 10% H56 rescue | n/a | {pct(h8_rescue.qk_reduction_median)} | {pct(h8_rescue.top128_recall, 3)} | {pct(h8_rescue.top512_recall, 3)} | {pct(h8_rescue.top2048_recall, 3)} | n/a | {h8_rescue.mla_relative_l2_p95:.4f} / {h8_rescue.mla_relative_l2_p99:.4f} | {h8_rescue.ppl_delta:+.4f} |")
    add("")
    add("The r0.1 method retains 80.9% of tokens on average, so its candidate traffic is not a small-repair regime, and newly-active recall is only 54.9% despite high Top-2048 recall. It is a useful upper-bound path for future approximate designs, not an exact substitute. Quality values are prior sidecar replay results and do not establish production DeepSeek-V3.2 quality.")
    add("")
    add("As an operation proxy, the approximate filter still performs N admission comparisons and forwards about 0.809N candidates into its downstream repair/Top-K stage. Exact downstream comparator operations were not isolated in the prior software trace, so no fabricated comparator-count reduction is claimed.")
    add("")
    add("## 11. Cost / Scaling Analysis")
    add("")
    add("| System | Exact? | 32K latency | 32K speedup | 64K latency | 64K speedup | Dense score off-chip | Main resource risk |")
    add("|---|---|---:|---:|---:|---:|---:|---|")
    add(f"| Optimized L40S GPU | Yes | {1000*g32.combined_ms:.3f} us | 1.000x | {1000*gpu[gpu.context.eq(65536)].iloc[0].combined_ms:.3f} us* | 1.000x | Yes | multi-pass global selection |")
    add(f"| HW-A conservative FPGA cold | Yes | {fpga_a32.latency_us:.3f} us | {fpga_a32.speedup_vs_gpu:.3f}x | {perf[(perf.system.eq('FPGA conservative HW-A')) & perf.context.eq(65536)].iloc[0].latency_us:.3f} us | {perf[(perf.system.eq('FPGA conservative HW-A')) & perf.context.eq(65536)].iloc[0].speedup_vs_gpu:.3f}x | No | 8,192 BF16 MAC lanes |")
    add(f"| HW-B conservative FPGA warm | Yes | {fpga_b32.latency_us:.3f} us | {fpga_b32.speedup_vs_gpu:.3f}x | {fpga_b64.latency_us:.3f} us | {fpga_b64.speedup_vs_gpu:.3f}x | No | same MAC floor; lower activity only |")
    add(f"| HW-B optimistic FPGA warm | Yes | {fpga_opt32.latency_us:.3f} us | {fpga_opt32.speedup_vs_gpu:.3f}x | {perf[(perf.system.eq('FPGA optimistic HW-B')) & perf.context.eq(65536)].iloc[0].latency_us:.3f} us | {perf[(perf.system.eq('FPGA optimistic HW-B')) & perf.context.eq(65536)].iloc[0].speedup_vs_gpu:.3f}x | No | 114.2% DSP proxy |")
    add(f"| HW-B ASIC warm | Yes | {asic32.latency_us:.3f} us | {asic32.speedup_vs_gpu:.3f}x | {asic64.latency_us:.3f} us | {asic64.speedup_vs_gpu:.3f}x | No | pre-floorplan power/wiring |")
    add("| HW-C temporal H8 repair | No | 104.608 us measured software | 0.832x | not measured | n/a | implementation-dependent | quality + irregularity |")
    add("")
    add("\\* 64K GPU result is a 2x projection from 32K isolated stages, not a measurement.")
    add("")
    add("At 32K the fused model removes 256 KiB of mandatory dense score write/read and up to 1.28 MiB under the multi-pass signature estimate, but still reads 8 MiB of indexer-K. Therefore the data-movement energy proxy is only 3.2% lower under the lower bound or 13.6% under the upper signature; no joule claim is made. Warm-start's 70.5% mean heap-admission reduction is a separate switching/SRAM-write energy proxy, not additive to off-chip byte savings without a power model.")
    add("")
    add("One layer of BF16 indexer-K storage is 2/4/8/16/32 MiB at 8K/16K/32K/64K/128K. Eight DSA layers do not all fit in the small selector state; HBM streaming, tiling, and time multiplexing are required. The throughput sweep is preserved in `throughput_sensitivity.csv`, rather than selecting only one favorable point.")
    add("")
    add("### All-context selected performance points")
    add("")
    add("| Context | GPU combined | FPGA HW-B P=1 | FPGA speedup | ASIC HW-B P=2 | ASIC speedup | HW off-chip bytes |")
    add("|---:|---:|---:|---:|---:|---:|---:|")
    for context in gpu.context.astype(int):
        g = gpu[gpu.context.eq(context)].iloc[0]
        f = perf[(perf.system.eq("FPGA conservative HW-B")) & perf.context.eq(context)].iloc[0]
        a = perf[(perf.system.eq("ASIC HW-B")) & perf.context.eq(context)].iloc[0]
        m = movement[(movement.context.eq(context)) & movement.system.str.startswith("Fused")].iloc[0]
        suffix = "*" if context == 65536 else ""
        add(f"| {context_label(context)} | {1000*g.combined_ms:.3f}{suffix} us | {f.latency_us:.3f} us | {f.speedup_vs_gpu:.3f}x | {a.latency_us:.3f} us | {a.speedup_vs_gpu:.3f}x | {m.total_offchip_bytes_lower/1024/1024:.3f} MiB |")
    add("")
    add("### 32K throughput/resource detail")
    add("")
    add("| Architecture | Input / admission throughput | Mean admissions/cycle | FIFO p99 | SRAM | Comparators | Pipeline | Total cycles | Stall |")
    add("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for name in ["FPGA conservative HW-A", "FPGA conservative HW-B", "FPGA optimistic HW-B", "ASIC HW-B"]:
        r = perf[(perf.system.eq(name)) & perf.context.eq(32768)].iloc[0]
        add(f"| {name} | {int(r.scores_per_cycle)} scores/cycle / {int(r.effective_admission_lanes)} admissions/cycle | {r.mean_candidate_admissions_per_cycle:.3f} | {int(r.required_fifo_entries_p99)} | {r.onchip_sram_bytes/1024:.1f} KiB | {int(r.total_comparator_units)} | {int(r.pipeline_depth_cycles)} cycles | {int(r.total_cycles):,} | {pct(r.stall_percentage)} |")
    add("")
    add("Merge throughput is represented by the effective admission-lane rate into the pipelined exact heap. Zero modeled stall at these selected points means the mean/p99 event model keeps up; it is not a worst-case proof.")
    add("")
    add("## 12. Reviewer-Style Weaknesses")
    add("")
    add("- **Prior-art/novelty risk.** GPU Top-K accelerators such as RadiK and temporal methods such as GVR already exploit radix selection, threshold refinement, and temporal guesses. RadiK: https://arxiv.org/abs/2501.14336. A publication claim cannot rest on 'faster Top-K' alone.")
    add("- **What is DSA-specific.** The architectural value is the fused 8,192-MAC-per-score Full64 reduction directly feeding an exact selector, with no N-score off-chip round trip, plus exact previous-state initialization. That system boundary is more than a sorting block, but score+selection fusion alone may still be incremental unless the implementation demonstrates a compelling bandwidth/energy/timing tradeoff.")
    add("- **Exact versus approximate.** HW-A/HW-B inspect all current scores and are exact in the replay/model. Chunk local r<B and HW-C H8 repair are approximate. Prior threshold never certifies a discard.")
    add("- **Model optimism.** One-admission/cycle/lane heap throughput, conflict-free SRAM banking, and comparator timing are unverified. Mean admissions plus p99 FIFO do not constitute a worst-case real-time bound.")
    add("- **Resource model incompleteness.** DSP counts omit routing, reduction trees, ReLU/weights, control, HBM controllers, and sparse-MLA integration. LUT comparator cost is only a unit count, not synthesis area.")
    add("- **GPU measurement sensitivity.** Locked and repeated scorer timings differ materially, particularly at 32K. The report preserves both instead of manufacturing a single precise number.")
    add("- **Traffic uncertainty.** There are no NCU DRAM/L2 counters. The 9-pass 32K quantity is a code/Nsight-signature upper estimate, not measured traffic.")
    add("- **Long-context evidence.** 32K is real; 64K is projected. The claim that hardware scaling wins at longer contexts is not yet empirically convincing for FPGA and actually weakens at the conservative point. A real 64K/128K trace is required.")
    add("- **External validity.** Traces are DeepSeek-V2-Lite sidecar data. They do not prove production DeepSeek-V3.2 latency, accuracy, or power.")
    add("")
    add("## 13. Final Recommendation")
    add("")
    add("**BUILD MORE DETAILED CYCLE MODEL FIRST.**")
    add("")
    add("The next experiment should combine a real 64K capture with an RTL-like event simulator that models bank addresses and hazards cycle by cycle, worst-case admission bursts, stable tie rules, HBM tile scheduling, score-reduction latency, and backpressure. In parallel, synthesize only small primitives—not the full design—to establish BF16 MACs/DSP, heap comparator frequency, and banked SRAM feasibility at 250/400/500 MHz. Re-evaluate GO only if a resource-feasible exact design sustains the scorer rate and reaches at least roughly 1.15–1.3x at real 32K/64K while preserving headroom.")
    add("")
    add("The current platform decisions are: FPGA Top-K-only **NO-GO**; fused FPGA Indexer+TopK **NO-GO FOR RTL under the conservative mapping**; exact ASIC selector **GO for detailed cycle/floorplan modeling, not RTL sign-off**; temporal H8 **secondary approximate research only**.")
    add("")
    add("### Compact summary")
    add("")
    add("```text")
    add("GPU bottleneck crossover: none observed; Top-K is already dominant at 8K")
    for row in gpu.itertuples():
        add(f"{context_label(int(row.context))} score/TopK fraction: {100*row.score_fraction_of_isolated_sum:.1f}% / {100*row.topk_fraction_of_isolated_sum:.1f}%" + (" (projected)" if int(row.context) == 65536 else ""))
    add("")
    add(f"32K GPU Full score latency: {1000*g32.score_ms:.3f} us")
    add(f"32K GPU TopK latency: {1000*g32.topk_ms:.3f} us")
    add(f"32K GPU combined: {1000*g32.combined_ms:.3f} us")
    add("")
    add(f"Previous Top-2048 overlap: {100*c32.overlap_mean:.2f}% mean at 32K")
    add(f"Mean new entries/step: {c32.new_entries_mean:.1f} at 32K")
    add(f"P95 new entries/step: {c32.new_entries_p95:.0f} at 32K")
    add("")
    add("Best exact streaming architecture: ASIC HW-B, previous-TopK warm-start, exact")
    add(f"Scores/cycle: {int(asic32.scores_per_cycle)}")
    add(f"TopK throughput: {int(asic32.scores_per_cycle)} input scores/cycle, {int(asic32.effective_admission_lanes)} admissions/cycle")
    add(f"On-chip SRAM: {asic32.onchip_sram_bytes/1024:.1f} KiB modeled; provision margin separately")
    add(f"Comparator/resource estimate: {int(asic32.total_comparator_units)} simplified comparator units + {int(asic32.full64_bf16_mac_lanes):,} BF16 MAC lanes")
    add(f"Estimated 32K latency: {asic32.latency_us:.3f} us (analytical, pre-floorplan)")
    add(f"Estimated 64K latency: {asic64.latency_us:.3f} us (analytical; GPU baseline projected)")
    add("")
    add("Speedup vs optimized GPU:")
    add(f"32K: {asic32.speedup_vs_gpu:.3f}x analytical")
    add(f"64K: {asic64.speedup_vs_gpu:.3f}x analytical")
    add("")
    add("Dense score traffic eliminated: 100% of dense score write/read")
    add(f"Estimated energy benefit: {100*traffic_lower:.1f}%–{100*traffic_upper:.1f}% total-byte proxy at 32K + 70.5% fewer warm heap admissions; no joule claim")
    add("")
    add("FPGA TopK-only offload: NO-GO")
    add("Fused FPGA Indexer+TopK: NO-GO FOR RTL under conservative 1-BF16-MAC/DSP mapping")
    add("Exact ASIC selector: GO for detailed cycle/floorplan model; NO-GO for RTL sign-off yet")
    add("")
    add("Temporal+H8 extension:")
    add(f"additional benefit: {100*h8_progressive.qk_reduction_median:.1f}% median QK reduction, {100*h8_progressive.candidate_fraction:.1f}% candidate fraction")
    add(f"quality loss: Top-2048 recall {100*h8_progressive.top2048_recall:.3f}%, new-active recall {100*h8_progressive.newly_active_token_recall:.1f}%, MLA RelL2 p95/p99 {h8_progressive.mla_relative_l2_p95:.4f}/{h8_progressive.mla_relative_l2_p99:.4f}")
    add("")
    add("Final verdict: PROMISING")
    add("Next experiment: real 64K trace + hazard/banking/HBM-aware detailed cycle model + primitive synthesis")
    add("```")
    add("")
    return "\n".join(lines)


def reproducibility(output: Path) -> None:
    payload = {
        "primary_baseline": "Full64 Triton scorer + torch.topk(K=2048, sorted=False)",
        "trace_inventory": "artifacts/h8_reconstruction/final/trace_inventory.csv",
        "trace_count": 144,
        "transitions": 18288,
        "real_contexts": [8192, 16384, 32768],
        "projected_contexts": [65536],
        "gpu": "NVIDIA L40S; new runs isolated to free device; locked measurements preserved",
        "commands": [
            "python scripts/analyze_full64_topk_churn.py --repo . --inventory artifacts/h8_reconstruction/final/trace_inventory.csv --output artifacts/streaming_topk/analysis/churn",
            "python scripts/simulate_exact_topk_heap_admissions.py --repo . --inventory artifacts/h8_reconstruction/final/trace_inventory.csv --output artifacts/streaming_topk/analysis/heap",
            "python scripts/extract_full64_topk_nsight.py --sqlite-16k artifacts/streaming_topk/raw/progressive_nsight/profile_16k.sqlite --sqlite-32k artifacts/streaming_topk/raw/progressive_nsight/profile_32k.sqlite --output artifacts/streaming_topk/analysis",
            "CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_full64_topk_decomposition.py --repo . --output artifacts/streaming_topk/raw/decomposition --layer 17 --warmup 100 --measurements 1000",
            "python scripts/simulate_streaming_topk_hardware.py --progressive-root artifacts/progressive_sw --decomposition-root artifacts/streaming_topk/raw --heap-root artifacts/streaming_topk/analysis/heap --output artifacts/streaming_topk/analysis/hardware",
            "python scripts/build_streaming_topk_report.py --analysis artifacts/streaming_topk/analysis --progressive-root artifacts/progressive_sw --h8-root artifacts/h8_reconstruction --output artifacts/streaming_topk/final",
        ],
        "limitations": [
            "64K GPU timing is a transparent two-times projection from 32K isolated stages",
            "ncu unavailable; traffic is code-level/analytical, not measured DRAM/L2 traffic",
            "heap hazard/banking model is not RTL validated",
            "sidecar evidence is not a production DeepSeek-V3.2 claim",
        ],
    }
    (output / "reproducibility.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DSA streaming Top-K feasibility report")
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--progressive-root", type=Path, required=True)
    parser.add_argument("--h8-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    h8 = make_h8_summary(args.progressive_root, args.h8_root)
    tables = copy_tables(args.analysis, args.output, h8)
    make_figures(tables, args.output)
    (args.output / "dsa_streaming_topk_feasibility.md").write_text(
        report_text(tables), encoding="utf-8"
    )
    verdict = json.loads((args.analysis / "hardware" / "hardware_verdict.json").read_text(encoding="utf-8"))
    (args.output / "hardware_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    reproducibility(args.output)


if __name__ == "__main__":
    main()
