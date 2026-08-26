from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "K1": "#64748B", "K2": "#2563EB", "K3": "#F59E0B",
    "K4": "#16A34A", "K5": "#7C3AED", "K6": "#DC2626",
}


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def metric(quality: pd.DataFrame, policy: str, name: str) -> float:
    return float(quality[quality.policy.eq(policy) & quality.metric.eq(name)].value.iloc[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    graphs = out / "graphs_fused_h8_sm89"
    graphs.mkdir(parents=True, exist_ok=True)

    k1 = read(out / "full64_sync_timing.csv")
    k2 = read(out / "full64_pipeline_timing.csv")
    k3 = read(out / "two_pass_h8_timing.csv")
    k4 = read(out / "fused_h8_mask_timing.csv")
    online = read(out / "fused_h8_online_timing.csv")
    promotion = read(out / "promotion_rate_sweep.csv")
    warp = read(out / "warp_config_sweep.csv")
    layout = read(out / "shared_layout_sweep.csv")
    nsight = read(out / "nsight_summary.csv")
    quality = read(out / "quality_results.csv")
    topk = read(out / "topk_integration_results.csv")

    def timing(frame: pd.DataFrame, context: int, method: str | None = None, cache: str = "cold_rotating") -> float:
        rows = frame[frame.context.eq(context) & frame.cache_state.eq(cache)]
        if method is not None:
            rows = rows[rows.method.eq(method)]
        if "timing_scope" in rows.columns:
            rows = rows[rows.timing_scope.eq("main_only_prepacked")]
        return float(rows.median_us.iloc[0])

    primary = {}
    for context in (8192, 16384, 32768):
        full_k1 = timing(k1, context)
        full_k2 = timing(k2, context)
        effective = min(full_k1, full_k2)
        entry = {
            "K1_us": full_k1, "K2_us": full_k2,
            "effective_full_us": effective,
            "K6_us": timing(online, context, "K6_fused_online_pipeline"),
        }
        if context in (8192, 16384, 32768):
            entry["K4_us"] = timing(k4, context, "K4_fused_mask")
            entry["K3_us"] = timing(k3, context, "K3_two_pass_precomputed")
        entry["K6_speedup_vs_K2"] = entry["K2_us"] / entry["K6_us"]
        entry["K6_speedup_vs_effective_full"] = effective / entry["K6_us"]
        entry["K4_speedup_vs_K3"] = entry["K3_us"] / entry["K4_us"]
        primary[str(context)] = entry

    p0 = "P0_global_top10_precomputed"
    p1 = "P1_validation_fixed_local_threshold"
    profile32 = nsight[nsight.context.eq(32768)].set_index("method")
    verdict = {
        "verdict": "NO-GO",
        "go": False,
        "reason": "Deployable K6 is slower than the fastest optimized full64 baseline in cold-cache 16K and 32K.",
        "primary_cold_cache": primary,
        "dataflow_fusion": {
            "K4_vs_K3_speedup_16k": primary["16384"]["K4_speedup_vs_K3"],
            "K4_vs_K3_speedup_32k": primary["32768"]["K4_speedup_vs_K3"],
            "dram_read_bytes_K3_32k": float(profile32.loc["K3", "dram_read_bytes"]),
            "dram_read_bytes_K4_32k": float(profile32.loc["K4", "dram_read_bytes"]),
            "dram_read_reduction": 1.0 - float(profile32.loc["K4", "dram_read_bytes"]) / float(profile32.loc["K3", "dram_read_bytes"]),
            "claim_scope": "DATAFLOW-ONLY / PRECOMPUTED-MASK",
        },
        "deployable_K6_profile_32k": {
            key: float(profile32.loc["K6", key])
            for key in (
                "dram_read_bytes", "l2_hit_rate_pct", "tensor_pipe_active_pct",
                "achieved_occupancy_pct", "registers_per_thread_max",
                "barrier_stall_per_issue", "long_scoreboard_stall_per_issue",
            )
        },
        "quality": {
            "P0_qk_reduction": metric(quality, p0, "net_qk_reduction_median"),
            "P0_mla_rell2_p95": metric(quality, p0, "mla_relative_l2_p95"),
            "P0_ppl_delta": metric(quality, p0, "ppl_delta"),
            "P1_actual_promotion_rate": metric(quality, p1, "actual_promotion_rate_of_cold"),
            "P1_qk_reduction": metric(quality, p1, "net_qk_reduction_median"),
            "P1_mla_rell2_p95": metric(quality, p1, "mla_relative_l2_p95"),
            "P1_ppl_delta": metric(quality, p1, "ppl_delta"),
            "P1_closed_loop_task_success": metric(quality, p1, "closed_loop_task_success"),
        },
        "topk_integration": json.loads(topk.to_json(orient="records")),
        "next_direction": "Compact K-sketch/traffic pruning or a dedicated progressive datapath; do not integrate this K6 software kernel.",
    }
    (out / "kernel_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")

    # 1. Primary latency.
    contexts = [16384, 32768]
    labels = ["16K", "32K"]
    methods = ["K1", "K2", "K3", "K4", "K6"]
    values = {
        "K1": [primary[str(c)]["K1_us"] for c in contexts],
        "K2": [primary[str(c)]["K2_us"] for c in contexts],
        "K3": [primary[str(c)]["K3_us"] for c in contexts],
        "K4": [primary[str(c)]["K4_us"] for c in contexts],
        "K6": [primary[str(c)]["K6_us"] for c in contexts],
    }
    fig, ax = plt.subplots(figsize=(8, 4.5)); x = np.arange(2); width = 0.15
    for i, method_name in enumerate(methods):
        ax.bar(x + (i - 2) * width, values[method_name], width, label=method_name, color=COLORS[method_name])
    ax.set_xticks(x, labels); ax.set_ylabel("Cold-cache latency (µs)"); ax.set_title("Full64 vs two-pass vs fused SM89 kernels"); ax.legend(ncol=5)
    save(fig, graphs / "01_full_two_pass_fused_pipeline_latency.png")

    # 2. Promotion sweep.
    fig, ax = plt.subplots(figsize=(7, 4.3))
    base32 = primary["32768"]["effective_full_us"]
    for pattern, style in (("random", "o-"), ("clustered", "s--")):
        f = promotion[promotion.context.eq(32768) & promotion.pattern.eq(pattern)].sort_values("promotion_rate")
        ax.plot(f.promotion_rate * 100, base32 / f.median_us, style, label=pattern)
    ax.axhline(1, color="#111827", lw=1); ax.set_xlabel("Promoted blocks (%)"); ax.set_ylabel("Speedup vs fastest full"); ax.set_title("Promotion rate vs K4 speedup (32K cold)"); ax.legend()
    save(fig, graphs / "02_promotion_rate_vs_speedup.png")

    # 3. Context speedup.
    fig, ax = plt.subplots(figsize=(7, 4.3))
    cs = [8192, 16384, 32768]
    ax.plot([8, 16, 32], [primary[str(c)]["K6_speedup_vs_effective_full"] for c in cs], "o-", color=COLORS["K6"], label="K6 online")
    ax.plot([8, 16, 32], [primary[str(c)]["K4_us"] and primary[str(c)]["effective_full_us"] / primary[str(c)]["K4_us"] for c in cs], "s--", color=COLORS["K4"], label="K4 mask")
    ax.axhline(1, color="#111827", lw=1); ax.set_xlabel("Context (K tokens)"); ax.set_ylabel("Speedup vs fastest full"); ax.set_title("Context length vs speedup (cold)"); ax.legend()
    save(fig, graphs / "03_context_length_vs_speedup.png")

    # 4. Hot vs cold.
    fig, ax = plt.subplots(figsize=(7, 4.3)); x = np.arange(2); width = 0.35
    hot_speed = []; cold_speed = []
    for c in contexts:
        hot_full = min(timing(k1, c, cache="hot_l2"), timing(k2, c, cache="hot_l2"))
        hot_speed.append(hot_full / timing(online, c, "K6_fused_online_pipeline", "hot_l2"))
        cold_speed.append(primary[str(c)]["K6_speedup_vs_effective_full"])
    ax.bar(x - width / 2, hot_speed, width, label="Hot L2", color="#60A5FA")
    ax.bar(x + width / 2, cold_speed, width, label="Cold rotating", color="#1D4ED8")
    ax.axhline(1, color="#111827", lw=1); ax.set_xticks(x, labels); ax.set_ylabel("K6 speedup vs fastest full"); ax.set_title("Hot-L2 vs cold-cache"); ax.legend()
    save(fig, graphs / "04_hot_vs_cold_speedup.png")

    # 5. Algorithmic vs measured reduction.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for c, marker in ((16384, "o"), (32768, "s")):
        for name, reduction, latency in (
            ("K4/P0", metric(quality, p0, "net_qk_reduction_median"), primary[str(c)]["K4_us"]),
            ("K6/P1", metric(quality, p1, "net_qk_reduction_median"), primary[str(c)]["K6_us"]),
        ):
            measured = 1 - latency / primary[str(c)]["effective_full_us"]
            ax.scatter(reduction * 100, measured * 100, marker=marker, s=70, label=f"{name} {c//1024}K")
    ax.plot([0, 30], [0, 30], "k--", lw=1, label="ideal 1:1"); ax.axhline(0, color="#9CA3AF", lw=1)
    ax.set_xlabel("Algorithmic QK reduction (%)"); ax.set_ylabel("Measured latency reduction (%)"); ax.set_title("MAC reduction did not become latency reduction"); ax.legend(fontsize=8)
    save(fig, graphs / "05_qk_vs_measured_latency_reduction.png")

    # 6-8 Nsight.
    for index, (column, ylabel, title, filename) in enumerate((
        ("dram_read_bytes", "DRAM read (MB)", "K reread removed by fusion", "06_k_reread_bytes_before_after.png"),
        ("dram_bandwidth_util_pct_of_864GBs", "Peak DRAM bandwidth (%)", "Measured DRAM bandwidth utilization", "07_dram_bandwidth_utilization.png"),
        ("tensor_pipe_active_pct", "Tensor pipe active (%)", "Tensor Core utilization remains low", "08_tensor_core_utilization.png"),
    )):
        fig, ax = plt.subplots(figsize=(7, 4.3)); x = np.arange(2); width = 0.2
        for i, method_name in enumerate(["K2", "K3", "K4", "K6"]):
            f = nsight[nsight.method.eq(method_name)].set_index("context")
            y = [float(f.loc[c, column]) / (1e6 if column == "dram_read_bytes" else 1) for c in contexts]
            ax.bar(x + (i - 1.5) * width, y, width, label=method_name, color=COLORS[method_name])
        ax.set_xticks(x, labels); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(ncol=4)
        save(fig, graphs / filename)

    # 9. Shared memory trade-off.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    f = layout[layout.cache_state.eq("cold_rotating")]
    for name, group in f.groupby("q_storage"):
        ax.scatter(group.shared_memory_bytes_per_cta / 1024, group.median_us, s=70, label=name)
        for _, row in group.iterrows(): ax.annotate(row.layout.replace("_", " "), (row.shared_memory_bytes_per_cta / 1024, row.median_us), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Shared memory / CTA (KiB)"); ax.set_ylabel("K1 latency (µs)"); ax.set_title("Q-shared and K-layout trade-off (32K cold)"); ax.legend()
    save(fig, graphs / "09_shared_memory_occupancy_tradeoff.png")

    # 10. Warp configuration.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method_name in ("K2_full_pipeline", "K6_fused_online_pipeline"):
        f = warp[warp.method.eq(method_name)]
        ax.scatter(f.total_warps + (f.ctas_per_sm - 2) * 0.08, f.median_us, s=70, label=method_name.split("_")[0])
    ax.set_xticks([4, 5, 8]); ax.set_xlabel("Total warps / CTA"); ax.set_ylabel("Latency (µs)"); ax.set_title("4/5/8-warp persistent configurations (32K cold)"); ax.legend()
    save(fig, graphs / "10_warp_configuration.png")

    # 11. Scheduling.
    fig, ax = plt.subplots(figsize=(7, 4.3))
    k2warp = warp[warp.method.eq("K2_full_pipeline")].groupby("ctas_per_sm").median_us.min()
    ax.bar(["one CTA/block\nK1", "persistent\n1 CTA/SM", "persistent\n2 CTA/SM", "persistent\n3 CTA/SM"], [primary["32768"]["K1_us"], *[k2warp.loc[i] for i in (1, 2, 3)]], color=[COLORS["K1"], COLORS["K2"], COLORS["K2"], COLORS["K2"]])
    ax.set_ylabel("Latency (µs)"); ax.set_title("One-CTA/block beats persistent async K2")
    save(fig, graphs / "11_one_cta_vs_persistent_pipeline.png")

    # 12. Timeline summary.
    fig, ax = plt.subplots(figsize=(7, 4.3))
    f = nsight[nsight.context.eq(32768)].set_index("method").loc[["K2", "K3", "K4", "K6"]]
    ax.barh(f.index, f.nsys_nvtx_avg_us, color=[COLORS[x] for x in f.index])
    for i, (_, row) in enumerate(f.iterrows()): ax.text(row.nsys_nvtx_avg_us + 0.4, i, f"{row.nsys_gpu_ops_per_iteration:.0f} GPU op(s)", va="center", fontsize=8)
    ax.set_xlabel("NVTX projected GPU time / iteration (µs)"); ax.set_title("Kernel-launch timeline summary (32K, 100 rotations)")
    save(fig, graphs / "12_kernel_launch_timeline.png")

    # 13. Quality-speed operating points.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    points = [
        ("P0/K4", metric(quality, p0, "net_qk_reduction_median"), metric(quality, p0, "mla_relative_l2_p95"), primary["32768"]["effective_full_us"] / primary["32768"]["K4_us"], COLORS["K4"]),
        ("P1/K6", metric(quality, p1, "net_qk_reduction_median"), metric(quality, p1, "mla_relative_l2_p95"), primary["32768"]["K6_speedup_vs_effective_full"], COLORS["K6"]),
    ]
    for name, reduction, rell2, speedup, color in points:
        ax.scatter(reduction * 100, rell2 * 100, s=250 * speedup, color=color); ax.annotate(f"{name}\n{speedup:.3f}×", (reduction * 100, rell2 * 100), xytext=(7, 5), textcoords="offset points")
    ax.set_xlabel("Net QK reduction (%)"); ax.set_ylabel("MLA RelL2 p95 (%)"); ax.set_title("Online promotion quality vs measured speed")
    save(fig, graphs / "13_online_promotion_quality_vs_speed.png")

    # 14-15 explicit early-stop panels.
    for filename, title, body in (
        ("14_scoring_vs_scoring_topk.png", "Scoring + Top-K was not integrated", "EARLY STOP\nK6 failed the 1.05× scoring gate\n16K: 0.833×   32K: 0.868×"),
        ("15_sidecar_tpot_impact.png", "Sidecar TPOT impact was not claimed", "NOT RUN\nNo model-runtime integration after scoring NO-GO\nNo production extrapolation"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4.2)); ax.axis("off"); ax.text(0.5, 0.68, title, ha="center", fontsize=15, weight="bold"); ax.text(0.5, 0.35, body, ha="center", va="center", fontsize=12, bbox=dict(boxstyle="round,pad=0.8", facecolor="#F3F4F6", edgecolor="#9CA3AF")); save(fig, graphs / filename)

    prep = online[online.timing_scope.eq("prep_only") & online.context.eq(32768)]
    report = f"""# L40S SM89 Fused Progressive H8 DSA CUDA Kernel Pilot

## Verdict — `NO-GO`

The single-load fused dataflow is real and removes the two-pass K reread, but the deployable online `cp.async` persistent kernel does **not** beat the fastest optimized Full64 kernel.  The requested software GO gate is therefore not met.

### First-page numbers

| Metric | 16K | 32K |
|---|---:|---:|
| K1 Full64 sync, cold | {primary['16384']['K1_us']:.3f} µs | {primary['32768']['K1_us']:.3f} µs |
| K2 Full64 async pipeline, cold | {primary['16384']['K2_us']:.3f} µs | {primary['32768']['K2_us']:.3f} µs |
| K3 two-pass precomputed, cold | {primary['16384']['K3_us']:.3f} µs | {primary['32768']['K3_us']:.3f} µs |
| K4 fused precomputed mask, cold | {primary['16384']['K4_us']:.3f} µs | {primary['32768']['K4_us']:.3f} µs |
| K6 fused online async, cold | {primary['16384']['K6_us']:.3f} µs | {primary['32768']['K6_us']:.3f} µs |
| K4 speedup vs K3 | {primary['16384']['K4_speedup_vs_K3']:.3f}× | {primary['32768']['K4_speedup_vs_K3']:.3f}× |
| K6 speedup vs K2 | {primary['16384']['K6_speedup_vs_K2']:.3f}× | {primary['32768']['K6_speedup_vs_K2']:.3f}× |
| K6 speedup vs fastest Full64 | {primary['16384']['K6_speedup_vs_effective_full']:.3f}× | {primary['32768']['K6_speedup_vs_effective_full']:.3f}× |

At 32K, measured DRAM read fell from **{profile32.loc['K3','dram_read_bytes']/1e6:.2f} MB (K3)** to **{profile32.loc['K4','dram_read_bytes']/1e6:.2f} MB (K4)**, a **{verdict['dataflow_fusion']['dram_read_reduction']*100:.1f}%** reduction.  This proves that fused same-tile continuation removed the physical reread.  It is a `DATAFLOW-ONLY / PRECOMPUTED-MASK` result, not a deployable policy claim.

K6 uses **{profile32.loc['K6','registers_per_thread_max']:.0f} registers/thread**, **{profile32.loc['K6','shared_memory_per_cta_bytes_max']/1024:.1f} KiB shared/CTA**, reaches only **{profile32.loc['K6','achieved_occupancy_pct']:.1f}%** achieved occupancy and **{profile32.loc['K6','tensor_pipe_active_pct']:.1f}%** Tensor-pipe activity.  Its DRAM read is {profile32.loc['K6','dram_read_bytes']/1e6:.2f} MB, essentially the same one-load traffic as K2, so the loss is dominated by register pressure, synchronization and under-utilization rather than an unremoved K reread.

### Quality operating points

| Policy | Actual cold promotion | Net QK reduction | MLA RelL2 p95 | Top-128 recall | PPL delta |
|---|---:|---:|---:|---:|---:|
| P0 global top-10%, precomputed | 10% budget | {metric(quality,p0,'net_qk_reduction_median')*100:.2f}% | {metric(quality,p0,'mla_relative_l2_p95')*100:.2f}% | {metric(quality,p0,'top128_recall')*100:.4f}% | {metric(quality,p0,'ppl_delta')*100:.3f}% |
| P1 validation-fixed local threshold | {metric(quality,p1,'actual_promotion_rate_of_cold')*100:.2f}% | {metric(quality,p1,'net_qk_reduction_median')*100:.2f}% | {metric(quality,p1,'mla_relative_l2_p95')*100:.2f}% | {metric(quality,p1,'top128_recall')*100:.4f}% | {metric(quality,p1,'ppl_delta')*100:.3f}% |

P1 maintained local answer-task success at {metric(quality,p1,'closed_loop_task_success')*100:.1f}%, matching the prior baseline aggregate, but long-code token accuracy remained a weak point.  These are locked prior replay results for the identical P0/P1 policy; this run revalidated mathematical kernel equivalence on actual sidecar traces (45/45 CUDA correctness rows passed).

## Required questions

1. **Was H8 mapped without padding?** Yes. Each compute warp executes exact `(16×128)·(128×8)` tiles with `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`; four warps cover B64.
2. **Did SASS generate Tensor Core instructions?** Yes. `cuobjdump` found 1,024 `HMMA` instructions in the extension; the correctness maximum absolute error stayed below 0.002.
3. **Was the K reread removed?** Yes for K4/K6. The 32K K3→K4 DRAM-read reduction was {verdict['dataflow_fusion']['dram_read_reduction']*100:.1f}%.
4. **Did `cp.async` overlap help?** `LDGSTS` (72 SASS instructions) and double buffers are present, but the measured pipeline did not create a latency win. K6's resource/synchronization cost outweighed overlap.
5. **Was the same pipeline applied to Full64?** Yes. K2 uses the identical async copy primitive, 32 KiB K double buffer and persistent scheduling. K1 is also disclosed because it was faster than K2 and is the honest effective baseline.
6. **Did MAC reduction become latency reduction?** No. P1 reduced QK by {metric(quality,p1,'net_qk_reduction_median')*100:.2f}% yet K6 was {1-primary['32768']['K6_speedup_vs_effective_full']:.1%} slower than fastest Full64 at 32K.
7. **Why not?** K6 is register/occupancy/synchronization-bound: 255 registers/thread, {profile32.loc['K6','achieved_occupancy_pct']:.1f}% achieved occupancy, {profile32.loc['K6','tensor_pipe_active_pct']:.1f}% Tensor-pipe activity, plus CTA-wide verifier barriers and irregular continuation.
8. **P0 vs P1 quality?** P1 preserved the P0 quality scale but drifted from the nominal 10% rescue to {metric(quality,p1,'actual_promotion_rate_of_cold')*100:.2f}%, reducing net QK savings from {metric(quality,p0,'net_qk_reduction_median')*100:.2f}% to {metric(quality,p1,'net_qk_reduction_median')*100:.2f}%.
9. **Did speedup survive Top-K?** Not tested by design: K6 failed the required 1.05× scoring gate. No Top-K or TPOT speedup is claimed.
10. **What direction is supported?** Keep the single-load fusion idea, but move to a compact K-sketch/traffic-pruning front end or a dedicated progressive datapath. This particular SM89 persistent software kernel should not be integrated.

## Additional observations

- Dynamic Top-8 selection + packing alone costs {float(prep[prep.head_scheme.eq('dynamic_abs_w')].median_us.iloc[0]):.3f} µs at 32K; fixed-head packing costs {float(prep[prep.head_scheme.eq('fixed_avg_abs_w')].median_us.iloc[0]):.3f} µs. Prep makes the software case worse.
- Q-shared plus padded K stride was the best shared-layout ablation, but it does not change the K6 verdict.
- Hot-L2 results were measured separately; cold rotating traces and a 4×L2 flush were used for the verdict.
- 64K/128K were performance-only synthetic key repeats. Quality claims remain restricted to 8K/16K/32K research-sidecar traces.
- No TMA, WGMMA, thread-block clusters, SM90 PTX or Hopper-only kernels were used.
"""
    (out / "fused_h8_sm89_report.md").write_text(report)
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
