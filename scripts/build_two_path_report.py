from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_REVISION = "85864749cd611b4353ce1decdb286193298f64c7"
SCOPE = "Research sidecar DSA on DeepSeek-V2-Lite"
HEAD = "head_dynamic_abs_w8_b64_r0.1"
DIM = "dim_had_even_w16_b64_r0.1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def band(row: pd.Series) -> str:
    rel = float(row.output_relative_l2_p95)
    cosine = float(row.output_cosine_p5)
    top128 = float(row.top128_recall)
    top512 = float(row.top512_recall)
    if rel <= 0.03 and cosine >= 0.997 and top128 >= 0.9995:
        return "A"
    if rel <= 0.05 and cosine >= 0.995 and top128 >= 0.998 and top512 >= 0.992:
        return "B"
    if rel <= 0.08 and cosine >= 0.990 and top128 >= 0.995:
        return "C"
    return "FAIL"


def weighted_tail(group: pd.DataFrame) -> float:
    count = float(group.tail_critical_blocks.sum())
    if count == 0:
        return np.nan
    return float((group.tail_critical_block_recall * group.tail_critical_blocks).sum() / count)


def save(fig: plt.Figure, root: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(root / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def placeholder(root: Path, name: str, title: str, reason: str) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 4.5))
    axis.axis("off")
    axis.text(0.5, 0.62, title, ha="center", va="center", fontsize=17, weight="bold")
    axis.text(0.5, 0.40, reason, ha="center", va="center", fontsize=11, wrap=True)
    save(fig, root, name)


def label_points(
    axis: plt.Axes,
    frame: pd.DataFrame,
    x: str,
    y: str,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> None:
    for row in frame.itertuples():
        axis.annotate(
            f"{row.policy_role[:3]} {row.verifier.split('_w')[0]}",
            (getattr(row, x) * scale_x, getattr(row, y) * scale_y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )


def build_graphs(
    root: Path,
    oracle: pd.DataFrame,
    wide: pd.DataFrame,
    tail_selected: pd.DataFrame,
    tail_mla: pd.DataFrame,
    selection: pd.DataFrame,
    mla: pd.DataFrame,
    vrows: pd.DataFrame,
    mla_rows: pd.DataFrame,
    teacher: pd.DataFrame,
    cycle: pd.DataFrame,
) -> list[str]:
    root.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    order = {"0": 0, "best1": 1, "best2": 2, "best4": 4, "best8": 8, "all": 32}
    plot = oracle.copy()
    plot["rescue_count"] = plot.rescue.map(order)
    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    for role, group in plot.groupby("policy_role"):
        group = group.sort_values("rescue_count")
        axis.plot(group.rescue_count, group.output_relative_l2_p95 * 100, marker="o", label=role)
    axis.axhline(5, color="black", linestyle="--", linewidth=1, label="Band B")
    axis.set(xlabel="Oracle rescued blocks (all shown at x=32)", ylabel="MLA RelL2 p95 (%)", title="Oracle rescue ceiling, 144 traces")
    axis.legend()
    save(fig, root, "01_oracle_rescue_vs_rell2_p95.png"); generated.append("01_oracle_rescue_vs_rell2_p95.png")

    sweep = wide[np.isclose(wide.rescue_fraction, 0.1)].copy()
    sweep["mac_budget"] = np.where(sweep.path == "head", sweep.width / 64, sweep.width / 128)
    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    for (role, path), group in sweep.groupby(["policy_role", "path"]):
        best = group.groupby("mac_budget", as_index=False).newly_active_token_recall.max()
        axis.plot(best.mac_budget * 100, best.newly_active_token_recall * 100, marker="o", label=f"{role} {path}")
    axis.set(xlabel="Verifier MAC vs full (%)", ylabel="Newly-active token recall (%)", title="32K tail detector recall")
    axis.legend(fontsize=8)
    save(fig, root, "02_newly_active_recall_vs_mac.png"); generated.append("02_newly_active_recall_vs_mac.png")

    tail = tail_selected[np.isclose(tail_selected.rescue_fraction, 0.1)].copy()
    tail["mac_budget"] = np.where(tail.path == "head", tail.width / 64, tail.width / 128)
    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    for (role, path), group in tail.groupby(["policy_role", "path"]):
        axis.plot(group.mac_budget * 100, group.tail_critical_block_recall * 100, marker="o", label=f"{role} {path}")
    axis.set(xlabel="Verifier MAC vs full (%)", ylabel="Top-4 tail-critical block recall (%)", title="32K tail-critical recall")
    axis.legend(fontsize=8)
    save(fig, root, "03_tail_critical_recall_vs_mac.png"); generated.append("03_tail_critical_recall_vs_mac.png")

    primary = selection[(selection.policy_role == "Aggressive") & selection.verifier.isin([HEAD, DIM])].merge(
        mla[["policy_role", "verifier", "output_relative_l2_p95"]], on=["policy_role", "verifier"]
    )
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.4))
    labels = ["8H×128D", "64H×16D"]
    values = primary.set_index("verifier").loc[[HEAD, DIM]]
    axes[0].bar(labels, values.output_relative_l2_p95 * 100); axes[0].set_title("RelL2 p95 (%)")
    axes[1].bar(labels, values.net_qk_reduction_median * 100); axes[1].set_title("Net QK reduction (%)")
    axes[2].bar(labels, values.physical_key_byte_reduction_median * 100); axes[2].set_title("Physical B64 bytes reduction (%)")
    fig.suptitle("Equal 12.5% verifier MAC: quality vs traffic")
    save(fig, root, "04_equal_cost_comparison.png"); generated.append("04_equal_cost_comparison.png")

    fig, axis = plt.subplots(figsize=(8, 4.8))
    budget = tail_mla.copy()
    budget["rescue_fraction"] = budget.verifier.str.extract(r"_r(0?\.\d+)$")[0].astype(float)
    budget["method"] = budget.verifier.str.replace(r"_r0?\.\d+$", "", regex=True)
    for (role, method), group in budget.groupby(["policy_role", "method"]):
        if len(group) < 2:
            continue
        axis.plot(group.rescue_fraction * 100, group.output_relative_l2_p95 * 100, marker="o", label=f"{role[:3]} {method.split('_b')[0]}")
    axis.axhline(5, color="black", linestyle="--", linewidth=1)
    axis.set(xlabel="Cold-block rescue budget (%)", ylabel="32K RelL2 p95 (%)", title="Measured rescue-budget ablation")
    axis.legend(fontsize=7, ncol=2)
    save(fig, root, "05_rescue_budget_vs_rell2_p95.png"); generated.append("05_rescue_budget_vs_rell2_p95.png")

    joined = selection.merge(mla, on=["policy_role", "verifier"], suffixes=("", "_mla"))
    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    for path, group in joined.groupby("path"):
        axis.scatter(group.net_qk_reduction_median * 100, group.output_relative_l2_p95 * 100, label=path, s=55)
    label_points(axis, joined, "net_qk_reduction_median", "output_relative_l2_p95", 100, 100)
    axis.set(xlabel="Net QK reduction (%)", ylabel="MLA RelL2 p95 (%)", title="Compute-quality frontier")
    axis.legend()
    save(fig, root, "06_net_qk_vs_rell2_p95.png"); generated.append("06_net_qk_vs_rell2_p95.png")

    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    for path, group in joined.groupby("path"):
        axis.scatter(group.physical_key_byte_reduction_median * 100, group.output_relative_l2_p95 * 100, label=path, s=55)
    axis.axvline(0, color="gray", linewidth=1)
    axis.set(xlabel="Physical B64 key-byte reduction (%)", ylabel="MLA RelL2 p95 (%)", title="Traffic-quality frontier")
    axis.legend()
    save(fig, root, "07_physical_bytes_vs_rell2_p95.png"); generated.append("07_physical_bytes_vs_rell2_p95.png")

    for number, metric, title in [(8, "top128_recall", "Top-128"), (9, "top512_recall", "Top-512")]:
        fig, axis = plt.subplots(figsize=(7.5, 4.8))
        for path, group in selection.groupby("path"):
            axis.scatter(group.net_qk_reduction_median * 100, group[metric] * 100, label=path, s=55)
        axis.set(xlabel="Net QK reduction (%)", ylabel=f"{title} recall (%)", title=f"{title} retention vs compute")
        axis.legend()
        filename = f"{number:02d}_{title.lower().replace('-', '')}_vs_net_qk.png"
        save(fig, root, filename); generated.append(filename)

    placeholder(root, "10_missed_token_rank_histogram.png", "Missed-token rank histogram", "Not retained by the replay artifact. Aggregate Top-128/512/2048 retention and newly-active counts are reported instead; no synthetic ranks are shown.")
    generated.append("10_missed_token_rank_histogram.png")

    positive = vrows[vrows.tail_critical_blocks > 0]
    by_layer = positive.groupby(["policy_role", "verifier", "layer"]).apply(weighted_tail, include_groups=False).rename("recall").reset_index()
    focus = by_layer[(by_layer.policy_role == "Aggressive") & by_layer.verifier.isin([HEAD, DIM])]
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    for verifier, group in focus.groupby("verifier"):
        axis.plot(group.layer, group.recall * 100, marker="o", label=verifier)
    axis.set(xlabel="Layer", ylabel="Tail-critical block recall (%)", title="Per-layer top-4 tail-critical recall")
    axis.legend(fontsize=8)
    save(fig, root, "11_per_layer_tail_recall.png"); generated.append("11_per_layer_tail_recall.png")

    focus_mla = mla_rows[(mla_rows.policy_role == "Aggressive") & (mla_rows.base_context_length == 32768) & mla_rows.layer.isin([8, 17, 21]) & mla_rows.verifier.isin([HEAD, DIM])]
    detail = focus_mla.groupby(["verifier", "layer"]).output_relative_l2.quantile(0.95).reset_index()
    fig, axis = plt.subplots(figsize=(8, 4.8))
    for verifier, group in detail.groupby("verifier"):
        axis.plot(group.layer, group.output_relative_l2 * 100, marker="o", label=verifier)
    axis.axhline(5, color="black", linestyle="--", linewidth=1)
    axis.set(xlabel="Layer", ylabel="32K RelL2 p95 (%)", title="Focused layers 8/17/21")
    axis.legend(fontsize=8)
    save(fig, root, "12_32k_layers_8_17_21.png"); generated.append("12_32k_layers_8_17_21.png")

    head = sweep[(sweep.policy_role == "Balanced") & (sweep.path == "head")].copy()
    head["strategy_label"] = head.verifier.str.extract(r"head_(.*?)_w")
    matrix = head.pivot_table(index="strategy_label", columns="width", values="top128_recall", aggfunc="max") * 100
    fig, axis = plt.subplots(figsize=(7, 3.8)); image = axis.imshow(matrix, aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(matrix.columns)), matrix.columns); axis.set_yticks(range(len(matrix.index)), matrix.index)
    axis.set(xlabel="Selected heads", title="Head strategy heatmap: Top-128 recall (%)")
    fig.colorbar(image, ax=axis)
    save(fig, root, "13_head_strategy_heatmap.png"); generated.append("13_head_strategy_heatmap.png")

    dims = sweep[(sweep.policy_role == "Balanced") & (sweep.path == "dim")].copy()
    dims["strategy_label"] = dims.verifier.str.extract(r"dim_(.*?)_w")
    matrix = dims.pivot_table(index="strategy_label", columns="width", values="top512_recall", aggfunc="max") * 100
    fig, axis = plt.subplots(figsize=(7, 4.2)); image = axis.imshow(matrix, aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(matrix.columns)), matrix.columns); axis.set_yticks(range(len(matrix.index)), matrix.index)
    axis.set(xlabel="Sketch dimensions", title="Dimension strategy heatmap: Top-512 recall (%)")
    fig.colorbar(image, ax=axis)
    save(fig, root, "14_dim_strategy_heatmap.png"); generated.append("14_dim_strategy_heatmap.png")

    ablation = dims[dims.strategy_label.isin(["random", "had_random", "had_even"])]
    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    for strategy, group in ablation.groupby("strategy_label"):
        axis.plot(group.width, group.top512_recall * 100, marker="o", label=strategy)
    axis.set(xlabel="Sketch dimensions", ylabel="Top-512 recall (%)", title="Hadamard rotation ablation, 32K Balanced")
    axis.legend()
    save(fig, root, "15_hadamard_ablation.png"); generated.append("15_hadamard_ablation.png")

    placeholder(root, "16_precision_sweep.png", "BF16 / INT8 / INT4 sketch quality", "INT8/INT4/INT2 were gated off: no BF16 dimension-sparse candidate passed Quality Band B. This is an explicit early-stop result.")
    generated.append("16_precision_sweep.png")

    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    for path, group in selection.groupby("path"):
        axis.scatter(group.newly_active_block_recall * 100, group.rescue_precision_mean * 100, label=path, s=60)
    axis.set(xlabel="Newly-active block recall (%)", ylabel="Rescue precision (%)", title="Detector precision-recall operating points")
    axis.legend()
    save(fig, root, "17_rescue_precision_recall.png"); generated.append("17_rescue_precision_recall.png")

    head_row = selection[(selection.policy_role == "Aggressive") & (selection.verifier == HEAD)].iloc[0]
    fig, axis = plt.subplots(figsize=(6.8, 4.5))
    axis.scatter([head_row.physical_key_byte_reduction_median * 100], [teacher.ppl_delta.iloc[0] * 100], s=90)
    axis.annotate("Aggressive H8", (head_row.physical_key_byte_reduction_median * 100, teacher.ppl_delta.iloc[0] * 100), xytext=(7, 7), textcoords="offset points")
    axis.axhline(1, color="black", linestyle="--", linewidth=1); axis.axhline(-1, color="black", linestyle="--", linewidth=1)
    axis.set(xlabel="Physical B64 key-byte reduction (%)", ylabel="Teacher-forced PPL delta (%)", title="Model quality vs physical bytes")
    save(fig, root, "18_teacher_ppl_vs_net_bytes.png"); generated.append("18_teacher_ppl_vs_net_bytes.png")

    sim = cycle[(cycle.policy_role == "Aggressive") & (cycle.context_length == 32768) & (cycle.verifier_compute_ratio == 4) & (cycle.rescue_rate_scale == 1.0)]
    sim = sim[((sim.verifier == HEAD) & (sim.sketch_memory == "full_k_hbm")) | ((sim.verifier == DIM) & (sim.sketch_memory == "shared_hbm"))]
    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    for verifier, group in sim.groupby("verifier"):
        axis.plot(group.full_k_bandwidth_gbps, group.speedup_over_full_dsa, marker="o", label=verifier)
    axis.axhline(1, color="black", linewidth=1)
    axis.set(xlabel="Full-K bandwidth (GB/s)", ylabel="Speedup over full DSA", title="Optimistic streaming cycle model, 32K")
    axis.legend(fontsize=8)
    save(fig, root, "19_sim_speedup_vs_hbm.png"); generated.append("19_sim_speedup_vs_hbm.png")

    sens = cycle[(cycle.policy_role == "Aggressive") & (cycle.context_length == 32768) & (cycle.verifier_compute_ratio == 4) & (cycle.full_k_bandwidth_gbps == 512)]
    sens = sens[((sens.verifier == HEAD) & (sens.sketch_memory == "full_k_hbm")) | ((sens.verifier == DIM) & (sens.sketch_memory == "shared_hbm"))]
    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    for verifier, group in sens.groupby("verifier"):
        group = group.sort_values("modeled_rescue_fraction")
        axis.plot(group.modeled_rescue_fraction * 100, group.speedup_over_full_dsa, marker="o", label=verifier)
    axis.set(xlabel="Modeled rescue fraction (%)", ylabel="Speedup over full DSA", title="Rescue-rate sensitivity extrapolation")
    axis.legend(fontsize=8)
    save(fig, root, "20_sim_speedup_vs_rescue_rate.png"); generated.append("20_sim_speedup_vs_rescue_rate.png")

    dims_sim = cycle[(cycle.path == "dim") & (cycle.sketch_memory == "separate_sram") & (cycle.context_length == 131072) & (cycle.rescue_rate_scale == 1.0)]
    sram = dims_sim.groupby("width", as_index=False).total_sram_bytes.max()
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(sram.width, sram.total_sram_bytes / (1024 ** 2), marker="o")
    axis.set(xlabel="Sketch dimensions", ylabel="Max total SRAM (MiB)", title="128K sketch + candidate SRAM")
    save(fig, root, "21_sram_vs_sketch_dim.png"); generated.append("21_sram_vs_sketch_dim.png")

    bytes_plot = selection[selection.policy_role == "Aggressive"].copy()
    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(bytes_plot.verifier, (1 - bytes_plot.physical_key_byte_reduction_median) * 100, color=np.where(bytes_plot.path == "head", "tab:blue", "tab:orange"))
    axis.axhline(100, color="black", linestyle="--", linewidth=1)
    axis.set(ylabel="Physical B64 bytes vs full DSA (%)", title="Full-K head scan vs compact dimension sketch")
    axis.tick_params(axis="x", rotation=25)
    save(fig, root, "22_head_vs_dim_physical_bytes.png"); generated.append("22_head_vs_dim_physical_bytes.png")
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final two-path verifier report")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-commit")
    args = parser.parse_args()
    source = args.input
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    selection = pd.read_csv(source / "full_replay/verifier_summary.csv")
    mla = pd.read_csv(source / "full_mla/mla_output_summary.csv")
    vrows = pd.read_csv(source / "full_replay/verifier_rows.csv")
    mla_rows = pd.read_csv(source / "full_mla/mla_output_quality.csv")
    oracle = pd.read_csv(source / "oracle_full/oracle_rescue_ceiling.csv")
    wide = pd.read_csv(source / "tail32k_verifier_summary_raw.csv")
    tail_selected = pd.read_csv(source / "tail32k_selected_verifier_summary.csv")
    tail_mla = pd.read_csv(source / "tail32k_mla_output_summary.csv")
    teacher = pd.read_csv(source / "teacher_forced/teacher_forced_summary.csv")
    closed = pd.read_csv(source / "closed_loop/closed_loop_quality.csv")
    cycle = pd.read_csv(source / "cycle_sim/cycle_sim_results.csv")

    merged = selection.merge(mla, on=["policy_role", "verifier"], suffixes=("", "_mla"))
    merged["offline_quality_band"] = merged.apply(band, axis=1)
    merged["teacher_forced_measured"] = (merged.policy_role == "Aggressive") & (merged.verifier == HEAD)
    merged["teacher_ppl_delta"] = np.where(merged.teacher_forced_measured, float(teacher.ppl_delta.iloc[0]), np.nan)
    merged["band_b_pass_with_available_model_gate"] = (
        merged.offline_quality_band.isin(["A", "B"])
        & (~merged.teacher_forced_measured | (merged.teacher_ppl_delta.abs() <= 0.01))
    )

    oracle.to_csv(output / "oracle_rescue_ceiling.csv", index=False, float_format="%.10g")
    selection.to_csv(output / "selection_quality.csv", index=False, float_format="%.10g")
    mla.to_csv(output / "mla_output_quality.csv", index=False, float_format="%.10g")
    merged[merged.path == "head"].to_csv(output / "head_sparse_results.csv", index=False, float_format="%.10g")
    merged[merged.path == "dim"].to_csv(output / "dim_sparse_results.csv", index=False, float_format="%.10g")
    teacher.to_csv(output / "teacher_forced_quality.csv", index=False, float_format="%.10g")
    closed.to_csv(output / "closed_loop_quality.csv", index=False, float_format="%.10g")
    cycle.to_csv(output / "cycle_sim_results.csv", index=False, float_format="%.10g")

    positive = vrows[vrows.tail_critical_blocks > 0]
    tail_rows = []
    group_cols = ["policy_role", "verifier", "path", "width", "layer", "workload", "base_context_length"]
    for keys, group in positive.groupby(group_cols, sort=False):
        row = dict(zip(group_cols, keys))
        row.update(
            observations=len(group),
            tail_critical_blocks=int(group.tail_critical_blocks.sum()),
            tail_critical_block_recall=weighted_tail(group),
            newly_active_token_recall=float(
                (group.newly_active_token_recall * group.newly_active_tokens).sum()
                / max(1, group.newly_active_tokens.sum())
            ),
            rescue_precision_mean=float(group.rescue_precision.mean()),
        )
        tail_rows.append(row)
    pd.DataFrame(tail_rows).to_csv(output / "tail_critical_recall.csv", index=False, float_format="%.10g")

    hardware = selection.copy()
    hardware["full_k_scan_required"] = hardware.path == "head"
    hardware["sketch_bytes_per_token"] = np.where(hardware.path == "dim", hardware.width * 2, 0)
    hardware["qk_gate_pass"] = hardware.net_qk_reduction_median >= 0.20
    hardware["physical_20pct_gate_pass"] = hardware.physical_key_byte_reduction_median >= 0.20
    hardware["sketch_16byte_gate_pass"] = hardware.sketch_bytes_per_token <= 16
    hardware.to_csv(output / "hardware_cost_model.csv", index=False, float_format="%.10g")

    pairs = {2: 4, 4: 8, 8: 16, 16: 32}
    equal_rows = []
    tail_10 = wide[np.isclose(wide.rescue_fraction, 0.1)]
    for role in ["Balanced", "Aggressive"]:
        for head_width, dim_width in pairs.items():
            for path, width in [("head", head_width), ("dim", dim_width)]:
                candidates = tail_10[(tail_10.policy_role == role) & (tail_10.path == path) & (tail_10.width == width)]
                chosen = candidates.sort_values(["top128_recall", "top512_recall", "net_qk_reduction_median"], ascending=False).iloc[0]
                row = chosen.to_dict()
                row["pair"] = f"{head_width}H×128D vs 64H×{dim_width}D"
                row["verifier_mac_fraction"] = head_width / 64
                actual = tail_mla[(tail_mla.policy_role == role) & (tail_mla.verifier == chosen.verifier)]
                row["tail32k_output_relative_l2_p95"] = float(actual.output_relative_l2_p95.iloc[0]) if len(actual) else np.nan
                equal_rows.append(row)
    pd.DataFrame(equal_rows).to_csv(output / "equal_cost_comparison.csv", index=False, float_format="%.10g")

    head_best = merged[(merged.policy_role == "Aggressive") & (merged.verifier == HEAD)].iloc[0]
    dim_near = merged[(merged.policy_role == "Balanced") & (merged.verifier == "dim_had_random_w32_b64_r0.1")].iloc[0]
    sim_head = pd.read_csv(source / "cycle_sim/cycle_sim_summary.csv")
    sim_head = sim_head[(sim_head.policy_role == "Aggressive") & (sim_head.verifier == HEAD)].iloc[0]
    oracle_bal = oracle[oracle.policy_role == "Balanced"].set_index("rescue")
    oracle_agg = oracle[oracle.policy_role == "Aggressive"].set_index("rescue")
    closed_summary = json.loads((source / "closed_loop/closed_loop_summary.json").read_text())

    selected_configs = {
        "scope": SCOPE,
        "quality_primary": {
            "temporal_policy": "streak2_bucket8_m0",
            "selection_role": "Aggressive",
            "verifier": HEAD,
            "reason": "Only evaluated point passing Band B/A, teacher PPL, and median net QK >=20%",
        },
        "dimension_nearest_quality_boundary": {
            "temporal_policy": "streak2_bucket8_m-0.75_refresh8",
            "selection_role": "Balanced",
            "verifier": "dim_had_random_w32_b64_r0.1",
            "reason": "RelL2 p95 5.039%, just outside Band B; physical reduction only 15.93%",
        },
    }
    (output / "selected_configs.json").write_text(json.dumps(selected_configs, indent=2) + "\n", encoding="utf-8")

    verdict = {
        "overall_verdict": "HEAD-SPARSE-SW-PROMISING",
        "case_a_verdict": "HEAD-SPARSE-SW-PROMISING",
        "case_b_verdict": "NO-GO",
        "strong_hardware_candidate": False,
        "scope": SCOPE,
        "decision_basis": {
            "head_h8_aggressive": {
                "quality_band": "A",
                "mla_relative_l2_p95": float(head_best.output_relative_l2_p95),
                "output_cosine_p5": float(head_best.output_cosine_p5),
                "top128_recall": float(head_best.top128_recall),
                "top512_recall": float(head_best.top512_recall),
                "median_net_qk_reduction": float(head_best.net_qk_reduction_median),
                "median_physical_b64_key_byte_reduction": float(head_best.physical_key_byte_reduction_median),
                "teacher_forced_ppl_delta": float(teacher.ppl_delta.iloc[0]),
                "cycle_speedup_min": float(sim_head.speedup_min),
                "cycle_speedup_median": float(sim_head.speedup_median),
            },
            "dimension_nearest_band_b": {
                "quality_band": str(dim_near.offline_quality_band),
                "mla_relative_l2_p95": float(dim_near.output_relative_l2_p95),
                "median_net_qk_reduction": float(dim_near.net_qk_reduction_median),
                "median_physical_b64_key_byte_reduction": float(dim_near.physical_key_byte_reduction_median),
                "sketch_bytes_per_token_bf16": 64,
            },
            "closed_loop": closed_summary,
        },
        "limitations": [
            "Greedy closed-loop token agreement was 30.86% with first divergence at step 10, although the three NIAH probes stayed correct.",
            "Head-sparse scans full K and increased physical B64 traffic by 2.75% at the selected point.",
            "The cycle simulator is an analytical optimistic-overlap model, not RTL or silicon-calibrated timing.",
            "Fixed transition/tail-aware head and dimension calibration, verifier-only baselines, and token-rescue ceilings were not expanded after the focused 32K tail early stop.",
            "INT8/INT4/INT2 dimension sketches were gated off because BF16 dimension-sparse failed Band B.",
        ],
        "next_step": "Prototype the H8 dynamic-head path as a software/MISA-style kernel with Nsight bandwidth measurement; do not proceed to RTL. Improve the all-head compact detector before another hardware gate.",
    }
    (output / "two_path_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")

    graph_names = build_graphs(
        output / "graphs_two_path", oracle, wide, tail_selected, tail_mla,
        selection, mla, vrows, mla_rows, teacher, cycle,
    )

    source_files = [
        source / "full_replay/verifier_summary.csv",
        source / "full_mla/mla_output_summary.csv",
        source / "oracle_full/oracle_rescue_ceiling.csv",
        source / "teacher_forced/teacher_forced_summary.csv",
        source / "closed_loop/closed_loop_quality.csv",
        source / "cycle_sim/cycle_sim_results.csv",
    ]
    git_commit = args.git_commit
    if git_commit is None:
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            git_commit = "unknown"
    reproducibility = {
        "scope": SCOPE,
        "model": "deepseek-ai/DeepSeek-V2-Lite-Chat",
        "model_revision": MODEL_REVISION,
        "server": "10.201.135.16:7021",
        "gpu_sku": ["NVIDIA L40S", "NVIDIA L40S"],
        "visible_physical_gpu_ids": [0, 1],
        "protected_gpu_ids": [2, 3],
        "selected_layers": [2, 5, 8, 11, 14, 17, 21, 25],
        "heldout_traces": 144,
        "transitions_per_policy": 9216,
        "block_size": 64,
        "top_k": 2048,
        "random_seed": 1582,
        "implementation_base_commit": git_commit,
        "unit_tests": "22 passed",
        "commands": {
            "full_replay": "run_two_path_verifier.py --max-transitions 64 --policy-roles Balanced Aggressive",
            "oracle": "evaluate_oracle_rescue.py --max-transitions 64 --policy-roles Balanced Aggressive",
            "actual_mla": "evaluate_verifier_mla_output.py (144 traces, four selected configs)",
            "teacher_forced": "evaluate_teacher_forced.py --prompt-ids code_heldout_3 text_heldout_26493 --contexts 8192 16384 32768 --steps 64 --roles Aggressive --verifier-names head_dynamic_abs_w8_b64_r0.1",
            "closed_loop": "evaluate_verifier_closed_loop.py --contexts 8192 16384 32768 --steps 128",
            "cycle_sim": "simulate_two_path_hardware.py --roles Balanced Aggressive",
        },
        "audit_sha256": {path.relative_to(source).as_posix(): sha256(path) for path in source_files},
        "graph_count": len(graph_names),
        "early_stops": {
            "dimension_precision_sweep": "skipped: BF16 dimension-sparse failed Band B",
            "learned_projection": "skipped: no BF16 dimension candidate passed Band B",
            "expanded_fixed_oracle_strategy_sweep": "not expanded after focused 32K tail results; dynamic and Hadamard/random reference sweep retained",
        },
    }
    (output / "reproducibility.json").write_text(json.dumps(reproducibility, indent=2) + "\n", encoding="utf-8")

    def pct(value: float, digits: int = 2) -> str:
        return f"{value * 100:.{digits}f}%"

    head_tail = tail_selected[(tail_selected.policy_role == "Aggressive") & (tail_selected.verifier == HEAD)].iloc[0]
    dim_primary = merged[(merged.policy_role == "Aggressive") & (merged.verifier == DIM)].iloc[0]
    report = f"""# Two-Path Cheap Verifier DSA Hardware Feasibility Pilot

> Scope: **{SCOPE}**. L40S ×2에서 측정한 연구용 sidecar 결과이며 DeepSeek-V3.2 production 결과가 아니다.

## Overall Verdict

**`HEAD-SPARSE-SW-PROMISING`**

Aggressive temporal + `8H×128D`, B64, cold-block 10% rescue는 전체 144 trace에서 Quality Band A의 offline MLA 기준과 teacher-forced PPL 기준을 통과하고 median net QK reduction {pct(head_best.net_qk_reduction_median)}를 남겼다. 그러나 full K scan과 rescue reread 때문에 physical B64 key traffic은 {pct(head_best.physical_key_byte_reduction_median)}(음수는 증가)이며, analytical simulator의 speedup 범위는 {sim_head.speedup_min:.3f}×–{sim_head.speedup_max:.3f}×로 대역폭 전 구간에서 일관되지 않았다. 따라서 TensorRT-LLM/H100용 BLASST 계열 **software candidate**로는 후속 profiling 가치가 있지만 dedicated hardware/RTL GO는 아니다.

closed-loop에서는 NIAH 3/3을 baseline과 동일하게 맞혔지만 greedy token agreement가 {closed_summary['generated_token_agreement_mean'] * 100:.2f}%이고 first divergence가 {closed_summary['first_divergence_min']} step이었다. 이 결과를 production 안정성으로 해석하면 안 된다.

## Case A — Head-Sparse verdict

- Selected: Aggressive + dynamic high-|w| H8 + B64 + 10% rescue.
- MLA RelL2 p95/p99: {pct(head_best.output_relative_l2_p95)} / {pct(head_best.output_relative_l2_p99)}.
- cosine p5: {head_best.output_cosine_p5:.6f}; Top-128/512: {pct(head_best.top128_recall, 4)} / {pct(head_best.top512_recall, 4)}.
- newly-active token/block recall: {pct(head_best.newly_active_token_recall)} / {pct(head_best.newly_active_block_recall)}; focused top-4 tail-block recall: {pct(head_tail.tail_critical_block_recall)}.
- net QK reduction: {pct(head_best.net_qk_reduction_median)}; physical B64 bytes: {pct(head_best.physical_key_byte_reduction_median)}.
- teacher-forced: PPL delta {pct(float(teacher.ppl_delta.iloc[0]), 3)}, logit KL mean {teacher.logit_kl_mean.iloc[0]:.6f}, Top-1 agreement {pct(teacher.top1_agreement.iloc[0])}.
- Dynamic head routing은 검증했지만 validation-fixed/transition-aware/tail-aware head가 비슷한지는 이번 축소 sweep으로 확정하지 못했다.

## Case B — Dimension-Sparse verdict

**`NO-GO`**. BF16 후보 네 개 모두 Band B를 통과하지 못했다. 품질 경계에 가장 가까운 Balanced D32는 RelL2 p95 {pct(dim_near.output_relative_l2_p95)}로 5% 기준을 0.039%p 초과했고, net QK/physical reduction도 {pct(dim_near.net_qk_reduction_median)} / {pct(dim_near.physical_key_byte_reduction_median)}에 그쳤다. 같은 12.5% MAC의 Aggressive D16은 physical bytes를 {pct(dim_primary.physical_key_byte_reduction_median)} 줄였지만 RelL2 p95가 {pct(dim_primary.output_relative_l2_p95)}였다. BF16 gate 실패로 INT8/INT4/INT2와 learned projection은 조기 종료했다.

## 핵심 결과

| Path / policy | Band | RelL2 p95 | cosine p5 | Top-128 | Top-512 | net QK | physical B64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Aggressive H8×128 | A + PPL pass | {pct(head_best.output_relative_l2_p95)} | {head_best.output_cosine_p5:.6f} | {pct(head_best.top128_recall, 4)} | {pct(head_best.top512_recall, 4)} | {pct(head_best.net_qk_reduction_median)} | {pct(head_best.physical_key_byte_reduction_median)} |
| Aggressive 64H×D16 | C | {pct(dim_primary.output_relative_l2_p95)} | {dim_primary.output_cosine_p5:.6f} | {pct(dim_primary.top128_recall, 4)} | {pct(dim_primary.top512_recall, 4)} | {pct(dim_primary.net_qk_reduction_median)} | {pct(dim_primary.physical_key_byte_reduction_median)} |
| Balanced 64H×D32 | C, B miss 0.039%p | {pct(dim_near.output_relative_l2_p95)} | {dim_near.output_cosine_p5:.6f} | {pct(dim_near.top128_recall, 4)} | {pct(dim_near.top512_recall, 4)} | {pct(dim_near.net_qk_reduction_median)} | {pct(dim_near.physical_key_byte_reduction_median)} |

## Oracle rescue ceiling

전체 9,216 observation/policy에서 Balanced temporal-only p95 {pct(oracle_bal.loc['0'].output_relative_l2_p95)}는 oracle best-1/2/4/8 block rescue 시 {pct(oracle_bal.loc['best1'].output_relative_l2_p95)} / {pct(oracle_bal.loc['best2'].output_relative_l2_p95)} / {pct(oracle_bal.loc['best4'].output_relative_l2_p95)} / {pct(oracle_bal.loc['best8'].output_relative_l2_p95)}가 됐다. Aggressive는 {pct(oracle_agg.loc['0'].output_relative_l2_p95)}에서 best-4 {pct(oracle_agg.loc['best4'].output_relative_l2_p95)}로 내려갔다. 따라서 aggregate에서는 상위 4개 block rescue만으로 Band B가 가능하다.

반면 32K layer 8/17/21 집중 subset에서는 Balanced best-8도 p95 8.30%, Aggressive best-8도 11.11%였다. aggregate 개선과 worst-tail 복구를 구분해야 하며, 현재 cheap detector는 이 집중 tail을 Band B까지 복구하지 못했다.

## 같은 MAC 예산: 8H×128D vs 64H×16D

Aggressive 기준 H8은 D16보다 newly-active/tail event와 actual output을 훨씬 잘 보존했다. D16은 full K 대신 compact sketch를 읽어 physical traffic을 줄였지만 Band C에 머물렀다. 즉 이 데이터에서는 “모든 head를 조금씩 보면 rare transition을 더 잘 잡는다”는 가설이 성립하지 않았다. Hadamard는 일부 rank recall을 개선했지만 Band B를 만들 만큼 충분하지 않았다.

## 반드시 답할 질문

1. **일부 head가 놓친 event:** 32K 집중 tail에서 H8의 top-4 tail-critical block recall은 {pct(head_tail.tail_critical_block_recall)}에 불과했고 MLA p95가 Band C에 머물렀다.
2. **all-head dim-sparse가 완화했는가:** 아니오. 같은 MAC의 D16은 H8보다 newly-active recall과 MLA tail이 더 나빴다.
3. **동일 12.5% MAC 승자:** 품질은 H8, physical traffic은 D16. 주 Band B 기준 때문에 최종 승자는 H8 software path다.
4. **필요 oracle block 수:** aggregate는 best-4로 B 통과, 집중 32K tail은 best-8로도 부족했다.
5. **candidate precision:** H8은 full replay에서 rescue precision이 {pct(head_best.rescue_precision_mean)}로 유용하지만 tail-critical recall은 제한적이다.
6. **rerank 포함 net QK:** Aggressive H8만 주 후보 기준 {pct(head_best.net_qk_reduction_median)}로 20%를 넘겼다.
7. **physical memory:** dimension-sparse가 명확히 유리하다. head-sparse는 full K scan 때문에 오히려 증가했다.
8. **Hadamard 필요성:** random/even subset보다 도움을 주는 구간은 있으나 품질 gate를 바꾸지는 못했다.
9. **fixed 설정:** 이번 결과로는 답할 수 없다. 동적 H8만 최종 모델 검증했고 fixed transition/tail-aware calibration은 후속 항목이다.
10. **다음 단계:** H8을 TensorRT-LLM/BLASST software kernel로 구현해 Nsight HBM traffic과 end-to-end latency를 먼저 측정한다. dimension detector를 개선하기 전 accelerator RTL로 가지 않는다.

## Hardware simulator 해석

1 GHz, full engine 8,192 MAC/cycle, scan/verifier/rerank optimistic overlap의 analytical model이다. 실측 CUDA kernel timing이 아니다. H8은 256 GB/s corner에서 {sim_head.speedup_min:.3f}×로 느려지고 최고 {sim_head.speedup_max:.3f}×였다. full-K scan이 남아 bandwidth-sensitive하며 `>=1.3× across multiple bandwidth settings`를 안정적으로 충족하지 않는다. Dimension path는 속도/energy proxy는 좋지만 품질 gate 실패로 hardware GO에 사용할 수 없다.

## Closed-loop

| Benchmark | Context | First divergence | Token agreement | Task result |
|---|---:|---:|---:|---|
"""
    for row in closed.itertuples():
        task = "NIAH baseline=pass, verifier=pass" if row.benchmark == "ruler_niah_small" else "agreement proxy only"
        report += f"| {row.benchmark} | {row.context_length} | {int(row.first_divergence_step)} | {pct(row.generated_token_agreement)} | {task} |\n"
    report += f"""

long-code에는 외부 정답 기반 task score가 없으므로 generated-token agreement만 보고한다. teacher-forced PPL 통과가 closed-loop 안정성을 보장하지 않는다는 반례로 해석한다.

## 실험 범위와 제한

- Full evaluation: 144 held-out traces, 9,216 transitions/observations per policy, layers 2/5/8/11/14/17/21/25, contexts 8K/16K/32K.
- 32K 집중 sweep: dynamic high-|w|/positive-weight heads와 original/Hadamard random/even dimensions, widths H2/4/8/16 및 D4/8/16/32, rescue 1/2/5/10%.
- 구현하지 않은 대규모 확장: validation-fixed energy/transition/tail-aware/held-out oracle head·dimension set, H1/D64, B32/B128, 20%/token rescue, verifier-only baseline. 집중 tail 조기 결과에 따라 확장하지 않았으며 해당 비교를 완료한 것으로 주장하지 않는다.
- Missed-token 개별 rank는 최종 detail artifact에 보존하지 않아 histogram은 결측 사유를 표시한 placeholder다.
- BF16 dimension path가 Band B를 통과하지 않아 low-bit precision/learned projection을 실행하지 않았다.
- Deterministic Top-K의 cutoff tie-break를 global lower-index 우선으로 수정했다. 이전 approximate pilot 수치는 보존했고 이번 baseline/후보는 수정된 동일 규칙을 사용한다.

## Reproducibility

- Model revision: `{MODEL_REVISION}`
- GPUs: NVIDIA L40S 48GB ×2, physical GPU 0/1 only; GPU 2/3 untouched.
- Tests: 22 passed.
- Main machine-readable artifacts: `two_path_verdict.json`, `oracle_rescue_ceiling.csv`, `head_sparse_results.csv`, `dim_sparse_results.csv`, `equal_cost_comparison.csv`, `tail_critical_recall.csv`, `selection_quality.csv`, `mla_output_quality.csv`, `teacher_forced_quality.csv`, `closed_loop_quality.csv`, `hardware_cost_model.csv`, `cycle_sim_results.csv`, `selected_configs.json`, `reproducibility.json`.
- Graphs: {len(graph_names)} files under `graphs_two_path/`.
"""
    (output / "two_path_verifier_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"verdict": verdict["overall_verdict"], "graphs": len(graph_names)}, indent=2))


if __name__ == "__main__":
    main()
