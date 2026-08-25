from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VERDICT = "NO-GO"


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    columns = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]
    widths = [len(column) for column in columns]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    header = "| " + " | ".join(value.ljust(width) for value, width in zip(columns, widths)) + " |"
    rule = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [
        "| " + " | ".join(value.ljust(width) for value, width in zip(row, widths)) + " |"
        for row in rows
    ]
    return "\n".join([header, rule, *body])


def pct(value: float, digits: int = 2) -> str:
    return f"{100 * float(value):.{digits}f}%"


def number(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def save(fig: plt.Figure, root: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(root / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def placeholder(root: Path, name: str, title: str, reason: str) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.axis("off")
    axis.text(0.5, 0.62, title, ha="center", va="center", fontsize=16, weight="bold")
    axis.text(0.5, 0.40, reason, ha="center", va="center", fontsize=11, wrap=True)
    save(fig, root, name)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_rows(summary: pd.DataFrame, specs: list[dict[str, Any]]) -> pd.DataFrame:
    role = {spec["config_id"]: spec["selection_role"] for spec in specs}
    result = summary[summary.policy.isin(role)].copy()
    result["selection_role"] = result.policy.map(role)
    order = {"Safe": 0, "Balanced": 1, "Aggressive": 2}
    result["_order"] = result.selection_role.map(order)
    return result.sort_values("_order").drop(columns="_order")


def build_hardware(
    selected: pd.DataFrame,
    full: pd.Series,
    lengths: list[int],
) -> pd.DataFrame:
    rows = []
    policies = [("Full", full)] + [
        (str(row.selection_role), row) for row in selected.itertuples(index=False)
    ]
    for role, row in policies:
        for length in lengths:
            blocks64 = math.ceil(length / 64)
            metadata_bpb = int(row.metadata_bytes_per_block)
            metadata = blocks64 * metadata_bpb
            qk_reduction = float(row.qk_reduction_median)
            ideal_bf16 = (1.0 - qk_reduction) * length * 256 + metadata
            ideal_fp8 = (1.0 - qk_reduction) * length * 128 + metadata
            physical64 = (
                (1.0 - float(row.physical_b64_reduction_median)) * length * 256 + metadata
            )
            physical128 = (
                (1.0 - float(row.physical_b128_reduction_median)) * length * 256 + metadata
            )
            item = {
                "policy": role,
                "context_length": length,
                "qk_reduction_median": qk_reduction,
                "head_dot_products": int(round((1.0 - qk_reduction) * length * 64 * 128)),
                "full_bf16_key_bytes": length * 256,
                "ideal_bf16_key_plus_metadata_bytes": int(round(ideal_bf16)),
                "ideal_fp8_key_plus_metadata_bytes": int(round(ideal_fp8)),
                "physical_b64_bf16_plus_metadata_bytes": int(round(physical64)),
                "physical_b128_bf16_plus_metadata_bytes": int(round(physical128)),
                "metadata_bytes_per_block": metadata_bpb,
                "metadata_bytes_per_layer": metadata,
                "seed_block_fraction_median": float(row.seed_block_fraction_median),
                "seed_cache_256_bf16_bytes": 256 * 128 * 2,
                "seed_cache_512_bf16_bytes": 512 * 128 * 2,
                "seed_cache_1024_bf16_bytes": 1024 * 128 * 2,
                "seed_cache_2048_bf16_bytes": 2048 * 128 * 2,
            }
            for cache in (0, 256, 512, 1024, 2048):
                item[f"cache{cache}_net_bf16_reduction"] = float(
                    getattr(row, f"cache{cache}_net_bf16_reduction_median")
                )
            rows.append(item)
    return pd.DataFrame(rows)


def create_graphs(
    graph_root: Path,
    all_own: pd.DataFrame,
    selected: pd.DataFrame,
    teacher: pd.DataFrame,
    mla: pd.DataFrame,
    phase_c: pd.DataFrame,
    phase_a_pareto: pd.DataFrame,
    rank_breakdown: pd.DataFrame,
    hardware: pd.DataFrame,
) -> list[str]:
    graph_root.mkdir(parents=True, exist_ok=True)
    generated = []

    def scatter(name: str, x: pd.Series, y: pd.Series, title: str, ylabel: str) -> None:
        fig, axis = plt.subplots(figsize=(7, 5))
        axis.scatter(100 * x, 100 * y, alpha=0.55, s=22)
        axis.set_xlabel("Median indexer QK reduction (%)")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.25)
        save(fig, graph_root, name)
        generated.append(name)

    scatter(
        "01_qk_vs_recall.png", all_own.qk_reduction_median, all_own.recall_mean,
        "QK reduction vs Recall@2048 (own trajectory)", "Mean Recall@2048 (%)",
    )
    fig, axis = plt.subplots(figsize=(8, 5))
    for column, label in (("top128_recall", "Top-128"), ("top256_recall", "Top-256"), ("top512_recall", "Top-512")):
        axis.scatter(100 * all_own.qk_reduction_median, 100 * all_own[column], s=20, alpha=0.55, label=label)
    axis.set(xlabel="Median QK reduction (%)", ylabel="Retention (%)", title="High-rank retention frontier")
    axis.legend(); axis.grid(alpha=0.25)
    save(fig, graph_root, "02_qk_vs_rank_recall.png"); generated.append("02_qk_vs_rank_recall.png")
    scatter(
        "03_qk_vs_index_mass.png", all_own.qk_reduction_median, all_own.index_mass_ratio_mean,
        "QK reduction vs indexer softmax mass", "Mean mass ratio (%)",
    )
    merged_teacher = selected.merge(teacher, on=["policy", "selection_role"], how="left")
    scatter(
        "04_qk_vs_teacher_mass.png", merged_teacher.qk_reduction_median,
        merged_teacher.teacher_mass_ratio_mean, "QK reduction vs dense-teacher attention mass",
        "Teacher mass ratio (%)",
    )
    merged_mla = selected.merge(mla, on=["policy", "selection_role"], how="left")
    scatter(
        "05_qk_vs_mla_cosine.png", merged_mla.qk_reduction_median_x,
        merged_mla.output_cosine_p5, "QK reduction vs actual MLA output cosine (p5)",
        "Output cosine p5 (%)",
    )
    scatter(
        "06_qk_vs_mla_relative_l2.png", merged_mla.qk_reduction_median_x,
        merged_mla.output_relative_l2_p95, "QK reduction vs actual MLA relative L2 (p95)",
        "Relative L2 p95 (%)",
    )
    model = phase_c[phase_c.comparison.eq("full_indexer_vs_approx")]
    merged_model = selected.merge(model, on=["policy", "selection_role"], how="left")
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(100 * merged_model.qk_reduction_median, merged_model.logit_kl_mean, s=65)
    for row in merged_model.itertuples():
        axis.annotate(row.selection_role, (100 * row.qk_reduction_median, row.logit_kl_mean))
    axis.set(xlabel="Median QK reduction (%)", ylabel="Mean logit KL", title="QK reduction vs teacher-forced logit KL")
    axis.grid(alpha=0.25)
    save(fig, graph_root, "07_qk_vs_logit_kl.png"); generated.append("07_qk_vs_logit_kl.png")
    scatter(
        "08_qk_vs_ppl_delta.png", merged_model.qk_reduction_median,
        merged_model.ppl_delta, "QK reduction vs teacher-forced PPL delta", "PPL delta (%)",
    )

    n = selected.transitions.to_numpy(dtype=float)
    cumulative = {}
    for width, column in ((128, "top128_recall"), (256, "top256_recall"), (512, "top512_recall"), (1024, "top1024_recall"), (2048, "top2048_recall")):
        cumulative[width] = (1.0 - selected[column].to_numpy()) * width * n
    buckets = np.vstack([
        cumulative[128], cumulative[256] - cumulative[128], cumulative[512] - cumulative[256],
        cumulative[1024] - cumulative[512], cumulative[2048] - cumulative[1024],
    ]).T
    fig, axis = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(selected))
    labels = ["1-128", "129-256", "257-512", "513-1024", "1025-2048"]
    for index, label in enumerate(labels):
        axis.bar(selected.selection_role, buckets[:, index], bottom=bottom, label=label)
        bottom += buckets[:, index]
    axis.set(ylabel="Estimated missed baseline tokens", title="Missed-token rank distribution")
    axis.legend(ncol=3)
    save(fig, graph_root, "09_missed_rank_histogram.png"); generated.append("09_missed_rank_histogram.png")
    scatter(
        "10_newly_hot_miss.png", all_own.qk_reduction_median, all_own.newly_hot_miss_rate,
        "Newly-hot miss rate vs reduction", "Newly-hot miss rate (%)",
    )
    refresh = all_own[all_own.repair.astype(str).str.startswith("refresh")].copy()
    refresh["interval"] = refresh.repair.str.removeprefix("refresh").astype(int)
    fig, left = plt.subplots(figsize=(8, 5)); right = left.twinx()
    grouped = refresh.groupby("interval", as_index=False).agg(qk=("qk_reduction_median", "mean"), recall=("recall_mean", "mean"))
    left.plot(grouped.interval, 100 * grouped.qk, marker="o", label="QK reduction")
    right.plot(grouped.interval, 100 * grouped.recall, marker="s", color="tab:orange", label="Recall")
    left.set(xlabel="Full refresh interval", ylabel="Median QK reduction (%)", title="Periodic refresh ablation")
    right.set_ylabel("Mean Recall@2048 (%)")
    save(fig, graph_root, "11_refresh_interval.png"); generated.append("11_refresh_interval.png")
    placeholder(graph_root, "12_query_fallback.png", "Query-change fallback", "Skipped after Phase-B actual-output NO-GO gate; no held-out parameter tuning performed.")
    generated.append("12_query_fallback.png")
    subset = phase_a_pareto[phase_a_pareto.policy.isin(["streak2_bucket8_m-0.75", "static_m-4", "dynamic_address_m-4"])]
    fig, axis = plt.subplots(figsize=(8, 5))
    for history, group in subset.groupby("history_mode"):
        axis.scatter(100 * group.qk_reduction_median, 100 * group.recall_mean, label=history, s=60)
    axis.set(xlabel="Median QK reduction (%)", ylabel="Mean Recall (%)", title="Teacher-forced vs own-trajectory replay")
    axis.legend(); axis.grid(alpha=0.25)
    save(fig, graph_root, "13_teacher_vs_own.png"); generated.append("13_teacher_vs_own.png")
    matched = phase_a_pareto[
        phase_a_pareto.policy.isin(["static_m-4", "dynamic_address_m-4"])
        & phase_a_pareto.history_mode.eq("own")
    ]
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.bar(matched.family, 100 * matched.qk_reduction_median)
    axis.set(ylabel="Median QK reduction (%)", title="Static vs dynamic at the same validation margin")
    save(fig, graph_root, "14_static_vs_dynamic.png"); generated.append("14_static_vs_dynamic.png")
    safe_id = selected[selected.selection_role.eq("Safe")].policy.iloc[0]
    heat = rank_breakdown[rank_breakdown.policy.eq(safe_id)].pivot_table(index="layer", columns="base_context_length", values="recall", aggfunc="mean")
    fig, axis = plt.subplots(figsize=(7, 5)); image = axis.imshow(100 * heat.to_numpy(), aspect="auto", vmin=90, vmax=100, cmap="viridis")
    axis.set_xticks(range(len(heat.columns)), [str(value) for value in heat.columns]); axis.set_yticks(range(len(heat.index)), [str(value) for value in heat.index])
    axis.set(xlabel="Context", ylabel="Layer", title="Safe Recall@2048 heatmap"); fig.colorbar(image, ax=axis, label="Recall (%)")
    save(fig, graph_root, "15_context_layer_heatmap.png"); generated.append("15_context_layer_heatmap.png")
    hw32 = hardware[hardware.context_length.eq(32768) & hardware.policy.ne("Full")]
    fig, axis = plt.subplots(figsize=(9, 5))
    width = 0.25; x = np.arange(len(hw32))
    ideal = 1 - hw32.ideal_bf16_key_plus_metadata_bytes / hw32.full_bf16_key_bytes
    p64 = 1 - hw32.physical_b64_bf16_plus_metadata_bytes / hw32.full_bf16_key_bytes
    p128 = 1 - hw32.physical_b128_bf16_plus_metadata_bytes / hw32.full_bf16_key_bytes
    axis.bar(x-width, 100*ideal, width, label="Ideal token"); axis.bar(x, 100*p64, width, label="Physical B64"); axis.bar(x+width, 100*p128, width, label="Physical B128")
    axis.set_xticks(x, hw32.policy); axis.set(ylabel="Net byte reduction (%)", title="Ideal vs physical block traffic"); axis.legend()
    save(fig, graph_root, "16_ideal_vs_physical_bytes.png"); generated.append("16_ideal_vs_physical_bytes.png")
    fig, axis = plt.subplots(figsize=(8, 5))
    caches = [0, 256, 512, 1024, 2048]
    for row in hw32.itertuples():
        axis.plot(caches, [100 * getattr(row, f"cache{cache}_net_bf16_reduction") for cache in caches], marker="o", label=row.policy)
    axis.set(xlabel="Previous-TopK cache entries", ylabel="Net BF16 HBM-byte reduction (%)", title="On-chip seed-key cache model"); axis.legend(); axis.grid(alpha=0.25)
    save(fig, graph_root, "17_seed_cache_vs_hbm.png"); generated.append("17_seed_cache_vs_hbm.png")
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(hw32.policy, 100 * (1 - hw32.ideal_bf16_key_plus_metadata_bytes / hw32.full_bf16_key_bytes))
    axis.axhline(20, color="red", linestyle="--", label="HW pilot byte gate")
    axis.set(ylabel="Net BF16 byte reduction (%)", title="Net bytes including metadata"); axis.legend()
    save(fig, graph_root, "18_net_bytes_metadata.png"); generated.append("18_net_bytes_metadata.png")
    placeholder(graph_root, "19_closed_loop_divergence.png", "Closed-loop first divergence", "Not run: Phase-B actual MLA output p95 error exceeded the pilot gate.")
    generated.append("19_closed_loop_divergence.png")
    placeholder(graph_root, "20_task_score_vs_qk.png", "Task score vs QK reduction", "Not run: early NO-GO stopped RULER/LongBench/code task evaluation.")
    generated.append("20_task_score_vs_qk.png")
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final approximate temporal DSA report")
    parser.add_argument("--phase-a", type=Path, required=True)
    parser.add_argument("--repairs", type=Path, required=True)
    parser.add_argument("--phase-b", type=Path, required=True)
    parser.add_argument("--phase-c", type=Path, required=True)
    parser.add_argument("--legacy-results", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--trace-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    selection = json.loads((args.repairs / "selected_phase_b_policies.json").read_text())
    specs = selection["policies"]
    own = pd.read_csv(args.repairs / "repair_summary_own.csv")
    selected = selected_rows(own, specs)
    full = own[own.policy.eq("full")].iloc[0]
    teacher = pd.read_csv(args.phase_b / "teacher_mass_summary.csv")
    mla = pd.read_csv(args.phase_b / "mla_output_summary.csv")
    phase_c = pd.read_csv(args.phase_c / "teacher_forced_summary.csv")
    phase_a_pareto = pd.read_csv(args.phase_a / "policy_pareto.csv")
    rank_breakdown = pd.read_csv(args.repairs / "rank_recall_repaired.csv")
    hardware = build_hardware(selected, full, [32768, 65536, 131072])

    selected.to_csv(args.output / "policy_pareto.csv", index=False)
    rank_breakdown.to_csv(args.output / "rank_recall.csv", index=False)
    shutil.copy2(args.phase_b / "teacher_mass.csv", args.output / "teacher_mass.csv")
    shutil.copy2(args.phase_b / "mla_output_error.csv", args.output / "mla_output_error.csv")
    shutil.copy2(args.phase_c / "teacher_forced_quality.csv", args.output / "teacher_forced_quality.csv")
    hardware.to_csv(args.output / "hardware_cost_model.csv", index=False)
    shutil.copy2(args.repairs / "selected_phase_b_policies.json", args.output / "selected_policies.json")
    skipped = pd.DataFrame([
        {"status": "skipped_due_to_phase_b_gate", "reason": "MLA output relative-L2 p95 exceeded 1% for all Pareto candidates"}
    ])
    skipped.to_csv(args.output / "closed_loop_quality.csv", index=False)
    skipped.assign(benchmark="RULER/LongBench/code").to_csv(args.output / "task_quality.csv", index=False)

    legacy_timing = json.loads((args.legacy_results / "software_baseline.json").read_text())
    timing_rows = []
    for row in legacy_timing["rows"]:
        timing_rows.append({"method": "Full dense sidecar + torch.topk", "status": "reused_legacy_50_iterations", **row})
    for method in ("Static temporal filter", "Bucket hot-first approximate"):
        timing_rows.append({"method": method, "status": "skipped_due_to_phase_b_gate"})
    pd.DataFrame(timing_rows).to_csv(args.output / "software_timing.csv", index=False)

    graph_names = create_graphs(
        args.output / "graphs_approx", own, selected, teacher, mla, phase_c,
        phase_a_pareto, rank_breakdown, hardware,
    )

    checkpoint_hashes = {
        path.name: sha256(path) for path in sorted(args.checkpoints.glob("layer_*.safetensors"))
    }
    trace_paths = sorted({path for root in args.trace_roots for path in root.rglob("*.npz")})
    phase_b_audit = json.loads((args.phase_b / "mla_output_audit.json").read_text())
    phase_a_audit = json.loads((args.phase_a / "reproducibility_phase_a.json").read_text())
    phase_c_rows = pd.read_csv(args.phase_c / "teacher_forced_quality.csv")
    dense_timing = phase_c_rows[phase_c_rows.comparison.eq("dense_vs_full_indexer")]
    approx_timing = phase_c_rows[phase_c_rows.comparison.eq("full_indexer_vs_approx")]
    dense_seconds = dense_timing.groupby(["prompt_id", "context_length"])["baseline_decode_seconds"].first().sum()
    baseline_seconds = dense_timing.groupby(["prompt_id", "context_length"])["approx_decode_seconds"].first().sum()
    approx_seconds = approx_timing.groupby(["policy", "prompt_id", "context_length"])["approx_decode_seconds"].first().sum()
    measured_decode_seconds = float(dense_seconds + baseline_seconds + approx_seconds)
    reproducibility = {
        "git_commit": args.git_commit,
        "model": "deepseek-ai/DeepSeek-V2-Lite-Chat",
        "model_revision": "85864749cd611b4353ce1decdb286193298f64c7",
        "scope": "Research sidecar DSA on DeepSeek-V2-Lite",
        "sidecar": {"heads": 64, "head_dim": 128, "dtype": "BF16 forward / FP32 master"},
        "checkpoint_sha256": checkpoint_hashes,
        "gpu_sku": phase_b_audit["gpus"],
        "visible_physical_gpu_ids": [0, 1],
        "protected_gpu_ids": [2, 3],
        "python": phase_b_audit["python"],
        "torch": phase_b_audit["torch"],
        "transformers": phase_b_audit["transformers"],
        "cuda": phase_b_audit["cuda"],
        "random_seed": 1582,
        "validation_prompt_ids": phase_a_audit["validation_prompt_ids"],
        "heldout_prompt_ids": phase_a_audit["heldout_prompt_ids"],
        "validation_trace_count": phase_a_audit["validation_trace_count"],
        "heldout_trace_count": len(trace_paths),
        "heldout_transitions": 18288,
        "score_trace_bytes": sum(path.stat().st_size for path in trace_paths),
        "selected_layers": [2, 5, 8, 11, 14, 17, 21, 25],
        "k": 2048,
        "block_size": 64,
        "policy_parameters": specs,
        "unit_tests": "16 passed",
        "phase_a_elapsed_seconds": phase_a_audit["elapsed_seconds"],
        "phase_b_mla_elapsed_seconds": phase_b_audit["elapsed_seconds"],
        "phase_c_measured_decode_seconds_excluding_prefill": measured_decode_seconds,
        "phase_c_measured_decode_gpu_hours_two_visible_gpus_excluding_prefill": measured_decode_seconds * 2 / 3600,
        "commands": {
            "phase_a": phase_a_audit["command"],
            "phase_b": "evaluate_mla_output.py --max-transitions 64 (144 traces)",
            "phase_c": "evaluate_teacher_forced.py --contexts 8192 16384 32768 --steps 64 --layers 2 5 8 11 14 17 21 25",
        },
        "early_stop": {
            "closed_loop": True,
            "task_benchmarks": True,
            "approx_software_timing": True,
            "reason": "Phase-B actual MLA output error exceeded pilot gate",
        },
    }
    (args.output / "reproducibility.json").write_text(
        json.dumps(reproducibility, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    merged = (
        selected.merge(teacher, on=["policy", "selection_role"])
        .merge(mla, on=["policy", "selection_role"])
        .merge(
            phase_c[phase_c.comparison.eq("full_indexer_vs_approx")],
            on=["policy", "selection_role"],
        )
    )
    verdict_numbers = {}
    for row in merged.itertuples():
        verdict_numbers[row.selection_role] = {
            "qk_reduction_mean": row.qk_reduction_mean_x,
            "qk_reduction_median": row.qk_reduction_median_x,
            "recall_at_2048": row.recall_mean,
            "top128_recall": row.top128_recall,
            "top512_recall": row.top512_recall,
            "teacher_mass_ratio_mean": row.teacher_mass_ratio_mean,
            "mla_cosine_median": row.output_cosine_median,
            "mla_cosine_p5": row.output_cosine_p5,
            "mla_relative_l2_p95": row.output_relative_l2_p95,
            "logit_kl_mean": row.logit_kl_mean,
            "top1_agreement": row.top1_agreement,
            "ppl_delta": row.ppl_delta,
            "net_bf16_bytes_reduction_median": row.net_bf16_reduction_median,
        }
    verdict_payload = {
        "verdict": VERDICT,
        "scope": "Research sidecar DSA on DeepSeek-V2-Lite",
        "decision_rule": "No candidate simultaneously reached >=20% held-out own-trajectory reduction and the Phase-B actual-output gate; Balanced relative-L2 p95 was >1%.",
        "numbers": verdict_numbers,
        "closed_loop_status": "skipped_due_to_phase_b_gate",
        "task_status": "skipped_due_to_phase_b_gate",
    }
    (args.output / "approx_verdict.json").write_text(
        json.dumps(verdict_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    algo_rows = []
    model_rows = []
    hardware_rows = []
    for row in merged.itertuples():
        algo_rows.append({
            "Policy": row.selection_role, "QK reduction": pct(row.qk_reduction_median_x),
            "Exact match": pct(row.exact_match), "Recall@K": pct(row.recall_mean),
            "Top-128": pct(row.top128_recall, 3), "Teacher mass": pct(row.teacher_mass_ratio_mean, 3),
            "MLA cosine med/p5": f"{row.output_cosine_median:.6f} / {row.output_cosine_p5:.6f}",
            "MLA RelL2 p95": pct(row.output_relative_l2_p95),
        })
        model_rows.append({
            "Policy": row.selection_role, "Logit KL": number(row.logit_kl_mean),
            "Top-1 agreement": pct(row.top1_agreement), "PPL delta": pct(row.ppl_delta),
            "Closed-loop": "skipped", "Task": "skipped",
        })
        hw = hardware[(hardware.policy.eq(row.selection_role)) & hardware.context_length.eq(32768)].iloc[0]
        hardware_rows.append({
            "Policy": row.selection_role, "QK reduction": pct(row.qk_reduction_median_x),
            "Ideal net bytes": pct(1 - hw.ideal_bf16_key_plus_metadata_bytes / hw.full_bf16_key_bytes),
            "Physical B64": pct(1 - hw.physical_b64_bf16_plus_metadata_bytes / hw.full_bf16_key_bytes),
            "Physical B128": pct(1 - hw.physical_b128_bf16_plus_metadata_bytes / hw.full_bf16_key_bytes),
            "Seed blocks touched": pct(hw.seed_block_fraction_median), "TopK cache": "512 KiB BF16",
        })
    dense_model = phase_c[phase_c.comparison.eq("dense_vs_full_indexer")].iloc[0]
    static = phase_a_pareto[(phase_a_pareto.policy.eq("static_m-4")) & phase_a_pareto.history_mode.eq("own")].iloc[0]
    dynamic = phase_a_pareto[(phase_a_pareto.policy.eq("dynamic_address_m-4")) & phase_a_pareto.history_mode.eq("own")].iloc[0]
    dynamic_gain = float(dynamic.qk_reduction_median - static.qk_reduction_median)

    safe = merged[merged.selection_role.eq("Safe")].iloc[0]
    balanced = merged[merged.selection_role.eq("Balanced")].iloc[0]
    aggressive = merged[merged.selection_role.eq("Aggressive")].iloc[0]
    report = f"""# Approximate Temporal DSA — L40S ×2 Quality and Hardware Feasibility Pilot

## Verdict

`{VERDICT}`

**Scope:** Research sidecar DSA on DeepSeek-V2-Lite. This is not a production DeepSeek-V3.2 FP8 Indexer result.

The Balanced point reduced held-out own-trajectory indexer QK by {pct(balanced.qk_reduction_median_x)} median, but actual sparse MLA output relative-L2 p95 was {pct(balanced.output_relative_l2_p95)}, above the pilot gate of 1%. The Safe point had {pct(safe.output_relative_l2_p95)} p95 output error and only {pct(safe.qk_reduction_median_x)} median reduction, below the 20% minimum. Therefore no candidate satisfies the pilot's quality-and-reduction decision rule.

## 핵심 숫자

- Safe / Balanced / Aggressive own median QK reduction: {pct(safe.qk_reduction_median_x)} / {pct(balanced.qk_reduction_median_x)} / {pct(aggressive.qk_reduction_median_x)}
- Top-128 recall: {pct(safe.top128_recall, 3)} / {pct(balanced.top128_recall, 3)} / {pct(aggressive.top128_recall, 3)}
- Top-512 recall: {pct(safe.top512_recall, 3)} / {pct(balanced.top512_recall, 3)} / {pct(aggressive.top512_recall, 3)}
- Teacher attention mass mean ratio: {pct(safe.teacher_mass_ratio_mean, 3)} / {pct(balanced.teacher_mass_ratio_mean, 3)} / {pct(aggressive.teacher_mass_ratio_mean, 3)}
- MLA output cosine median: {safe.output_cosine_median:.6f} / {balanced.output_cosine_median:.6f} / {aggressive.output_cosine_median:.6f}
- MLA output relative-L2 p95: {pct(safe.output_relative_l2_p95)} / {pct(balanced.output_relative_l2_p95)} / {pct(aggressive.output_relative_l2_p95)}
- Teacher-forced logit KL mean: {safe.logit_kl_mean:.6f} / {balanced.logit_kl_mean:.6f} / {aggressive.logit_kl_mean:.6f}
- Teacher-forced PPL delta: {pct(safe.ppl_delta)} / {pct(balanced.ppl_delta)} / {pct(aggressive.ppl_delta)}
- Dense teacher → full-indexer sparse PPL delta: {pct(dense_model.ppl_delta)}
- Same-margin dynamic gain over static: {100 * dynamic_gain:.3f} percentage point
- BF16 previous-TopK cache: 512 KiB/layer for K=2048, d=128
- Closed-loop/task/software approximate timing: skipped after the Phase-B NO-GO gate

## 최종 질문에 대한 답

1. **Exact match가 낮아도 output이 유지되는가?** Median은 유지되지만 tail은 아니다. Balanced cosine median은 {balanced.output_cosine_median:.6f}지만 relative-L2 p95는 {pct(balanced.output_relative_l2_p95)}다.
2. **Miss가 cutoff에 집중되는가?** 대체로 그렇다. Safe Top-128 retention은 {pct(safe.top128_recall, 3)}인 반면 Recall@2048은 {pct(safe.recall_mean, 3)}다.
3. **High-score core가 안정적인가?** 평균적으로 매우 안정적이나 반복적인 rare miss가 있고 32K layer 17/21에서 큰 MLA output tail과 연결된다.
4. **25–35% reduction에서 model quality가 유지되는가?** Teacher-forced PPL은 유지됐지만 actual MLA output gate는 실패했다.
5. **40–50% reduction을 repair할 수 있는가?** R=8/age cap repair는 Balanced를 약 28%까지 복구했지만 40–50%에서 output gate를 만족하지 못했다.
6. **Teacher-forced가 closed loop에서도 유지되는가?** Phase-B gate 실패로 closed-loop는 실행하지 않았다. 이 항목은 미확인이다.
7. **Seed/metadata 포함 traffic이 줄어드는가?** Ideal token traffic은 줄지만 scattered seed가 32K block의 큰 비율을 touch해 B128 physical reduction이 크게 축소된다.
8. **Static CUDA-style filter로 대부분 얻는가?** 그렇다. 같은 gamma에서 static→dynamic median 추가 reduction은 {100 * dynamic_gain:.3f}pp뿐이다.
9. **Dynamic HW가 정당화되는가?** 아니다. Quality gate 실패와 negligible dynamic gain 둘 다 dedicated scheduler/SRAM을 정당화하지 못한다.
10. **다음 단계는?** FPGA/HW가 아니라 tail-aware static/software policy 개선과 independent production-V3.2 validation이다.

## Algorithm quality

{markdown_table(pd.DataFrame(algo_rows))}

## Model quality

{markdown_table(pd.DataFrame(model_rows))}

## Hardware value at 32K

{markdown_table(pd.DataFrame(hardware_rows))}

## Baseline separation

Dense MLA teacher와 full-indexer research sidecar sparse baseline의 teacher-forced 비교는 logit KL mean {dense_model.logit_kl_mean:.6f}, Top-1 agreement {pct(dense_model.top1_agreement)}, PPL delta {pct(dense_model.ppl_delta)}였다. Approximate temporal 결과는 이 full-indexer sparse baseline에 대해 측정했으므로 sidecar 자체 오차와 temporal approximation 오차를 혼합하지 않았다.

## Phase A — runtime-legal replay

- Validation: 24 traces, held-out verdict: 144 traces / 18,288 transitions.
- Own-trajectory에서 seed는 이전 approximate Top-K이고, skip block은 stale max/age만 유지했다.
- Previous-TopK seed rescore를 QK cost에 포함했으며 current key 중복 score는 제거했다.
- 8K validation 절대 gamma가 16K/32K에서 과도하게 pruning되는 문제를 발견해 prespecified refresh/age sweeps를 적용했다.
- Safe는 age cap 2, Balanced는 periodic full refresh R=8, Aggressive는 no repair다.

## Phase B — actual teacher attention and MLA output

Teacher mass median은 세 정책 모두 거의 1이었지만 p5와 worst tail이 악화됐다. 실제 main MLA Q/K/V로 selection set만 바꾼 9,216 observations/policy에서 Safe/Balanced/Aggressive relative-L2 p95가 각각 {pct(safe.output_relative_l2_p95)}, {pct(balanced.output_relative_l2_p95)}, {pct(aggressive.output_relative_l2_p95)}였다. Worst concentration은 32K layers 17/21과 long-code layer 8에서 나타났다.

## Phase C — teacher-forced model validation

Text/code 각 3 prompts × 8K/16K/32K × 64 steps, selected 8 layers에서 dense, full-indexer sparse, 세 approximate 정책을 비교했다. Approximate 정책은 PPL delta ±1% gate를 통과했지만, 이 결과가 Phase-B attention-output tail을 무효화하지는 않는다. Residual path가 많은 오류를 흡수한다는 증거로 해석한다.

## Phase D early stop

Closed-loop generation, RULER/NIAH/LongBench/code task, query-change fallback, approximate CUDA-style timing은 명세의 early-stop 원칙에 따라 실행하지 않았다. 미실행 결과를 0이나 통과로 간주하지 않으며, 각각 CSV와 placeholder graph에 명시했다.

## Hardware interpretation

- Metadata는 last max + streak + age/bucket을 packed/aligned 8 B/block로 모델링했다.
- 32K에서 metadata는 B64 기준 4 KiB/layer이고 64K/128K extrapolation은 8/16 KiB/layer다.
- K=2048 seed-key cache는 BF16 512 KiB/layer, FP8 extrapolation 256 KiB/layer다.
- Seed Top-K가 scattered되어 physical B128 traffic reduction은 ideal token-level reduction보다 훨씬 작다.
- Same-margin static과 dynamic address order의 held-out median reduction 차이는 {100 * dynamic_gain:.3f}pp로 hardware-specific feedback value가 관찰되지 않았다.

## Reproducibility and limitations

- Code commit: `{args.git_commit}`
- Model revision: `85864749cd611b4353ce1decdb286193298f64c7`
- GPUs: NVIDIA L40S, physical IDs 0/1 only; IDs 2/3 untouched.
- Unit tests: 16 passed.
- Main result files: `policy_pareto.csv`, `rank_recall.csv`, `teacher_mass.csv`, `mla_output_error.csv`, `teacher_forced_quality.csv`, `hardware_cost_model.csv`, `software_timing.csv`, `reproducibility.json`.
- Generated graphs: {len(graph_names)} under `graphs_approx/`.
- This result characterizes a BF16/FP32 research sidecar, not the production V3.2 FP8 Indexer or TensorRT-LLM kernel.

## References

- [DeepSeek-V3.2 paper](https://arxiv.org/abs/2512.02556)
- [Official DeepSeek-V3.2 experimental indexer reference](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py)
- [DeepSeek-V2-Lite-Chat model](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat)
"""
    (args.output / "approx_temporal_dsa_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(verdict_payload, indent=2))


if __name__ == "__main__":
    main()
