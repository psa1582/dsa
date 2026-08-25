from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, output: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(output / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def create_required_graphs(
    temporal: pd.DataFrame,
    blocks: pd.DataFrame,
    replay: pd.DataFrame,
    output: Path,
) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    names: list[str] = []

    fig, ax = plt.subplots()
    summary = temporal.groupby("k")["topk_overlap"].agg(["mean", "median"])
    ax.plot(summary.index, summary["mean"], marker="o", label="mean")
    ax.plot(summary.index, summary["median"], marker="s", label="median")
    ax.set(xlabel="K", ylabel="Adjacent Top-K overlap", xscale="log", ylim=(0, 1))
    ax.legend()
    names.append("01_adjacent_topk_overlap.png"); _save(fig, output, names[-1])

    fig, ax = plt.subplots()
    pivot = temporal.pivot_table(index="layer", columns="k", values="topk_overlap", aggfunc="median")
    image = ax.imshow(pivot.values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set(xticks=range(len(pivot.columns)), xticklabels=pivot.columns, yticks=range(len(pivot.index)), yticklabels=pivot.index, xlabel="K", ylabel="Layer")
    fig.colorbar(image, ax=ax, label="Median overlap")
    names.append("02_overlap_layer_heatmap.png"); _save(fig, output, names[-1])

    fig, ax = plt.subplots()
    for workload, group in temporal.groupby("workload"):
        rank = group.groupby("k")["topk_overlap"].quantile([0.05, 0.5, 0.95]).unstack()
        ax.plot(rank.index, rank[0.5], marker="o", label=str(workload))
        ax.fill_between(rank.index, rank[0.05], rank[0.95], alpha=0.15)
    ax.set(xlabel="K", ylabel="Rank-set stability (p5/p50/p95)", xscale="log", ylim=(0, 1))
    ax.legend()
    names.append("03_rank_stability_by_workload.png"); _save(fig, output, names[-1])

    fig, ax = plt.subplots()
    correlations = temporal.drop_duplicates(["layer", "workload", "prompt_id", "step"])
    ax.hist(correlations["score_pearson"].dropna(), bins=30, alpha=0.6, label="Pearson")
    ax.hist(correlations["score_spearman"].dropna(), bins=30, alpha=0.6, label="Spearman")
    ax.set(xlabel="Adjacent score correlation", ylabel="Transitions", xlim=(-1, 1)); ax.legend()
    names.append("04_score_correlation.png"); _save(fig, output, names[-1])

    fig, ax = plt.subplots()
    ax.hist(correlations["delta_p99_abs"].dropna(), bins=30)
    ax.set(xlabel="p99 |score delta|", ylabel="Transitions")
    names.append("05_score_delta_distribution.png"); _save(fig, output, names[-1])

    fig, ax = plt.subplots()
    block_unique = blocks.drop_duplicates(
        ["layer", "workload", "prompt_id", "step", "block_size"]
    )
    block_summary = block_unique.groupby("block_size")[["delta_max_p5", "delta_max_median", "delta_max_p95"]].median()
    ax.errorbar(
        block_summary.index,
        block_summary["delta_max_median"],
        yerr=[
            block_summary["delta_max_median"] - block_summary["delta_max_p5"],
            block_summary["delta_max_p95"] - block_summary["delta_max_median"],
        ],
        fmt="o-",
    )
    ax.set(xlabel="Block size", ylabel="Block-max delta (p5/p50/p95)")
    names.append("06_block_max_delta.png"); _save(fig, output, names[-1])

    fig, ax = plt.subplots()
    streaks = [1, 2, 4, 8]
    cold_k = 512 if 512 in set(blocks["k"]) else int(blocks["k"].min())
    for block_size, group in blocks[blocks["k"] == cold_k].groupby("block_size"):
        ax.plot(
            streaks,
            [group[f"cold_fraction_{streak}"].mean() for streak in streaks],
            marker="o",
            label=f"B{block_size}",
        )
    ax.set(xlabel="Consecutive cold steps", ylabel=f"Fraction cold vs Top-{cold_k}", yscale="log")
    ax.legend()
    names.append("07_cold_persistence.png"); _save(fig, output, names[-1])

    fig, ax = plt.subplots()
    thresholds = replay[(replay["method"] == "static") & (replay["gamma_sigma"] == replay["gamma_sigma"].min())]
    ax.scatter(thresholds["tau_final"], thresholds["tau_seed"], s=5, alpha=0.25)
    finite = np.concatenate([thresholds["tau_final"].to_numpy(), thresholds["tau_seed"].to_numpy()])
    finite = finite[np.isfinite(finite)]
    if finite.size:
        ax.plot([finite.min(), finite.max()], [finite.min(), finite.max()], "k--", linewidth=1)
    ax.set(xlabel="Final kth score", ylabel="Previous-Top-K seed score")
    names.append("08_seed_vs_final_threshold.png"); _save(fig, output, names[-1])

    fig, ax = plt.subplots()
    for (method, order), group in replay.groupby(["method", "scan_order"]):
        reduced = group.groupby("gamma_sigma")[["qk_reduction", "recall"]].mean()
        ax.plot(reduced["qk_reduction"], reduced["recall"], marker="o", label=f"{method}/{order}")
    ax.set(xlabel="QK reduction", ylabel="Recall@K", xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(fontsize=6)
    names.append("09_pruning_recall_frontier.png"); _save(fig, output, names[-1])

    fig, ax = plt.subplots()
    comparison = replay.groupby(["method", "scan_order"])[["qk_reduction", "net_byte_reduction"]].median().sort_values("qk_reduction")
    labels = [f"{a}\n{b}" for a, b in comparison.index]
    x = np.arange(len(labels)); width = 0.4
    ax.bar(x-width/2, comparison["qk_reduction"], width, label="QK")
    ax.bar(x+width/2, comparison["net_byte_reduction"], width, label="Net bytes")
    ax.set(xticks=x, xticklabels=labels, ylabel="Median reduction"); ax.tick_params(axis="x", labelrotation=45); ax.legend()
    names.append("10_method_hardware_comparison.png"); _save(fig, output, names[-1])
    return names


def choose_verdict(replay: pd.DataFrame, quality_gate: dict[str, Any], policy: dict[str, float]) -> tuple[str, dict[str, float]]:
    if not quality_gate.get("passed", False):
        return "INCONCLUSIVE", {}
    evidence = replay[~replay["workload"].astype(str).str.contains("smoke", case=False)]
    if evidence.empty:
        return "INCONCLUSIVE", {"smoke_only": 1.0}
    replay = evidence
    selected_gamma = -1.0 if (-1.0 in set(replay["gamma_sigma"])) else replay["gamma_sigma"].max()
    dynamic = replay[
        (replay["method"] == "dynamic")
        & (replay["scan_order"] == "previous_hot")
        & (replay["gamma_sigma"] == selected_gamma)
    ]
    static = replay[
        (replay["method"] == "static")
        & (replay["gamma_sigma"] == selected_gamma)
    ]
    hot = replay[
        (replay["method"] == "static_hot_first")
        & (replay["gamma_sigma"] == selected_gamma)
    ]
    if dynamic.empty:
        return "INCONCLUSIVE", {}
    numbers = {
        "selected_gamma_policy": selected_gamma,
        "dynamic_qk_reduction": float(dynamic["qk_reduction"].median()),
        "dynamic_exact_rate": float(dynamic["exact_match"].mean()),
        "dynamic_false_cold_rate": float(dynamic["false_cold_rate"].mean()),
        "dynamic_net_byte_reduction": float(dynamic["net_byte_reduction"].median()),
        "dynamic_gain_over_static": float(dynamic["qk_reduction"].median() - static["qk_reduction"].median()),
        "hot_first_discovery_gain": float(static["discovery_fraction"].median() - hot["discovery_fraction"].median()),
    }
    safe = (
        numbers["dynamic_false_cold_rate"] <= policy["max_false_cold_rate"]
        and numbers["dynamic_exact_rate"] == 1.0
    )
    useful = numbers["dynamic_qk_reduction"] >= policy["min_exact_qk_reduction"]
    feedback = numbers["dynamic_gain_over_static"] >= policy["min_dynamic_gain_over_static"]
    scheduling = numbers["hot_first_discovery_gain"] >= policy["min_hot_first_gain_over_address"]
    if safe and useful and feedback and scheduling and numbers["dynamic_qk_reduction"] >= 0.50:
        verdict = "STRONG-HW-CANDIDATE"
    elif safe and useful and (feedback or scheduling) and numbers["dynamic_net_byte_reduction"] > 0:
        verdict = "HW-PROMISING"
    elif useful or scheduling:
        verdict = "SOFTWARE-ONLY"
    else:
        verdict = "NO-GO"
    return verdict, numbers


def hardware_extrapolation(replay: pd.DataFrame, lengths: list[int], block_sizes: list[int], key_bytes: int, metadata_bytes_per_block: int) -> pd.DataFrame:
    selected_gamma = -1.0 if (-1.0 in set(replay["gamma_sigma"])) else replay["gamma_sigma"].max()
    reduction = float(replay[(replay["method"] == "dynamic") & (replay["scan_order"] == "previous_hot") & (replay["gamma_sigma"] == selected_gamma)]["qk_reduction"].median())
    rows = []
    for length in lengths:
        for block_size in block_sizes:
            blocks = math.ceil(length / block_size)
            metadata = blocks * metadata_bytes_per_block
            full = length * key_bytes
            estimated = int(round(length * (1 - reduction))) * key_bytes + metadata
            rows.append({"context_length": length, "block_size": block_size, "blocks": blocks, "full_key_bytes": full, "metadata_bytes": metadata, "estimated_bytes": estimated, "estimated_net_reduction": 1-estimated/full})
    return pd.DataFrame(rows)


def write_report(
    output: Path,
    verdict: str,
    numbers: dict[str, float],
    quality_gate: dict[str, Any],
    graph_names: list[str],
    hardware: pd.DataFrame,
    audit: dict[str, Any],
) -> Path:
    failed = quality_gate.get("failed_layers", [])
    lines = [
        "# Temporal DSA Hardware Feasibility",
        "",
        f"## Verdict: {verdict}",
        "",
        "This verdict applies to a DeepSeek-V2-Lite **research sidecar**, not the production V3.2 Indexer or a TensorRT-LLM kernel. Oracle scan order is a ceiling only.",
        "",
        "## Quality gate",
        "",
        f"- Passed: `{quality_gate.get('passed', False)}`",
        f"- Reason: {quality_gate.get('reason', 'not supplied')}",
        f"- Failed layers: {failed or 'none'}",
        "- The numeric gate is pilot policy, not an official DeepSeek acceptance threshold.",
        "",
        "## Decision numbers",
        "",
    ]
    lines.extend(f"- {key}: `{value:.6g}`" for key, value in sorted(numbers.items()))
    lines.extend([
        "",
        "Exact-match results and approximate empirical-margin results must be read separately. A zero observed miss rate is not a mathematical certificate.",
        "",
        "## Hardware extrapolation",
        "",
        _markdown_table(hardware),
        "",
        "## Required figures",
        "",
    ])
    lines.extend(f"- [{name}](graphs/{name})" for name in graph_names)
    lines.extend([
        "",
        "## Reproducibility audit",
        "",
        "```json",
        json.dumps(audit, indent=2, sort_keys=True),
        "```",
        "",
        "## Interpretation constraints",
        "",
        "- Dense MLA teacher targets are per-head softmax probabilities summed across heads and L1-normalized.",
        "- Static/dynamic empirical bounds can miss current Top-K keys; only the separate Lipschitz/Cauchy bound is certified.",
        "- Hot-first changes threshold discovery order; it is not itself a reduction in mathematical QK work unless combined with running feedback.",
        "- Software timings do not establish dedicated hardware speedup.",
    ])
    path = output / "temporal_dsa_hw_feasibility.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
