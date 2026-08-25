from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--quality-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    temporal = pd.read_csv(args.report_root / "temporal_stats.csv")
    replay = pd.read_csv(args.report_root / "replay_rows.csv")
    blocks = pd.read_csv(args.report_root / "block_stats.csv")
    quality = pd.read_csv(args.quality_csv)
    for frame in (temporal, replay, blocks, quality):
        frame["context_bucket"] = ((frame["context_length"] - 1) // 8192 * 8192).astype(int)

    temporal_k = temporal.groupby("k")["topk_overlap"].agg(["mean", "median", "min"]).reset_index()
    temporal_context = temporal.groupby(["context_bucket", "workload", "k"])["topk_overlap"].agg(["mean", "median"]).reset_index()
    temporal_layer = temporal.groupby(["layer", "k"])["topk_overlap"].agg(["mean", "median", "min"]).reset_index()
    correlations = temporal.drop_duplicates(
        ["context_bucket", "layer", "workload", "prompt_id", "step"]
    )
    correlation_summary = correlations.groupby(["context_bucket", "workload"])[["score_pearson", "score_spearman", "delta_p99_abs"]].agg(["mean", "median"]).reset_index()
    correlation_summary.columns = ["_".join(str(part) for part in column if part) for column in correlation_summary.columns]

    method = replay.groupby(["method", "scan_order", "gamma_sigma"])[["qk_reduction", "recall", "exact_match", "false_cold_rate", "discovery_fraction", "net_byte_reduction"]].mean().reset_index()
    selected_gamma = -1.0 if -1.0 in set(replay["gamma_sigma"]) else 1.0
    selected = replay[(replay["method"] == "dynamic") & (replay["scan_order"] == "previous_hot") & (replay["gamma_sigma"] == selected_gamma)]
    selected_detail = selected.groupby(["context_bucket", "workload", "k", "block_size"])[["qk_reduction", "recall", "exact_match", "false_cold_rate", "net_byte_reduction"]].mean().reset_index()
    oracle = replay[replay["method"] == "oracle"].groupby(["context_bucket", "k", "block_size"])[["qk_reduction", "exact_match", "net_byte_reduction"]].mean().reset_index()
    safe_dynamic = replay[
        (replay["method"] == "dynamic")
        & (replay["scan_order"] != "oracle_current")
        & (replay["exact_match"] == 1.0)
        & (replay["false_cold_rate"] == 0.0)
    ]
    safe_summary = {
        "rows": int(len(safe_dynamic)),
        "fraction_of_dynamic_rows": float(len(safe_dynamic) / max(1, len(replay[(replay["method"] == "dynamic") & (replay["scan_order"] != "oracle_current")]))),
        "median_qk_reduction": float(safe_dynamic["qk_reduction"].median()) if len(safe_dynamic) else None,
        "max_qk_reduction": float(safe_dynamic["qk_reduction"].max()) if len(safe_dynamic) else None,
    }
    quality_summary = quality.groupby(["layer", "workload", "k"])[["recall", "normalized_lift"]].agg(["mean", "median", "min"]).reset_index()
    quality_summary.columns = ["_".join(str(part) for part in column if part) for column in quality_summary.columns]
    cold = blocks.groupby(["context_bucket", "workload", "k", "block_size"])[["cold_fraction_1", "cold_fraction_2", "cold_fraction_4", "cold_fraction_8"]].mean().reset_index()

    payload = {
        "temporal_by_k": records(temporal_k),
        "temporal_by_context_workload_k": records(temporal_context),
        "temporal_by_layer_k": records(temporal_layer),
        "correlations": records(correlation_summary),
        "method_comparison": records(method),
        "selected_dynamic_detail": records(selected_detail),
        "oracle_ceiling": records(oracle),
        "fully_exact_dynamic": safe_summary,
        "selected_gamma_policy": "validation_max_margin" if selected_gamma == -1.0 else f"{selected_gamma:g}_sigma",
        "quality_by_layer_workload_k": records(quality_summary),
        "cold_persistence": records(cold),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
