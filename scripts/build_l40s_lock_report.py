from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


VERDICT = "PROTOCOL-SENSITIVE-BUT-EXPLAINED"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_row(
    repo: Path,
    category: str,
    artifact: str,
    relative: str,
    content: str,
    scope: str,
    protocol: str,
    status: str = "LOCKED",
) -> dict[str, Any]:
    path = repo / relative
    if path.is_dir():
        files = [item for item in path.rglob("*") if item.is_file()]
        byte_count = sum(item.stat().st_size for item in files)
        digest = "directory; see member artifacts"
        file_count = len(files)
    elif path.is_file():
        byte_count = path.stat().st_size
        digest = sha256(path)
        file_count = 1
    else:
        byte_count = 0
        digest = ""
        file_count = 0
        status = "MISSING"
    return {
        "category": category,
        "artifact": artifact,
        "path": relative,
        "exists": path.exists(),
        "file_count": file_count,
        "bytes": byte_count,
        "sha256": digest,
        "content": content,
        "scope": scope,
        "protocol": protocol,
        "status": status,
    }


def build_inventory(repo: Path, output: Path) -> list[dict[str, Any]]:
    specs = [
        (
            "score",
            "Full64 score traces",
            "artifacts/pilot/scores_b/traces",
            "scores[Q,N] FP16, lengths, source metadata",
            "held-out real traces",
            "exact stored Full64",
        ),
        (
            "verifier",
            "Dynamic H8 IDs and scores",
            "artifacts/h8_reconstruction/cache/test",
            "h8 FP32, rho, head_ids[Q,8], lengths",
            "96 test traces",
            "dynamic stable |w| Top8",
        ),
        (
            "temporal",
            "Prior temporal selection details",
            "artifacts/progressive_threshold/selection_details",
            "approximate, baseline stable Top-2048, rescued_blocks; no direct mask stored",
            "selected fixed-threshold replay",
            "masks reproducible from score+policy; now sealed in bundles",
        ),
        (
            "temporal",
            "Temporal policy selection",
            "artifacts/progressive_sw/prior_approx/selected_policies.json",
            "streak2_bucket8_m0 and gamma scale",
            "runtime-legal policy",
            "gamma=0, cold_streak=2, bucket8",
        ),
        (
            "promotion",
            "B64 thresholds",
            "artifacts/progressive_sw/threshold/runtime_config.json",
            "block size, H8 width, layer thresholds",
            "validation-calibrated",
            "fixed-threshold production/runtime-legal",
        ),
        (
            "promotion",
            "Promotion replay rows",
            "artifacts/progressive_sw/threshold/threshold_replay_rows.csv",
            "nominal/actual promotion and B64 replay metrics",
            "18,288 held-out transitions per scheme",
            "fixed-threshold",
        ),
        (
            "quality",
            "Selected promotion quality",
            "artifacts/progressive_sw/threshold/promotion_quality.csv",
            "QK reduction, Top-K and newly-active recall",
            "selected dynamic H8",
            "fixed-threshold",
        ),
        (
            "quality",
            "H8+10% H56 and T1 results",
            "artifacts/h8_reconstruction/final/main_results.csv",
            "ranking plus available MLA/KL/PPL status",
            "96 held-out traces",
            "existing locked evaluation",
        ),
        (
            "reconstruction",
            "T1 coefficients",
            "artifacts/h8_reconstruction/final/coefficients.json",
            "per-layer current-H8 + previous-Full64 affine coefficients",
            "calibration-only fit",
            "checkpoint-independent coefficients",
        ),
        (
            "quality",
            "MLA output lock",
            "artifacts/progressive_sw/mla/mla_output_summary.csv",
            "RelL2 p95/p99 and cosine",
            "selected fixed-threshold H8",
            "existing GPU replay",
        ),
        (
            "quality",
            "Teacher-forced lock",
            "artifacts/progressive_sw/final/teacher_forced_quality.csv",
            "logit KL and PPL delta",
            "384 tokens",
            "existing GPU replay",
        ),
        (
            "quality",
            "Closed-loop lock",
            "artifacts/progressive_sw/closed_loop/closed_loop_summary.json",
            "agreement, first divergence, NIAH success",
            "15 existing probes",
            "no rerun",
        ),
        (
            "latency",
            "Prior Full64 timing",
            "artifacts/progressive_sw/final/full_baseline_timing.csv",
            "CUDA-event Full64 and Full64+Top-K",
            "8K/16K/32K",
            "50 warmup, 500 measure, 64 MiB flush, 3 rotations",
        ),
        (
            "latency",
            "Prior fused progressive timing",
            "artifacts/progressive_sw/final/fused_progressive_timing.csv",
            "CUDA-event fused dense/compact and Top-K",
            "8K/16K/32K",
            "global-budget performance emulation",
        ),
        (
            "nsight",
            "Prior Nsight 16K/32K reports",
            "artifacts/progressive_nsight",
            "Nsight Systems reports/sqlite",
            "prior torch 2.9.1 path",
            "100 representative iterations",
        ),
        (
            "replay",
            "Portable DSA replay bundles",
            "artifacts/l40s_dsa_lock/replay_bundles",
            "Q/K/w, H8 IDs, masks, scores, stable Top-2048",
            "3 real contexts, 10 observations",
            "checkpoint-independent runtime",
        ),
    ]
    rows = [inventory_row(repo, *spec) for spec in specs]
    write_csv(
        output / "prior_artifact_inventory.csv",
        rows,
        [
            "category",
            "artifact",
            "path",
            "exists",
            "file_count",
            "bytes",
            "sha256",
            "content",
            "scope",
            "protocol",
            "status",
        ],
    )
    return rows


def build_latency(repo: Path, output: Path) -> list[dict[str, Any]]:
    fields = [
        "record_type",
        "source_study",
        "runtime",
        "context",
        "method_id",
        "method",
        "protocol",
        "observation_category",
        "promotion_policy",
        "nominal_promotion_rate",
        "actual_promotion_rate",
        "block_expanded_actual_promotion_rate",
        "warmup",
        "measurements",
        "mean_us",
        "median_us",
        "p5_us",
        "p95_us",
        "min_us",
        "std_us",
        "fixed_input_addresses",
        "preallocated_outputs",
        "notes",
    ]
    rows: list[dict[str, Any]] = []
    for row in read_csv(output / "latency_replay_raw.csv"):
        rows.append(
            {
                "record_type": "current_lock",
                "source_study": "portable_real_trace_replay",
                "runtime": f"torch {row['torch']}; CUDA {row['cuda']}; Triton {row['triton']}",
                "context": row["nominal_context"],
                "method_id": row["method_id"],
                "method": row["method"],
                "protocol": row["protocol"],
                "observation_category": row["observation_category"],
                "promotion_policy": "validation_fixed_threshold",
                "nominal_promotion_rate": row["nominal_promotion_rate"],
                "actual_promotion_rate": row["reference_actual_promotion_rate"],
                "block_expanded_actual_promotion_rate": row["runtime_actual_promotion_rate"],
                "warmup": row["warmup"],
                "measurements": row["measurements"],
                "mean_us": row["mean_us"],
                "median_us": row["median_us"],
                "p5_us": row["p5_us"],
                "p95_us": row["p95_us"],
                "min_us": row["min_us"],
                "std_us": row["std_us"],
                "fixed_input_addresses": row["fixed_input_addresses"],
                "preallocated_outputs": row["preallocated_outputs"],
                "notes": (
                    "Near-threshold real observation; block-expanded rate may differ from "
                    "token-granular production reference."
                ),
            }
        )

    prior_specs = [
        (
            repo / "artifacts" / "progressive_sw" / "final" / "full_baseline_timing.csv",
            "full_plus_topk",
            "L2",
            "full64_topk",
        ),
        (
            repo / "artifacts" / "progressive_sw" / "final" / "fused_progressive_timing.csv",
            "fused_dense_plus_topk",
            "L7",
            "fused_topk",
        ),
    ]
    for path, prior_method, method_id, method in prior_specs:
        for row in read_csv(path):
            if not (
                row["method"] == prior_method
                and row["block_size"] == "64"
                and row["promotion_target"] == "0.1"
                and row["topk"] == "2048"
            ):
                continue
            rows.append(
                {
                    "record_type": "prior_audit_reference",
                    "source_study": "progressive_sw_locked",
                    "runtime": "torch 2.9.1+cu128; CUDA 12.8",
                    "context": row["context"],
                    "method_id": method_id,
                    "method": method,
                    "protocol": "cache-flush-rotating3",
                    "observation_category": "three real captures",
                    "promotion_policy": "global-budget performance emulation",
                    "nominal_promotion_rate": row["promotion_target"],
                    "actual_promotion_rate": "",
                    "block_expanded_actual_promotion_rate": "",
                    "warmup": row["warmup"],
                    "measurements": row["measurements"],
                    "mean_us": float(row["mean_ms"]) * 1000,
                    "median_us": float(row["median_ms"]) * 1000,
                    "p5_us": float(row["p5_ms"]) * 1000,
                    "p95_us": float(row["p95_ms"]) * 1000,
                    "min_us": "",
                    "std_us": float(row["std_ms"]) * 1000,
                    "fixed_input_addresses": False,
                    "preallocated_outputs": False,
                    "notes": "Audit reference only; not forced to agree with the new protocol.",
                }
            )

    repeat_path = repo / "artifacts" / "streaming_topk" / "final" / "gpu_decomposition_repeat_audit.csv"
    repeat_ids = {
        "full64_score_only": ("L0", "full64"),
        "stock_topk_only": ("L1", "topk_only"),
        "full64_score_plus_stock_topk": ("L2", "full64_topk"),
    }
    for row in read_csv(repeat_path):
        method_id, method = repeat_ids[row["method"]]
        rows.append(
            {
                "record_type": "prior_repeat_audit",
                "source_study": "exact_streaming_topk_repeat",
                "runtime": "torch 2.11.0+cu130; CUDA 13.0",
                "context": row["context"],
                "method_id": method_id,
                "method": method,
                "protocol": "cache-flush-rotating3",
                "observation_category": "three real captures",
                "promotion_policy": "not applicable",
                "warmup": 100,
                "measurements": "1000 x 2 repeats",
                "mean_us": row["median_us_mean"],
                "median_us": row["median_us_mean"],
                "min_us": row["median_us_min"],
                "fixed_input_addresses": False,
                "preallocated_outputs": False,
                "notes": f"Repeat-median maximum {row['median_us_max']} us; no sample p5/p95 here.",
            }
        )
    rows.sort(
        key=lambda row: (
            int(row["context"]),
            str(row["record_type"]),
            str(row["protocol"]),
            str(row["method_id"]),
        )
    )
    write_csv(output / "latency_protocols.csv", rows, fields)
    return rows


def build_quality(repo: Path, output: Path) -> list[dict[str, Any]]:
    fields = [
        "policy",
        "promotion_policy",
        "nominal_promotion_rate",
        "actual_promotion_rate",
        "qk_reduction_mean",
        "qk_reduction_median",
        "top128_recall",
        "top512_recall",
        "top2048_recall",
        "newly_active_token_recall",
        "mla_relative_l2_p95",
        "mla_relative_l2_p99",
        "mla_cosine_p5",
        "logit_kl_mean",
        "teacher_forced_ppl_delta",
        "closed_loop_token_agreement_mean",
        "closed_loop_first_divergence_min",
        "closed_loop_niah_success_rate",
        "ranking_status",
        "mla_kl_ppl_status",
        "source",
        "notes",
    ]
    promotion = next(
        row
        for row in read_csv(
            repo / "artifacts" / "progressive_sw" / "threshold" / "promotion_quality.csv"
        )
        if row["head_scheme"] == "dynamic_abs_w" and row["promotion_target"] == "0.1"
    )
    mla = next(
        row
        for row in read_csv(
            repo / "artifacts" / "progressive_sw" / "mla" / "mla_output_summary.csv"
        )
        if row["verifier"] == "head_dynamic_abs_w_w8_b64_threshold_r0.1"
    )
    teacher = read_csv(
        repo / "artifacts" / "progressive_sw" / "final" / "teacher_forced_quality.csv"
    )[0]
    closed = json.loads(
        (repo / "artifacts" / "progressive_sw" / "closed_loop" / "closed_loop_summary.json")
        .read_text(encoding="utf-8")
    )
    rows: list[dict[str, Any]] = [
        {
            "policy": "streak2_bucket8_m0 + dynamic H8 fixed threshold",
            "promotion_policy": promotion["promotion_policy"],
            "nominal_promotion_rate": promotion["promotion_target"],
            "actual_promotion_rate": promotion["actual_promotion_rate"],
            "qk_reduction_mean": promotion["net_qk_reduction_mean"],
            "qk_reduction_median": promotion["net_qk_reduction_median"],
            "top128_recall": promotion["top128_recall"],
            "top512_recall": promotion["top512_recall"],
            "top2048_recall": promotion["top2048_recall"],
            "newly_active_token_recall": promotion["newly_active_token_recall"],
            "mla_relative_l2_p95": mla["output_relative_l2_p95"],
            "mla_relative_l2_p99": mla["output_relative_l2_p99"],
            "mla_cosine_p5": mla["output_cosine_p5"],
            "logit_kl_mean": teacher["logit_kl_mean"],
            "teacher_forced_ppl_delta": teacher["ppl_delta"],
            "closed_loop_token_agreement_mean": closed["generated_token_agreement_mean"],
            "closed_loop_first_divergence_min": closed["first_divergence_min"],
            "closed_loop_niah_success_rate": closed["niah_approx_success_rate"],
            "ranking_status": "available",
            "mla_kl_ppl_status": "available; reused, not rerun",
            "source": "progressive_sw threshold/MLA/teacher-forced/closed-loop",
            "notes": "Primary runtime-legal quality lock; aggregate over 18,288 transitions.",
        }
    ]
    reconstruction = read_csv(
        repo / "artifacts" / "h8_reconstruction" / "final" / "main_results.csv"
    )
    rescue = next(row for row in reconstruction if row["method"] == "Existing H8 + 10% H56 rescue")
    t1 = next(row for row in reconstruction if row["method"] == "Per-layer T1 H8 + previous Full64")
    prior_closed = json.loads(
        (repo / "artifacts" / "progressive_sw" / "prior_two_path" / "two_path_verdict.json")
        .read_text(encoding="utf-8")
    )["decision_basis"]["closed_loop"]
    rows.append(
        {
            "policy": rescue["method"],
            "promotion_policy": "prior H8 global-budget 10% H56 rescue",
            "nominal_promotion_rate": 0.1,
            "actual_promotion_rate": "not separately locked",
            "qk_reduction_mean": rescue["qk_reduction"],
            "top128_recall": rescue["top128_recall"],
            "top512_recall": rescue["top512_recall"],
            "top2048_recall": rescue["top2048_recall"],
            "mla_relative_l2_p95": rescue["mla_relative_l2_p95"],
            "mla_relative_l2_p99": rescue["mla_relative_l2_p99"],
            "mla_cosine_p5": rescue["mla_cosine_p5"],
            "logit_kl_mean": rescue["logit_kl_mean"],
            "teacher_forced_ppl_delta": rescue["ppl_delta"],
            "closed_loop_token_agreement_mean": prior_closed["generated_token_agreement_mean"],
            "closed_loop_first_divergence_min": prior_closed["first_divergence_min"],
            "closed_loop_niah_success_rate": prior_closed["niah_approx_success_rate"],
            "ranking_status": "available",
            "mla_kl_ppl_status": "available; prior GPU replay",
            "source": "h8_reconstruction/final/main_results.csv",
            "notes": "Preserved separately from the fixed-threshold production lock.",
        }
    )
    rows.append(
        {
            "policy": t1["method"],
            "promotion_policy": "no H56",
            "nominal_promotion_rate": 0.0,
            "actual_promotion_rate": 0.0,
            "qk_reduction_mean": t1["qk_reduction"],
            "top128_recall": t1["top128_recall"],
            "top512_recall": t1["top512_recall"],
            "top2048_recall": t1["top2048_recall"],
            "ranking_status": "available",
            "mla_kl_ppl_status": "GPU FOLLOW-UP REQUIRED",
            "source": "h8_reconstruction/final/main_results.csv",
            "notes": "MLA/KL/PPL intentionally remain unavailable; no new GPU quality sweep was run.",
        }
    )
    write_csv(output / "quality_lock.csv", rows, fields)
    return rows


def build_kernel_inventory(repo: Path, output: Path) -> list[dict[str, Any]]:
    fields = [
        "record_type",
        "context",
        "operation_order",
        "operation_type",
        "category",
        "stage",
        "short_name",
        "launches_per_iteration",
        "duration_us_first_iteration",
        "gap_from_previous_us_first_iteration",
        "active_duration_us_per_iteration",
        "operation_count_per_iteration",
        "prior_operation_count_per_iteration",
        "status",
        "path",
        "sha256",
        "notes",
    ]
    rows: list[dict[str, Any]] = []
    sources = [
        ("progressive kernel definitions", "src/temporal_dsa/progressive_kernel.py"),
        ("runtime-legal temporal replay", "src/temporal_dsa/approx.py"),
        ("promotion verifier", "src/temporal_dsa/verifier.py"),
        ("H8 scorer and encoder", "src/temporal_dsa/verifier_scoring.py"),
        ("portable benchmark", "scripts/benchmark_dsa_replay.py"),
        ("bundle builder", "scripts/build_l40s_replay_bundles.py"),
        ("Nsight extractor", "scripts/extract_full64_topk_nsight.py"),
    ]
    for notes, relative in sources:
        path = repo / relative
        rows.append(
            {
                "record_type": "source",
                "path": relative,
                "sha256": sha256(path),
                "status": "LOCKED",
                "notes": notes,
            }
        )
    timeline = read_csv(output / "nsight" / "gpu_topk_timeline.csv")
    aggregate = read_csv(output / "nsight" / "gpu_topk_operation_summary.csv")
    for item in timeline:
        rows.append(
            {
                "record_type": "current_nsight_timeline",
                "context": item["context"],
                "operation_order": item["operation_order"],
                "operation_type": item["operation_type"],
                "category": item["category"],
                "stage": item["stage"],
                "short_name": item["short_name"],
                "launches_per_iteration": 1,
                "duration_us_first_iteration": item["duration_us_first_iteration"],
                "gap_from_previous_us_first_iteration": item[
                    "gap_from_previous_us_first_iteration"
                ],
                "status": "MEASURED",
                "path": item["source"],
                "notes": "One representative iteration inside the 100-iteration NVTX range.",
            }
        )
    for item in aggregate:
        rows.append(
            {
                "record_type": "current_nsight_aggregate",
                "context": item["context"],
                "operation_type": item["operation_type"],
                "category": item["category"],
                "stage": item["stage"],
                "short_name": item["short_name"],
                "launches_per_iteration": item["launches_per_iteration"],
                "active_duration_us_per_iteration": item["active_duration_us_per_iteration"],
                "status": "MEASURED",
                "path": item["source"],
                "notes": "Aggregate across 100 iterations.",
            }
        )
    prior_counts = {16384: 2, 32768: 18}
    for context in (16384, 32768):
        count = sum(1 for item in timeline if int(item["context"]) == context)
        rows.append(
            {
                "record_type": "operation_count_summary",
                "context": context,
                "operation_count_per_iteration": count,
                "prior_operation_count_per_iteration": prior_counts[context],
                "status": "REPRODUCED" if count == prior_counts[context] else "CHANGED",
                "path": f"artifacts/l40s_dsa_lock/nsight/full64_topk_{context//1024}k.sqlite",
                "notes": (
                    "32K path changed from 18 to 22 operations under torch 2.11/CUDA 13; "
                    "radixFindKthValues became digit-count and digit-cumulative-sum stages."
                    if context == 32768
                    else "16K remains one Full64 score kernel plus one gatherTopK kernel."
                ),
            }
        )
    write_csv(output / "kernel_inventory.csv", rows, fields)
    return rows


def latency_map(rows: list[dict[str, Any]], protocol: str) -> dict[tuple[int, str], float]:
    return {
        (int(row["context"]), str(row["method_id"])): float(row["median_us"])
        for row in rows
        if row["record_type"] == "current_lock" and row["protocol"] == protocol
    }


def build_report(
    output: Path,
    latency: list[dict[str, Any]],
    quality: list[dict[str, Any]],
) -> None:
    warm = latency_map(latency, "warm-cache")
    flush = latency_map(latency, "cache-flush")
    prior = {
        (int(row["context"]), str(row["method_id"])): float(row["median_us"])
        for row in latency
        if row["record_type"] == "prior_audit_reference"
    }
    method_names = {
        "L0": "Full64 score",
        "L1": "precomputed score Top-K",
        "L2": "Full64 + Top-K",
        "L3": "dynamic H8 score",
        "L4": "two-pass progressive H8",
        "L5": "fused progressive dense",
        "L6": "fused progressive compact",
        "L7": "fused progressive + Top-K",
        "L8": "T1 H8 + previous Full64",
    }
    lines = [
        "# L40S DSA measurement lock",
        "",
        f"Final verdict: **{VERDICT}**",
        "",
        "This package locks the DeepSeek-V2-Lite research-sidecar DSA operator and existing "
        "quality evidence. It does not claim TensorRT-LLM integration or H100/SM120 silicon results.",
        "",
        "## Environment and scope",
        "",
        "- GPU: NVIDIA L40S ×4 physical; measurement isolated to physical GPU 0. GPUs 2/3 "
        "were protected because they had existing jobs.",
        "- Measurement runtime: Python 3.10.12, torch 2.11.0+cu130, CUDA runtime 13.0, "
        "Triton 3.6.0, driver 580.173.02.",
        "- Prior locked/runtime materialization environment: torch 2.9.1+cu128, CUDA 12.8.",
        "- Authoritative local source base: commit `130148b4fe6e9a601a29cb40a8485bca77503158`, "
        "branch `exp/l40s-dsa-lock`; measurement host is a non-git source snapshot.",
        "- `ncu` is unavailable. Nsight Systems 2025.3.2 was used; DRAM/L2/occupancy counters "
        "are therefore not invented.",
        "",
        "Complete trace roots, result directories, kernel paths and Nsight report paths are in "
        "`environment.json`; exact prior artifact roles and hashes are in "
        "`prior_artifact_inventory.csv`.",
        "",
        "## Locked shapes",
        "",
        "Primary: B=1, Q=1, H=64, D=128, Top-K=2048, N=8K/16K/32K, BF16 Q/K. "
        "The real replay lengths include the decode offset (for example 8K near-threshold is "
        "8,285 tokens). Synthetic 64K and 128K operator-only tests also passed without model "
        "inference; see `synthetic_shape_audit.json`.",
        "",
        "## New fixed-address timing lock",
        "",
        "CUDA Events, preallocated explicit outputs, fixed Q/K/w addresses, 200 warmups and "
        "2,000 measurements were used. Flush is a preallocated 64 MiB device write outside the "
        "timed event. Warm-cache and cache-flush are not averaged.",
        "",
        "| Method | 8K warm / flush µs | 16K warm / flush µs | 32K warm / flush µs |",
        "|---|---:|---:|---:|",
    ]
    for method_id, name in method_names.items():
        values = [
            f"{warm[(context, method_id)]:.3f} / {flush[(context, method_id)]:.3f}"
            for context in (8192, 16384, 32768)
        ]
        lines.append(f"| {method_id} {name} | {values[0]} | {values[1]} | {values[2]} |")
    lines.extend(
        [
            "",
            "All mean/median/p5/p95/min/std samples, promotion rates and pointer-lock fields are "
            "in `latency_protocols.csv`.",
            "",
            "## Prior audit references versus new replay",
            "",
            "| Path | 8K prior → new flush µs | 16K prior → new flush µs | 32K prior → new flush µs |",
            "|---|---:|---:|---:|",
        ]
    )
    for method_id, label in (("L2", "Full64 + Top-K"), ("L7", "fused progressive + Top-K")):
        values = [
            f"{prior[(context, method_id)]:.3f} → {flush[(context, method_id)]:.3f}"
            for context in (8192, 16384, 32768)
        ]
        lines.append(f"| {label} | {values[0]} | {values[1]} | {values[2]} |")
    lines.extend(
        [
            "",
            "The differences are explained, not normalized away:",
            "",
            "1. The prior timing used torch 2.9.1/CUDA 12.8, 50/500, three rotating traces, "
            "stock Top-K output allocation, and a global-budget performance mask.",
            "2. The new timing uses torch 2.11/CUDA 13, 200/2,000, one fixed real replay address, "
            "preallocated Top-K outputs, and the validation-fixed production threshold.",
            "3. The flush write also raises the L40S clock state. Score-only kernels are faster "
            "under the nominal cache-flush protocol than warm-cache, so this protocol is not a "
            "pure cache-state intervention.",
            "4. The near-threshold observations intentionally expose numerical sensitivity: "
            "reference/runtime promotion rates are 20.0/40.0% at 8K, 46.34/46.34% at 16K, and "
            "1.047/1.047% at 32K. Aggregate quality remains the separately locked 29.36% actual "
            "promotion rate.",
            "",
            "## Nsight Systems lock",
            "",
            "At 16K the current path remains two GPU operations: `_full_score_kernel` and "
            "`gatherTopK`. At 32K the prior 18-operation path is **not** reproduced: the current "
            "torch 2.11/CUDA 13 path has 22 operations. It contains one score kernel, one fill, "
            "two memsets, four radix digit-histogram stages, four radix digit-cumulative-sum "
            "stages, four within-K count stages, one Kth-count stage, two scan initializers, two "
            "scans, and one gather. These are selection stages; they are not all called sorting "
            "kernels. Per-operation durations and launch gaps are in `kernel_inventory.csv` and "
            "the raw 100-iteration reports are under `nsight/`.",
            "",
            "## Quality lock (no new expensive sweep)",
            "",
            "The primary runtime-legal dynamic-H8 fixed-threshold result is: actual promotion "
            "29.3597%, mean/median QK reduction 15.8913%/13.9827%, Top-128/512/2048 recall "
            "99.9980%/99.9702%/99.5875%, newly-active token recall 54.9154%, MLA RelL2 "
            "p95/p99 2.6129%/9.1845%, logit KL 0.0024889, and teacher-forced PPL delta "
            "-0.0044188. Existing closed-loop agreement is 50.2083%, first divergence 6; NIAH "
            "success stayed 100% for both baseline and approximate paths.",
            "",
            "The prior H8 + 10% H56 result is preserved separately (20.1252% QK reduction, "
            "Top-2048 99.5555%, MLA RelL2 p95 2.5958%, logit KL 0.0025926). T1 without H56 "
            "has ranking evidence (87.5% QK reduction, Top-128/512/2048 "
            "99.9402%/99.3853%/82.0076%) but still has no MLA/KL/PPL result. See "
            "`quality_lock.csv`.",
            "",
            "## Portable replay contract",
            "",
            "Three `.pt` files contain ten real observations across easy, newly-active, "
            "near-threshold, and 32K-tail categories. Each stores BF16 Q/K, FP32 w, stable "
            "dynamic H8 IDs, token- and B64-block temporal masks, promotion blocks/threshold, "
            "Full64/H8/T1 scores, previous Full64 and stable exact Top-2048 IDs. Source trace, "
            "capture, cache and checkpoint hashes are retained, but runtime replay requires no "
            "checkpoint.",
            "",
            "The common runner is `cross_platform_runner/benchmark_dsa_replay.py` and supports "
            "`full64`, `topk_only`, `full64_topk`, `h8`, `progressive_h8`, `fused_dense`, "
            "`fused_compact`, `fused_topk`, and `t1`, with JSON and CSV output and no hard-coded "
            "platform path.",
            "",
            "Example:",
            "",
            "```bash",
            "python benchmark_dsa_replay.py --bundle <bundle.pt> --method full64 --top-k 2048 "
            "--warmup 200 --iters 2000",
            "```",
            "",
            f"Locked Full64 baseline: L2 prior 40.784/63.488/87.040 µs; new flush {flush[(8192, 'L2')]:.3f}/{flush[(16384, 'L2')]:.3f}/{flush[(32768, 'L2')]:.3f} µs",
            f"Locked Top-K baseline: L1 new flush {flush[(8192, 'L1')]:.3f}/{flush[(16384, 'L1')]:.3f}/{flush[(32768, 'L1')]:.3f} µs",
            f"Locked progressive H8 result: L7 prior 62.464/70.656/104.608 µs; new flush {flush[(8192, 'L7')]:.3f}/{flush[(16384, 'L7')]:.3f}/{flush[(32768, 'L7')]:.3f} µs",
            "Locked quality result: fixed-threshold H8 mean QK reduction 15.8913%, Top-2048 recall 99.5875%, MLA RelL2 p95 2.6129%",
            "Main protocol sensitivity: runtime/Top-K path, GPU clock effect of flush, fixed-address versus rotating input, and production threshold versus global budget",
            "Portable replay bundle count: 3 files / 10 real observations",
            "Ready for SM120: Yes for portable replay and compile-time validation; silicon measurement pending",
            "Ready for H100: Yes for the common CUDA-event replay protocol; silicon measurement pending",
            "Remaining missing artifact: T1 no-H56 MLA/KL/PPL replay and Nsight Compute DRAM/L2/SM counters",
        ]
    )
    (output / "l40s_dsa_lock.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def install_runner(repo: Path, output: Path) -> None:
    destination = output / "cross_platform_runner"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(repo / "scripts" / "benchmark_dsa_replay.py", destination / "benchmark_dsa_replay.py")
    (destination / "README.md").write_text(
        "# Portable DSA replay runner\n\n"
        "Requires Python, PyTorch with CUDA, Triton, and the `temporal_dsa` package/source tree.\n\n"
        "```bash\n"
        "PYTHONPATH=src python artifacts/l40s_dsa_lock/cross_platform_runner/benchmark_dsa_replay.py "
        "--bundle artifacts/l40s_dsa_lock/replay_bundles/dsa_replay_c32768_layer17_code_heldout_3.pt "
        "--method full64 --top-k 2048 --warmup 200 --iters 2000\n"
        "```\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final L40S DSA lock package")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    build_inventory(repo, output)
    latency = build_latency(repo, output)
    quality = build_quality(repo, output)
    build_kernel_inventory(repo, output)
    install_runner(repo, output)
    build_report(output, latency, quality)
    print(json.dumps({"verdict": VERDICT, "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
