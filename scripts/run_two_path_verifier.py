from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from temporal_dsa.approx import ApproxPolicy
from temporal_dsa.metrics import stable_topk
from temporal_dsa.trace import load_trace
from temporal_dsa.verifier import VerifierConfig, replay_verifier_trace
from temporal_dsa.verifier_scoring import (
    dynamic_head_indices,
    load_sidecar_encoded,
    score_dim_sparse,
    score_head_sparse,
)


def family_policy(family: str, *, name: str, gamma: float) -> ApproxPolicy:
    common: dict[str, Any] = {"name": name, "gamma": gamma}
    if family == "static":
        return ApproxPolicy(**common)
    if family == "dynamic_address":
        return ApproxPolicy(**common, dynamic_threshold=True)
    if family == "dynamic_previous_max":
        return ApproxPolicy(**common, dynamic_threshold=True, order="previous_max")
    if family in {"dynamic_bucket8", "dynamic_bucket16"}:
        return ApproxPolicy(
            **common, dynamic_threshold=True, order=family.removeprefix("dynamic_")
        )
    if family in {"streak2_bucket8", "streak4_bucket8"}:
        streak = int(family.removeprefix("streak").split("_", 1)[0])
        return ApproxPolicy(
            **common, dynamic_threshold=True, order="bucket8", cold_streak=streak
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


def build_policy(spec: dict[str, Any], layer: int, scales: dict[int, float]) -> ApproxPolicy:
    policy = family_policy(
        str(spec["family"]),
        name=str(spec["config_id"]),
        gamma=float(spec["multiplier"]) * scales[layer],
    )
    return replace(policy, **spec.get("overrides", {}))


def discover(files: list[Path] | None, roots: list[Path] | None) -> list[Path]:
    paths = list(files or [])
    for root in roots or []:
        paths.extend(sorted(root.rglob("*.npz")))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError("no score traces")
    return paths


def expand_configs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    configs = [dict(value) for value in payload.get("configs", [])]
    for sweep in payload.get("sweeps", []):
        for width in sweep["widths"]:
            for strategy_spec in sweep["strategies"]:
                for rescue in sweep["rescue_fractions"]:
                    spec = {
                        **{key: value for key, value in sweep.items() if key not in {
                            "widths", "strategies", "rescue_fractions"
                        }},
                        **strategy_spec,
                        "width": int(width),
                        "rescue_fraction": float(rescue),
                    }
                    label = str(strategy_spec.get("label", strategy_spec["strategy"]))
                    spec["name"] = (
                        f"{spec['path']}_{label}_w{width}_b{spec.get('block_size', 64)}_"
                        f"r{float(rescue):g}"
                    )
                    configs.append(spec)
    names = [str(value["name"]) for value in configs]
    if len(names) != len(set(names)):
        raise ValueError("verifier configuration names must be unique")
    return configs


def score_group(spec: dict[str, Any]) -> str:
    ignored = {"name", "rescue_fraction", "block_size", "retain_candidate_keys"}
    return json.dumps(
        {key: value for key, value in spec.items() if key not in ignored}, sort_keys=True
    )


def fixed_ids(spec: dict[str, Any], layer: int, size: int) -> np.ndarray:
    by_layer = spec.get("ids_by_layer", {})
    values = by_layer.get(str(layer))
    if values is not None:
        result = np.asarray(values, dtype=np.int64)
    elif spec["strategy"] == "random_fixed":
        generator = np.random.default_rng(int(spec.get("seed", 1582)) + layer * 1009)
        result = np.sort(generator.choice(size, int(spec["width"]), replace=False))
    elif spec["strategy"] == "first":
        result = np.arange(int(spec["width"]), dtype=np.int64)
    elif spec["strategy"] == "evenly_spaced":
        result = np.linspace(0, size - 1, int(spec["width"]), dtype=np.int64)
    else:
        raise ValueError(
            f"strategy {spec['strategy']} requires ids_by_layer for layer {layer}"
        )
    if result.size != int(spec["width"]) or np.unique(result).size != result.size:
        raise ValueError(f"invalid fixed IDs for {spec['name']} layer {layer}")
    if result.min() < 0 or result.max() >= size:
        raise ValueError(f"fixed IDs out of range for {spec['name']}")
    return result


def verifier_config(spec: dict[str, Any]) -> VerifierConfig:
    path = str(spec["path"])
    width = int(spec["width"])
    ratio = width / 64 if path == "head" else width / 128
    precision = str(spec.get("precision", "bf16"))
    bits = {"bf16": 16, "int8": 8, "int4": 4, "int2": 2}[precision]
    sketch_bytes = width * bits / 8 if path == "dim" else 0.0
    return VerifierConfig(
        name=str(spec["name"]),
        path=path,
        width=width,
        score_ratio=ratio,
        block_size=int(spec.get("block_size", 64)),
        rescue_fraction=float(spec["rescue_fraction"]),
        sketch_bytes_per_token=sketch_bytes,
        retain_candidate_keys=bool(spec.get("retain_candidate_keys", False)),
    )


def compute_scores(spec: dict[str, Any], layer: int, queries, weights, keys):
    if spec["path"] == "head":
        strategy = str(spec["strategy"])
        if strategy in {"high_weight", "positive_weight"}:
            ids = dynamic_head_indices(weights, int(spec["width"]), strategy)
        else:
            ids = fixed_ids(spec, layer, 64)
        return score_head_sparse(queries, weights, keys, ids)
    dimensions = fixed_ids(spec, layer, 128)
    precision = str(spec.get("precision", "bf16"))
    bits = None if precision == "bf16" else int(precision.removeprefix("int"))
    return score_dim_sparse(
        queries,
        weights,
        keys,
        dimensions,
        rotation=str(spec.get("rotation", "none")),
        bits=bits,
        query_scale=spec.get("query_scale_by_layer", {}).get(str(layer)),
        key_scale=spec.get("key_scale_by_layer", {}).get(str(layer)),
    )


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["policy_role", "policy", "verifier", "path", "width", "rescue_fraction"]
    for keys, group in frame.groupby(group_columns, sort=False):
        row = dict(zip(group_columns, keys))
        newly_tokens = int(group.newly_active_tokens.sum())
        newly_blocks = int(group.newly_active_blocks.sum())
        tail_blocks = int(group.tail_critical_blocks.sum())
        row.update(
            {
                "observations": len(group),
                "net_qk_reduction_mean": group.net_qk_reduction.mean(),
                "net_qk_reduction_median": group.net_qk_reduction.median(),
                "physical_key_byte_reduction_median": group.physical_key_byte_reduction.median(),
                "exact_match": group.exact_match.mean(),
                "recall_mean": group.recall.mean(),
                "top128_recall": group.top128_recall.mean(),
                "top256_recall": group.top256_recall.mean(),
                "top512_recall": group.top512_recall.mean(),
                "top1024_recall": group.top1024_recall.mean(),
                "top2048_recall": group.top2048_recall.mean(),
                "index_mass_ratio_mean": group.index_mass_ratio.mean(),
                "newly_active_token_recall": (
                    float((group.newly_active_token_recall * group.newly_active_tokens).sum() / newly_tokens)
                    if newly_tokens else 1.0
                ),
                "newly_active_block_recall": (
                    float((group.newly_active_block_recall * group.newly_active_blocks).sum() / newly_blocks)
                    if newly_blocks else 1.0
                ),
                "tail_critical_block_recall": (
                    float((group.tail_critical_block_recall * group.tail_critical_blocks).sum() / tail_blocks)
                    if tail_blocks else np.nan
                ),
                "detector_ap_mean": group.detector_ap.mean(),
                "rescue_precision_mean": group.rescue_precision.mean(),
                "rescue_block_fraction_mean": group.rescue_blocks.sum()
                / np.maximum(1, np.ceil(group.context_length / group.block_size).sum()),
                "verifier_key_bytes": group.verifier_key_bytes.sum(),
                "sketch_bytes": group.sketch_bytes.sum(),
                "rescue_reread_key_bytes": group.rescue_reread_key_bytes.sum(),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def save_details(path: Path, details: dict[int, dict[str, np.ndarray]]) -> None:
    steps = np.asarray(sorted(details), dtype=np.int16)
    np.savez_compressed(
        path,
        steps=steps,
        approximate=np.stack([details[int(step)]["approximate"] for step in steps]),
        baseline=np.stack([details[int(step)]["baseline"] for step in steps]),
        rescued_blocks=np.asarray(
            [details[int(step)]["rescued_blocks"] for step in steps], dtype=object
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-path current-query verifier replay")
    parser.add_argument("--trace-files", type=Path, nargs="+")
    parser.add_argument("--trace-roots", type=Path, nargs="+")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--verifier-configs", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-roles", nargs="+", default=["Balanced", "Aggressive"])
    parser.add_argument("--max-transitions", type=int)
    parser.add_argument("--save-details", action="store_true")
    parser.add_argument("--smoke-reconstruction", action="store_true")
    parser.add_argument("--tail-labels", type=Path)
    parser.add_argument(
        "--tail-definition",
        choices=["top1", "top2", "top4", "gain_at_least_1pp", "gain_at_least_25pct"],
        default="top4",
    )
    args = parser.parse_args()

    if torch.cuda.device_count() != 2:
        raise RuntimeError(
            f"expected CUDA_VISIBLE_DEVICES=0,1, found {torch.cuda.device_count()} GPUs"
        )
    gpu_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("L40" in value for value in gpu_names):
        raise RuntimeError(f"L40 guard rejected {gpu_names}")
    paths = discover(args.trace_files, args.trace_roots)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    scales = {int(key): float(value) for key, value in selection["gamma_scale"].items()}
    policy_specs = [
        value for value in selection["policies"] if value["selection_role"] in args.policy_roles
    ]
    if not policy_specs:
        raise ValueError("policy role filter selected no temporal policies")
    config_payload = json.loads(args.verifier_configs.read_text(encoding="utf-8"))
    config_specs = expand_configs(config_payload)
    if not config_specs:
        raise ValueError("verifier config file contains no configurations")
    tail_frame = None
    if args.tail_labels is not None:
        tail_frame = pd.read_csv(args.tail_labels)
        required = {
            "policy", "step", "layer", "prompt_id", "base_context_length", "block_id",
            args.tail_definition,
        }
        if not required.issubset(tail_frame.columns):
            raise ValueError(f"tail label file is missing {sorted(required - set(tail_frame.columns))}")
        tail_frame = tail_frame[tail_frame[args.tail_definition].astype(bool)]

    args.output.mkdir(parents=True, exist_ok=True)
    detail_root = args.output / "selection_details"
    if args.save_details:
        detail_root.mkdir(exist_ok=True)
    started = time.perf_counter()
    rows: list[pd.DataFrame] = []
    reconstruction: list[dict[str, Any]] = []
    for trace_index, trace_path in enumerate(paths, start=1):
        trace = load_trace(trace_path)
        checkpoint = args.checkpoint_root / f"layer_{trace.layer:02d}.safetensors"
        queries, weights, keys, capture = load_sidecar_encoded(
            trace.metadata["source_capture"], checkpoint, trace.lengths, device="cuda:0"
        )
        if args.smoke_reconstruction:
            count = min(trace.scores.shape[0], (args.max_transitions or 16) + 1)
            reconstructed = score_dim_sparse(
                queries[:count], weights[:count], keys, np.arange(128)
            ).cpu().numpy()
            deltas = []
            references = []
            topk_recalls = []
            for step in range(count):
                length = int(trace.lengths[step])
                reference = trace.row(step)
                candidate = reconstructed[step, :length]
                deltas.append(np.abs(candidate - reference))
                references.append(np.abs(reference))
                topk = min(int(selection["k"]), length)
                expected_ids = stable_topk(reference, topk)
                actual_ids = stable_topk(candidate, topk)
                topk_recalls.append(np.isin(expected_ids, actual_ids).mean())
            all_delta = np.concatenate(deltas)
            all_reference = np.concatenate(references)
            reconstruction.append(
                {
                    "trace": str(trace_path),
                    "rows": count,
                    "max_abs": float(all_delta.max()),
                    "p99_abs": float(np.quantile(all_delta, 0.99)),
                    "mean_abs": float(all_delta.mean()),
                    "reference_abs_p99": float(np.quantile(all_reference, 0.99)),
                    "mean_abs_over_reference_abs_p99": float(
                        all_delta.mean() / max(np.quantile(all_reference, 0.99), 1e-12)
                    ),
                    "topk_recall_mean": float(np.mean(topk_recalls)),
                    "topk_recall_worst": float(np.min(topk_recalls)),
                }
            )
            del reconstructed, deltas, references, all_delta, all_reference

        score_cache: dict[str, np.ndarray] = {}
        for config_index, spec in enumerate(config_specs, start=1):
            group = score_group(spec)
            if group not in score_cache:
                score_cache[group] = compute_scores(
                    spec, trace.layer, queries, weights, keys
                ).cpu().numpy()
            partial = score_cache[group]
            config = verifier_config(spec)
            for policy_spec in policy_specs:
                policy = build_policy(policy_spec, trace.layer, scales)
                detail_rows: dict[int, dict[str, np.ndarray]] = {}
                tail_labels = None
                if tail_frame is not None:
                    matched = tail_frame[
                        (tail_frame.policy == policy.name)
                        & (tail_frame.layer == trace.layer)
                        & (tail_frame.prompt_id == trace.prompt_id)
                        & (
                            tail_frame.base_context_length
                            == int(trace.lengths[0]) - 1
                        )
                    ]
                    tail_labels = {
                        int(step): group.block_id.to_numpy(dtype=np.int64)
                        for step, group in matched.groupby("step")
                    }

                def collect(step: int, value: dict[str, np.ndarray]) -> None:
                    detail_rows[step] = value

                frame = replay_verifier_trace(
                    trace,
                    partial,
                    policy=policy,
                    config=config,
                    k=int(selection["k"]),
                    max_transitions=args.max_transitions,
                    tail_labels=tail_labels,
                    detail_callback=collect if args.save_details else None,
                )
                frame["policy_role"] = policy_spec["selection_role"]
                frame["strategy"] = spec["strategy"]
                frame["rotation"] = spec.get("rotation", "none")
                frame["precision"] = spec.get("precision", "bf16")
                rows.append(frame)
                if args.save_details:
                    save_details(
                        detail_root
                        / f"{trace_path.stem}__{policy_spec['selection_role']}__{spec['name']}.npz",
                        detail_rows,
                    )
            torch.cuda.empty_cache()
            print(
                f"[{trace_index}/{len(paths)} config {config_index}/{len(config_specs)}] "
                f"{trace_path.name} {spec['name']}",
                flush=True,
            )
        del queries, weights, keys, capture, score_cache
        torch.cuda.empty_cache()

    frame = pd.concat(rows, ignore_index=True)
    frame.to_csv(args.output / "verifier_rows.csv", index=False)
    summary = aggregate(frame)
    summary.to_csv(args.output / "verifier_summary.csv", index=False)
    audit = {
        "command": " ".join(__import__("sys").argv),
        "elapsed_seconds": time.perf_counter() - started,
        "gpus": gpu_names,
        "visible_gpu_ids": [0, 1],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "trace_count": len(paths),
        "config_count": len(config_specs),
        "policy_roles": args.policy_roles,
        "max_transitions": args.max_transitions,
        "score_reconstruction": reconstruction,
        "tail_definition": args.tail_definition if args.tail_labels else None,
        "tail_label_file": str(args.tail_labels) if args.tail_labels else None,
        "oracle_leakage": False,
        "duplicate_qk_accounting": False,
    }
    (args.output / "verifier_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
