from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from evaluate_mla_output import (
    distribution_divergence,
    prepare_main_attention,
    vector_metrics,
)
from temporal_dsa.trace import load_trace
from temporal_dsa.v2_collect import MODEL_REVISION, fixed_device_map


def discover(files: list[Path] | None, roots: list[Path] | None) -> list[Path]:
    paths = list(files or [])
    for root in roots or []:
        paths.extend(sorted(root.rglob("*.npz")))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError("no score traces")
    return paths


def load_details(root: Path, trace_stem: str) -> list[tuple[str, str, dict[str, np.ndarray]]]:
    result = []
    for path in sorted(root.glob(f"{trace_stem}__*__*.npz")):
        prefix, policy_role, verifier = path.stem.split("__", 2)
        if prefix != trace_stem:
            continue
        with np.load(path, allow_pickle=True) as payload:
            result.append(
                (
                    policy_role,
                    verifier,
                    {key: payload[key] for key in payload.files},
                )
            )
    return result


def head_metrics(base: torch.Tensor, approximate: torch.Tensor) -> tuple[float, float]:
    base32 = base.float()
    delta = approximate.float() - base32
    relative = torch.linalg.vector_norm(delta, dim=-1) / torch.linalg.vector_norm(
        base32, dim=-1
    ).clamp_min(1e-12)
    return float(relative.mean().cpu()), float(relative.max().cpu())


@torch.no_grad()
def selected_output(module, logits, values, ids: np.ndarray):
    selected = torch.as_tensor(ids, device=logits.device, dtype=torch.long)
    probabilities = torch.softmax(logits[:, selected].float(), dim=-1).to(values.dtype)
    heads = torch.matmul(probabilities.unsqueeze(1), values[:, selected]).squeeze(1)
    projected = module.o_proj(heads.reshape(1, 1, -1)).squeeze()
    return heads, projected


@torch.no_grad()
def evaluate_trace(model, trace_path: Path, detail_root: Path) -> list[dict[str, Any]]:
    trace = load_trace(trace_path)
    methods = load_details(detail_root, trace_path.stem)
    if not methods:
        raise FileNotFoundError(f"no selection detail for {trace_path.stem}")
    capture = torch.load(trace.metadata["source_capture"], map_location="cpu", weights_only=True)
    hidden = capture["hidden"]
    steps = methods[0][2]["steps"].astype(np.int64)
    positions = np.asarray([int(trace.lengths[step]) - 1 for step in steps], dtype=np.int64)
    module = model.model.layers[trace.layer].self_attn
    apply_rotary = module.__class__.forward.__globals__["apply_rotary_pos_emb"]
    queries, keys, values = prepare_main_attention(module, hidden, positions, apply_rotary)
    rows: list[dict[str, Any]] = []
    for local_step, step in enumerate(steps):
        length = int(trace.lengths[step])
        logits = torch.einsum("hd,hkd->hk", queries[:, local_step], keys[:, :length])
        logits = logits * float(module.softmax_scale)
        baseline = methods[0][2]["baseline"][local_step].astype(np.int64)
        base_heads, base_projected = selected_output(
            module, logits, values[:, :length], baseline
        )
        base_norm = float(torch.linalg.vector_norm(base_projected.float()).cpu())
        for policy_role, verifier, details in methods:
            if int(details["steps"][local_step]) != int(step):
                raise RuntimeError("selection detail steps are misaligned")
            if not np.array_equal(baseline, details["baseline"][local_step]):
                raise RuntimeError("baseline selection differs across verifier methods")
            approximate = details["approximate"][local_step].astype(np.int64)
            approx_heads, approx_projected = selected_output(
                module, logits, values[:, :length], approximate
            )
            cosine, relative_l2, max_abs = vector_metrics(base_projected, approx_projected)
            pre_cosine, pre_relative_l2, _ = vector_metrics(base_heads, approx_heads)
            per_head_mean, per_head_worst = head_metrics(base_heads, approx_heads)
            js, kl = distribution_divergence(logits, baseline, approximate)
            rows.append(
                {
                    "policy_role": policy_role,
                    "verifier": verifier,
                    "step": int(step),
                    "layer": trace.layer,
                    "prompt_id": trace.prompt_id,
                    "workload": trace.workload,
                    "base_context_length": int(trace.lengths[0]) - 1,
                    "context_length": length,
                    "output_cosine": cosine,
                    "output_relative_l2": relative_l2,
                    "output_max_abs": max_abs,
                    "output_norm_ratio": float(
                        torch.linalg.vector_norm(approx_projected.float()).cpu()
                        / max(base_norm, 1e-12)
                    ),
                    "preprojection_cosine": pre_cosine,
                    "preprojection_relative_l2": pre_relative_l2,
                    "per_head_relative_l2_mean": per_head_mean,
                    "per_head_relative_l2_worst": per_head_worst,
                    "attention_js": js,
                    "attention_kl_eps1e8": kl,
                }
            )
    del hidden, queries, keys, values
    torch.cuda.empty_cache()
    return rows


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["policy_role", "verifier"], as_index=False).agg(
        observations=("output_cosine", "size"),
        output_cosine_p50=("output_cosine", "median"),
        output_cosine_p5=("output_cosine", lambda x: x.quantile(0.05)),
        output_cosine_p1=("output_cosine", lambda x: x.quantile(0.01)),
        output_relative_l2_p50=("output_relative_l2", "median"),
        output_relative_l2_p90=("output_relative_l2", lambda x: x.quantile(0.90)),
        output_relative_l2_p95=("output_relative_l2", lambda x: x.quantile(0.95)),
        output_relative_l2_p99=("output_relative_l2", lambda x: x.quantile(0.99)),
        output_relative_l2_max=("output_relative_l2", "max"),
        output_max_abs_p95=("output_max_abs", lambda x: x.quantile(0.95)),
        output_norm_ratio_p5=("output_norm_ratio", lambda x: x.quantile(0.05)),
        output_norm_ratio_p95=("output_norm_ratio", lambda x: x.quantile(0.95)),
        per_head_relative_l2_mean=("per_head_relative_l2_mean", "mean"),
        per_head_relative_l2_worst_p95=(
            "per_head_relative_l2_worst", lambda x: x.quantile(0.95)
        ),
        attention_js_mean=("attention_js", "mean"),
        attention_kl_mean=("attention_kl_eps1e8", "mean"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Actual MLA output for verifier selections")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--trace-files", type=Path, nargs="+")
    parser.add_argument("--trace-roots", type=Path, nargs="+")
    parser.add_argument("--detail-root", type=Path, required=True)
    parser.add_argument("--verifier-rows", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    from transformers import AutoConfig, AutoModelForCausalLM

    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"expected two visible GPUs, found {torch.cuda.device_count()}")
    gpu_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("L40" in value for value in gpu_names):
        raise RuntimeError(f"L40 guard rejected {gpu_names}")
    paths = discover(args.trace_files, args.trace_roots)
    config = AutoConfig.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=fixed_device_map(config.num_hidden_layers),
        attn_implementation="eager",
    ).eval()
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(paths, start=1):
        rows.extend(evaluate_trace(model, path, args.detail_root))
        print(f"[{index}/{len(paths)}] {path}", flush=True)
    frame = pd.DataFrame(rows)
    if args.verifier_rows is not None:
        replay = pd.read_csv(args.verifier_rows)
        join = [
            "policy_role", "verifier", "step", "layer", "prompt_id",
            "base_context_length", "context_length",
        ]
        costs = replay[join + [
            "net_qk_reduction", "physical_key_byte_reduction", "top128_recall",
            "top512_recall", "top2048_recall", "tail_critical_block_recall",
        ]]
        frame = frame.merge(costs, on=join, how="left", validate="one_to_one")
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "mla_output_quality.csv", index=False)
    summary = summarize(frame)
    if "net_qk_reduction" in frame:
        cost_summary = frame.groupby(["policy_role", "verifier"], as_index=False).agg(
            net_qk_reduction_median=("net_qk_reduction", "median"),
            physical_key_byte_reduction_median=("physical_key_byte_reduction", "median"),
            top128_recall=("top128_recall", "mean"),
            top512_recall=("top512_recall", "mean"),
            top2048_recall=("top2048_recall", "mean"),
            tail_critical_block_recall=("tail_critical_block_recall", "mean"),
        )
        summary = summary.merge(cost_summary, on=["policy_role", "verifier"], how="left")
    summary.to_csv(args.output / "mla_output_summary.csv", index=False)
    audit = {
        "command": " ".join(__import__("sys").argv),
        "elapsed_seconds": time.perf_counter() - started,
        "gpus": gpu_names,
        "visible_gpu_ids": [0, 1],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "trace_count": len(paths),
        "method_count": int(frame[["policy_role", "verifier"]].drop_duplicates().shape[0]),
        "model_revision": args.revision,
    }
    (args.output / "mla_output_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
