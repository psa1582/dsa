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
from torch.nn import functional as F

from temporal_dsa.approx import ApproxPolicy, replay_approx_trace
from temporal_dsa.trace import load_trace
from temporal_dsa.v2_collect import MODEL_REVISION, fixed_device_map


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


def selection_trajectories(
    trace: Any,
    specs: list[dict[str, Any]],
    scales: dict[int, float],
    k: int,
    block_size: int,
    max_transitions: int | None,
) -> tuple[dict[str, dict[int, dict[str, np.ndarray]]], dict[str, pd.DataFrame]]:
    details: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    frames: dict[str, pd.DataFrame] = {}
    for spec in specs:
        policy = build_policy(spec, trace.layer, scales)
        policy_details: dict[int, dict[str, np.ndarray]] = {}

        def collect(step: int, value: dict[str, np.ndarray]) -> None:
            policy_details[step] = value

        frame = replay_approx_trace(
            trace,
            policy=policy,
            k=k,
            block_size=block_size,
            history_mode="own",
            detail_callback=collect,
            max_transitions=max_transitions,
        )
        details[str(spec["config_id"])] = policy_details
        frames[str(spec["config_id"])] = frame.set_index("step")
    return details, frames


@torch.no_grad()
def prepare_main_attention(
    module: torch.nn.Module,
    hidden_cpu: torch.Tensor,
    query_positions: np.ndarray,
    apply_rotary: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = module.o_proj.weight.device
    hidden = hidden_cpu.to(device=device, dtype=torch.bfloat16).unsqueeze(0)
    query_ids = torch.as_tensor(query_positions, device=device, dtype=torch.long)
    query_hidden = hidden[:, query_ids]
    if module.q_lora_rank is None:
        q = module.q_proj(query_hidden)
    else:
        q = module.q_b_proj(module.q_a_layernorm(module.q_a_proj(query_hidden)))
    q = q.view(1, query_ids.numel(), module.num_heads, module.q_head_dim).transpose(1, 2)
    q_nope, q_pe = torch.split(
        q, [module.qk_nope_head_dim, module.qk_rope_head_dim], dim=-1
    )

    compressed_kv = module.kv_a_proj_with_mqa(hidden)
    compressed_kv, k_pe = torch.split(
        compressed_kv, [module.kv_lora_rank, module.qk_rope_head_dim], dim=-1
    )
    k_pe = k_pe.view(1, hidden.shape[1], 1, module.qk_rope_head_dim).transpose(1, 2)
    kv = (
        module.kv_b_proj(module.kv_a_layernorm(compressed_kv))
        .view(
            1,
            hidden.shape[1],
            module.num_heads,
            module.qk_nope_head_dim + module.v_head_dim,
        )
        .transpose(1, 2)
    )
    k_nope, values = torch.split(kv, [module.qk_nope_head_dim, module.v_head_dim], dim=-1)
    cos, sin = module.rotary_emb(values, seq_len=hidden.shape[1])
    query_position_tensor = query_ids.view(1, -1)
    key_position_tensor = torch.arange(hidden.shape[1], device=device).view(1, -1)
    q_pe, _ = apply_rotary(q_pe, q_pe, cos, sin, query_position_tensor)
    _, k_pe = apply_rotary(k_pe, k_pe, cos, sin, key_position_tensor)
    queries = torch.cat((q_nope, q_pe), dim=-1).squeeze(0)
    keys = torch.cat((k_nope, k_pe.expand(-1, module.num_heads, -1, -1)), dim=-1).squeeze(0)
    return queries, keys, values.squeeze(0)


def vector_metrics(base: torch.Tensor, approx: torch.Tensor) -> tuple[float, float, float]:
    base32 = base.float().flatten()
    approx32 = approx.float().flatten()
    cosine = float(F.cosine_similarity(base32, approx32, dim=0, eps=1e-12).cpu())
    delta = approx32 - base32
    relative_l2 = float((torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(base32).clamp_min(1e-12)).cpu())
    max_abs = float(delta.abs().max().cpu())
    return cosine, relative_l2, max_abs


def distribution_divergence(
    logits: torch.Tensor, baseline: np.ndarray, approximate: np.ndarray
) -> tuple[float, float]:
    union = np.union1d(baseline, approximate)
    union_tensor = torch.as_tensor(union, device=logits.device, dtype=torch.long)
    union_logits = logits[:, union_tensor].float()
    base_mask = torch.as_tensor(np.isin(union, baseline), device=logits.device)
    approx_mask = torch.as_tensor(np.isin(union, approximate), device=logits.device)
    p = torch.softmax(union_logits.masked_fill(~base_mask, -torch.inf), dim=-1)
    q = torch.softmax(union_logits.masked_fill(~approx_mask, -torch.inf), dim=-1)
    midpoint = 0.5 * (p + q)
    eps = 1e-12
    js = 0.5 * (
        (p * ((p + eps).log() - (midpoint + eps).log())).sum(dim=-1)
        + (q * ((q + eps).log() - (midpoint + eps).log())).sum(dim=-1)
    )
    q_smooth = q + 1e-8
    q_smooth = q_smooth / q_smooth.sum(dim=-1, keepdim=True)
    kl = (p * ((p + eps).log() - q_smooth.log())).sum(dim=-1)
    return float(js.mean().cpu()), float(kl.mean().cpu())


@torch.no_grad()
def evaluate_trace(
    model: torch.nn.Module,
    trace_path: Path,
    specs: list[dict[str, Any]],
    scales: dict[int, float],
    k: int,
    block_size: int,
    max_transitions: int | None,
) -> list[dict[str, Any]]:
    trace = load_trace(trace_path)
    details, frames = selection_trajectories(
        trace, specs, scales, k, block_size, max_transitions
    )
    source_capture = Path(str(trace.metadata["source_capture"]))
    capture = torch.load(source_capture, map_location="cpu", weights_only=True)
    hidden = capture["hidden"]
    steps = sorted(next(iter(details.values())))
    positions = np.asarray([int(trace.lengths[step]) - 1 for step in steps], dtype=np.int64)
    module = model.model.layers[trace.layer].self_attn
    apply_rotary = module.__class__.forward.__globals__["apply_rotary_pos_emb"]
    queries, keys, values = prepare_main_attention(module, hidden, positions, apply_rotary)
    rows: list[dict[str, Any]] = []
    baseline_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for local_step, step in enumerate(steps):
        length = int(trace.lengths[step])
        logits = torch.einsum("hd,hkd->hk", queries[:, local_step], keys[:, :length])
        logits = logits * float(module.softmax_scale)
        for spec in specs:
            name = str(spec["config_id"])
            detail = details[name][step]
            baseline = detail["baseline"]
            approximate = detail["approximate"]
            if step not in baseline_cache:
                base_ids = torch.as_tensor(baseline, device=logits.device, dtype=torch.long)
                base_prob = torch.softmax(logits[:, base_ids].float(), dim=-1).to(values.dtype)
                base_heads = torch.matmul(base_prob.unsqueeze(1), values[:, base_ids]).squeeze(1)
                base_projected = module.o_proj(base_heads.reshape(1, 1, -1)).squeeze()
                baseline_cache[step] = (base_heads, base_projected, base_prob)
            base_heads, base_projected, _ = baseline_cache[step]
            approx_ids = torch.as_tensor(approximate, device=logits.device, dtype=torch.long)
            approx_prob = torch.softmax(logits[:, approx_ids].float(), dim=-1).to(values.dtype)
            approx_heads = torch.matmul(approx_prob.unsqueeze(1), values[:, approx_ids]).squeeze(1)
            approx_projected = module.o_proj(approx_heads.reshape(1, 1, -1)).squeeze()
            cosine, relative_l2, max_abs = vector_metrics(base_projected, approx_projected)
            pre_cosine, pre_relative_l2, _ = vector_metrics(base_heads, approx_heads)
            js, kl = distribution_divergence(logits, baseline, approximate)
            replay_row = frames[name].loc[step]
            rows.append(
                {
                    "policy": name,
                    "selection_role": spec["selection_role"],
                    "step": step,
                    "layer": trace.layer,
                    "prompt_id": trace.prompt_id,
                    "workload": trace.workload,
                    "base_context_length": int(trace.lengths[0]) - 1,
                    "context_length": length,
                    "qk_reduction": float(replay_row.qk_reduction),
                    "recall": float(replay_row.recall),
                    "output_cosine": cosine,
                    "output_relative_l2": relative_l2,
                    "output_max_abs": max_abs,
                    "preprojection_cosine": pre_cosine,
                    "preprojection_relative_l2": pre_relative_l2,
                    "attention_js": js,
                    "attention_kl_eps1e8": kl,
                }
            )
    del hidden, queries, keys, values, baseline_cache
    torch.cuda.empty_cache()
    return rows


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.groupby(["policy", "selection_role"], as_index=False).agg(
        observations=("output_cosine", "size"),
        qk_reduction_mean=("qk_reduction", "mean"),
        qk_reduction_median=("qk_reduction", "median"),
        output_cosine_mean=("output_cosine", "mean"),
        output_cosine_p1=("output_cosine", lambda x: x.quantile(0.01)),
        output_cosine_p5=("output_cosine", lambda x: x.quantile(0.05)),
        output_cosine_median=("output_cosine", "median"),
        output_cosine_worst=("output_cosine", "min"),
        output_relative_l2_mean=("output_relative_l2", "mean"),
        output_relative_l2_median=("output_relative_l2", "median"),
        output_relative_l2_p95=("output_relative_l2", lambda x: x.quantile(0.95)),
        output_relative_l2_p99=("output_relative_l2", lambda x: x.quantile(0.99)),
        output_relative_l2_worst=("output_relative_l2", "max"),
        output_max_abs_p95=("output_max_abs", lambda x: x.quantile(0.95)),
        attention_js_mean=("attention_js", "mean"),
        attention_js_p95=("attention_js", lambda x: x.quantile(0.95)),
        attention_kl_eps1e8_mean=("attention_kl_eps1e8", "mean"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Reference sparse MLA output validation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--trace-files", type=Path, nargs="+")
    parser.add_argument("--trace-roots", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-transitions", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    from transformers import AutoConfig, AutoModelForCausalLM

    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"expected CUDA_VISIBLE_DEVICES=0,1, found {torch.cuda.device_count()} GPUs")
    gpu_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("L40" in name for name in gpu_names):
        raise RuntimeError(f"L40 guard rejected {gpu_names}")
    selection = json.loads(args.selection.read_text())
    specs = selection["policies"]
    scales = {int(key): float(value) for key, value in selection["gamma_scale"].items()}
    k, block_size = int(selection["k"]), int(selection["block_size"])
    paths = list(args.trace_files or [])
    for root in args.trace_roots or []:
        paths.extend(sorted(root.rglob("*.npz")))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError("no score trace files")
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
        rows.extend(
            evaluate_trace(
                model,
                path,
                specs,
                scales,
                k,
                block_size,
                args.max_transitions,
            )
        )
        print(f"[{index}/{len(paths)}] {path}", flush=True)
    frame = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "mla_output_error.csv", index=False)
    summary = summarize(frame)
    summary.to_csv(args.output / "mla_output_summary.csv", index=False)
    frame.groupby(
        ["policy", "selection_role", "layer", "base_context_length", "workload"],
        as_index=False,
    ).agg(
        observations=("output_cosine", "size"),
        qk_reduction=("qk_reduction", "median"),
        output_cosine_median=("output_cosine", "median"),
        output_cosine_p5=("output_cosine", lambda x: x.quantile(0.05)),
        output_relative_l2_median=("output_relative_l2", "median"),
        output_relative_l2_p95=("output_relative_l2", lambda x: x.quantile(0.95)),
        attention_js_mean=("attention_js", "mean"),
    ).to_csv(args.output / "mla_output_breakdown.csv", index=False)
    audit = {
        "model_revision": args.revision,
        "gpus": gpu_names,
        "visible_gpu_ids": [0, 1],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda": torch.version.cuda,
        "trace_count": len(paths),
        "max_transitions": args.max_transitions,
        "elapsed_seconds": time.perf_counter() - started,
        "reference_only": True,
    }
    (args.output / "mla_output_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
