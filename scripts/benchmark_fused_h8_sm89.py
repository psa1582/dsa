from __future__ import annotations

import argparse
import json
import math
import platform
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable

import numpy as np
import pandas as pd
import torch

from temporal_dsa.fused_h8_sm89 import BLOCK_SIZE, load_extension, pack_query
from temporal_dsa.verifier_scoring import load_sidecar_encoded


CONTEXTS = (8192, 16384, 32768, 65536, 131072)
REAL_CONTEXTS = (8192, 16384, 32768)
DIRECT_FRACTIONS = {8192: 0.945519, 16384: 0.703342, 32768: 0.594227}
CAPTURES = ("code_heldout_3", "text_heldout_28444", "code_heldout_5")
PROMOTION_RATES = (0.0, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0)


def stat_row(values_ms: list[float]) -> dict[str, float]:
    values = np.asarray(values_ms, dtype=np.float64) * 1000.0
    return {
        "median_us": float(np.median(values)),
        "mean_us": float(values.mean()),
        "p5_us": float(np.percentile(values, 5)),
        "p95_us": float(np.percentile(values, 95)),
        "std_us": float(values.std()),
        "measurements": int(values.size),
    }


def cuda_bench(
    fn: Callable[[int], None],
    *,
    variants: int,
    flush: torch.Tensor | None,
    warmup: int,
    measure: int,
) -> dict[str, float]:
    for iteration in range(warmup):
        if flush is not None:
            flush.add_(1)
        fn(iteration % variants)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(measure)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(measure)]
    for iteration, (start, end) in enumerate(zip(starts, ends)):
        if flush is not None:
            flush.add_(1)
        start.record()
        fn(iteration % variants)
        end.record()
    torch.cuda.synchronize()
    return stat_row([start.elapsed_time(end) for start, end in zip(starts, ends)])


@dataclass
class Item:
    q: torch.Tensor
    w: torch.Tensor
    keys: torch.Tensor
    packed_q: torch.Tensor
    packed_w: torch.Tensor
    packed_ids: torch.Tensor
    output: torch.Tensor
    h8: torch.Tensor
    maxima: torch.Tensor
    accepted: torch.Tensor
    direct: torch.Tensor
    p0_keep: torch.Tensor
    local_threshold: float

    @property
    def blocks(self) -> int:
        return self.maxima.numel()


def load_real_variant(repo: Path, context: int, name: str, layer: int) -> tuple[torch.Tensor, ...]:
    capture = repo / "artifacts" / "pilot" / f"trace_capture_{context // 1024}k" / name / f"layer_{layer:02d}.pt"
    checkpoint = repo / "artifacts" / "pilot" / "indexers_1000" / "checkpoints" / f"layer_{layer:02d}.safetensors"
    metadata = torch.load(capture, map_location="cpu", weights_only=True)
    length = min(context, int(metadata["hidden"].shape[0]))
    del metadata
    q, w, keys, _ = load_sidecar_encoded(
        capture, checkpoint, np.asarray([length - 1]), device="cuda:0"
    )
    return (
        q[0].to(torch.bfloat16).contiguous(),
        w[0].float().contiguous(),
        keys[:length].to(torch.bfloat16).contiguous(),
    )


def deterministic_mask(blocks: int, count: int, seed: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(blocks, generator=gen)
    mask = torch.zeros(blocks, dtype=torch.bool)
    mask[order[:count]] = True
    return mask.to(device)


def prepare_item(
    ext,
    q: torch.Tensor,
    w: torch.Tensor,
    keys: torch.Tensor,
    *,
    context: int,
    seed: int,
    local_threshold: float,
) -> Item:
    packed = pack_query(q, w)
    length = keys.shape[0]
    blocks = math.ceil(length / BLOCK_SIZE)
    output = torch.empty(length, device=keys.device, dtype=torch.float32)
    h8 = torch.empty_like(output)
    maxima = torch.empty(blocks, device=keys.device, dtype=torch.float32)
    accepted = torch.empty(blocks, device=keys.device, dtype=torch.bool)
    direct_fraction = DIRECT_FRACTIONS.get(context, DIRECT_FRACTIONS[32768])
    direct = deterministic_mask(blocks, round(blocks * direct_fraction), seed, keys.device)
    ext.h8_pass(packed.q, packed.w, keys, h8, maxima)
    torch.cuda.synchronize()
    cold = ~direct
    rescue_count = min(int(cold.sum()), max(1, round(0.10 * int(cold.sum()))))
    cold_max = maxima.masked_fill(~cold, -math.inf)
    rescue = torch.topk(cold_max, rescue_count, sorted=False).indices
    p0_keep = direct.clone()
    p0_keep[rescue] = True
    return Item(
        q, w, keys, packed.q, packed.w, packed.head_ids, output, h8, maxima,
        accepted, direct, p0_keep, local_threshold,
    )


def load_items(repo: Path, ext, layer: int, local_threshold: float) -> dict[int, list[Item]]:
    result: dict[int, list[Item]] = {}
    for context in REAL_CONTEXTS:
        result[context] = []
        for variant, name in enumerate(CAPTURES):
            q, w, keys = load_real_variant(repo, context, name, layer)
            result[context].append(
                prepare_item(
                    ext, q, w, keys, context=context, seed=1582 + context + variant,
                    local_threshold=local_threshold,
                )
            )
    base = result[32768]
    for context in (65536, 131072):
        result[context] = []
        for variant, item in enumerate(base):
            repeats = math.ceil(context / item.keys.shape[0])
            keys = item.keys.repeat((repeats, 1))[:context].contiguous()
            result[context].append(
                prepare_item(
                    ext, item.q, item.w, keys, context=context,
                    seed=1582 + context + variant, local_threshold=local_threshold,
                )
            )
    return result


def trace_mask_pool(repo: Path, context: int, device: torch.device) -> list[torch.Tensor]:
    root = repo / "artifacts" / "progressive_threshold" / "selection_details"
    candidates = sorted(root.glob(f"*c{context}_layer_17__Aggressive__head_dynamic_abs_w_w8_b64_threshold_r0.1.npz"))
    if not candidates:
        return []
    detail = np.load(candidates[0], allow_pickle=True)
    rows = pd.read_csv(repo / "artifacts" / "progressive_threshold" / "threshold_replay_rows.csv")
    stem = candidates[0].stem
    prompt = re.match(r"(.+)_c\d+_layer_17", stem).group(1)
    rows = rows[
        rows.prompt_id.eq(prompt)
        & rows.base_context_length.eq(context)
        & rows.layer.eq(17)
        & rows.head_scheme.eq("dynamic_abs_w")
        & rows.promotion_target.eq(0.10)
        & rows.policy_role.eq("Aggressive")
    ].sort_values("step")
    masks: list[torch.Tensor] = []
    for index, step in enumerate(detail["steps"]):
        matched = rows[rows.step.eq(int(step))]
        if matched.empty:
            continue
        row = matched.iloc[0]
        # Normalize growing decode steps back to the requested benchmark
        # context so every rotated mask has the same launch shape.
        blocks = math.ceil(context / BLOCK_SIZE)
        direct_target = min(blocks, math.ceil(int(row.temporal_exact_tokens) / BLOCK_SIZE))
        rescue = np.asarray(detail["rescued_blocks"][index], dtype=np.int64)
        seeds = np.unique(np.asarray(detail["approximate"][index], dtype=np.int64) // BLOCK_SIZE)
        direct: list[int] = []
        seen = set(int(x) for x in rescue)
        for block in seeds:
            block = int(block)
            if block not in seen and block < blocks:
                direct.append(block)
                seen.add(block)
            if len(direct) >= direct_target:
                break
        generator = np.random.default_rng(1582 + int(step))
        for block in generator.permutation(blocks):
            block = int(block)
            if block not in seen:
                direct.append(block)
                seen.add(block)
            if len(direct) >= direct_target:
                break
        mask = torch.zeros(blocks, dtype=torch.bool)
        if direct:
            mask[torch.as_tensor(direct[:direct_target])] = True
        if rescue.size:
            mask[torch.as_tensor(rescue[rescue < blocks])] = True
        masks.append(mask.to(device))
    return masks


def method_timing(
    ext,
    items: list[Item],
    method: str,
    *,
    cache_state: str,
    flush: torch.Tensor,
    warmup: int,
    measure: int,
    ctas_per_sm: int = 1,
    producer_warps: int = 0,
    masks: list[torch.Tensor] | None = None,
) -> dict[str, float | int | str]:
    masks = masks or []

    def call(index: int) -> None:
        item = items[index % len(items)]
        mask = masks[index % len(masks)] if masks else item.p0_keep
        if mask.numel() != item.blocks:
            mask = item.p0_keep
        if method == "K1_full_sync":
            ext.full64_sync(item.packed_q, item.packed_w, item.keys, item.output)
        elif method == "K2_full_pipeline":
            ext.full64_pipeline(
                item.packed_q, item.packed_w, item.keys, item.output,
                ctas_per_sm, producer_warps,
            )
        elif method == "K3_two_pass_precomputed":
            ext.h8_pass(item.packed_q, item.packed_w, item.keys, item.h8, item.maxima)
            ext.h56_pass(item.packed_q, item.packed_w, item.keys, mask, item.h8, item.output)
        elif method == "K3_two_pass_online_local":
            ext.h8_pass(item.packed_q, item.packed_w, item.keys, item.h8, item.maxima)
            ext.select_threshold(item.maxima, item.direct, item.local_threshold, item.accepted)
            ext.h56_pass(item.packed_q, item.packed_w, item.keys, item.accepted, item.h8, item.output)
        elif method == "K4_fused_mask":
            ext.fused_mask_sync(item.packed_q, item.packed_w, item.keys, mask, item.output, item.maxima)
        elif method == "K5_fused_online_sync":
            ext.fused_online_sync(
                item.packed_q, item.packed_w, item.keys, item.direct,
                item.local_threshold, item.output, item.accepted, item.maxima,
            )
        elif method == "K6_fused_online_pipeline":
            ext.fused_online_pipeline(
                item.packed_q, item.packed_w, item.keys, item.direct,
                item.local_threshold, item.output, item.accepted, item.maxima,
                ctas_per_sm, producer_warps,
            )
        else:
            raise ValueError(method)

    variants = max(len(items), len(masks)) if masks else len(items)
    stats = cuda_bench(
        call, variants=variants,
        flush=flush if cache_state == "cold_rotating" else None,
        warmup=warmup, measure=measure,
    )
    return {"method": method, "cache_state": cache_state, **stats}


def correctness_rows(ext, items_by_context: dict[int, list[Item]]) -> list[dict]:
    rows: list[dict] = []
    for context in REAL_CONTEXTS:
        for variant, item in enumerate(items_by_context[context]):
            reference = (torch.relu(item.q.float() @ item.keys.float().T) * item.w[:, None]).sum(0)
            methods = {
                "K1_full_sync": lambda: ext.full64_sync(item.packed_q, item.packed_w, item.keys, item.output),
                "K2_full_pipeline": lambda: ext.full64_pipeline(item.packed_q, item.packed_w, item.keys, item.output, 1, 0),
                "K3_two_pass": lambda: (
                    ext.h8_pass(item.packed_q, item.packed_w, item.keys, item.h8, item.maxima),
                    ext.h56_pass(item.packed_q, item.packed_w, item.keys, item.p0_keep, item.h8, item.output),
                ),
                "K4_fused_mask": lambda: ext.fused_mask_sync(item.packed_q, item.packed_w, item.keys, item.p0_keep, item.output, item.maxima),
                "K6_fused_online": lambda: ext.fused_online_pipeline(
                    item.packed_q, item.packed_w, item.keys, item.direct, math.inf,
                    item.output, item.accepted, item.maxima, 1, 0,
                ),
            }
            for name, fn in methods.items():
                fn()
                torch.cuda.synchronize()
                keep = item.p0_keep if name in {"K3_two_pass", "K4_fused_mask"} else (
                    item.direct if name == "K6_fused_online" else torch.ones(item.blocks, device="cuda", dtype=torch.bool)
                )
                token_keep = keep.repeat_interleave(BLOCK_SIZE)[: item.keys.shape[0]]
                error = (item.output[token_keep] - reference[token_keep]).abs()
                top = min(128, int(token_keep.sum()))
                actual_ids = torch.topk(item.output, top).indices
                expected_masked = reference.masked_fill(~token_keep, -math.inf)
                expected_ids = torch.topk(expected_masked, top).indices
                recall = torch.isin(expected_ids, actual_ids).float().mean()
                rows.append({
                    "context": context,
                    "variant": variant,
                    "method": name,
                    "max_abs_error": float(error.max()) if error.numel() else 0.0,
                    "mean_abs_error": float(error.mean()) if error.numel() else 0.0,
                    "top128_recall_vs_reference": float(recall),
                    "rejected_all_negative_inf": bool(torch.isneginf(item.output[~token_keep]).all()) if (~token_keep).any() else True,
                    "passed": bool((not error.numel() or float(error.max()) <= 2e-3) and float(recall) == 1.0),
                })
    return rows


def command_text(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--measure", type=int, default=1000)
    parser.add_argument("--layer", type=int, default=17)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("performance process requires exactly one visible GPU")
    if torch.cuda.get_device_capability(0) != (8, 9):
        raise RuntimeError("SM89 is required")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    ext = load_extension(verbose=False)
    runtime_config = json.loads(
        (args.repo / "artifacts" / "progressive_sw" / "threshold" / "runtime_config.json").read_text()
    )
    threshold = float(runtime_config["configs"][0]["promotion_threshold_by_layer"][str(args.layer)])
    items_by_context = load_items(args.repo, ext, args.layer, threshold)
    l2_bytes = int(ext.device_properties()["l2_cache_bytes"])
    flush = torch.empty((4 * l2_bytes) // 4, device="cuda", dtype=torch.float32)
    mask_pools = {context: trace_mask_pool(args.repo, context, torch.device("cuda")) for context in REAL_CONTEXTS}

    correctness = pd.DataFrame(correctness_rows(ext, items_by_context))
    correctness.to_csv(args.output / "correctness_results.csv", index=False)

    timing: dict[str, list[dict]] = {name: [] for name in (
        "full64_sync_timing", "full64_pipeline_timing", "two_pass_h8_timing",
        "fused_h8_mask_timing", "fused_h8_online_timing",
    )}
    # Select the best persistent full configuration using the required 32K cold sweep.
    warp_rows = []
    for ctas in (1, 2, 3):
        for producers in (0, 1, 4):
            for method in ("K2_full_pipeline", "K6_fused_online_pipeline"):
                row = method_timing(
                    ext, items_by_context[32768], method, cache_state="cold_rotating",
                    flush=flush, warmup=args.warmup, measure=args.measure,
                    ctas_per_sm=ctas, producer_warps=producers,
                    masks=mask_pools[32768],
                )
                warp_rows.append({
                    "context": 32768, "ctas_per_sm": ctas,
                    "producer_warps": producers, "total_warps": 4 + producers,
                    **row,
                })
    warp_frame = pd.DataFrame(warp_rows)
    warp_frame.to_csv(args.output / "warp_config_sweep.csv", index=False)
    k2_best = warp_frame[warp_frame.method.eq("K2_full_pipeline")].sort_values("median_us").iloc[0]
    k6_best = warp_frame[warp_frame.method.eq("K6_fused_online_pipeline")].sort_values("median_us").iloc[0]
    k2_config = (int(k2_best.ctas_per_sm), int(k2_best.producer_warps))
    k6_config = (int(k6_best.ctas_per_sm), int(k6_best.producer_warps))

    for context in CONTEXTS:
        for cache_state in ("hot_l2", "cold_rotating"):
            base = {"context": context, "dtype": "bf16", "block_size": 64}
            row = method_timing(
                ext, items_by_context[context], "K1_full_sync", cache_state=cache_state,
                flush=flush, warmup=args.warmup, measure=args.measure,
            )
            timing["full64_sync_timing"].append({**base, **row})
            row = method_timing(
                ext, items_by_context[context], "K2_full_pipeline", cache_state=cache_state,
                flush=flush, warmup=args.warmup, measure=args.measure,
                ctas_per_sm=k2_config[0], producer_warps=k2_config[1],
            )
            timing["full64_pipeline_timing"].append({
                **base, "ctas_per_sm": k2_config[0], "producer_warps": k2_config[1], **row,
            })
            if context in REAL_CONTEXTS:
                masks = mask_pools[context]
                for method in ("K3_two_pass_precomputed", "K3_two_pass_online_local"):
                    row = method_timing(
                        ext, items_by_context[context], method, cache_state=cache_state,
                        flush=flush, warmup=args.warmup, measure=args.measure, masks=masks,
                    )
                    timing["two_pass_h8_timing"].append({**base, **row})
                row = method_timing(
                    ext, items_by_context[context], "K4_fused_mask", cache_state=cache_state,
                    flush=flush, warmup=args.warmup, measure=args.measure, masks=masks,
                )
                timing["fused_h8_mask_timing"].append({
                    **base, "claim_scope": "DATAFLOW-ONLY / PRECOMPUTED-MASK", **row,
                })
                for method in ("K5_fused_online_sync", "K6_fused_online_pipeline"):
                    row = method_timing(
                        ext, items_by_context[context], method, cache_state=cache_state,
                        flush=flush, warmup=args.warmup, measure=args.measure,
                        ctas_per_sm=k6_config[0], producer_warps=k6_config[1], masks=masks,
                    )
                    timing["fused_h8_online_timing"].append({
                        **base, "promotion_policy": "validation_fixed_threshold",
                        "threshold": threshold, "ctas_per_sm": k6_config[0],
                        "producer_warps": k6_config[1], **row,
                    })
    for name, rows in timing.items():
        pd.DataFrame(rows).to_csv(args.output / f"{name}.csv", index=False)

    # Synthetic random/clustered continuation patterns isolate branch behavior.
    promotion_rows = []
    for context in (16384, 32768):
        items = items_by_context[context]
        for rate in PROMOTION_RATES:
            for pattern in ("random", "clustered"):
                masks = []
                for variant, item in enumerate(items):
                    count = round(rate * item.blocks)
                    if pattern == "clustered":
                        mask = torch.zeros(item.blocks, device="cuda", dtype=torch.bool)
                        mask[:count] = True
                    else:
                        mask = deterministic_mask(item.blocks, count, 8000 + int(rate * 100) + variant, item.keys.device)
                    masks.append(mask)
                row = method_timing(
                    ext, items, "K4_fused_mask", cache_state="cold_rotating",
                    flush=flush, warmup=args.warmup, measure=args.measure, masks=masks,
                )
                promotion_rows.append({
                    "context": context, "promotion_rate": rate, "pattern": pattern,
                    "accepted_fraction": rate,
                    "qk_reduction": 1.0 - (8.0 + 56.0 * rate) / 64.0,
                    **row,
                })
    promotion_frame = pd.DataFrame(promotion_rows)
    k2_cold = pd.DataFrame(timing["full64_pipeline_timing"])
    promotion_frame["k2_median_us"] = promotion_frame.context.map(
        k2_cold[k2_cold.cache_state.eq("cold_rotating")].set_index("context").median_us
    )
    promotion_frame["speedup_vs_k2"] = promotion_frame.k2_median_us / promotion_frame.median_us
    promotion_frame.to_csv(args.output / "promotion_rate_sweep.csv", index=False)

    # Shared-layout/Q-placement ablation on K1, 32K.
    shared_rows = []
    layout_names = {0: "plain_row_major", 1: "padded_stride_136", 2: "xor_vector_swizzle"}
    items = items_by_context[32768]
    for layout_id, layout_name in layout_names.items():
        for q_shared in (False, True):
            for cache_state in ("hot_l2", "cold_rotating"):
                def variant_call(index: int, layout_id=layout_id, q_shared=q_shared):
                    item = items[index % len(items)]
                    ext.full64_sync_variant(
                        item.packed_q, item.packed_w, item.keys, item.output,
                        layout_id, q_shared,
                    )
                stats = cuda_bench(
                    variant_call, variants=len(items),
                    flush=flush if cache_state == "cold_rotating" else None,
                    warmup=args.warmup, measure=args.measure,
                )
                stride = 136 if layout_id == 1 else 128
                shared_rows.append({
                    "context": 32768, "layout": layout_name,
                    "q_storage": "shared" if q_shared else "readonly_global_cache",
                    "cache_state": cache_state,
                    "shared_memory_bytes_per_cta": 64 * stride * 2 + (64 * 128 * 2 if q_shared else 0),
                    **stats,
                })
    pd.DataFrame(shared_rows).to_csv(args.output / "shared_layout_sweep.csv", index=False)

    cache_rows = []
    for key in timing:
        for row in timing[key]:
            if row["context"] in (16384, 32768):
                cache_rows.append(row)
    pd.DataFrame(cache_rows).to_csv(args.output / "cache_state_results.csv", index=False)

    full_frame = pd.concat([
        pd.DataFrame(timing["full64_sync_timing"]),
        pd.DataFrame(timing["full64_pipeline_timing"]),
    ])
    online_frame = pd.DataFrame(timing["fused_h8_online_timing"])
    primary_speedups = []
    for context in (16384, 32768):
        baseline = full_frame[
            full_frame.context.eq(context) & full_frame.cache_state.eq("cold_rotating")
        ].median_us.min()
        proposed = online_frame[
            online_frame.context.eq(context)
            & online_frame.cache_state.eq("cold_rotating")
            & online_frame.method.eq("K6_fused_online_pipeline")
        ].median_us.iloc[0]
        primary_speedups.append(baseline / proposed)
    topk_rows = [{
        "status": "SKIPPED_BY_EARLY_STOP" if min(primary_speedups) < 1.05 else "RUN",
        "reason": "K6 scoring did not reach 1.05x versus fastest optimized full baseline" if min(primary_speedups) < 1.05 else "scoring gate passed",
        "context": context,
        "scoring_speedup_vs_fastest_full": speedup,
        "scoring_plus_topk_speedup": math.nan,
        "sidecar_tpot_impact": math.nan,
    } for context, speedup in zip((16384, 32768), primary_speedups)]
    pd.DataFrame(topk_rows).to_csv(args.output / "topk_integration_results.csv", index=False)

    device_info = dict(ext.device_properties())
    device_info.update({
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "python": platform.python_version(),
        "nvcc": command_text(["/usr/local/cuda-12.8/bin/nvcc", "--version"]),
        "nvidia_smi_query": command_text([
            "nvidia-smi", "--query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu,temperature.gpu,clocks.sm,clocks.mem",
            "--format=csv,noheader",
        ]),
        "performance_gpu_visible_index": 0,
        "physical_gpu_policy": "CUDA_VISIBLE_DEVICES=0; physical GPU 2/3 untouched",
    })
    (args.output / "device_info.json").write_text(json.dumps(device_info, indent=2) + "\n")
    selected = {
        "compile_target": "-arch=sm_89",
        "block_size": 64,
        "tensor_core_instruction": "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
        "k2": {"ctas_per_sm": k2_config[0], "producer_warps": k2_config[1]},
        "k6": {"ctas_per_sm": k6_config[0], "producer_warps": k6_config[1]},
        "local_threshold_layer17": threshold,
        "mask_pool_sizes": {str(k): len(v) for k, v in mask_pools.items()},
        "effective_full_baseline": "min(K1_sync,K2_pipeline), disclosed per context",
    }
    (args.output / "selected_kernel_config.json").write_text(json.dumps(selected, indent=2) + "\n")
    reproducibility = {
        "command": " ".join(__import__("sys").argv),
        "started_unix": started,
        "elapsed_seconds": time.time() - started,
        "seed": 1582,
        "warmup": args.warmup,
        "measurements": args.measure,
        "timing": "CUDA events; flush outside timed interval",
        "cold_cache": f"{4*l2_bytes} byte flush buffer (4x detected L2)",
        "trace_rotation": list(CAPTURES),
        "mask_protocol": "127-step trace-detail/replay-count reconstructed masks when available",
        "quality_source": "locked prior P0/P1 replay artifacts; CUDA mathematical equivalence rechecked",
    }
    (args.output / "reproducibility.json").write_text(json.dumps(reproducibility, indent=2) + "\n")
    print(json.dumps({"selected": selected, "speedups": primary_speedups}, indent=2))


if __name__ == "__main__":
    main()
