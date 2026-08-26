from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on CPU-only development hosts
    triton = None
    tl = None

from temporal_dsa.progressive_kernel import (
    HEAD_DIM,
    _full_score_kernel,
    _h8_block_max_kernel,
    _masked_full_kernel,
    _progressive_kernel,
    split_head_ids,
)


METHODS = (
    "full64",
    "topk_only",
    "full64_topk",
    "h8",
    "progressive_h8",
    "fused_dense",
    "fused_compact",
    "fused_topk",
    "t1",
)
METHOD_IDS = {
    "full64": "L0",
    "topk_only": "L1",
    "full64_topk": "L2",
    "h8": "L3",
    "progressive_h8": "L4",
    "fused_dense": "L5",
    "fused_compact": "L6",
    "fused_topk": "L7",
    "t1": "L8",
}


if triton is not None:

    @triton.jit
    def _t1_reconstruct_kernel(
        h8_ptr,
        previous_ptr,
        coefficients_ptr,
        output_ptr,
        length,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < length
        current_h8 = tl.load(h8_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
        previous = tl.load(previous_ptr + offsets, mask=valid, other=-float("inf")).to(
            tl.float32
        )
        a = tl.load(coefficients_ptr)
        b = tl.load(coefficients_ptr + 1)
        c = tl.load(coefficients_ptr + 2)
        tl.store(output_ptr + offsets, a * current_h8 + b * previous + c, mask=valid)


def discover_bundles(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    bundles = sorted(path.glob("*.pt"))
    if not bundles:
        raise FileNotFoundError(f"no .pt replay bundles under {path}")
    return bundles


def select_observation(bundle: dict[str, Any], selector: str) -> int:
    observations = bundle["metadata"]["observations"]
    if selector.isdigit():
        index = int(selector)
        if not 0 <= index < len(observations):
            raise IndexError(f"observation {index} outside [0, {len(observations)})")
        return index
    for index, row in enumerate(observations):
        if row["category"] == selector:
            return index
    available = [row["category"] for row in observations]
    raise KeyError(f"category {selector!r} unavailable; choose from {available}")


def load_item(path: Path, observation: str, device: torch.device, topk: int) -> dict[str, Any]:
    bundle = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if bundle.get("schema_version") != "dsa-replay-v1":
        raise ValueError(f"unsupported bundle schema in {path}")
    index = select_observation(bundle, observation)
    metadata = bundle["metadata"]
    observation_metadata = metadata["observations"][index]
    length = int(bundle["lengths"][index])
    block_size = int(metadata["block_size"])
    blocks = math.ceil(length / block_size)
    if topk > length:
        raise ValueError(f"Top-K {topk} exceeds replay length {length}")
    q = bundle["q_indexer"][index].to(device=device).contiguous()
    k = bundle["k_indexer"][:length].to(device=device).contiguous()
    weights = bundle["head_weight"][index].to(device=device).contiguous()
    heads_cpu, remaining_cpu = split_head_ids(bundle["dynamic_h8_ids"][index])
    heads = heads_cpu.to(device=device).contiguous()
    remaining = remaining_cpu.to(device=device).contiguous()
    direct = (
        bundle["temporal_block_mask"][index, :blocks].to(device=device).contiguous()
    )
    promotion_reference = (
        bundle["promotion_block_mask"][index, :blocks].to(device=device).contiguous()
    )
    threshold = bundle["promotion_threshold"][index : index + 1].to(device=device)
    full_reference = bundle["full64_score"][index, :length].to(device=device).contiguous()
    h8_reference = bundle["h8_score"][index, :length].to(device=device).contiguous()
    previous_reference = (
        bundle["previous_full64_score"][index, :length].to(device=device).contiguous()
    )
    coefficients = bundle["t1_coefficients"][index].to(device=device).contiguous()
    t1_reference = bundle["t1_score"][index, :length].to(device=device).contiguous()
    return {
        "path": path,
        "metadata": metadata,
        "observation_metadata": observation_metadata,
        "observation_index": index,
        "q": q,
        "k": k,
        "weights": weights,
        "heads": heads,
        "remaining": remaining,
        "direct": direct,
        "cold": ~direct,
        "all_cold": torch.ones(blocks, device=device, dtype=torch.bool),
        "promotion_reference": promotion_reference,
        "threshold": threshold,
        "full_reference": full_reference,
        "h8_reference": h8_reference,
        "previous_reference": previous_reference,
        "coefficients": coefficients,
        "t1_reference": t1_reference,
        "length": length,
        "blocks": blocks,
        "block_size": block_size,
        "full_out": torch.empty(length, device=device, dtype=torch.float32),
        "h8_out": torch.empty(length, device=device, dtype=torch.float32),
        "maxima": torch.empty(blocks, device=device, dtype=torch.float32),
        "promote": torch.empty(blocks, device=device, dtype=torch.bool),
        "keep": torch.empty(blocks, device=device, dtype=torch.bool),
        "progressive_out": torch.empty(length, device=device, dtype=torch.float32),
        "fused_out": torch.empty(length, device=device, dtype=torch.float32),
        "compact_out": torch.empty(length, device=device, dtype=torch.float32),
        "compact_indices": torch.empty(length, device=device, dtype=torch.int32),
        "dummy_indices": torch.empty(length, device=device, dtype=torch.int32),
        "accepted": torch.empty(blocks, device=device, dtype=torch.int8),
        "block_max": torch.empty(blocks, device=device, dtype=torch.float32),
        "counter": torch.zeros(1, device=device, dtype=torch.int32),
        "t1_out": torch.empty(length, device=device, dtype=torch.float32),
        "topk_values": torch.empty(topk, device=device, dtype=torch.float32),
        "topk_indices": torch.empty(topk, device=device, dtype=torch.int64),
    }


def launch_full(item: dict[str, Any]) -> None:
    _full_score_kernel[(item["blocks"],)](
        item["q"],
        item["weights"],
        item["k"],
        item["full_out"],
        item["length"],
        D=HEAD_DIM,
        BLOCK_N=item["block_size"],
        num_warps=8,
    )


def launch_topk(item: dict[str, Any], source: torch.Tensor, topk: int) -> None:
    torch.topk(
        source,
        topk,
        sorted=False,
        out=(item["topk_values"], item["topk_indices"]),
    )


def launch_h8(item: dict[str, Any], cold: torch.Tensor) -> None:
    _h8_block_max_kernel[(item["blocks"],)](
        item["q"],
        item["weights"],
        item["k"],
        item["heads"],
        cold,
        item["h8_out"],
        item["maxima"],
        item["length"],
        D=HEAD_DIM,
        BLOCK_N=item["block_size"],
        num_warps=8,
    )


def launch_progressive(item: dict[str, Any]) -> None:
    launch_h8(item, item["cold"])
    torch.ge(item["maxima"], item["threshold"], out=item["promote"])
    torch.logical_or(item["direct"], item["promote"], out=item["keep"])
    _masked_full_kernel[(item["blocks"],)](
        item["q"],
        item["weights"],
        item["k"],
        item["keep"],
        item["progressive_out"],
        item["length"],
        D=HEAD_DIM,
        BLOCK_N=item["block_size"],
        num_warps=8,
    )


def launch_fused(item: dict[str, Any], *, compact: bool) -> None:
    if compact:
        item["counter"].zero_()
    output = item["compact_out"] if compact else item["fused_out"]
    indices = item["compact_indices"] if compact else item["dummy_indices"]
    _progressive_kernel[(item["blocks"],)](
        item["q"],
        item["weights"],
        item["k"],
        item["heads"],
        item["remaining"],
        item["direct"],
        item["threshold"],
        output,
        indices,
        item["accepted"],
        item["block_max"],
        item["counter"],
        item["length"],
        D=HEAD_DIM,
        BLOCK_N=item["block_size"],
        COMPACT=compact,
        num_warps=8,
    )


def launch_t1(item: dict[str, Any]) -> None:
    if triton is None:
        raise RuntimeError("Triton is required")
    launch_h8(item, item["all_cold"])
    _t1_reconstruct_kernel[(triton.cdiv(item["length"], 256),)](
        item["h8_out"],
        item["previous_reference"],
        item["coefficients"],
        item["t1_out"],
        item["length"],
        BLOCK=256,
        num_warps=4,
    )


def method_launcher(method: str, item: dict[str, Any], topk: int) -> Callable[[], None]:
    if method == "full64":
        return lambda: launch_full(item)
    if method == "topk_only":
        return lambda: launch_topk(item, item["full_reference"], topk)
    if method == "full64_topk":
        return lambda: (launch_full(item), launch_topk(item, item["full_out"], topk))
    if method == "h8":
        return lambda: launch_h8(item, item["all_cold"])
    if method == "progressive_h8":
        return lambda: launch_progressive(item)
    if method == "fused_dense":
        return lambda: launch_fused(item, compact=False)
    if method == "fused_compact":
        return lambda: launch_fused(item, compact=True)
    if method == "fused_topk":
        return lambda: (
            launch_fused(item, compact=False),
            launch_topk(item, item["fused_out"], topk),
        )
    if method == "t1":
        return lambda: launch_t1(item)
    raise ValueError(method)


def measure(
    function: Callable[[], None],
    *,
    protocol: str,
    flush: torch.Tensor,
    warmup: int,
    iterations: int,
) -> np.ndarray:
    for _ in range(warmup):
        if protocol == "cache-flush":
            flush.add_(1)
        function()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        if protocol == "cache-flush":
            flush.add_(1)
        start.record()
        function()
        end.record()
    torch.cuda.synchronize()
    return np.asarray(
        [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)],
        dtype=np.float64,
    )


def summarize(values_us: np.ndarray) -> dict[str, float]:
    return {
        "mean_us": mean(values_us),
        "median_us": float(np.median(values_us)),
        "p5_us": float(np.percentile(values_us, 5)),
        "p95_us": float(np.percentile(values_us, 95)),
        "min_us": float(values_us.min()),
        "std_us": pstdev(values_us),
    }


def correctness(item: dict[str, Any], topk: int) -> dict[str, float]:
    launch_full(item)
    launch_fused(item, compact=False)
    launch_t1(item)
    torch.cuda.synchronize()
    full = item["full_out"].float()
    reference = item["full_reference"].float()
    full_ids = torch.topk(full, topk, sorted=False).indices
    reference_ids = torch.topk(reference, topk, sorted=False).indices
    keep = item["accepted"].bool().repeat_interleave(item["block_size"])[: item["length"]]
    progressive_error = (item["fused_out"][keep] - full[keep]).abs()
    fused_ids = torch.topk(item["fused_out"], topk, sorted=False).indices
    actual_promoted = item["accepted"].bool() & ~item["direct"]
    reference_promoted = item["promotion_reference"] & ~item["direct"]
    common = int(item["observation_metadata"]["previous_length"])
    return {
        "full64_reference_max_abs": float((full - reference).abs().max().item()),
        "full64_reference_topk_recall": float(
            torch.isin(reference_ids, full_ids).float().mean().item()
        ),
        "fused_vs_full_max_abs_on_candidates": float(
            progressive_error.max().item() if progressive_error.numel() else 0.0
        ),
        "fused_topk_recall_vs_runtime_full64": float(
            torch.isin(full_ids, fused_ids).float().mean().item()
        ),
        "nominal_promotion_rate": 0.1,
        "reference_actual_promotion_rate": float(
            reference_promoted.sum().item() / max(1, item["cold"].sum().item())
        ),
        "runtime_actual_promotion_rate": float(
            actual_promoted.sum().item() / max(1, item["cold"].sum().item())
        ),
        "h8_reference_max_abs": float(
            (item["h8_out"] - item["h8_reference"]).abs().max().item()
        ),
        "t1_reference_max_abs": float(
            (item["t1_out"][:common] - item["t1_reference"][:common]).abs().max().item()
        ),
    }


def write_outputs(rows: list[dict[str, Any]], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Portable DeepSeek DSA operator replay benchmark")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--method", action="append", choices=(*METHODS, "all"), default=[])
    parser.add_argument("--top-k", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--protocol", choices=("warm-cache", "cache-flush", "both"), default="warm-cache")
    parser.add_argument("--observation", default="near_threshold")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--expected-gpu-count", type=int)
    parser.add_argument("--gpu-name-contains")
    parser.add_argument("--flush-mib", type=int, default=64)
    parser.add_argument("--csv", type=Path, default=Path("dsa_replay_results.csv"))
    parser.add_argument("--json", type=Path, default=Path("dsa_replay_results.json"))
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--profile-iterations", type=int, default=100)
    args = parser.parse_args()

    if triton is None:
        raise RuntimeError("Triton is required for GPU replay")
    gpu_count = torch.cuda.device_count()
    if args.expected_gpu_count is not None and gpu_count != args.expected_gpu_count:
        raise RuntimeError(f"expected {args.expected_gpu_count} visible GPUs, found {gpu_count}")
    if not 0 <= args.device < gpu_count:
        raise RuntimeError(f"device {args.device} outside [0, {gpu_count})")
    gpu_names = [torch.cuda.get_device_name(index) for index in range(gpu_count)]
    if args.gpu_name_contains and args.gpu_name_contains not in gpu_names[args.device]:
        raise RuntimeError(
            f"GPU name guard {args.gpu_name_contains!r} rejected {gpu_names[args.device]!r}"
        )
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    requested = list(METHODS) if not args.method or "all" in args.method else args.method
    protocols = (
        ("warm-cache", "cache-flush") if args.protocol == "both" else (args.protocol,)
    )
    flush = torch.empty(
        args.flush_mib * 1024 * 1024 // 4, device=device, dtype=torch.float32
    )
    rows: list[dict[str, Any]] = []
    for bundle_path in discover_bundles(args.bundle):
        item = load_item(bundle_path, args.observation, device, args.top_k)
        audit = correctness(item, args.top_k)
        for method in requested:
            function = method_launcher(method, item, args.top_k)
            for _ in range(args.warmup):
                function()
            torch.cuda.synchronize()
            if args.profile_only:
                torch.cuda.nvtx.range_push(
                    f"{METHOD_IDS[method]}_{method}_c{item['metadata']['nominal_context']}"
                )
                for _ in range(args.profile_iterations):
                    function()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
                continue
            for protocol in protocols:
                values = measure(
                    function,
                    protocol=protocol,
                    flush=flush,
                    warmup=args.warmup,
                    iterations=args.iters,
                )
                row = {
                    "method_id": METHOD_IDS[method],
                    "method": method,
                    "protocol": protocol,
                    "bundle": bundle_path.name,
                    "observation_index": item["observation_index"],
                    "observation_category": item["observation_metadata"]["category"],
                    "nominal_context": item["metadata"]["nominal_context"],
                    "length": item["length"],
                    "batch": 1,
                    "query_length": 1,
                    "heads": 64,
                    "head_dimension": 128,
                    "dtype": str(item["q"].dtype).removeprefix("torch."),
                    "topk": args.top_k,
                    "block_size": item["block_size"],
                    "warmup": args.warmup,
                    "measurements": args.iters,
                    "flush_mib": args.flush_mib if protocol == "cache-flush" else 0,
                    "fixed_input_addresses": True,
                    "preallocated_outputs": True,
                    "timed_region_explicit_output_allocations": 0,
                    "gpu": gpu_names[args.device],
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "triton": triton.__version__,
                    "q_data_ptr": hex(item["q"].data_ptr()),
                    "k_data_ptr": hex(item["k"].data_ptr()),
                    **audit,
                    **summarize(values),
                }
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
    if args.profile_only:
        return
    if not rows:
        raise RuntimeError("no benchmark rows produced")
    write_outputs(rows, args.csv, args.json)


if __name__ == "__main__":
    main()
