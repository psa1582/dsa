from __future__ import annotations

import argparse
from pathlib import Path

import torch

from benchmark_progressive_dsa import (
    DIRECT_FRACTIONS,
    launch_full,
    launch_fused,
    launch_two_pass,
    load_trace_variants,
    prepare_items,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Nsight Systems progressive DSA profile target")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--context", type=int, choices=[16384, 32768], required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--layer", type=int, default=17)
    args = parser.parse_args()
    if torch.cuda.device_count() != 2:
        raise RuntimeError("expected CUDA_VISIBLE_DEVICES=0,1")
    names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("L40" in name for name in names):
        raise RuntimeError(f"L40 guard rejected {names}")
    checkpoint = (
        args.repo / "artifacts" / "pilot" / "indexers_1000" / "checkpoints"
        / f"layer_{args.layer:02d}.safetensors"
    )
    variants = load_trace_variants(
        args.repo, args.context, args.layer, ["code_heldout_3"], checkpoint
    )
    item = prepare_items(
        variants,
        block_size=64,
        promotion_rate=0.10,
        direct_fraction=DIRECT_FRACTIONS[args.context],
        seed=1582 + args.context,
    )[0]
    topk = 2048
    methods = {
        "optimized_full": lambda: launch_full(item, topk),
        "h8_two_pass": lambda: launch_two_pass(item, topk),
        "fused_progressive_dense": lambda: launch_fused(item, topk, False),
        "fused_progressive_compact": lambda: launch_fused(item, topk, True),
    }
    for function in methods.values():
        for _ in range(20):
            item["counter"].zero_()
            function()
    torch.cuda.synchronize()
    for name, function in methods.items():
        torch.cuda.nvtx.range_push(f"{name}_c{args.context}")
        for _ in range(args.iterations):
            if "compact" in name:
                item["counter"].zero_()
            function()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()


if __name__ == "__main__":
    main()
