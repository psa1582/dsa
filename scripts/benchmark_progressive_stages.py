from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from benchmark_progressive_dsa import (
    DIRECT_FRACTIONS,
    cuda_bench,
    launch_full,
    launch_fused,
    load_trace_variants,
    prepare_items,
    stats,
)
from temporal_dsa.progressive_kernel import (
    HEAD_DIM,
    _h8_block_max_kernel,
    _masked_full_kernel,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Progressive DSA primary stage timing")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--measure", type=int, default=500)
    parser.add_argument("--layer", type=int, default=17)
    args = parser.parse_args()
    if torch.cuda.device_count() != 2:
        raise RuntimeError("expected CUDA_VISIBLE_DEVICES=0,1")
    checkpoint = (
        args.repo / "artifacts" / "pilot" / "indexers_1000" / "checkpoints"
        / f"layer_{args.layer:02d}.safetensors"
    )
    flush = torch.empty(64 * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
    capture_names = ["code_heldout_3", "text_heldout_28444", "code_heldout_5"]
    rows = []
    for context in (8192, 16384, 32768):
        variants = load_trace_variants(
            args.repo, context, args.layer, capture_names, checkpoint
        )
        items = prepare_items(
            variants,
            block_size=64,
            promotion_rate=0.10,
            direct_fraction=DIRECT_FRACTIONS[context],
            seed=1582 + context,
        )
        for item in items:
            launch_full(item, None)
            launch_fused(item, None, False)
            item["counter"].zero_()
            launch_fused(item, None, True)
        torch.cuda.synchronize()

        def h8(index: int) -> None:
            item = items[index]
            _h8_block_max_kernel[(item["blocks"],)](
                item["q"], item["w"], item["k"], item["heads"], item["cold"],
                item["h8_out"], item["maxima"], item["length"], D=HEAD_DIM,
                BLOCK_N=64, num_warps=8,
            )

        def rescue(index: int) -> None:
            item = items[index]
            torch.logical_or(
                item["direct"], item["maxima"] >= item["threshold"], out=item["keep"]
            )

        def rerank(index: int) -> None:
            item = items[index]
            _masked_full_kernel[(item["blocks"],)](
                item["q"], item["w"], item["k"], item["keep"], item["two_out"],
                item["length"], D=HEAD_DIM, BLOCK_N=64, num_warps=8,
            )

        stages = {
            "head_select_topk64x8": lambda i: torch.topk(
                items[i]["w"].abs(), 8, sorted=False
            ),
            "head_select_stable_argsort64": lambda i: torch.argsort(
                items[i]["w"].abs(), descending=True, stable=True
            )[:8],
            "h8_verifier_scan": h8,
            "rescue_threshold_mask": rescue,
            "full_candidate_rerank": rerank,
            "topk_full_vector": lambda i: torch.topk(
                items[i]["full_out"], 2048, sorted=False
            ),
            "topk_fused_dense_vector": lambda i: torch.topk(
                items[i]["fused_out"], 2048, sorted=False
            ),
            "topk_compact_candidates": lambda i: torch.topk(
                items[i]["compact_out"][: items[i]["candidate_count"]],
                min(2048, items[i]["candidate_count"]),
                sorted=False,
            ),
        }
        for stage, function in stages.items():
            values = cuda_bench(
                function,
                variants=len(items),
                flush=flush,
                warmup=args.warmup,
                measure=args.measure,
            )
            row = {
                "context": context,
                "layer": args.layer,
                "stage": stage,
                "promotion_target": 0.10,
                "candidate_fraction_mean": sum(
                    item["candidate_count"] / item["length"] for item in items
                ) / len(items),
                "warmup": args.warmup,
                "measurements": args.measure,
                **stats(values),
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
        del items
        for promotion_rate in (0.05, 0.15, 0.20):
            rate_items = prepare_items(
                variants,
                block_size=64,
                promotion_rate=promotion_rate,
                direct_fraction=DIRECT_FRACTIONS[context],
                seed=1582 + context,
            )
            for item in rate_items:
                item["counter"].zero_()
                launch_fused(item, None, True)
            torch.cuda.synchronize()
            values = cuda_bench(
                lambda i: torch.topk(
                    rate_items[i]["compact_out"][: rate_items[i]["candidate_count"]],
                    min(2048, rate_items[i]["candidate_count"]),
                    sorted=False,
                ),
                variants=len(rate_items),
                flush=flush,
                warmup=args.warmup,
                measure=args.measure,
            )
            row = {
                "context": context,
                "layer": args.layer,
                "stage": "topk_compact_candidates",
                "promotion_target": promotion_rate,
                "candidate_fraction_mean": sum(
                    item["candidate_count"] / item["length"] for item in rate_items
                ) / len(rate_items),
                "warmup": args.warmup,
                "measurements": args.measure,
                **stats(values),
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            del rate_items
        del variants
        torch.cuda.empty_cache()
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output / "stage_timing.csv", index=False)


if __name__ == "__main__":
    main()
