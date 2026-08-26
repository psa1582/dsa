from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from benchmark_progressive_dsa import load_trace_variants, stats

from temporal_dsa.progressive_kernel import HEAD_DIM, _full_score_kernel

CONTEXTS = (8192, 16384, 32768)
CAPTURES = ("code_heldout_3", "text_heldout_28444", "code_heldout_5")


def benchmark(
    function: Callable[[int], None],
    *,
    variants: int,
    flush: torch.Tensor,
    warmup: int,
    measurements: int,
) -> list[float]:
    for iteration in range(warmup):
        flush.add_(1)
        function(iteration % variants)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(measurements)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(measurements)]
    for iteration, (start, end) in enumerate(zip(starts, ends)):
        flush.add_(1)
        start.record()
        function(iteration % variants)
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Full64 score/Top-K latency decomposition")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--measurements", type=int, default=1000)
    args = parser.parse_args()

    if torch.cuda.device_count() != 2:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES guard expected exactly GPUs 0,1; got {torch.cuda.device_count()}"
        )
    names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("L40" in name for name in names):
        raise RuntimeError(f"L40S guard rejected {names}")
    torch.cuda.set_device(0)
    checkpoint = (
        args.repo
        / "artifacts"
        / "pilot"
        / "indexers_1000"
        / "checkpoints"
        / f"layer_{args.layer:02d}.safetensors"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    flush = torch.empty(64 * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
    rows: list[dict] = []

    for context in CONTEXTS:
        variants = load_trace_variants(
            args.repo, context, args.layer, list(CAPTURES), checkpoint
        )
        items = []
        for query, weights, keys in variants:
            length = int(keys.shape[0])
            output = torch.empty(length, device="cuda", dtype=torch.float32)
            item = {
                "query": query,
                "weights": weights,
                "keys": keys,
                "length": length,
                "blocks": (length + 63) // 64,
                "output": output,
            }
            _full_score_kernel[(item["blocks"],)](
                query,
                weights,
                keys,
                output,
                length,
                D=HEAD_DIM,
                BLOCK_N=64,
                num_warps=8,
            )
            items.append(item)
        torch.cuda.synchronize()

        def score_only(index: int, batch: list[dict] = items) -> None:
            item = batch[index]
            _full_score_kernel[(item["blocks"],)](
                item["query"],
                item["weights"],
                item["keys"],
                item["output"],
                item["length"],
                D=HEAD_DIM,
                BLOCK_N=64,
                num_warps=8,
            )

        def topk_only(index: int, batch: list[dict] = items) -> None:
            torch.topk(batch[index]["output"], 2048, sorted=False)

        def combined(index: int) -> None:
            score_only(index)
            topk_only(index)

        for method, function in [
            ("full64_score_only", score_only),
            ("stock_topk_only", topk_only),
            ("full64_score_plus_stock_topk", combined),
        ]:
            values = benchmark(
                function,
                variants=len(items),
                flush=flush,
                warmup=args.warmup,
                measurements=args.measurements,
            )
            row = {
                "context": context,
                "layer": args.layer,
                "method": method,
                "k": 2048,
                "warmup": args.warmup,
                "measurements": args.measurements,
                **stats(values),
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
        del variants
        torch.cuda.empty_cache()

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "full64_topk_decomposition_raw.csv", index=False)
    piv = frame.pivot(index="context", columns="method", values="median_ms")
    summary = piv.reset_index().rename_axis(columns=None)
    isolated_sum = summary.full64_score_only + summary.stock_topk_only
    summary["score_fraction_of_isolated_sum"] = summary.full64_score_only / isolated_sum
    summary["topk_fraction_of_isolated_sum"] = summary.stock_topk_only / isolated_sum
    summary["topk_fraction_of_combined"] = (
        summary.stock_topk_only / summary.full64_score_plus_stock_topk
    )
    summary["isolated_sum_vs_combined_ratio"] = (
        isolated_sum / summary.full64_score_plus_stock_topk
    )
    summary.to_csv(args.output / "full64_topk_decomposition.csv", index=False)
    metadata = {
        "gpu_names": names,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "contexts": list(CONTEXTS),
        "capture_names": list(CAPTURES),
        "layer": args.layer,
        "topk": 2048,
        "full64_kernel": "temporal_dsa.progressive_kernel._full_score_kernel",
        "topk_implementation": "torch.topk(sorted=False)",
        "timing": "CUDA events",
        "cache_protocol": "64 MiB device flush before each untimed start event; rotate three real captures",
        "scope_limit": "No real 64K/128K sidecar capture exists; those contexts are not synthetically benchmarked.",
    }
    (args.output / "benchmark_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
