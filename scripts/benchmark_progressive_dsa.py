from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable

import numpy as np
import pandas as pd
import torch

from temporal_dsa.progressive_kernel import (
    HEAD_DIM,
    _full_score_kernel,
    _h8_block_max_kernel,
    _masked_full_kernel,
    _progressive_kernel,
    dynamic_head_ids,
    full_scores_triton,
    h8_block_max_triton,
    progressive_qk_reduction,
    progressive_scores_triton,
    split_head_ids,
    two_pass_scores_triton,
)
from temporal_dsa.verifier_scoring import load_sidecar_encoded


DIRECT_FRACTIONS = {8192: 0.945519, 16384: 0.703342, 32768: 0.594227}
PROMOTION_RATES = (0.05, 0.10, 0.15, 0.20)
BLOCK_SIZES = (32, 64, 128)
TOPKS = (512, 1024, 2048)


def stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median_ms": float(np.median(array)),
        "p5_ms": float(np.percentile(array, 5)),
        "p95_ms": float(np.percentile(array, 95)),
        "mean_ms": mean(values),
        "std_ms": pstdev(values),
    }


def cuda_bench(
    fn: Callable[[int], None],
    *,
    variants: int,
    flush: torch.Tensor,
    warmup: int,
    measure: int,
) -> list[float]:
    for iteration in range(warmup):
        flush.add_(1)
        fn(iteration % variants)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(measure)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(measure)]
    for iteration, (start, end) in enumerate(zip(starts, ends)):
        # The flush and trace rotation are deliberately outside the timed interval.
        flush.add_(1)
        start.record()
        fn(iteration % variants)
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends)]


def direct_mask(length: int, block_size: int, fraction: float, seed: int) -> torch.Tensor:
    blocks = math.ceil(length / block_size)
    count = round(fraction * blocks)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(blocks, generator=generator)
    mask = torch.zeros(blocks, dtype=torch.bool)
    mask[order[:count]] = True
    return mask.cuda()


def threshold_for_rate(maxima: torch.Tensor, cold: torch.Tensor, rate: float) -> float:
    values = maxima[cold]
    if values.numel() == 0:
        return math.inf
    # kthvalue gives a deterministic threshold without copying all values to the host.
    promoted = max(1, round(rate * values.numel()))
    rank = max(1, values.numel() - promoted + 1)
    return float(torch.kthvalue(values, rank).values.item())


def load_trace_variants(
    repo: Path, context: int, layer: int, names: list[str], checkpoint: Path
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    root = repo / "artifacts" / "pilot" / f"trace_capture_{context // 1024}k"
    variants = []
    for name in names:
        capture = root / name / f"layer_{layer:02d}.pt"
        if not capture.exists():
            raise FileNotFoundError(capture)
        metadata = torch.load(capture, map_location="cpu", weights_only=True)
        length = min(context, int(metadata["hidden"].shape[0]))
        del metadata
        queries, weights, keys, _ = load_sidecar_encoded(
            capture, checkpoint, np.asarray([length - 1]), device="cuda:0"
        )
        variants.append(
            (
                queries[0].to(torch.bfloat16).contiguous(),
                weights[0].contiguous(),
                keys[:length].to(torch.bfloat16).contiguous(),
            )
        )
    return variants


def launch_full(item: dict, topk: int | None) -> None:
    _full_score_kernel[(item["blocks"],)](
        item["q"], item["w"], item["k"], item["full_out"], item["length"],
        D=HEAD_DIM, BLOCK_N=item["block_size"], num_warps=8,
    )
    if topk is not None:
        torch.topk(item["full_out"], topk, sorted=False)


def launch_temporal(item: dict, topk: int) -> None:
    _masked_full_kernel[(item["blocks"],)](
        item["q"], item["w"], item["k"], item["direct"], item["temporal_out"],
        item["length"], D=HEAD_DIM, BLOCK_N=item["block_size"], num_warps=8,
    )
    torch.topk(item["temporal_out"], topk, sorted=False)


def launch_h8(item: dict, topk: int) -> None:
    _h8_block_max_kernel[(item["blocks"],)](
        item["q"], item["w"], item["k"], item["heads"], item["all_cold"],
        item["h8_out"], item["maxima"], item["length"], D=HEAD_DIM,
        BLOCK_N=item["block_size"], num_warps=8,
    )
    torch.topk(item["h8_out"], topk, sorted=False)


def launch_two_pass(item: dict, topk: int | None) -> None:
    _h8_block_max_kernel[(item["blocks"],)](
        item["q"], item["w"], item["k"], item["heads"], item["cold"],
        item["h8_out"], item["maxima"], item["length"], D=HEAD_DIM,
        BLOCK_N=item["block_size"], num_warps=8,
    )
    torch.logical_or(item["direct"], item["maxima"] >= item["threshold"], out=item["keep"])
    _masked_full_kernel[(item["blocks"],)](
        item["q"], item["w"], item["k"], item["keep"], item["two_out"],
        item["length"], D=HEAD_DIM, BLOCK_N=item["block_size"], num_warps=8,
    )
    if topk is not None:
        torch.topk(item["two_out"], topk, sorted=False)


def launch_fused(item: dict, topk: int | None, compact: bool) -> None:
    output = item["compact_out"] if compact else item["fused_out"]
    indices = item["compact_indices"] if compact else item["dummy_indices"]
    _progressive_kernel[(item["blocks"],)](
        item["q"], item["w"], item["k"], item["heads"], item["remaining"],
        item["direct"], item["threshold"], output, indices, item["accepted"],
        item["block_max"], item["counter"], item["length"], D=HEAD_DIM,
        BLOCK_N=item["block_size"], COMPACT=compact, num_warps=8,
    )
    if topk is not None:
        candidate_count = item["candidate_count"] if compact else item["length"]
        torch.topk(output[:candidate_count], min(topk, candidate_count), sorted=False)


def prepare_items(
    variants: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    block_size: int,
    promotion_rate: float,
    direct_fraction: float,
    seed: int,
) -> list[dict]:
    items = []
    for variant, (q, w, k) in enumerate(variants):
        length = k.shape[0]
        blocks = math.ceil(length / block_size)
        selected_cpu, remaining_cpu = split_head_ids(dynamic_head_ids(w).cpu())
        selected = selected_cpu.cuda()
        remaining = remaining_cpu.cuda()
        direct = direct_mask(length, block_size, direct_fraction, seed + variant)
        cold = ~direct
        _, maxima = h8_block_max_triton(q, w, k, selected, cold, block_size=block_size)
        threshold = threshold_for_rate(maxima, cold, promotion_rate)
        threshold_tensor = torch.tensor([threshold], device="cuda", dtype=torch.float32)
        probe = progressive_scores_triton(
            q, w, k, selected, direct, threshold_tensor, block_size=block_size,
            remaining_heads=remaining,
        )
        promoted = int((probe.accepted_blocks.bool() & cold).sum().item())
        accepted = int(probe.accepted_blocks.sum().item())
        candidate_count = min(length, accepted * block_size)
        item = {
            "q": q,
            "w": w,
            "k": k,
            "length": length,
            "blocks": blocks,
            "block_size": block_size,
            "heads": selected,
            "remaining": remaining,
            "direct": direct,
            "cold": cold,
            "all_cold": torch.ones(blocks, device="cuda", dtype=torch.bool),
            "threshold": threshold_tensor,
            "full_out": torch.empty(length, device="cuda", dtype=torch.float32),
            "temporal_out": torch.empty(length, device="cuda", dtype=torch.float32),
            "h8_out": torch.empty(length, device="cuda", dtype=torch.float32),
            "maxima": torch.empty(blocks, device="cuda", dtype=torch.float32),
            "keep": torch.empty(blocks, device="cuda", dtype=torch.bool),
            "two_out": torch.empty(length, device="cuda", dtype=torch.float32),
            "fused_out": torch.empty(length, device="cuda", dtype=torch.float32),
            "compact_out": torch.empty(length, device="cuda", dtype=torch.float32),
            "compact_indices": torch.empty(length, device="cuda", dtype=torch.int32),
            "dummy_indices": torch.empty(length, device="cuda", dtype=torch.int32),
            "accepted": torch.empty(blocks, device="cuda", dtype=torch.int8),
            "block_max": torch.empty(blocks, device="cuda", dtype=torch.float32),
            "counter": torch.zeros(1, device="cuda", dtype=torch.int32),
            "candidate_count": candidate_count,
            "direct_count": int(direct.sum().item()),
            "promoted_count": promoted,
            "threshold_value": threshold,
        }
        items.append(item)
    return items


def correctness_rows(items: list[dict], context: int, topk: int) -> list[dict]:
    rows = []
    for variant, item in enumerate(items):
        reference = full_scores_triton(
            item["q"], item["w"], item["k"], block_size=item["block_size"]
        )
        fused = progressive_scores_triton(
            item["q"], item["w"], item["k"], item["heads"], item["direct"],
            item["threshold"], block_size=item["block_size"],
            remaining_heads=item["remaining"],
        )
        two = two_pass_scores_triton(
            item["q"], item["w"], item["k"], item["heads"], item["direct"],
            item["threshold"], block_size=item["block_size"],
        )
        keep = fused.accepted_blocks.bool().repeat_interleave(item["block_size"])[
            : item["length"]
        ]
        error = (fused.scores[keep] - reference[keep]).abs()
        fused_ids = torch.topk(fused.scores, topk, sorted=False).indices
        full_ids = torch.topk(reference, topk, sorted=False).indices
        recall = torch.isin(full_ids, fused_ids).float().mean().item()
        rows.append(
            {
                "context": context,
                "variant": variant,
                "block_size": item["block_size"],
                "topk": topk,
                "threshold": item["threshold_value"],
                "direct_blocks": item["direct_count"],
                "promoted_cold_blocks": item["promoted_count"],
                "candidate_count": item["candidate_count"],
                "max_abs_full_error": float(error.max().item()) if error.numel() else 0.0,
                "mean_abs_full_error": float(error.mean().item()) if error.numel() else 0.0,
                "two_fused_max_abs": float(
                    torch.nan_to_num(two.scores - fused.scores).abs().max().item()
                ),
                "topk_recall": recall,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="L40S progressive DSA CUDA-event benchmark")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--measure", type=int, default=500)
    parser.add_argument("--primary-only", action="store_true")
    parser.add_argument("--seed", type=int, default=1582)
    args = parser.parse_args()

    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES guard expected 2 GPUs, got {torch.cuda.device_count()}")
    names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("L40" in name for name in names):
        raise RuntimeError(f"L40S guard rejected {names}")
    torch.cuda.set_device(0)
    checkpoint = (
        args.repo / "artifacts" / "pilot" / "indexers_1000" / "checkpoints"
        / f"layer_{args.layer:02d}.safetensors"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    # 64 MiB is larger than a single 32K K-index tensor and prevents a one-tile hot-cache result.
    flush = torch.empty(64 * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
    timing_rows: list[dict] = []
    quality_rows: list[dict] = []
    calibration_rows: list[dict] = []
    contexts = (8192, 16384, 32768)
    capture_names = ["code_heldout_3", "text_heldout_28444", "code_heldout_5"]
    for context in contexts:
        variants = load_trace_variants(args.repo, context, args.layer, capture_names, checkpoint)
        combinations = [(64, 0.10, 2048)] if args.primary_only else [
            (block, rate, topk)
            for block in BLOCK_SIZES
            for rate in PROMOTION_RATES
            for topk in TOPKS
        ]
        for block_size, promotion_rate, topk in combinations:
            items = prepare_items(
                variants,
                block_size=block_size,
                promotion_rate=promotion_rate,
                direct_fraction=DIRECT_FRACTIONS[context],
                seed=args.seed + context + block_size,
            )
            calibration_rows.extend(
                {
                    "context": context,
                    "variant": index,
                    "block_size": block_size,
                    "promotion_target": promotion_rate,
                    "threshold": item["threshold_value"],
                    "direct_fraction": item["direct_count"] / item["blocks"],
                    "promotion_fraction_of_cold": item["promoted_count"]
                    / max(1, item["blocks"] - item["direct_count"]),
                    "candidate_fraction": item["candidate_count"] / item["length"],
                    "qk_reduction": progressive_qk_reduction(
                        item["length"], item["direct_count"], item["promoted_count"],
                        block_size=block_size,
                    ),
                }
                for index, item in enumerate(items)
            )
            if block_size == 64 and promotion_rate == 0.10 and topk == 2048:
                quality_rows.extend(correctness_rows(items, context, topk))
            methods: dict[str, tuple[Callable[[dict], None], int, str]] = {
                "full_kernel": (lambda item: launch_full(item, None), 1, "kernel"),
                "two_pass_kernel": (lambda item: launch_two_pass(item, None), 4, "kernel"),
                "fused_dense_kernel": (lambda item: launch_fused(item, None, False), 1, "kernel"),
                "full_plus_topk": (lambda item: launch_full(item, topk), 2, "indexer_topk"),
                "temporal_plus_topk": (lambda item: launch_temporal(item, topk), 2, "indexer_topk"),
                "h8_plus_topk": (lambda item: launch_h8(item, topk), 2, "indexer_topk"),
                "two_pass_plus_topk": (lambda item: launch_two_pass(item, topk), 5, "indexer_topk"),
                "fused_dense_plus_topk": (lambda item: launch_fused(item, topk, False), 2, "indexer_topk"),
                "fused_compact_plus_topk": (lambda item: launch_fused(item, topk, True), 3, "indexer_topk"),
            }
            for method, (launcher, launches, scope) in methods.items():
                # Compact output requires a real counter-reset launch each invocation.
                def invoke(index: int, call=launcher, name=method) -> None:
                    item = items[index]
                    if "compact" in name:
                        item["counter"].zero_()
                    call(item)

                values = cuda_bench(
                    invoke, variants=len(items), flush=flush,
                    warmup=args.warmup, measure=args.measure,
                )
                row = {
                    "context": context,
                    "layer": args.layer,
                    "method": method,
                    "scope": scope,
                    "block_size": block_size,
                    "promotion_target": promotion_rate,
                    "topk": topk,
                    "warmup": args.warmup,
                    "measurements": args.measure,
                    "kernel_launches": launches,
                    **stats(values),
                }
                timing_rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
            del items
            torch.cuda.empty_cache()
        del variants

    pd.DataFrame(timing_rows).to_csv(args.output / "timing.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(args.output / "calibration.csv", index=False)
    pd.DataFrame(quality_rows).to_csv(args.output / "correctness.csv", index=False)
    metadata = {
        "gpu_names": names,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "warmup": args.warmup,
        "measurements": args.measure,
        "timing": "CUDA events",
        "cache_protocol": "64 MiB device flush before each untimed start event; rotate 3 real traces",
        "capture_names": capture_names,
        "layer": args.layer,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
