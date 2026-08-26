from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from temporal_dsa.progressive_kernel import (
    dynamic_head_ids,
    full_scores_triton,
    h8_block_max_triton,
    progressive_scores_triton,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic 64K/128K DSA operator shape audit")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", type=int, nargs="+", default=[65536, 131072])
    parser.add_argument("--seed", type=int, default=1582)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    query = torch.randn((64, 128), generator=generator, device=device, dtype=torch.bfloat16)
    weights = torch.randn((64,), generator=generator, device=device, dtype=torch.float32)
    selected = dynamic_head_ids(weights)
    rows = []
    for context in args.contexts:
        keys = torch.randn(
            (context, 128), generator=generator, device=device, dtype=torch.bfloat16
        )
        blocks = math.ceil(context / 64)
        direct = torch.zeros(blocks, device=device, dtype=torch.bool)
        direct[::3] = True
        cold = ~direct
        full = full_scores_triton(query, weights, keys)
        h8, maxima = h8_block_max_triton(query, weights, keys, selected, cold)
        finite_maxima = maxima[torch.isfinite(maxima)]
        threshold = torch.quantile(finite_maxima, 0.9)
        fused = progressive_scores_triton(query, weights, keys, selected, direct, threshold)
        keep = fused.accepted_blocks.bool().repeat_interleave(64)[:context]
        error = (fused.scores[keep] - full[keep]).abs()
        torch.cuda.synchronize()
        rows.append(
            {
                "context": context,
                "batch": 1,
                "query_length": 1,
                "heads": 64,
                "head_dimension": 128,
                "dtype": "bfloat16",
                "topk": 2048,
                "block_size": 64,
                "full64_shape": list(full.shape),
                "h8_shape": list(h8.shape),
                "block_max_shape": list(maxima.shape),
                "fused_dense_shape": list(fused.scores.shape),
                "accepted_block_shape": list(fused.accepted_blocks.shape),
                "accepted_blocks": int(fused.accepted_blocks.sum().item()),
                "max_abs_fused_vs_full_on_candidates": float(error.max().item()),
                "passed": bool(
                    full.shape == h8.shape == fused.scores.shape == (context,)
                    and fused.accepted_blocks.shape == (blocks,)
                    and error.max().item() < 1e-4
                ),
                "model_inference_run": False,
                "synthetic": True,
            }
        )
    payload = {
        "gpu": torch.cuda.get_device_name(args.device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "seed": args.seed,
        "rows": rows,
        "all_passed": all(row["passed"] for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
