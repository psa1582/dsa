from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from benchmark_fused_h8_sm89 import load_real_variant, prepare_item
from temporal_dsa.fused_h8_sm89 import load_extension


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("K2", "K3", "K4", "K6"), required=True)
    parser.add_argument("--context", type=int, choices=(16384, 32768), required=True)
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()
    ext = load_extension()
    runtime = json.loads(
        (args.repo / "artifacts/progressive_sw/threshold/runtime_config.json").read_text()
    )
    threshold = float(runtime["configs"][0]["promotion_threshold_by_layer"]["17"])
    selected = json.loads((args.output / "selected_kernel_config.json").read_text())
    q, w, keys = load_real_variant(args.repo, args.context, "code_heldout_3", 17)
    item = prepare_item(
        ext, q, w, keys, context=args.context, seed=18158,
        local_threshold=threshold,
    )
    flush = torch.empty(
        int(ext.device_properties()["l2_cache_bytes"]), device="cuda", dtype=torch.float32
    )

    def run() -> None:
        if args.method == "K2":
            cfg = selected["k2"]
            ext.full64_pipeline(
                item.packed_q, item.packed_w, item.keys, item.output,
                int(cfg["ctas_per_sm"]), int(cfg["producer_warps"]),
            )
        elif args.method == "K3":
            ext.h8_pass(item.packed_q, item.packed_w, item.keys, item.h8, item.maxima)
            ext.h56_pass(
                item.packed_q, item.packed_w, item.keys, item.p0_keep,
                item.h8, item.output,
            )
        elif args.method == "K4":
            ext.fused_mask_sync(
                item.packed_q, item.packed_w, item.keys, item.p0_keep,
                item.output, item.maxima,
            )
        else:
            cfg = selected["k6"]
            ext.fused_online_pipeline(
                item.packed_q, item.packed_w, item.keys, item.direct, threshold,
                item.output, item.accepted, item.maxima,
                int(cfg["ctas_per_sm"]), int(cfg["producer_warps"]),
            )

    for _ in range(20):
        run()
    torch.cuda.synchronize()
    for _ in range(args.iterations):
        flush.add_(1)
        torch.cuda.nvtx.range_push(f"{args.method}_{args.context}_cold")
        run()
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    print(args.method, args.context, args.iterations)


if __name__ == "__main__":
    main()
