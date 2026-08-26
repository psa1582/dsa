from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from benchmark_fused_h8_sm89 import (
    CAPTURES,
    cuda_bench,
    load_real_variant,
    prepare_item,
)
from temporal_dsa.fused_h8_sm89 import load_extension


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--measure", type=int, default=1000)
    args = parser.parse_args()
    ext = load_extension()
    runtime = json.loads(
        (args.repo / "artifacts/progressive_sw/threshold/runtime_config.json").read_text()
    )
    threshold = float(runtime["configs"][0]["promotion_threshold_by_layer"]["17"])
    thresholds = json.loads(
        (args.repo / "artifacts/progressive_threshold/thresholds.json").read_text()
    )
    fixed_sets = thresholds["fixed_head_ids_by_layer"]["17"]
    selected = json.loads((args.output / "selected_kernel_config.json").read_text())
    ctas = int(selected["k6"]["ctas_per_sm"])
    producers = int(selected["k6"]["producer_warps"])
    l2_bytes = int(ext.device_properties()["l2_cache_bytes"])
    flush = torch.empty(l2_bytes, device="cuda", dtype=torch.float32)
    rows = []
    for context in (16384, 32768):
        items = []
        for variant, capture in enumerate(CAPTURES):
            q, w, keys = load_real_variant(args.repo, context, capture, 17)
            items.append(
                prepare_item(
                    ext, q, w, keys, context=context,
                    seed=9900 + context + variant, local_threshold=threshold,
                )
            )
        q_packed = torch.empty_like(items[0].packed_q)
        w_packed = torch.empty_like(items[0].packed_w)
        packed_ids = torch.empty_like(items[0].packed_ids)
        schemes = {
            "dynamic_abs_w": torch.empty(0, device="cuda", dtype=torch.int32),
            "fixed_avg_abs_w": torch.tensor(
                fixed_sets["fixed_avg_abs_w"], device="cuda", dtype=torch.int32
            ),
            "fixed_transition_aware": torch.tensor(
                fixed_sets["fixed_transition_aware"], device="cuda", dtype=torch.int32
            ),
        }
        for scheme, fixed in schemes.items():
            def prep(index: int) -> None:
                item = items[index % len(items)]
                ext.pack_qw(item.q, item.w, fixed, q_packed, w_packed, packed_ids)

            def prep_main(index: int) -> None:
                item = items[index % len(items)]
                ext.pack_qw(item.q, item.w, fixed, q_packed, w_packed, packed_ids)
                ext.fused_online_pipeline(
                    q_packed, w_packed, item.keys, item.direct, threshold,
                    item.output, item.accepted, item.maxima, ctas, producers,
                )

            for scope, fn in (("prep_only", prep), ("prep_plus_main", prep_main)):
                stats = cuda_bench(
                    fn, variants=len(items), flush=flush,
                    warmup=args.warmup, measure=args.measure,
                )
                rows.append({
                    "context": context,
                    "dtype": "bf16",
                    "block_size": 64,
                    "promotion_policy": "validation_fixed_threshold",
                    "threshold": threshold,
                    "ctas_per_sm": ctas,
                    "producer_warps": producers,
                    "method": "QW_pack" if scope == "prep_only" else "K6_fused_online_pipeline",
                    "cache_state": "cold_rotating",
                    "timing_scope": scope,
                    "head_scheme": scheme,
                    "prep_included": True,
                    **stats,
                })
    path = args.output / "fused_h8_online_timing.csv"
    prior = pd.read_csv(path)
    prior["timing_scope"] = "main_only_prepacked"
    prior["head_scheme"] = "dynamic_abs_w"
    prior["prep_included"] = False
    pd.concat([prior, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)


if __name__ == "__main__":
    main()
