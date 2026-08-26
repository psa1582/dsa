from __future__ import annotations

import math
from statistics import median

import torch

from temporal_dsa.fused_h8_sm89 import (
    full64_pipeline,
    full64_sync,
    fused_mask,
    fused_online,
    h8_two_pass,
    pack_query,
)


def bench(fn, warmup: int = 100, measure: int = 1000) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(measure)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(measure)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return median(start.elapsed_time(end) * 1000 for start, end in zip(starts, ends))


def main() -> None:
    torch.manual_seed(9)
    q = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(64, device="cuda")
    packed = pack_query(q, w)
    for length, direct_fraction in [(8192, 0.945519), (16384, 0.703342), (32768, 0.594227)]:
        keys = torch.randn(length, 128, device="cuda", dtype=torch.bfloat16)
        blocks = math.ceil(length / 64)
        direct_count = round(direct_fraction * blocks)
        promoted_cold = round(0.1 * (blocks - direct_count))
        mask = torch.zeros(blocks, device="cuda", dtype=torch.bool)
        mask[: direct_count + promoted_cold] = True
        direct = torch.zeros_like(mask)
        direct[:direct_count] = True
        values = {
            "K1": bench(lambda: full64_sync(packed, keys)),
            "K2": bench(lambda: full64_pipeline(packed, keys)),
            "K3": bench(lambda: h8_two_pass(packed, keys, mask)),
            "K4": bench(lambda: fused_mask(packed, keys, mask)),
            "K5": bench(lambda: fused_online(packed, keys, direct, math.inf, pipeline=False)),
            "K6": bench(lambda: fused_online(packed, keys, direct, math.inf, pipeline=True)),
        }
        print(length, values, "K2/K6", values["K2"] / values["K6"])
        if length == 32768:
            for ctas in (1, 2, 3):
                for producers in (0, 1, 4):
                    full_us = bench(
                        lambda: full64_pipeline(
                            packed, keys, ctas_per_sm=ctas, producer_warps=producers
                        ),
                        measure=300,
                    )
                    fused_us = bench(
                        lambda: fused_online(
                            packed,
                            keys,
                            direct,
                            math.inf,
                            pipeline=True,
                            ctas_per_sm=ctas,
                            producer_warps=producers,
                        ),
                        measure=300,
                    )
                    print("sweep", ctas, producers, full_us, fused_us, full_us / fused_us)


if __name__ == "__main__":
    main()
