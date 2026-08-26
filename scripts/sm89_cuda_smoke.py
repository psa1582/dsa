from __future__ import annotations

import math

import torch

from temporal_dsa.fused_h8_sm89 import (
    BLOCK_SIZE,
    full64_pipeline,
    full64_sync,
    fused_mask,
    fused_online,
    h8_two_pass,
    pack_query,
    reference_scores,
)


def compare(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    error = (actual[finite] - expected[finite]).abs()
    print(name, "max_abs", float(error.max()) if error.numel() else 0.0)


def main() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda:0")
    q = torch.randn(64, 128, device=device, dtype=torch.bfloat16)
    w = torch.randn(64, device=device, dtype=torch.float32)
    keys = torch.randn(193, 128, device=device, dtype=torch.bfloat16)
    packed = pack_query(q, w)
    reference = reference_scores(q, w, keys)
    sync = full64_sync(packed, keys)
    pipeline = full64_pipeline(packed, keys)
    compare("K1", sync, reference)
    compare("K2", pipeline, reference)

    blocks = math.ceil(keys.shape[0] / BLOCK_SIZE)
    mask = torch.tensor([True, False, True, False], device=device)[:blocks]
    fused, _ = fused_mask(packed, keys, mask)
    two, _, _ = h8_two_pass(packed, keys, mask)
    expanded = mask.repeat_interleave(BLOCK_SIZE)[: keys.shape[0]]
    compare("K3", two[expanded], reference[expanded])
    compare("K4", fused[expanded], reference[expanded])
    assert torch.isneginf(two[~expanded]).all()
    assert torch.isneginf(fused[~expanded]).all()

    direct = mask.clone()
    online = fused_online(packed, keys, direct, math.inf, pipeline=False)
    online_pipe = fused_online(packed, keys, direct, math.inf, pipeline=True)
    compare("K5", online.scores[expanded], reference[expanded])
    compare("K6", online_pipe.scores[expanded], reference[expanded])
    assert torch.equal(online.accepted, direct)
    assert torch.equal(online_pipe.accepted, direct)
    print("head_ids", packed.head_ids[:8].tolist())


if __name__ == "__main__":
    main()
