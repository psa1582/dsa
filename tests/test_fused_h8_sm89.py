from __future__ import annotations

import math

import pytest
import torch

from temporal_dsa.fused_h8_sm89 import (
    BLOCK_SIZE,
    full64_pipeline,
    full64_sync,
    full64_sync_variant,
    fused_mask,
    fused_online,
    h8_two_pass,
    pack_query,
    reference_scores,
)


SM89 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability(0) != (8, 9),
    reason="requires an SM89 CUDA device",
)


def inputs(length: int, seed: int = 1, weight_mode: str = "normal"):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn((64, 128), generator=generator, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn((length, 128), generator=generator, device="cuda", dtype=torch.bfloat16)
    if weight_mode == "zero":
        w = torch.zeros(64, device="cuda")
    elif weight_mode == "negative":
        w = -torch.rand(64, generator=generator, device="cuda")
    elif weight_mode == "signed":
        w = torch.linspace(-2, 2, 64, device="cuda")
    else:
        w = torch.randn(64, generator=generator, device="cuda")
    return q, w, keys


def assert_close(actual: torch.Tensor, expected: torch.Tensor):
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


@SM89
def test_dynamic_pack_top8_and_unique():
    q, w, _ = inputs(64)
    packed = pack_query(q, w)
    expected = torch.argsort(torch.abs(w), descending=True, stable=True)[:8]
    assert torch.equal(packed.head_ids[:8].long(), expected)
    assert packed.head_ids.unique().numel() == 64


@SM89
def test_fixed_pack_preserves_requested_group0():
    q, w, _ = inputs(64)
    fixed = [9, 7, 5, 3, 1, 11, 13, 15]
    packed = pack_query(q, w, fixed)
    assert packed.head_ids[:8].tolist() == fixed


@SM89
def test_pack_rejects_duplicate_heads():
    q, w, _ = inputs(64)
    with pytest.raises(ValueError):
        pack_query(q, w, [0] * 8)


@pytest.mark.parametrize("length", [1, 7, 31, 63, 64, 65, 127, 193, 8192])
@SM89
def test_full64_sync_tail_and_context(length: int):
    q, w, keys = inputs(length, seed=length)
    assert_close(full64_sync(pack_query(q, w), keys), reference_scores(q, w, keys))


@pytest.mark.parametrize("length", [17, 64, 129, 1025])
@SM89
def test_full64_pipeline_tail(length: int):
    q, w, keys = inputs(length, seed=length + 3)
    assert_close(full64_pipeline(pack_query(q, w), keys), reference_scores(q, w, keys))


@pytest.mark.parametrize("layout_id", [0, 1, 2])
@pytest.mark.parametrize("q_shared", [False, True])
@SM89
def test_shared_layout_variants(layout_id: int, q_shared: bool):
    q, w, keys = inputs(193, seed=60 + layout_id + int(q_shared))
    assert_close(
        full64_sync_variant(pack_query(q, w), keys, layout_id=layout_id, q_shared=q_shared),
        reference_scores(q, w, keys),
    )


@pytest.mark.parametrize("weight_mode", ["zero", "negative", "signed"])
@SM89
def test_signed_weights_and_relu_position(weight_mode: str):
    q, w, keys = inputs(193, seed=17, weight_mode=weight_mode)
    assert_close(full64_sync(pack_query(q, w), keys), reference_scores(q, w, keys))


@pytest.mark.parametrize("pattern", ["reject", "promote", "alternating"])
@SM89
def test_fused_mask_semantics(pattern: str):
    q, w, keys = inputs(257, seed=22)
    blocks = math.ceil(keys.shape[0] / BLOCK_SIZE)
    if pattern == "reject":
        mask = torch.zeros(blocks, device="cuda", dtype=torch.bool)
    elif pattern == "promote":
        mask = torch.ones(blocks, device="cuda", dtype=torch.bool)
    else:
        mask = torch.arange(blocks, device="cuda") % 2 == 0
    output, _ = fused_mask(pack_query(q, w), keys, mask)
    expanded = mask.repeat_interleave(BLOCK_SIZE)[: keys.shape[0]]
    assert torch.isneginf(output[~expanded]).all()
    assert_close(output[expanded], reference_scores(q, w, keys)[expanded])


@SM89
def test_two_pass_and_fused_reuse_h8_exactly_once():
    q, w, keys = inputs(385, seed=29)
    blocks = math.ceil(keys.shape[0] / BLOCK_SIZE)
    mask = torch.arange(blocks, device="cuda") % 3 != 1
    packed = pack_query(q, w)
    two, _, _ = h8_two_pass(packed, keys, mask)
    fused, _ = fused_mask(packed, keys, mask)
    assert_close(two, fused)


@SM89
def test_online_all_reject_and_all_promote():
    q, w, keys = inputs(320, seed=31)
    blocks = math.ceil(keys.shape[0] / BLOCK_SIZE)
    direct = torch.zeros(blocks, device="cuda", dtype=torch.bool)
    packed = pack_query(q, w)
    rejected = fused_online(packed, keys, direct, math.inf, pipeline=False)
    promoted = fused_online(packed, keys, direct, -math.inf, pipeline=False)
    assert not rejected.accepted.any()
    assert promoted.accepted.all()
    assert torch.isneginf(rejected.scores).all()
    assert_close(promoted.scores, reference_scores(q, w, keys))


@pytest.mark.parametrize("producer_warps", [0, 1, 4])
@SM89
def test_pipeline_warp_configs(producer_warps: int):
    q, w, keys = inputs(513, seed=37)
    blocks = math.ceil(keys.shape[0] / BLOCK_SIZE)
    direct = torch.arange(blocks, device="cuda") % 2 == 0
    packed = pack_query(q, w)
    result = fused_online(
        packed,
        keys,
        direct,
        math.inf,
        pipeline=True,
        ctas_per_sm=1,
        producer_warps=producer_warps,
    )
    expanded = direct.repeat_interleave(BLOCK_SIZE)[: keys.shape[0]]
    assert torch.equal(result.accepted, direct)
    assert_close(result.scores[expanded], reference_scores(q, w, keys)[expanded])


@pytest.mark.parametrize("ctas_per_sm", [1, 2, 3])
@SM89
def test_pipeline_cta_configs(ctas_per_sm: int):
    q, w, keys = inputs(2049, seed=41)
    packed = pack_query(q, w)
    assert_close(
        full64_pipeline(packed, keys, ctas_per_sm=ctas_per_sm),
        reference_scores(q, w, keys),
    )
