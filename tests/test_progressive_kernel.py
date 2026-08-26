from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from temporal_dsa.progressive_kernel import (
    HEADS,
    HEAD_DIM,
    dynamic_head_ids,
    full_scores_triton,
    progressive_qk_reduction,
    progressive_scores_triton,
    split_head_ids,
    torch_full_scores,
    torch_progressive_scores,
    two_pass_scores_triton,
)


def cpu_inputs(length: int = 129) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1582)
    queries = torch.randn(HEADS, HEAD_DIM, generator=generator)
    weights = torch.randn(HEADS, generator=generator)
    keys = torch.randn(length, HEAD_DIM, generator=generator)
    return queries, weights, keys


def test_split_head_ids_returns_complement() -> None:
    selected, remaining = split_head_ids(torch.tensor([0, 2, 4, 6, 8, 10, 12, 63]))
    assert selected.numel() == 8
    assert remaining.numel() == 56
    assert set(selected.tolist()).isdisjoint(remaining.tolist())
    assert sorted(selected.tolist() + remaining.tolist()) == list(range(64))


@pytest.mark.parametrize(
    "values",
    [
        [0, 1],
        [0, 1, 2, 3, 4, 5, 6, 64],
        [0, 1, 2, 3, 4, 5, 6, 6],
    ],
)
def test_split_head_ids_rejects_invalid(values: list[int]) -> None:
    with pytest.raises(ValueError):
        split_head_ids(torch.tensor(values))


def test_dynamic_head_ids_is_stable_for_ties() -> None:
    weights = torch.zeros(64)
    weights[20] = -4
    weights[3] = 4
    result = dynamic_head_ids(weights)
    assert result[:2].tolist() == [3, 20]
    assert result[2:].tolist() == [0, 1, 2, 4, 5, 6]


def test_torch_full_scores_matches_explicit_sum() -> None:
    queries, weights, keys = cpu_inputs(17)
    actual = torch_full_scores(queries, weights, keys)
    expected = []
    for key in keys:
        expected.append(sum(weights[h] * torch.relu(torch.dot(queries[h], key)) for h in range(64)))
    torch.testing.assert_close(actual, torch.stack(expected), rtol=1e-5, atol=1e-4)


def test_torch_progressive_all_promoted_equals_full() -> None:
    queries, weights, keys = cpu_inputs(129)
    direct = torch.zeros(math.ceil(len(keys) / 64), dtype=torch.bool)
    output, accepted, _ = torch_progressive_scores(
        queries, weights, keys, dynamic_head_ids(weights), direct, -math.inf
    )
    torch.testing.assert_close(output, torch_full_scores(queries, weights, keys))
    assert accepted.all()


def test_torch_progressive_rejects_cold_blocks() -> None:
    queries, weights, keys = cpu_inputs(128)
    direct = torch.tensor([True, False])
    output, accepted, maxima = torch_progressive_scores(
        queries, weights, keys, dynamic_head_ids(weights), direct, math.inf
    )
    assert accepted.tolist() == [True, False]
    assert torch.isfinite(output[:64]).all()
    assert torch.isneginf(output[64:]).all()
    assert torch.isfinite(maxima).all()


def test_torch_progressive_handles_ragged_last_block() -> None:
    queries, weights, keys = cpu_inputs(65)
    direct = torch.tensor([False, True])
    output, accepted, _ = torch_progressive_scores(
        queries, weights, keys, torch.arange(8), direct, math.inf
    )
    assert output.shape == (65,)
    assert accepted.tolist() == [False, True]
    assert torch.isfinite(output[-1])


def test_progressive_qk_reduction_no_promotion() -> None:
    reduction = progressive_qk_reduction(8192, 0, 0)
    assert reduction == pytest.approx(0.875)


def test_progressive_qk_reduction_all_direct() -> None:
    reduction = progressive_qk_reduction(8192, 128, 0)
    assert reduction == pytest.approx(0.0)


def test_progressive_qk_reduction_with_promotion() -> None:
    reduction = progressive_qk_reduction(8192, 32, 16)
    expected_work = 2048 * 64 + 6144 * 8 + 1024 * 56
    assert reduction == pytest.approx(1 - expected_work / (8192 * 64))


def test_progressive_qk_reduction_rejects_bad_counts() -> None:
    with pytest.raises(ValueError):
        progressive_qk_reduction(8192, 129, 0)
    with pytest.raises(ValueError):
        progressive_qk_reduction(8192, -1, 0)


gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for Triton kernel tests"
)


def cuda_inputs(length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(1582)
    queries = torch.randn(HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16) * 0.1
    weights = torch.randn(HEADS, device="cuda", dtype=torch.float32)
    keys = torch.randn(length, HEAD_DIM, device="cuda", dtype=torch.bfloat16) * 0.1
    return queries, weights, keys


@gpu
def test_triton_full_matches_torch_reference() -> None:
    queries, weights, keys = cuda_inputs(257)
    actual = full_scores_triton(queries, weights, keys)
    expected = torch_full_scores(queries, weights, keys)
    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)


@gpu
def test_triton_progressive_all_promoted_matches_full_kernel() -> None:
    queries, weights, keys = cuda_inputs(257)
    direct = torch.zeros(math.ceil(len(keys) / 64), device="cuda", dtype=torch.int8)
    result = progressive_scores_triton(
        queries, weights, keys, dynamic_head_ids(weights.cpu()), direct, -math.inf
    )
    expected = full_scores_triton(queries, weights, keys)
    torch.testing.assert_close(result.scores, expected, rtol=1e-5, atol=1e-5)
    assert result.accepted_blocks.bool().all()


@gpu
def test_triton_progressive_direct_and_rejected_blocks() -> None:
    queries, weights, keys = cuda_inputs(128)
    direct = torch.tensor([1, 0], device="cuda", dtype=torch.int8)
    result = progressive_scores_triton(
        queries, weights, keys, torch.arange(8), direct, math.inf
    )
    expected = full_scores_triton(queries, weights, keys)
    torch.testing.assert_close(result.scores[:64], expected[:64], rtol=1e-5, atol=1e-5)
    assert torch.isneginf(result.scores[64:]).all()


@gpu
def test_two_pass_matches_fused_decisions() -> None:
    queries, weights, keys = cuda_inputs(256)
    direct = torch.tensor([1, 0, 0, 1], device="cuda", dtype=torch.int8)
    heads = dynamic_head_ids(weights.cpu())
    probe = progressive_scores_triton(queries, weights, keys, heads, direct, math.inf)
    threshold = torch.quantile(probe.block_max[1:3], 0.5)
    fused = progressive_scores_triton(queries, weights, keys, heads, direct, threshold)
    two_pass = two_pass_scores_triton(queries, weights, keys, heads, direct, threshold)
    assert torch.equal(fused.accepted_blocks, two_pass.accepted_blocks)
    torch.testing.assert_close(fused.scores, two_pass.scores, rtol=1e-5, atol=1e-5)


@gpu
def test_compact_output_matches_dense_candidates() -> None:
    queries, weights, keys = cuda_inputs(192)
    direct = torch.tensor([1, 0, 1], device="cuda", dtype=torch.int8)
    heads = dynamic_head_ids(weights.cpu())
    dense = progressive_scores_triton(queries, weights, keys, heads, direct, math.inf)
    compact = progressive_scores_triton(
        queries, weights, keys, heads, direct, math.inf, compact=True
    )
    assert compact.candidate_count == 128
    assert compact.candidate_indices is not None
    order = torch.argsort(compact.candidate_indices.long())
    indices = compact.candidate_indices[order].long()
    torch.testing.assert_close(compact.scores[order], dense.scores[indices])
