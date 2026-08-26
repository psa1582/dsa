from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on non-GPU development hosts
    triton = None
    tl = None


HEADS = 64
HEAD_DIM = 128
VERIFIER_HEADS = 8


def split_head_ids(selected: torch.Tensor | np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.as_tensor(selected, dtype=torch.int32)
    if values.ndim != 1 or values.numel() != VERIFIER_HEADS:
        raise ValueError("selected head IDs must have shape [8]")
    if values.min().item() < 0 or values.max().item() >= HEADS:
        raise ValueError("head ID outside [0, 64)")
    if torch.unique(values).numel() != VERIFIER_HEADS:
        raise ValueError("selected head IDs must be unique")
    mask = torch.ones(HEADS, dtype=torch.bool)
    mask[values.long()] = False
    remaining = torch.arange(HEADS, dtype=torch.int32)[mask]
    return values.contiguous(), remaining.contiguous()


def dynamic_head_ids(weights: torch.Tensor, width: int = VERIFIER_HEADS) -> torch.Tensor:
    if weights.shape != (HEADS,):
        raise ValueError("weights must have shape [64]")
    if not 0 < width <= HEADS:
        raise ValueError("invalid head width")
    return torch.argsort(weights.abs(), descending=True, stable=True)[:width].to(torch.int32)


def torch_full_scores(
    queries: torch.Tensor,
    weights: torch.Tensor,
    keys: torch.Tensor,
) -> torch.Tensor:
    if queries.shape != (HEADS, HEAD_DIM) or weights.shape != (HEADS,):
        raise ValueError("expected queries [64,128] and weights [64]")
    if keys.ndim != 2 or keys.shape[1] != HEAD_DIM:
        raise ValueError("expected keys [L,128]")
    dots = torch.matmul(queries.float(), keys.float().transpose(0, 1))
    return (torch.relu(dots) * weights.float().unsqueeze(1)).sum(dim=0)


def torch_progressive_scores(
    queries: torch.Tensor,
    weights: torch.Tensor,
    keys: torch.Tensor,
    selected_heads: torch.Tensor,
    direct_full_blocks: torch.Tensor,
    threshold: float,
    *,
    block_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected, remaining = split_head_ids(selected_heads.cpu())
    selected = selected.to(queries.device).long()
    remaining = remaining.to(queries.device).long()
    length = keys.shape[0]
    block_count = math.ceil(length / block_size)
    if direct_full_blocks.numel() != block_count:
        raise ValueError("direct_full_blocks length mismatch")
    output = torch.full((length,), -torch.inf, device=keys.device, dtype=torch.float32)
    accepted = direct_full_blocks.to(device=keys.device, dtype=torch.bool).clone()
    block_max = torch.full((block_count,), -torch.inf, device=keys.device, dtype=torch.float32)
    for block_id in range(block_count):
        start = block_id * block_size
        stop = min(length, start + block_size)
        tile = keys[start:stop]
        partial_dots = torch.matmul(queries[selected].float(), tile.float().transpose(0, 1))
        partial = (torch.relu(partial_dots) * weights[selected].float().unsqueeze(1)).sum(0)
        block_max[block_id] = partial.max()
        if accepted[block_id] or block_max[block_id] >= threshold:
            accepted[block_id] = True
            rest_dots = torch.matmul(queries[remaining].float(), tile.float().transpose(0, 1))
            rest = (torch.relu(rest_dots) * weights[remaining].float().unsqueeze(1)).sum(0)
            output[start:stop] = partial + rest
    return output, accepted, block_max


def progressive_qk_reduction(
    length: int,
    direct_full_blocks: int,
    promoted_cold_blocks: int,
    *,
    block_size: int = 64,
) -> float:
    block_count = math.ceil(length / block_size)
    if min(direct_full_blocks, promoted_cold_blocks) < 0:
        raise ValueError("block counts must be non-negative")
    if direct_full_blocks + promoted_cold_blocks > block_count:
        raise ValueError("accepted block counts exceed total blocks")
    direct_tokens = min(length, direct_full_blocks * block_size)
    promoted_tokens = min(length - direct_tokens, promoted_cold_blocks * block_size)
    cold_tokens = length - direct_tokens
    work = direct_tokens * HEADS + cold_tokens * VERIFIER_HEADS + promoted_tokens * (
        HEADS - VERIFIER_HEADS
    )
    return 1.0 - work / (length * HEADS)


@dataclass(frozen=True)
class ProgressiveResult:
    scores: torch.Tensor
    accepted_blocks: torch.Tensor
    block_max: torch.Tensor
    candidate_indices: torch.Tensor | None = None
    candidate_count: int | None = None


if triton is not None:

    @triton.jit
    def _selected_contribution(
        q_ptr,
        w_ptr,
        k_tile,
        head_ids_ptr,
        offs_d,
        offs_n,
        valid_n,
        D: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offs_m = tl.arange(0, 16)
        valid_m = offs_m < 8
        head_ids = tl.load(head_ids_ptr + offs_m, mask=valid_m, other=0)
        q = tl.load(
            q_ptr + head_ids[:, None] * D + offs_d[None, :],
            mask=valid_m[:, None],
            other=0.0,
        )
        dots = tl.dot(q, k_tile)
        weights = tl.load(w_ptr + head_ids, mask=valid_m, other=0.0).to(tl.float32)
        return tl.sum(tl.maximum(dots, 0.0) * weights[:, None], axis=0)


    @triton.jit
    def _remaining_contribution(
        q_ptr,
        w_ptr,
        k_tile,
        remaining_ids_ptr,
        offs_d,
        D: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        total = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for start in tl.static_range(0, 64, 16):
            offs_m = start + tl.arange(0, 16)
            valid_m = offs_m < 56
            head_ids = tl.load(remaining_ids_ptr + offs_m, mask=valid_m, other=0)
            q = tl.load(
                q_ptr + head_ids[:, None] * D + offs_d[None, :],
                mask=valid_m[:, None],
                other=0.0,
            )
            dots = tl.dot(q, k_tile)
            weights = tl.load(w_ptr + head_ids, mask=valid_m, other=0.0).to(tl.float32)
            total += tl.sum(tl.maximum(dots, 0.0) * weights[:, None], axis=0)
        return total


    @triton.jit
    def _full_contribution(
        q_ptr,
        w_ptr,
        k_tile,
        offs_d,
        D: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        total = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for start in tl.static_range(0, 64, 16):
            offs_m = start + tl.arange(0, 16)
            q = tl.load(q_ptr + offs_m[:, None] * D + offs_d[None, :])
            dots = tl.dot(q, k_tile)
            weights = tl.load(w_ptr + offs_m).to(tl.float32)
            total += tl.sum(tl.maximum(dots, 0.0) * weights[:, None], axis=0)
        return total


    @triton.jit
    def _full_score_kernel(
        q_ptr,
        w_ptr,
        k_ptr,
        out_ptr,
        length,
        D: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        block_id = tl.program_id(0)
        offs_n = block_id * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, D)
        valid_n = offs_n < length
        k_tile = tl.load(
            k_ptr + offs_d[:, None] + offs_n[None, :] * D,
            mask=valid_n[None, :],
            other=0.0,
        )
        total = _full_contribution(q_ptr, w_ptr, k_tile, offs_d, D, BLOCK_N)
        tl.store(out_ptr + offs_n, total, mask=valid_n)


    @triton.jit
    def _h8_block_max_kernel(
        q_ptr,
        w_ptr,
        k_ptr,
        head_ids_ptr,
        cold_blocks_ptr,
        partial_ptr,
        block_max_ptr,
        length,
        D: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        block_id = tl.program_id(0)
        offs_n = block_id * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, D)
        valid_n = offs_n < length
        is_cold = tl.load(cold_blocks_ptr + block_id) != 0
        k_tile = tl.load(
            k_ptr + offs_d[:, None] + offs_n[None, :] * D,
            mask=valid_n[None, :],
            other=0.0,
        )
        partial = _selected_contribution(
            q_ptr, w_ptr, k_tile, head_ids_ptr, offs_d, offs_n, valid_n, D, BLOCK_N
        )
        partial = tl.where(is_cold & valid_n, partial, -float("inf"))
        tl.store(partial_ptr + offs_n, partial, mask=valid_n)
        tl.store(block_max_ptr + block_id, tl.max(partial, axis=0))


    @triton.jit
    def _masked_full_kernel(
        q_ptr,
        w_ptr,
        k_ptr,
        keep_blocks_ptr,
        out_ptr,
        length,
        D: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        block_id = tl.program_id(0)
        offs_n = block_id * BLOCK_N + tl.arange(0, BLOCK_N)
        valid_n = offs_n < length
        keep = tl.load(keep_blocks_ptr + block_id) != 0
        if keep:
            offs_d = tl.arange(0, D)
            k_tile = tl.load(
                k_ptr + offs_d[:, None] + offs_n[None, :] * D,
                mask=valid_n[None, :],
                other=0.0,
            )
            total = _full_contribution(q_ptr, w_ptr, k_tile, offs_d, D, BLOCK_N)
            tl.store(out_ptr + offs_n, total, mask=valid_n)
        else:
            tl.store(out_ptr + offs_n, -float("inf"), mask=valid_n)


    @triton.jit
    def _progressive_kernel(
        q_ptr,
        w_ptr,
        k_ptr,
        head_ids_ptr,
        remaining_ids_ptr,
        direct_full_ptr,
        threshold_ptr,
        out_scores_ptr,
        out_indices_ptr,
        accepted_ptr,
        block_max_ptr,
        counter_ptr,
        length,
        D: tl.constexpr,
        BLOCK_N: tl.constexpr,
        COMPACT: tl.constexpr,
    ):
        block_id = tl.program_id(0)
        offs_n = block_id * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, D)
        valid_n = offs_n < length
        k_tile = tl.load(
            k_ptr + offs_d[:, None] + offs_n[None, :] * D,
            mask=valid_n[None, :],
            other=0.0,
        )
        direct = tl.load(direct_full_ptr + block_id) != 0
        threshold = tl.load(threshold_ptr)
        maximum = -float("inf")
        promote = False
        accept = True
        total = tl.zeros((BLOCK_N,), dtype=tl.float32)
        if direct:
            total = _full_contribution(q_ptr, w_ptr, k_tile, offs_d, D, BLOCK_N)
        else:
            partial = _selected_contribution(
                q_ptr,
                w_ptr,
                k_tile,
                head_ids_ptr,
                offs_d,
                offs_n,
                valid_n,
                D,
                BLOCK_N,
            )
            maximum = tl.max(tl.where(valid_n, partial, -float("inf")), axis=0)
            promote = maximum >= threshold
            accept = promote
            total = partial
            if promote:
                total += _remaining_contribution(
                    q_ptr, w_ptr, k_tile, remaining_ids_ptr, offs_d, D, BLOCK_N
                )
        tl.store(accepted_ptr + block_id, accept.to(tl.int8))
        tl.store(block_max_ptr + block_id, maximum)
        if COMPACT:
            if accept:
                slot = tl.atomic_add(counter_ptr, 1)
                output_offsets = slot * BLOCK_N + tl.arange(0, BLOCK_N)
                tl.store(out_scores_ptr + output_offsets, total, mask=valid_n)
                tl.store(out_indices_ptr + output_offsets, offs_n, mask=valid_n)
        else:
            tl.store(
                out_scores_ptr + offs_n,
                tl.where(accept, total, -float("inf")),
                mask=valid_n,
            )


def _check_cuda_inputs(queries: torch.Tensor, weights: torch.Tensor, keys: torch.Tensor) -> None:
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not queries.is_cuda or not weights.is_cuda or not keys.is_cuda:
        raise ValueError("Triton kernels require CUDA tensors")
    if queries.shape != (HEADS, HEAD_DIM) or weights.shape != (HEADS,):
        raise ValueError("expected queries [64,128] and weights [64]")
    if keys.ndim != 2 or keys.shape[1] != HEAD_DIM:
        raise ValueError("expected keys [L,128]")
    if queries.dtype not in {torch.float16, torch.bfloat16} or keys.dtype != queries.dtype:
        raise ValueError("queries and keys must share FP16/BF16 dtype")


def full_scores_triton(
    queries: torch.Tensor,
    weights: torch.Tensor,
    keys: torch.Tensor,
    *,
    block_size: int = 64,
) -> torch.Tensor:
    _check_cuda_inputs(queries, weights, keys)
    output = torch.empty(keys.shape[0], device=keys.device, dtype=torch.float32)
    _full_score_kernel[(triton.cdiv(keys.shape[0], block_size),)](
        queries, weights, keys, output, keys.shape[0], D=HEAD_DIM, BLOCK_N=block_size,
        num_warps=8,
    )
    return output


def h8_block_max_triton(
    queries: torch.Tensor,
    weights: torch.Tensor,
    keys: torch.Tensor,
    selected_heads: torch.Tensor,
    cold_blocks: torch.Tensor,
    *,
    block_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    _check_cuda_inputs(queries, weights, keys)
    block_count = triton.cdiv(keys.shape[0], block_size)
    if cold_blocks.numel() != block_count:
        raise ValueError("cold block mask length mismatch")
    if selected_heads.is_cuda:
        if selected_heads.numel() != VERIFIER_HEADS:
            raise ValueError("selected head IDs must have shape [8]")
        selected = selected_heads.to(device=keys.device, dtype=torch.int32).contiguous()
    else:
        selected, _ = split_head_ids(selected_heads)
        selected = selected.to(keys.device)
    partial = torch.empty(keys.shape[0], device=keys.device, dtype=torch.float32)
    maxima = torch.empty(block_count, device=keys.device, dtype=torch.float32)
    _h8_block_max_kernel[(block_count,)](
        queries, weights, keys, selected, cold_blocks, partial, maxima, keys.shape[0],
        D=HEAD_DIM, BLOCK_N=block_size, num_warps=8,
    )
    return partial, maxima


def masked_full_scores_triton(
    queries: torch.Tensor,
    weights: torch.Tensor,
    keys: torch.Tensor,
    keep_blocks: torch.Tensor,
    *,
    block_size: int = 64,
) -> torch.Tensor:
    _check_cuda_inputs(queries, weights, keys)
    block_count = triton.cdiv(keys.shape[0], block_size)
    if keep_blocks.numel() != block_count:
        raise ValueError("keep block mask length mismatch")
    output = torch.empty(keys.shape[0], device=keys.device, dtype=torch.float32)
    _masked_full_kernel[(block_count,)](
        queries, weights, keys, keep_blocks, output, keys.shape[0], D=HEAD_DIM,
        BLOCK_N=block_size, num_warps=8,
    )
    return output


def progressive_scores_triton(
    queries: torch.Tensor,
    weights: torch.Tensor,
    keys: torch.Tensor,
    selected_heads: torch.Tensor,
    direct_full_blocks: torch.Tensor,
    threshold: float | torch.Tensor,
    *,
    block_size: int = 64,
    compact: bool = False,
    remaining_heads: torch.Tensor | None = None,
) -> ProgressiveResult:
    _check_cuda_inputs(queries, weights, keys)
    if remaining_heads is None:
        selected, remaining = split_head_ids(selected_heads.cpu())
        selected = selected.to(keys.device)
        remaining = remaining.to(keys.device)
    else:
        if selected_heads.numel() != VERIFIER_HEADS or remaining_heads.numel() != HEADS - VERIFIER_HEADS:
            raise ValueError("expected 8 selected and 56 remaining head IDs")
        selected = selected_heads.to(device=keys.device, dtype=torch.int32).contiguous()
        remaining = remaining_heads.to(device=keys.device, dtype=torch.int32).contiguous()
    block_count = triton.cdiv(keys.shape[0], block_size)
    if direct_full_blocks.numel() != block_count:
        raise ValueError("direct full block mask length mismatch")
    threshold_tensor = torch.as_tensor(
        threshold, device=keys.device, dtype=torch.float32
    ).reshape(1)
    accepted = torch.empty(block_count, device=keys.device, dtype=torch.int8)
    block_max = torch.empty(block_count, device=keys.device, dtype=torch.float32)
    counter = torch.zeros(1, device=keys.device, dtype=torch.int32)
    scores = torch.empty(keys.shape[0], device=keys.device, dtype=torch.float32)
    indices = (
        torch.empty(keys.shape[0], device=keys.device, dtype=torch.int32) if compact else None
    )
    placeholder_indices = scores if indices is None else indices
    _progressive_kernel[(block_count,)](
        queries, weights, keys, selected, remaining, direct_full_blocks, threshold_tensor,
        scores, placeholder_indices, accepted, block_max, counter, keys.shape[0],
        D=HEAD_DIM, BLOCK_N=block_size, COMPACT=compact, num_warps=8,
    )
    candidate_count = None
    if compact:
        accepted_blocks = int(counter.item())
        candidate_count = min(keys.shape[0], accepted_blocks * block_size)
        scores = scores[:candidate_count]
        assert indices is not None
        indices = indices[:candidate_count]
    return ProgressiveResult(scores, accepted, block_max, indices, candidate_count)


def two_pass_scores_triton(
    queries: torch.Tensor,
    weights: torch.Tensor,
    keys: torch.Tensor,
    selected_heads: torch.Tensor,
    direct_full_blocks: torch.Tensor,
    threshold: float | torch.Tensor,
    *,
    block_size: int = 64,
) -> ProgressiveResult:
    cold = ~direct_full_blocks.to(torch.bool)
    _, maxima = h8_block_max_triton(
        queries, weights, keys, selected_heads, cold, block_size=block_size
    )
    threshold_tensor = torch.as_tensor(threshold, device=keys.device, dtype=torch.float32)
    accepted = direct_full_blocks.to(torch.bool) | (maxima >= threshold_tensor)
    scores = masked_full_scores_triton(
        queries, weights, keys, accepted, block_size=block_size
    )
    return ProgressiveResult(scores, accepted.to(torch.int8), maxima)
