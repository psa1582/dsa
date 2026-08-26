"""PyTorch bindings for the L40S SM89 progressive H8 CUDA pilot.

The extension is intentionally JIT-built outside the source tree.  All scoring
kernels use the same exact BF16 ``mma.sync.m16n8k16`` primitive; the proposed
path differs only in scheduling and whether H56 is conditionally continued.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.cpp_extension import load


HEADS = 64
HEAD_DIM = 128
GROUP_HEADS = 8
BLOCK_SIZE = 64


_EXTENSION = None


def load_extension(*, build_directory: Path | None = None, verbose: bool = False):
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    root = Path(__file__).resolve().parents[2]
    cuda_root = root / "cuda" / "fused_h8_sm89"
    if build_directory is None:
        build_directory = root / ".torch_extensions" / "fused_h8_sm89"
    build_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    _EXTENSION = load(
        name="temporal_dsa_fused_h8_sm89",
        sources=[str(cuda_root / "bindings.cpp"), str(cuda_root / "kernels.cu")],
        extra_include_paths=[str(cuda_root)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "-arch=sm_89",
            "--use_fast_math",
            "-lineinfo",
            "-Xptxas=-v",
        ],
        build_directory=str(build_directory),
        verbose=verbose,
    )
    return _EXTENSION


@dataclass(frozen=True)
class PackedQuery:
    q: torch.Tensor
    w: torch.Tensor
    head_ids: torch.Tensor


@dataclass(frozen=True)
class ProgressiveResult:
    scores: torch.Tensor
    accepted: torch.Tensor
    block_max: torch.Tensor


def _check_inputs(q: torch.Tensor, w: torch.Tensor, keys: torch.Tensor | None = None) -> None:
    if q.shape != (HEADS, HEAD_DIM) or q.dtype != torch.bfloat16 or not q.is_cuda:
        raise ValueError("q must be CUDA BF16 [64,128]")
    if w.shape != (HEADS,) or w.dtype != torch.float32 or not w.is_cuda:
        raise ValueError("w must be CUDA FP32 [64]")
    if keys is not None and (
        keys.ndim != 2
        or keys.shape[1] != HEAD_DIM
        or keys.dtype != torch.bfloat16
        or not keys.is_cuda
    ):
        raise ValueError("keys must be CUDA BF16 [L,128]")


def pack_query(
    q: torch.Tensor,
    w: torch.Tensor,
    fixed_heads: Sequence[int] | torch.Tensor | None = None,
) -> PackedQuery:
    _check_inputs(q, w)
    ext = load_extension()
    if fixed_heads is None:
        fixed = torch.empty(0, device=q.device, dtype=torch.int32)
    else:
        fixed = torch.as_tensor(fixed_heads, device=q.device, dtype=torch.int32).contiguous()
        if fixed.shape != (GROUP_HEADS,) or fixed.unique().numel() != GROUP_HEADS:
            raise ValueError("fixed_heads must contain 8 unique head IDs")
        if int(fixed.min()) < 0 or int(fixed.max()) >= HEADS:
            raise ValueError("fixed head IDs must be in [0,64)")
    q_packed = torch.empty((8, HEAD_DIM, GROUP_HEADS), device=q.device, dtype=q.dtype)
    w_packed = torch.empty((8, GROUP_HEADS), device=q.device, dtype=w.dtype)
    ids = torch.empty(HEADS, device=q.device, dtype=torch.int32)
    ext.pack_qw(q.contiguous(), w.contiguous(), fixed, q_packed, w_packed, ids)
    return PackedQuery(q_packed, w_packed, ids)


def reference_scores(q: torch.Tensor, w: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    _check_inputs(q, w, keys)
    return (torch.relu(q.float() @ keys.float().T) * w[:, None]).sum(dim=0)


def full64_sync(packed: PackedQuery, keys: torch.Tensor) -> torch.Tensor:
    output = torch.empty(keys.shape[0], device=keys.device, dtype=torch.float32)
    load_extension().full64_sync(packed.q, packed.w, keys.contiguous(), output)
    return output


def full64_sync_variant(
    packed: PackedQuery,
    keys: torch.Tensor,
    *,
    layout_id: int,
    q_shared: bool,
) -> torch.Tensor:
    output = torch.empty(keys.shape[0], device=keys.device, dtype=torch.float32)
    load_extension().full64_sync_variant(
        packed.q, packed.w, keys.contiguous(), output, layout_id, q_shared
    )
    return output


def full64_pipeline(
    packed: PackedQuery,
    keys: torch.Tensor,
    *,
    ctas_per_sm: int = 1,
    producer_warps: int = 0,
) -> torch.Tensor:
    output = torch.empty(keys.shape[0], device=keys.device, dtype=torch.float32)
    load_extension().full64_pipeline(
        packed.q, packed.w, keys.contiguous(), output, ctas_per_sm, producer_warps
    )
    return output


def h8_two_pass(
    packed: PackedQuery, keys: torch.Tensor, promotion_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    length = keys.shape[0]
    blocks = (length + BLOCK_SIZE - 1) // BLOCK_SIZE
    if promotion_mask.shape != (blocks,) or promotion_mask.dtype != torch.bool:
        raise ValueError("promotion_mask must be CUDA bool [ceil(L/64)]")
    h8 = torch.empty(length, device=keys.device, dtype=torch.float32)
    maxima = torch.empty(blocks, device=keys.device, dtype=torch.float32)
    output = torch.empty(length, device=keys.device, dtype=torch.float32)
    ext = load_extension()
    ext.h8_pass(packed.q, packed.w, keys.contiguous(), h8, maxima)
    ext.h56_pass(packed.q, packed.w, keys.contiguous(), promotion_mask.contiguous(), h8, output)
    return output, h8, maxima


def fused_mask(
    packed: PackedQuery, keys: torch.Tensor, promotion_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    length = keys.shape[0]
    blocks = (length + BLOCK_SIZE - 1) // BLOCK_SIZE
    output = torch.empty(length, device=keys.device, dtype=torch.float32)
    maxima = torch.empty(blocks, device=keys.device, dtype=torch.float32)
    load_extension().fused_mask_sync(
        packed.q, packed.w, keys.contiguous(), promotion_mask.contiguous(), output, maxima
    )
    return output, maxima


def fused_online(
    packed: PackedQuery,
    keys: torch.Tensor,
    direct_mask: torch.Tensor,
    threshold: float,
    *,
    pipeline: bool,
    ctas_per_sm: int = 1,
    producer_warps: int = 0,
) -> ProgressiveResult:
    length = keys.shape[0]
    blocks = (length + BLOCK_SIZE - 1) // BLOCK_SIZE
    output = torch.empty(length, device=keys.device, dtype=torch.float32)
    accepted = torch.empty(blocks, device=keys.device, dtype=torch.bool)
    maxima = torch.empty(blocks, device=keys.device, dtype=torch.float32)
    ext = load_extension()
    if pipeline:
        ext.fused_online_pipeline(
            packed.q,
            packed.w,
            keys.contiguous(),
            direct_mask.contiguous(),
            threshold,
            output,
            accepted,
            maxima,
            ctas_per_sm,
            producer_warps,
        )
    else:
        ext.fused_online_sync(
            packed.q,
            packed.w,
            keys.contiguous(),
            direct_mask.contiguous(),
            threshold,
            output,
            accepted,
            maxima,
        )
    return ProgressiveResult(output, accepted, maxima)
