from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def normalized_hadamard(tensor):
    """Apply an orthonormal Walsh-Hadamard transform on the last axis."""

    import torch

    size = tensor.shape[-1]
    if size <= 0 or size & (size - 1):
        raise ValueError("Hadamard dimension must be a positive power of two")
    output = tensor
    stride = 1
    while stride < size:
        shape = (*output.shape[:-1], -1, stride * 2)
        grouped = output.reshape(shape)
        left = grouped[..., :stride]
        right = grouped[..., stride:]
        output = torch.cat((left + right, left - right), dim=-1).reshape_as(output)
        stride *= 2
    return output * (size**-0.5)


def fake_symmetric_quantize(tensor, bits: int, scale: float | None = None):
    """Reference fake quantization with a fixed symmetric tensor scale."""

    import torch

    if bits not in {2, 4, 8}:
        raise ValueError("bits must be one of 2, 4, 8")
    limit = 2 ** (bits - 1) - 1
    if scale is None:
        scale = float(tensor.float().abs().max().cpu()) / max(limit, 1)
    scale = max(float(scale), 1e-12)
    quantized = torch.clamp(torch.round(tensor.float() / scale), -limit, limit)
    return (quantized * scale).to(tensor.dtype), scale


def _chunks(size: int, chunk: int) -> Iterable[tuple[int, int]]:
    for start in range(0, size, chunk):
        yield start, min(size, start + chunk)


def score_head_sparse(
    queries,
    weights,
    keys,
    head_indices,
    *,
    query_chunk_size: int = 1,
    key_chunk_size: int = 8192,
):
    """Compute selected-head full-dimension verifier scores.

    ``head_indices`` may be a fixed ``[h_v]`` vector or a dynamic ``[Q,h_v]``
    matrix.  Output is float32 ``[Q,K]``.
    """

    import torch
    from torch.nn import functional as F

    if queries.ndim != 3 or weights.shape != queries.shape[:2] or keys.ndim != 2:
        raise ValueError("expected queries [Q,H,D], weights [Q,H], keys [K,D]")
    head_indices = torch.as_tensor(head_indices, device=queries.device, dtype=torch.long)
    if head_indices.ndim not in {1, 2}:
        raise ValueError("head_indices must be [h_v] or [Q,h_v]")
    if head_indices.ndim == 2 and head_indices.shape[0] != queries.shape[0]:
        raise ValueError("dynamic head_indices must have one row per query")
    output = torch.empty(
        (queries.shape[0], keys.shape[0]), device=queries.device, dtype=torch.float32
    )
    for q_start, q_stop in _chunks(queries.shape[0], query_chunk_size):
        ids = head_indices if head_indices.ndim == 1 else head_indices[q_start:q_stop]
        if ids.ndim == 1:
            selected_q = queries[q_start:q_stop, ids]
            selected_w = weights[q_start:q_stop, ids]
        else:
            rows = torch.arange(q_stop - q_start, device=queries.device).unsqueeze(-1)
            selected_q = queries[q_start:q_stop][rows, ids]
            selected_w = weights[q_start:q_stop][rows, ids]
        for k_start, k_stop in _chunks(keys.shape[0], key_chunk_size):
            key_chunk = keys[k_start:k_stop].to(selected_q.dtype)
            dots = torch.einsum("qhd,kd->qhk", selected_q, key_chunk)
            output[q_start:q_stop, k_start:k_stop] = (
                F.relu(dots).float() * selected_w.float().unsqueeze(-1)
            ).sum(dim=1)
    return output


def score_dim_sparse(
    queries,
    weights,
    keys,
    dimension_indices,
    *,
    rotation: str = "none",
    bits: int | None = None,
    query_scale: float | None = None,
    key_scale: float | None = None,
    query_chunk_size: int = 1,
    key_chunk_size: int = 8192,
):
    """Compute all-head partial-dimension verifier scores."""

    import torch
    from torch.nn import functional as F

    if queries.ndim != 3 or weights.shape != queries.shape[:2] or keys.ndim != 2:
        raise ValueError("expected queries [Q,H,D], weights [Q,H], keys [K,D]")
    if rotation == "hadamard":
        queries = normalized_hadamard(queries)
        keys = normalized_hadamard(keys)
    elif rotation != "none":
        raise ValueError("rotation must be 'none' or 'hadamard'")
    dimensions = torch.as_tensor(
        dimension_indices, device=queries.device, dtype=torch.long
    )
    selected_q = queries.index_select(-1, dimensions)
    selected_k = keys.index_select(-1, dimensions).to(selected_q.dtype)
    if bits is not None:
        selected_q, _ = fake_symmetric_quantize(selected_q, bits, query_scale)
        selected_k, _ = fake_symmetric_quantize(selected_k, bits, key_scale)
    output = torch.empty(
        (queries.shape[0], keys.shape[0]), device=queries.device, dtype=torch.float32
    )
    for q_start, q_stop in _chunks(queries.shape[0], query_chunk_size):
        q_chunk = selected_q[q_start:q_stop]
        w_chunk = weights[q_start:q_stop]
        for k_start, k_stop in _chunks(keys.shape[0], key_chunk_size):
            dots = torch.einsum("qhd,kd->qhk", q_chunk, selected_k[k_start:k_stop])
            output[q_start:q_stop, k_start:k_stop] = (
                F.relu(dots).float() * w_chunk.float().unsqueeze(-1)
            ).sum(dim=1)
    return output


def dynamic_head_indices(weights, width: int, strategy: str):
    """Return per-query runtime-visible head routing from current weights."""

    import torch

    if not 0 < width <= weights.shape[-1]:
        raise ValueError("invalid head width")
    if strategy == "high_weight":
        values = weights.abs()
    elif strategy == "positive_weight":
        values = torch.clamp_min(weights, 0)
    else:
        raise ValueError("strategy must be high_weight or positive_weight")
    address = torch.arange(weights.shape[-1], device=weights.device)
    # Stable argsort preserves lower head IDs for deterministic ties.
    order = torch.argsort(values, dim=-1, descending=True, stable=True)
    del address
    return order[:, :width]


def layer_energy_dimensions(queries, keys, width: int) -> np.ndarray:
    """Select a fixed layer subset using validation q/k energy only."""

    energy = queries.float().square().mean(dim=(0, 1)) * keys.float().square().mean(dim=0)
    return (
        energy.argsort(descending=True, stable=True)[:width]
        .sort()
        .values.cpu().numpy().astype(np.int64)
    )


def load_sidecar_encoded(
    capture_path: str | Path,
    checkpoint_path: str | Path,
    lengths: np.ndarray,
    *,
    device: str = "cuda",
):
    """Reconstruct the exact sidecar q/w/K tensors from an existing capture."""

    import torch
    from safetensors.torch import load_file

    from .sidecar import LightningIndexerSidecar

    capture = torch.load(capture_path, map_location="cpu", weights_only=True)
    target = torch.device(device)
    if target.type not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported sidecar replay device: {target.type}")
    hidden = capture["hidden"].to(device=target, dtype=torch.bfloat16).unsqueeze(0)
    indexer = LightningIndexerSidecar().to(device=device, dtype=torch.float32).eval()
    indexer.load_state_dict(load_file(str(checkpoint_path), device=device))
    positions = torch.arange(hidden.shape[1], device=target).view(1, -1)
    query_positions = torch.as_tensor(lengths - 1, device=target, dtype=torch.long)
    with torch.no_grad(), torch.autocast(device_type=target.type, dtype=torch.bfloat16):
        keys = indexer.encode_keys(hidden, positions).squeeze(0)
        # Match trace collection's one-query GEMM shape.  Batched projection is
        # mathematically equivalent but can differ by a few BF16 tie-breaks.
        query_rows = []
        weight_rows = []
        for position in query_positions.tolist():
            query, weight = indexer.encode_queries(
                hidden[:, position : position + 1],
                torch.tensor([[position]], device=device),
            )
            query_rows.append(query)
            weight_rows.append(weight)
        queries = torch.cat(query_rows, dim=1)
        weights = torch.cat(weight_rows, dim=1)
    return queries.squeeze(0), weights.squeeze(0), keys, capture
