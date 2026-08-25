from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def apply_noninterleaved_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    *,
    theta: float = 10_000.0,
) -> torch.Tensor:
    """Apply the non-interleaved RoPE layout used by the V3.2 Indexer."""
    if x.shape[-1] % 2:
        raise ValueError("RoPE dimension must be even")
    half = x.shape[-1] // 2
    inv_freq = theta ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    angles = positions.to(torch.float32).unsqueeze(-1) * inv_freq
    cos = torch.cos(angles).to(x.dtype)
    sin = torch.sin(angles).to(x.dtype)
    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    first, second = x[..., :half], x[..., half:]
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)


class LightningIndexerSidecar(nn.Module):
    """BF16 research sidecar matching the V3.2 Lightning Indexer score equation.

    V3.2 obtains its query from the attention q-LoRA latent.  V2-Lite has no
    q-LoRA rank, so this pilot deliberately projects the layer input directly.
    The deviation is explicit and checkpoints are not production-compatible.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        heads: int = 64,
        head_dim: int = 128,
        rope_dim: int = 64,
        rope_theta: float = 10_000.0,
    ) -> None:
        super().__init__()
        if not 0 <= rope_dim <= head_dim or rope_dim % 2:
            raise ValueError("rope_dim must be even and no larger than head_dim")
        self.hidden_size = hidden_size
        self.heads = heads
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.rope_theta = rope_theta
        self.wq = nn.Linear(hidden_size, heads * head_dim, bias=False)
        self.wk = nn.Linear(hidden_size, head_dim, bias=False)
        self.k_norm = nn.LayerNorm(head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(hidden_size, heads, bias=False, dtype=torch.float32)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.wq.weight, mean=0.0, std=self.hidden_size**-0.5)
        nn.init.normal_(self.wk.weight, mean=0.0, std=self.hidden_size**-0.5)
        nn.init.normal_(self.weights_proj.weight, mean=0.0, std=self.hidden_size**-0.5)

    def _rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self.rope_dim == 0:
            return x
        rope = apply_noninterleaved_rope(
            x[..., : self.rope_dim], positions, theta=self.rope_theta
        )
        return torch.cat((rope, x[..., self.rope_dim :]), dim=-1)

    def encode_keys(self, hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        keys = self.k_norm(self.wk(hidden))
        return self._rope(keys, positions)

    def encode_queries(
        self, hidden: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        queries = self.wq(hidden).view(*hidden.shape[:-1], self.heads, self.head_dim)
        queries = self._rope(queries, positions)
        device_type = hidden.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            weights = self.weights_proj(hidden.float())
        weights = weights * (self.heads * self.head_dim) ** -0.5
        return queries, weights

    def score_encoded(
        self,
        queries: torch.Tensor,
        weights: torch.Tensor,
        keys: torch.Tensor,
        *,
        key_chunk_size: int | None = None,
    ) -> torch.Tensor:
        if queries.ndim != 4 or weights.shape != queries.shape[:-1]:
            raise ValueError("queries must be [batch,query,heads,dim]")
        if keys.ndim != 3 or keys.shape[0] != queries.shape[0]:
            raise ValueError("keys must be [batch,key,dim]")
        chunk = key_chunk_size or keys.shape[1]
        outputs = []
        for start in range(0, keys.shape[1], chunk):
            key_chunk = keys[:, start : start + chunk]
            dots = torch.einsum("bqhd,bkd->bqhk", queries, key_chunk)
            outputs.append((F.relu(dots) * weights.unsqueeze(-1)).sum(dim=2))
        return torch.cat(outputs, dim=-1)

    def forward(
        self,
        query_hidden: torch.Tensor,
        key_hidden: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
        *,
        key_chunk_size: int | None = None,
    ) -> torch.Tensor:
        queries, weights = self.encode_queries(query_hidden, query_positions)
        keys = self.encode_keys(key_hidden, key_positions)
        return self.score_encoded(queries, weights, keys, key_chunk_size=key_chunk_size)


def indexer_kl_loss(scores: torch.Tensor, teacher_probabilities: torch.Tensor) -> torch.Tensor:
    """D_KL(p_teacher || softmax(index_score)), averaged over batch/query."""
    teacher = teacher_probabilities.float()
    teacher = teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    log_prediction = F.log_softmax(scores.float(), dim=-1)
    return F.kl_div(log_prediction, teacher, reduction="batchmean")


def teacher_from_attention_logits(attention_logits: torch.Tensor) -> torch.Tensor:
    """Official dense warm-up target: sum per-head probabilities, then L1-normalize."""
    if attention_logits.ndim < 3:
        raise ValueError("attention logits need a head and key dimension")
    probabilities = torch.softmax(attention_logits.float(), dim=-1)
    target = probabilities.sum(dim=-2)
    return target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
