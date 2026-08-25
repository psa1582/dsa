import torch

from temporal_dsa.sidecar import (
    LightningIndexerSidecar,
    indexer_kl_loss,
    teacher_from_attention_logits,
)


def test_sidecar_shapes_and_loss() -> None:
    torch.manual_seed(1)
    indexer = LightningIndexerSidecar(hidden_size=16, heads=4, head_dim=8, rope_dim=4)
    query = torch.randn(2, 3, 16)
    keys = torch.randn(2, 7, 16)
    q_pos = torch.arange(3).repeat(2, 1)
    k_pos = torch.arange(7).repeat(2, 1)
    scores = indexer(query, keys, q_pos, k_pos, key_chunk_size=3)
    assert scores.shape == (2, 3, 7)
    target = torch.softmax(torch.randn(2, 3, 7), dim=-1)
    assert torch.isfinite(indexer_kl_loss(scores, target))


def test_teacher_is_head_probability_average_not_raw_logit_sum() -> None:
    logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    target = teacher_from_attention_logits(logits)
    assert torch.allclose(target, torch.tensor([[0.5, 0.5]]))

