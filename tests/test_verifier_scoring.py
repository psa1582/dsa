import pytest

torch = pytest.importorskip("torch")

from temporal_dsa.verifier_scoring import (  # noqa: E402
    normalized_hadamard,
    score_dim_sparse,
    score_head_sparse,
)


def test_hadamard_preserves_dot_product_at_full_width() -> None:
    generator = torch.Generator().manual_seed(1582)
    q = torch.randn(3, 4, 8, generator=generator)
    k = torch.randn(7, 8, generator=generator)
    direct = torch.einsum("qhd,kd->qhk", q, k)
    rotated = torch.einsum(
        "qhd,kd->qhk", normalized_hadamard(q), normalized_hadamard(k)
    )
    assert torch.allclose(direct, rotated, atol=1e-5)


def test_full_head_and_dimension_paths_reconstruct_same_score() -> None:
    generator = torch.Generator().manual_seed(1582)
    q = torch.randn(3, 4, 8, generator=generator)
    k = torch.randn(11, 8, generator=generator)
    w = torch.randn(3, 4, generator=generator)
    head = score_head_sparse(q, w, k, torch.arange(4), query_chunk_size=2, key_chunk_size=5)
    dim = score_dim_sparse(q, w, k, torch.arange(8), query_chunk_size=2, key_chunk_size=5)
    assert torch.allclose(head, dim, atol=1e-6)
