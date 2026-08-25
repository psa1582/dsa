import numpy as np

from temporal_dsa.metrics import normalized_recall_lift, spearman, stable_topk, topk_recall


def test_stable_topk_uses_index_tie_break() -> None:
    values = np.array([1.0, 3.0, 3.0, 2.0])
    assert stable_topk(values, 2).tolist() == [1, 2]


def test_recall_and_random_normalization() -> None:
    assert topk_recall(np.array([1, 2]), np.array([2, 3])) == 0.5
    assert normalized_recall_lift(0.5, 2, 4) == 0.0


def test_spearman_perfect_monotone() -> None:
    assert np.isclose(spearman(np.array([1, 2, 3]), np.array([10, 20, 30])), 1.0)

