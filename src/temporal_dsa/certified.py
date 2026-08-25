from __future__ import annotations

import numpy as np


def relu_index_score(q: np.ndarray, keys: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Reference I_s = sum_j w_j ReLU(q_j dot k_s)."""
    q = np.asarray(q, dtype=np.float64)
    keys = np.asarray(keys, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if q.ndim != 2 or keys.ndim != 2 or q.shape[1] != keys.shape[1]:
        raise ValueError("q must be [heads,dim] and keys [tokens,dim]")
    if weights.shape != (q.shape[0],):
        raise ValueError("weights must have one value per head")
    return (np.maximum(q @ keys.T, 0.0) * weights[:, None]).sum(axis=0)


def temporal_score_radius(
    q_now: np.ndarray,
    q_prev: np.ndarray,
    w_now: np.ndarray,
    w_prev: np.ndarray,
    block_key_norm: np.ndarray,
) -> np.ndarray:
    """Certified |I_t-I_(t-1)| radius for every key block.

    Negative indexer weights are handled with absolute values.  The bound uses
    ReLU's 1-Lipschitz property and Cauchy-Schwarz:

      |w_t ReLU(q_t k) - w_p ReLU(q_p k)|
      <= (|w_t-w_p| ||q_p|| + |w_t| ||q_t-q_p||) ||k||.
    """
    q_now = np.asarray(q_now, dtype=np.float64)
    q_prev = np.asarray(q_prev, dtype=np.float64)
    w_now = np.asarray(w_now, dtype=np.float64)
    w_prev = np.asarray(w_prev, dtype=np.float64)
    block_key_norm = np.asarray(block_key_norm, dtype=np.float64)
    if q_now.shape != q_prev.shape or q_now.ndim != 2:
        raise ValueError("query tensors must have shape [heads,dim]")
    if w_now.shape != w_prev.shape or w_now.shape != (q_now.shape[0],):
        raise ValueError("weight tensors must have shape [heads]")
    if block_key_norm.ndim != 2 or block_key_norm.shape[1] != q_now.shape[0]:
        raise ValueError("block_key_norm must have shape [blocks,heads]")
    coefficient = (
        np.abs(w_now - w_prev) * np.linalg.norm(q_prev, axis=1)
        + np.abs(w_now) * np.linalg.norm(q_now - q_prev, axis=1)
    )
    return block_key_norm @ coefficient


def block_key_norms(keys: np.ndarray, block_size: int) -> np.ndarray:
    """Maximum key L2 norm per block and head for keys [tokens,heads,dim]."""
    keys = np.asarray(keys, dtype=np.float64)
    if keys.ndim != 3:
        raise ValueError("keys must have shape [tokens,heads,dim]")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    rows = []
    for start in range(0, keys.shape[0], block_size):
        rows.append(np.linalg.norm(keys[start : start + block_size], axis=2).max(axis=0))
    return np.stack(rows)

