import numpy as np

from temporal_dsa.certified import block_key_norms, relu_index_score, temporal_score_radius


def test_temporal_radius_covers_every_key_with_signed_weights() -> None:
    rng = np.random.default_rng(1582)
    keys = rng.normal(size=(37, 8, 16))
    q_previous = rng.normal(size=(8, 16))
    q_current = q_previous + 0.05 * rng.normal(size=(8, 16))
    w_previous = rng.normal(size=8)
    w_current = w_previous + 0.05 * rng.normal(size=8)
    # The production score has one shared key vector across heads.  Use head 0
    # repeated only for the scalar reference and preserve per-head norms below.
    shared = keys[:, 0]
    previous = relu_index_score(q_previous, shared, w_previous)
    current = relu_index_score(q_current, shared, w_current)
    repeated = np.repeat(shared[:, None, :], 8, axis=1)
    norms = block_key_norms(repeated, 7)
    radius = temporal_score_radius(q_current, q_previous, w_current, w_previous, norms)
    for block, start in enumerate(range(0, shared.shape[0], 7)):
        observed = np.abs(current[start : start + 7] - previous[start : start + 7])
        assert np.all(observed <= radius[block] + 1e-10)
