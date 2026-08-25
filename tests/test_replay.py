import numpy as np

from temporal_dsa.replay import replay_transition


def test_large_margin_replay_is_exact() -> None:
    rng = np.random.default_rng(4)
    previous = rng.normal(size=512)
    current = np.concatenate([previous + 0.001 * rng.normal(size=512), [0.0]])
    result = replay_transition(
        previous,
        current,
        k=32,
        block_size=16,
        gamma_sigma=10.0,
        method="dynamic",
        scan_order="previous_hot",
    )
    assert result.exact_match
    assert result.recall == 1.0
    assert 0 <= result.qk_reduction <= 1


def test_oracle_is_exact_and_never_scores_more_than_full() -> None:
    rng = np.random.default_rng(5)
    previous = rng.normal(size=257)
    current = np.concatenate([rng.normal(size=257), [3.0]])
    result = replay_transition(
        previous,
        current,
        k=16,
        block_size=32,
        gamma_sigma=0.0,
        method="oracle",
        scan_order="oracle_current",
    )
    assert result.exact_match
    assert result.qk_scored <= current.size


def test_empirical_zero_margin_reports_miss() -> None:
    previous = np.zeros(64)
    previous[:4] = 2.0
    current = previous.copy()
    current[40:44] = 3.0
    result = replay_transition(
        previous,
        current,
        k=4,
        block_size=8,
        gamma_sigma=0.0,
        method="dynamic",
        scan_order="previous_hot",
    )
    assert not result.exact_match
    assert result.false_cold_blocks > 0

