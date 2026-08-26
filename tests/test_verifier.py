import numpy as np

from temporal_dsa.approx import ApproxPolicy, initialize_state
from temporal_dsa.verifier import VerifierConfig, verifier_step


def _config(rescue_fraction: float = 0.5) -> VerifierConfig:
    return VerifierConfig(
        name="head8",
        path="head",
        width=8,
        score_ratio=0.125,
        block_size=8,
        rescue_fraction=rescue_fraction,
    )


def test_verifier_rescues_newly_active_block_without_duplicate_scoring() -> None:
    previous = np.zeros(32, dtype=np.float32)
    previous[:4] = 3
    current = previous.copy()
    current[16:20] = 10
    partial = np.zeros_like(current)
    partial[16:20] = 5
    state = initialize_state(previous, k=4, block_size=8)
    state, metrics, details = verifier_step(
        state,
        current,
        partial,
        policy=ApproxPolicy(name="static", gamma=0),
        config=_config(),
        k=4,
        step=1,
        previous_length=32,
    )
    assert metrics["recall"] == 1
    assert metrics["newly_active_token_recall"] == 1
    assert metrics["duplicate_exact_tokens"] == 0
    assert 2 in details["rescued_blocks"]
    assert state.last_max[2] == 10
    assert state.age[2] == 0


def test_unrescued_block_does_not_leak_current_score() -> None:
    previous = np.zeros(32, dtype=np.float32)
    previous[:4] = 3
    current = previous.copy()
    current[16:20] = 10
    state = initialize_state(previous, k=4, block_size=8)
    state, metrics, _ = verifier_step(
        state,
        current,
        np.zeros_like(current),
        policy=ApproxPolicy(name="static", gamma=0),
        config=_config(rescue_fraction=0),
        k=4,
        step=1,
        previous_length=32,
    )
    assert metrics["recall"] == 0
    assert state.last_max[2] == 0
    assert state.age[2] == 1


def test_cost_includes_verifier_and_candidate_rerank() -> None:
    previous = np.arange(32, dtype=np.float32)
    state = initialize_state(previous, k=4, block_size=8)
    _, metrics, _ = verifier_step(
        state,
        previous.copy(),
        previous.copy(),
        policy=ApproxPolicy(name="static", gamma=1e9),
        config=_config(rescue_fraction=0.5),
        k=4,
        step=1,
        previous_length=32,
    )
    expected = (
        metrics["temporal_exact_tokens"]
        + metrics["rescue_exact_tokens"]
        + metrics["cold_tokens_scanned"] * 0.125
    )
    assert metrics["qk_full_equivalent"] == expected
    assert metrics["candidate_union_tokens"] == (
        metrics["temporal_exact_tokens"] + metrics["rescue_exact_tokens"]
    )
    assert metrics["physical_b64_key_byte_reduction"] <= 0


def test_fixed_threshold_promotes_all_blocks_at_or_above_cutoff() -> None:
    previous = np.zeros(32, dtype=np.float32)
    previous[:4] = 3
    current = previous.copy()
    current[16:20] = 10
    verifier = np.zeros_like(current)
    verifier[8:16] = 4
    verifier[16:24] = 5
    state = initialize_state(previous, k=4, block_size=8)
    _, metrics, details = verifier_step(
        state,
        current,
        verifier,
        policy=ApproxPolicy(name="static", gamma=0),
        config=_config(rescue_fraction=0),
        k=4,
        step=1,
        previous_length=32,
        promotion_threshold=5,
    )
    assert metrics["promotion_policy"] == "fixed_threshold"
    assert metrics["promotion_threshold"] == 5
    assert np.array_equal(details["rescued_blocks"], np.asarray([2]))


def test_fixed_threshold_tie_break_is_block_address_stable() -> None:
    values = np.zeros(32, dtype=np.float32)
    values[:4] = 3
    state = initialize_state(values, k=4, block_size=8)
    _, _, details = verifier_step(
        state,
        values,
        np.ones_like(values),
        policy=ApproxPolicy(name="static", gamma=0),
        config=_config(rescue_fraction=0),
        k=4,
        step=1,
        previous_length=32,
        promotion_threshold=1,
    )
    assert np.array_equal(details["rescued_blocks"], np.arange(1, 4))
