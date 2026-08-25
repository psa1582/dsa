import numpy as np

from temporal_dsa.approx import ApproxPolicy, initialize_state, replay_step


def test_full_refresh_matches_baseline_and_counts_every_key() -> None:
    previous = np.linspace(0, 1, 64, dtype=np.float32)
    current = previous[::-1].copy()
    state = initialize_state(previous, k=8, block_size=8)
    state, metrics, details = replay_step(
        state,
        current,
        policy=ApproxPolicy(name="full"),
        k=8,
        block_size=8,
        step=1,
        previous_length=64,
    )
    assert metrics["exact_match"]
    assert metrics["qk_scored"] == 64
    assert np.array_equal(np.sort(details["approximate"]), np.sort(details["baseline"]))


def test_seed_rescore_is_not_double_counted() -> None:
    previous = np.arange(64, dtype=np.float32)
    current = previous.copy()
    state = initialize_state(previous, k=8, block_size=8)
    _, metrics, _ = replay_step(
        state,
        current,
        policy=ApproxPolicy(name="static", gamma=1e9),
        k=8,
        block_size=8,
        step=1,
        previous_length=64,
    )
    assert metrics["qk_scored"] == 64


def test_own_trajectory_never_updates_skipped_block_with_truth() -> None:
    previous = np.zeros(32, dtype=np.float32)
    previous[:4] = 3
    state = initialize_state(previous, k=4, block_size=8)
    current = previous.copy()
    current[16:20] = 10
    state, metrics, _ = replay_step(
        state,
        current,
        policy=ApproxPolicy(name="static", gamma=0),
        k=4,
        block_size=8,
        step=1,
        previous_length=32,
    )
    assert metrics["recall"] == 0
    assert state.last_max[2] == 0  # Current ground truth maximum (10) did not leak.
    assert state.age[2] == 1


def test_age_cap_forces_stale_block_refresh() -> None:
    previous = np.zeros(32, dtype=np.float32)
    previous[:4] = 3
    state = initialize_state(previous, k=4, block_size=8)
    current = previous.copy()
    current[16:20] = 10
    policy = ApproxPolicy(name="age", gamma=0, age_cap=1)
    state, first, _ = replay_step(
        state, current, policy=policy, k=4, block_size=8, step=1, previous_length=32
    )
    state, second, _ = replay_step(
        state, current, policy=policy, k=4, block_size=8, step=2, previous_length=32
    )
    assert first["recall"] == 0
    assert second["recall"] == 1


def test_scattered_seed_physical_traffic_is_reported() -> None:
    previous = np.zeros(64, dtype=np.float32)
    previous[::8] = 5
    state = initialize_state(previous, k=8, block_size=8)
    _, metrics, _ = replay_step(
        state,
        previous.copy(),
        policy=ApproxPolicy(name="static", gamma=0),
        k=8,
        block_size=8,
        step=1,
        previous_length=64,
    )
    assert metrics["seed_blocks_touched"] == 8
    assert metrics["physical_b16_blocks"] == 4
