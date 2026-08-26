from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "streaming_topk_hardware", ROOT / "scripts" / "simulate_streaming_topk_hardware.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_chunk_topr_sweep_never_calls_r_less_than_b_exact() -> None:
    frame = MODULE.architecture_a_sweep()
    assert (frame.worst_case_exact == (frame.effective_local_r >= frame.chunk_size)).all()
    assert frame.loc[frame.worst_case_exact, "candidate_fraction"].eq(1.0).all()
    assert frame.loc[~frame.worst_case_exact, "classification"].eq("APPROXIMATE").all()


def test_dense_score_materialization_is_eliminated_only_in_fused_system() -> None:
    gpu = MODULE.pd.DataFrame(
        [
            {
                "context": 32768,
                "combined_ms": 0.08704,
                "score_ms": 0.011264,
                "topk_ms": 0.063584,
            }
        ]
    )
    frame = MODULE.data_movement(gpu)
    dense = frame[frame.system.str.startswith("GPU")].iloc[0]
    fused = frame[frame.system.str.startswith("Fused")].iloc[0]
    assert dense.dense_score_write_bytes == 32768 * 4
    assert dense.selection_score_read_bytes_lower == 32768 * 4
    assert fused.dense_score_write_bytes == 0
    assert fused.selection_score_read_bytes_upper == 0
    assert fused.final_result_bytes == 2048 * 4


def test_pcie_payload_includes_scores_and_final_ids() -> None:
    gpu = MODULE.pd.DataFrame(
        [
            {"context": n, "topk_ms": 0.05}
            for n in MODULE.CONTEXTS
        ]
    )
    frame = MODULE.pcie_model(gpu)
    row = frame[(frame.context == 32768) & (frame.pcie == "Gen4 x16")].iloc[0]
    assert row.gpu_to_fpga_score_bytes == 32768 * 4
    assert row.fpga_to_gpu_index_bytes == 2048 * 4
    assert row.verdict == "NO-GO"
