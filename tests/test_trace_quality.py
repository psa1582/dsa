from pathlib import Path

import numpy as np
import pandas as pd

from temporal_dsa.quality import apply_quality_gate
from temporal_dsa.trace import ScoreTrace, load_trace, save_trace


def test_trace_round_trip(tmp_path: Path) -> None:
    trace = ScoreTrace(
        scores=np.arange(18, dtype=np.float16).reshape(3, 6),
        lengths=np.array([4, 5, 6]),
        layer=2,
        workload="code",
        prompt_id="p0",
        metadata={"seed": 1},
    )
    path = tmp_path / "trace.npz"
    save_trace(path, trace)
    loaded = load_trace(path)
    assert loaded.layer == 2
    assert np.array_equal(loaded.row(1), np.array([6, 7, 8, 9, 10], dtype=np.float32))


def test_quality_gate_lists_failed_layer() -> None:
    rows = pd.DataFrame(
        {
            "layer": [2, 2, 5, 5],
            "k": [512, 512, 512, 512],
            "normalized_lift": [0.8, 0.7, 0.0, 0.1],
        }
    )
    gate = apply_quality_gate(rows, min_layer_pass_fraction=0.5)
    assert gate.passed
    assert gate.failed_layers == [5]

