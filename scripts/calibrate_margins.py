from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from temporal_dsa.trace import load_trace


def block_max(values: np.ndarray, block_size: int) -> np.ndarray:
    return np.maximum.reduceat(values, np.arange(0, values.size, block_size))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[16, 32, 64, 128])
    args = parser.parse_args()
    observations: dict[tuple[int, int], list[np.ndarray]] = {}
    trace_paths = sorted(args.trace_root.rglob("*.npz"))
    for path in trace_paths:
        trace = load_trace(path)
        for step in range(1, trace.scores.shape[0]):
            previous = trace.row(step - 1)
            current = trace.row(step)[: previous.size]
            for block_size in args.block_sizes:
                delta = block_max(current, block_size) - block_max(previous, block_size)
                observations.setdefault((trace.layer, block_size), []).append(delta)
    margins: dict[str, dict[str, float]] = {}
    diagnostics = []
    for (layer, block_size), chunks in sorted(observations.items()):
        values = np.concatenate(chunks).astype(np.float64)
        positive = np.maximum(values, 0.0)
        margin = float(np.max(positive))
        margins.setdefault(str(layer), {})[str(block_size)] = margin
        diagnostics.append(
            {
                "layer": layer,
                "block_size": block_size,
                "observations": int(values.size),
                "positive_max": margin,
                "positive_p99": float(np.percentile(positive, 99)),
                "positive_p999": float(np.percentile(positive, 99.9)),
            }
        )
    payload = {
        "policy": "maximum positive adjacent block-max increase on validation traces",
        "mathematical_certificate": False,
        "trace_root": str(args.trace_root),
        "trace_count": len(trace_paths),
        "margins": margins,
        "diagnostics": diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

