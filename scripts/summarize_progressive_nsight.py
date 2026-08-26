from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize progressive DSA Nsight Systems NVTX ranges")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    rows = []
    for context in (16384, 32768):
        path = args.input / f"nvtx_{context // 1024}k_nvtx_gpu_proj_sum.csv"
        frame = pd.read_csv(path)
        frame = frame[frame.Range.str.contains(f"_c{context}", regex=False)]
        for row in frame.itertuples(index=False):
            range_name = str(row.Range).lstrip(":")
            method = range_name.rsplit("_c", 1)[0]
            projected_ns = float(getattr(row, "_2"))
            range_ns = float(getattr(row, "_3"))
            gpu_ops = int(getattr(row, "_10"))
            rows.append(
                {
                    "context": context,
                    "method": method,
                    "iterations": args.iterations,
                    "gpu_time_ms": projected_ns / 1e6,
                    "gpu_time_per_iteration_us": projected_ns / args.iterations / 1e3,
                    "nvtx_wall_time_per_iteration_us": range_ns / args.iterations / 1e3,
                    "host_gap_per_iteration_us": (range_ns - projected_ns) / args.iterations / 1e3,
                    "total_gpu_ops": gpu_ops,
                    "gpu_ops_per_iteration": gpu_ops / args.iterations,
                    "memcpy_in_profiled_range": 0,
                    "dram_read_bytes": "unavailable_ncu_not_installed",
                    "dram_write_bytes": "unavailable_ncu_not_installed",
                    "l2_read_bytes": "unavailable_ncu_not_installed",
                    "l2_write_bytes": "unavailable_ncu_not_installed",
                    "sm_utilization": "unavailable_ncu_not_installed",
                    "achieved_occupancy": "unavailable_ncu_not_installed",
                }
            )
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output / "nsight_summary.csv", index=False)


if __name__ == "__main__":
    main()
