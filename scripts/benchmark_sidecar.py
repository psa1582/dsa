from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from temporal_dsa.sidecar import LightningIndexerSidecar
from temporal_dsa.training import load_capture


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[8192, 16384, 32768])
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    sample = load_capture(args.capture)
    indexer = LightningIndexerSidecar().to(device=device, dtype=torch.float32)
    indexer.load_state_dict(load_file(args.checkpoint, device=str(device)))
    indexer.eval()
    hidden = sample.hidden.to(device=device, dtype=torch.bfloat16).unsqueeze(0)
    positions = torch.arange(hidden.shape[1], device=device).view(1, -1)
    rows = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        encoded_keys = indexer.encode_keys(hidden, positions)
        for length in args.lengths:
            if length > hidden.shape[1]:
                raise ValueError(f"capture has only {hidden.shape[1]} hidden states")
            query_hidden = hidden[:, length - 1 : length]
            query_position = torch.tensor([[length - 1]], device=device)
            queries, weights = indexer.encode_queries(query_hidden, query_position)

            def operation() -> None:
                scores = indexer.score_encoded(queries, weights, encoded_keys[:, :length])
                torch.topk(scores, min(args.k, length), dim=-1)

            for _ in range(args.warmups):
                operation()
            torch.cuda.synchronize()
            timings = []
            for _ in range(args.iterations):
                started = time.perf_counter()
                operation()
                torch.cuda.synchronize()
                timings.append((time.perf_counter() - started) * 1000)
            rows.append(
                {
                    "context_length": length,
                    "k": min(args.k, length),
                    "warmups": args.warmups,
                    "iterations": args.iterations,
                    "latency_ms_mean": statistics.mean(timings),
                    "latency_ms_median": statistics.median(timings),
                    "latency_ms_p5": percentile(timings, 0.05),
                    "latency_ms_p95": percentile(timings, 0.95),
                    "key_bytes_bf16": length * 128 * 2,
                }
            )
    payload = {
        "benchmark": "dense BF16 research-sidecar score plus torch.topk; cached keys",
        "not_a_sparse_kernel": True,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "capture": str(args.capture),
        "checkpoint": str(args.checkpoint),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

